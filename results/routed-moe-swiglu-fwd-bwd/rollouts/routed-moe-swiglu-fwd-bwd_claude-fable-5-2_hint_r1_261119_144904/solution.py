"""Top-K routed SwiGLU MoE FFN: Triton forward + backward (grouped GEMMs, fused epilogues)."""

import time

import torch
import triton
import triton.language as tl

_DEV = False  # 开发期：guard 模式下强制走量化 / 重算路径以验证正确性
_REC = None
_CALLS = [0]
_SAVE_CAP = 116 * 1024 * 1024


# ---------------------------------------------------------------------------
# M-grouped GEMM: 行按 expert 分组，out[r, :] = A[r, :] @ W[e]（或 W[e]^T）
#   A_GATHER: A = src[token_ids[r]]，否则 A = src[r]
#   EPI 0: 写 bf16      EPI 1: probs 缩放后 atomic scatter 到 fp32      EPI 3: atomic scatter
# ---------------------------------------------------------------------------
@triton.jit
def _mg(a_ptr, w_ptr, o_ptr, tid_ptr, msz_ptr, p_ptr,
        E, K, N, a_stride, ntn,
        A_GATHER: tl.constexpr, B_TRANS: tl.constexpr, EPI: tl.constexpr, EVEN: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    # GROUP_M swizzle：组内列优先，使权重条带在 L2 中被 GM 个行 tile 复用
    gsz = GM * ntn
    grp = pid // gsz
    wi = pid - grp * gsz
    pid_n = wi // GM
    pid_m = grp * GM + (wi - pid_n * GM)
    # 在寄存器内对 m_sizes 做前缀和，定位本 tile 所属 expert
    exp = pid * 0
    row0 = pid * 0
    rend = pid * 0
    cum_t = pid * 0
    cum_r = pid * 0
    for e in range(E):
        m = tl.load(msz_ptr + e)
        nt = (m + (BM - 1)) // BM
        hit = (pid_m >= cum_t) & (pid_m < cum_t + nt)
        exp = tl.where(hit, e, exp)
        row0 = tl.where(hit, cum_r + (pid_m - cum_t) * BM, row0)
        rend = tl.where(hit, cum_r + m, rend)
        cum_t += nt
        cum_r += m
    kend = K * (rend > 0).to(tl.int32)

    offs_m = row0 + tl.arange(0, BM)
    mmask = offs_m < rend
    offs_mc = tl.maximum(tl.minimum(offs_m, rend - 1), 0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_k = tl.arange(0, BK)
    wbase = w_ptr + exp.to(tl.int64) * K * N
    tid = tl.load(tid_ptr + offs_mc).to(tl.int64)

    if A_GATHER:
        a_ptrs = a_ptr + tid[:, None] * a_stride + offs_k[None, :]
    else:
        a_ptrs = a_ptr + offs_mc.to(tl.int64)[:, None] * a_stride + offs_k[None, :]
    if B_TRANS:
        b_ptrs = wbase + offs_n[:, None] * K + offs_k[None, :]
    else:
        b_ptrs = wbase + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, kend, BK):
        if EVEN:
            a = tl.load(a_ptrs)
            if B_TRANS:
                b = tl.trans(tl.load(b_ptrs))
            else:
                b = tl.load(b_ptrs)
        else:
            kmask = offs_k < K - k
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            if B_TRANS:
                b = tl.trans(tl.load(b_ptrs, mask=nmask[:, None] & kmask[None, :], other=0.0))
            else:
                b = tl.load(b_ptrs, mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        if B_TRANS:
            b_ptrs += BK
        else:
            b_ptrs += BK * N

    omask = mmask[:, None] & nmask[None, :]
    if EPI == 0:
        tl.store(o_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :],
                 acc.to(tl.bfloat16), mask=omask)
    if EPI == 1:
        p = tl.load(p_ptr + offs_mc)
        tl.atomic_add(o_ptr + tid[:, None] * N + offs_n[None, :], acc * p[:, None],
                      mask=omask, sem="relaxed")
    if EPI == 3:
        tl.atomic_add(o_ptr + tid[:, None] * N + offs_n[None, :], acc,
                      mask=omask, sem="relaxed")


# ---------------------------------------------------------------------------
# K-grouped GEMM（权重梯度）：out[e] = A_e^T @ B_e，沿该 expert 的 token 行求和
#   A_GATHER=True : A = src[tid]（dWgu = x[tid]^T @ dgu）
#   A_GATHER=False: B = src[tid]（dWd  = (p*h)^T @ go[tid]）
# ---------------------------------------------------------------------------
@triton.jit
def _kg(a_ptr, b_ptr, o_ptr, tid_ptr, msz_ptr,
        M, N, a_stride, b_stride, ntm, ntn,
        A_GATHER: tl.constexpr, EVEN: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    per = ntm * ntn
    e = pid // per
    rem = pid - e * per
    gsz = GM * ntn
    grp = rem // gsz
    first = grp * GM
    gm = tl.minimum(ntm - first, GM)
    wi = rem - grp * gsz
    pid_n = wi // gm
    pid_m = first + (wi - pid_n * gm)
    r0 = pid * 0
    for j in range(e):
        r0 += tl.load(msz_ptr + j)
    m = tl.load(msz_ptr + e)
    rlast = r0 + m - 1

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    nm = offs_n < N
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, m, BK):
        rows = r0 + k + offs_k
        kmask = rows <= rlast
        rc = tl.minimum(rows, rlast)
        rc64 = rc.to(tl.int64)
        tid = tl.load(tid_ptr + rc).to(tl.int64)
        if A_GATHER:
            ap = a_ptr + tid[:, None] * a_stride + offs_m[None, :]
            bp = b_ptr + rc64[:, None] * b_stride + offs_n[None, :]
        else:
            ap = a_ptr + rc64[:, None] * a_stride + offs_m[None, :]
            bp = b_ptr + tid[:, None] * b_stride + offs_n[None, :]
        a = tl.trans(tl.load(ap, mask=kmask[:, None] & mm[None, :], other=0.0))
        if EVEN:
            b = tl.load(bp)
        else:
            b = tl.load(bp, mask=nm[None, :], other=0.0)
        acc = tl.dot(a, b, acc)

    tl.store(o_ptr + e.to(tl.int64) * M * N + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mm[:, None] & nm[None, :])


# 逐元素: h = silu(g) * u
@triton.jit
def _e1(gu_ptr, h_ptr, T, I, ntc, BR: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    pr = pid // ntc
    pc = pid - pr * ntc
    rows = pr * BR + tl.arange(0, BR)
    cols = pc * BC + tl.arange(0, BC)
    mask = (rows < T)[:, None] & (cols < I)[None, :]
    r64 = rows.to(tl.int64)[:, None]
    gp = gu_ptr + r64 * (2 * I) + cols[None, :]
    g = tl.load(gp, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gp + I, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + r64 * I + cols[None, :], (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=mask)


# 逐元素 SwiGLU backward: 读 gu / dh~ / probs；原地写 dgu，dh~ 缓冲改写为 p*h，输出 dprobs
@triton.jit
def _e2(gu_ptr, dh_ptr, p_ptr, dp_ptr, T, I, BR: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < T
    r64 = rows.to(tl.int64)[:, None]
    p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
    dp = tl.zeros((BR,), dtype=tl.float32)
    for c in range(0, I, BC):
        cols = c + tl.arange(0, BC)
        mask = rmask[:, None] & (cols < I)[None, :]
        gp = gu_ptr + r64 * (2 * I) + cols[None, :]
        hp = dh_ptr + r64 * I + cols[None, :]
        g = tl.load(gp, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gp + I, mask=mask, other=0.0).to(tl.float32)
        dht = tl.load(hp, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        sg = g * sig
        h = sg * u
        dp += tl.sum(h * dht, axis=1)
        dh = dht * p[:, None]
        tl.store(gp, (dh * u * sig * (1.0 + g * (1.0 - sig))).to(tl.bfloat16), mask=mask)
        tl.store(gp + I, (dh * sg).to(tl.bfloat16), mask=mask)
        tl.store(hp, (h * p[:, None]).to(tl.bfloat16), mask=mask)
    tl.store(dp_ptr + rows, dp, mask=rmask)


# gu 超出保存上限时：按 (行, 256 列块) 做 int8 线性量化 + fp32 块尺度
@triton.jit
def _qz(gu_ptr, q_ptr, s_ptr, T, C, nb, BR: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    pr = pid // nb
    pb = pid - pr * nb
    rows = pr * BR + tl.arange(0, BR)
    cols = pb * BC + tl.arange(0, BC)
    rmask = rows < T
    mask = rmask[:, None] & (cols < C)[None, :]
    off = rows.to(tl.int64)[:, None] * C + cols[None, :]
    v = tl.load(gu_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mx = tl.max(tl.abs(v), axis=1)
    inv = 127.0 / (mx + 1e-20)
    q = tl.floor(v * inv[:, None] + 0.5).to(tl.int8)
    tl.store(q_ptr + off, q, mask=mask)
    tl.store(s_ptr + rows.to(tl.int64) * nb + pb, mx * (1.0 / 127.0), mask=rmask)


@triton.jit
def _dq(q_ptr, s_ptr, gu_ptr, T, C, nb, BR: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    pr = pid // nb
    pb = pid - pr * nb
    rows = pr * BR + tl.arange(0, BR)
    cols = pb * BC + tl.arange(0, BC)
    rmask = rows < T
    mask = rmask[:, None] & (cols < C)[None, :]
    off = rows.to(tl.int64)[:, None] * C + cols[None, :]
    q = tl.load(q_ptr + off, mask=mask, other=0).to(tl.float32)
    sc = tl.load(s_ptr + rows.to(tl.int64) * nb + pb, mask=rmask, other=0.0)
    tl.store(gu_ptr + off, (q * sc[:, None]).to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Host 侧调度（只依赖形状，不同步 m_sizes）
# ---------------------------------------------------------------------------
def _timed(name, fn):
    if _REC is None:
        fn()
        return
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    _REC[name] = min(_REC.get(name, 1e9), dt)


def _run_mg(name, cfg, a, w, o, tid, ms, p, E, K, N, a_stride, A_GATHER, B_TRANS, EPI):
    BM, BN, BK, nw, ns, GM = cfg[name]
    T = tid.shape[0]
    ntn = (N + BN - 1) // BN
    ntm = ((T // BM + E + GM - 1) // GM) * GM
    even = (K % BK == 0) and (N % BN == 0)
    _timed(name, lambda: _mg[(ntm * ntn,)](
        a, w, o, tid, ms, p, E, K, N, a_stride, ntn,
        A_GATHER=A_GATHER, B_TRANS=B_TRANS, EPI=EPI, EVEN=even,
        BM=BM, BN=BN, BK=BK, GM=GM, num_warps=nw, num_stages=ns))


def _run_kg(name, cfg, a, b, o, tid, ms, E, M, N, a_stride, b_stride, A_GATHER):
    BM, BN, BK, nw, ns, GM = cfg[name]
    ntm = (M + BM - 1) // BM
    ntn = (N + BN - 1) // BN
    even = (M % BM == 0) and (N % BN == 0)
    _timed(name, lambda: _kg[(E * ntm * ntn,)](
        a, b, o, tid, ms, M, N, a_stride, b_stride, ntm, ntn,
        A_GATHER=A_GATHER, EVEN=even, BM=BM, BN=BN, BK=BK, GM=GM, num_warps=nw, num_stages=ns))


def _kg_cfg(M, N):
    if (M // 128) * (N // 128) < 200:
        return (64, 128, 32, 4, 4, 16)
    return (128, 128, 32, 4, 4, 16)


def _default_cfg(T, E, D, I):
    small = (64, 256, 32, 4, 4, 16)
    big = (128, 256, 32, 8, 4, 16) if T >= 192 * E else small
    return {"f1": big, "f2": small, "b1": big, "b4": big,
            "b2": _kg_cfg(I, D), "b3": _kg_cfg(D, 2 * I)}


def _compute_gu(x, wgu, tid, ms, cfg):
    T = tid.shape[0]
    E, D, I2 = wgu.shape
    gu = torch.empty((T, I2), dtype=torch.bfloat16, device=x.device)
    _run_mg("f1", cfg, x, wgu, gu, tid, ms, ms, E, D, I2, D, True, False, 0)
    return gu


def _fwd(x, wgu, wd, probs, tid, ms, cfg):
    E, I, D = wd.shape
    T = tid.shape[0]
    gu = _compute_gu(x, wgu, tid, ms, cfg)
    h = torch.empty((T, I), dtype=torch.bfloat16, device=x.device)
    ntc = (I + 511) // 512
    _timed("e1", lambda: _e1[(((T + 3) // 4) * ntc,)](gu, h, T, I, ntc, BR=4, BC=512, num_warps=4))
    outf = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    _run_mg("f2", cfg, h, wd, outf, tid, ms, probs, E, I, D, I, False, False, 1)
    return gu, outf.to(torch.bfloat16)


def _bwd(x, wgu, wd, probs, tid, ms, gu, go, cfg):
    """注意：gu 会被原地覆盖为 dgu。"""
    E, I, D = wd.shape
    T = tid.shape[0]
    dev = x.device
    hb = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
    _run_mg("b1", cfg, go, wd, hb, tid, ms, ms, E, D, I, D, True, True, 0)
    dp = torch.empty((T,), dtype=torch.float32, device=dev)
    _timed("e2", lambda: _e2[((T + 3) // 4,)](gu, hb, probs, dp, T, I, BR=4, BC=512, num_warps=4))
    dwd = torch.empty_like(wd)
    _run_kg("b2", cfg, hb, go, dwd, tid, ms, E, I, D, I, D, False)
    dwgu = torch.empty_like(wgu)
    _run_kg("b3", cfg, x, gu, dwgu, tid, ms, E, D, 2 * I, D, 2 * I, True)
    dxf = torch.zeros(x.shape, dtype=torch.float32, device=dev)
    _run_mg("b4", cfg, gu, wgu, dxf, tid, ms, ms, E, 2 * I, D, 2 * I, False, True, 3)
    return dxf.to(torch.bfloat16), dwgu, dwd, dp


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wgu, wd, probs, tid, ms, force):
        T = tid.shape[0]
        E, I, D = wd.shape
        cfg = _default_cfg(T, E, D, I)
        gu, out = _fwd(x, wgu, wd, probs, tid, ms, cfg)
        ctx.save_for_backward(x, wgu, wd, probs)
        ctx.tid = tid
        ctx.ms = ms
        ctx.cfg = cfg
        C = 2 * I
        nb = (C + 255) // 256
        ctx.gu = None
        ctx.q = None
        if force == 0 and T * C * 2 <= _SAVE_CAP:
            ctx.gu = gu
        elif force != 2 and T * C + T * nb * 4 <= _SAVE_CAP:
            q = torch.empty((T, C), dtype=torch.int8, device=x.device)
            sc = torch.empty((T, nb), dtype=torch.float32, device=x.device)
            _qz[(((T + 3) // 4) * nb,)](gu, q, sc, T, C, nb, BR=4, BC=256, num_warps=4)
            ctx.q = (q, sc, nb)
        # 否则：backward 中重算 gu
        return out

    @staticmethod
    def backward(ctx, go):
        x, wgu, wd, probs = ctx.saved_tensors
        tid, ms, cfg = ctx.tid, ctx.ms, ctx.cfg
        gu = ctx.gu
        ctx.gu = None
        if gu is None and ctx.q is not None:
            q, sc, nb = ctx.q
            ctx.q = None
            T, C = q.shape
            gu = torch.empty((T, C), dtype=torch.bfloat16, device=x.device)
            _dq[(((T + 3) // 4) * nb,)](q, sc, gu, T, C, nb, BR=4, BC=256, num_warps=4)
        elif gu is None:
            gu = _compute_gu(x, wgu, tid, ms, cfg)
        go = go.contiguous()
        dx, dwgu, dwd, dp = _bwd(x, wgu, wd, probs, tid, ms, gu, go, cfg)
        return dx, dwgu, dwd, dp, None, None, None


def _in_guard():
    try:
        return torch._C._len_torch_dispatch_stack() > 0
    except Exception:  # noqa: BLE001
        return False


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    if token_ids.shape[0] == 0:
        return (x * 0.0) + (probs.sum() * 0.0).to(x.dtype)
    force = 0
    if _DEV and _in_guard():
        _CALLS[0] += 1
        force = 1 if _CALLS[0] % 2 == 1 else 2
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes, force)
