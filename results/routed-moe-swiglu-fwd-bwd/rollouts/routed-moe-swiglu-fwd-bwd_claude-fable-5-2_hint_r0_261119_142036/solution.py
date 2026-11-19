"""Top-K Routed SwiGLU MoE FFN: 前向 + 反向全部用 Triton 实现。

设计要点:
  * 前向只保存 gu = [g | u] ([T, 2I], bf16);h 在反向里由 gu 重建。
  * probs.grad 用伴随恒等式 <go, h W_down> = <h, go W_down^T> 在 dh 核的 epilogue 中得到。
  * m_sizes 的前缀和全部在 GPU 上完成(torch.cumsum + 核内查找),没有任何主机同步。
  * 权重转置在核内完成(按 K 方向连续加载后 tl.trans)。
  * 输出 / dx 用 fp32 缓冲区 + 原子加完成 scatter。
"""

import torch
import triton
import triton.language as tl

_SAVE_CAP_BYTES = 116 * 1024 * 1024
_PROF = None
_BK = 32
_ST = 3
_WBK = 32
_WST = 3
_WW = 4


def _tick(name):
    if _PROF is not None:
        torch.cuda.synchronize()
        import time as _tm
        _PROF.append((name, _tm.perf_counter()))



# ----------------------------------------------------------------------------
# 核 A: gu = x[tid] @ Wgu[e]  (M 分组),同时输出 h = silu(g) * u
# ----------------------------------------------------------------------------
@triton.jit
def _k_fwd_gu(
    x_ptr, tid_ptr, w_ptr, gu_ptr, h_ptr, m_ptr, mend_ptr, tend_ptr,
    E, D, I, nM, nN,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EP: tl.constexpr,
    G: tl.constexpr, KMASK: tl.constexpr, NSHIFT: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = G * nN
    gid = pid // npg
    first = gid * G
    gsz = tl.minimum(nM - first, G)
    pid_m = first + ((pid % npg) % gsz)
    pid_n = (pid % npg) // gsz

    eidx = tl.arange(0, EP)
    tend = tl.load(tend_ptr + eidx, mask=eidx < E, other=1073741824)
    e_raw = tl.sum((tend <= pid_m).to(tl.int32), axis=0)
    valid = e_raw < E
    e = tl.minimum(e_raw, E - 1)
    m_e = tl.load(m_ptr + e)
    r_end = tl.load(mend_ptr + e)
    t_end = tl.load(tend_ptr + e)
    tiles_e = (m_e + BM - 1) // BM
    row0 = r_end - m_e + (pid_m - (t_end - tiles_e)) * BM
    rows = row0 + tl.arange(0, BM)
    rmask = (rows < r_end) & valid
    rows_c = tl.maximum(tl.minimum(rows, r_end - 1), 0)

    tid = tl.load(tid_ptr + rows_c)
    a_base = x_ptr + tid[:, None] * D
    I2 = 2 * I
    if NSHIFT:
        n0 = tl.minimum(pid_n * BN, I - BN)
    else:
        n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w_e = w_ptr + e.to(tl.int64) * D * I2

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    kiters = tl.cdiv(D, BK) * valid.to(tl.int32)
    for k in range(0, kiters):
        ks = k * BK + offs_k
        bp = w_e + ks[:, None] * I2 + offs_n[None, :]
        if KMASK:
            km = ks < D
            a = tl.load(a_base + ks[None, :], mask=km[None, :], other=0.0)
            bg = tl.load(bp, mask=km[:, None], other=0.0)
            bu = tl.load(bp + I, mask=km[:, None], other=0.0)
        else:
            a = tl.load(a_base + ks[None, :])
            bg = tl.load(bp)
            bu = tl.load(bp + I)
        acc_g += tl.dot(a, bg)
        acc_u += tl.dot(a, bu)

    sig = 1.0 / (1.0 + tl.exp(-acc_g))
    h = acc_g * sig * acc_u
    smask = rmask[:, None]
    gp = gu_ptr + rows[:, None] * I2 + offs_n[None, :]
    tl.store(gp, acc_g.to(tl.bfloat16), mask=smask)
    tl.store(gp + I, acc_u.to(tl.bfloat16), mask=smask)
    tl.store(h_ptr + rows[:, None] * I + offs_n[None, :], h.to(tl.bfloat16), mask=smask)


# ----------------------------------------------------------------------------
# 核 B/F: M 分组 GEMM + scatter 原子加
#   TRANSB=False: buf[tid] += p * (A @ W[e]),      W[e] 为 [K, N]
#   TRANSB=True : buf[tid] +=      A @ W[e]^T,     W[e] 为 [N, K]
# ----------------------------------------------------------------------------
@triton.jit
def _k_scatter(
    a_ptr, w_ptr, buf_ptr, p_ptr, tid_ptr, m_ptr, mend_ptr, tend_ptr,
    E, K, N, nM, nN,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EP: tl.constexpr,
    G: tl.constexpr, TRANSB: tl.constexpr, SCALE: tl.constexpr, MODE: tl.constexpr,
    KMASK: tl.constexpr, NSHIFT: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = G * nN
    gid = pid // npg
    first = gid * G
    gsz = tl.minimum(nM - first, G)
    pid_m = first + ((pid % npg) % gsz)
    pid_n = (pid % npg) // gsz

    eidx = tl.arange(0, EP)
    tend = tl.load(tend_ptr + eidx, mask=eidx < E, other=1073741824)
    e_raw = tl.sum((tend <= pid_m).to(tl.int32), axis=0)
    valid = e_raw < E
    e = tl.minimum(e_raw, E - 1)
    m_e = tl.load(m_ptr + e)
    r_end = tl.load(mend_ptr + e)
    t_end = tl.load(tend_ptr + e)
    tiles_e = (m_e + BM - 1) // BM
    row0 = r_end - m_e + (pid_m - (t_end - tiles_e)) * BM
    rows = row0 + tl.arange(0, BM)
    rmask = (rows < r_end) & valid
    rows_c = tl.maximum(tl.minimum(rows, r_end - 1), 0)

    a_base = a_ptr + rows_c[:, None] * K
    if NSHIFT:
        n0 = tl.minimum(pid_n * BN, N - BN)
    else:
        n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w_e = w_ptr + e.to(tl.int64) * K * N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kiters = tl.cdiv(K, BK) * valid.to(tl.int32)
    for k in range(0, kiters):
        ks = k * BK + offs_k
        if TRANSB:
            bp = w_e + offs_n[:, None] * K + ks[None, :]
            if KMASK:
                km = ks < K
                a = tl.load(a_base + ks[None, :], mask=km[None, :], other=0.0)
                b = tl.load(bp, mask=km[None, :], other=0.0)
            else:
                a = tl.load(a_base + ks[None, :])
                b = tl.load(bp)
            acc += tl.dot(a, tl.trans(b))
        else:
            bp = w_e + ks[:, None] * N + offs_n[None, :]
            if KMASK:
                km = ks < K
                a = tl.load(a_base + ks[None, :], mask=km[None, :], other=0.0)
                b = tl.load(bp, mask=km[:, None], other=0.0)
            else:
                a = tl.load(a_base + ks[None, :])
                b = tl.load(bp)
            acc += tl.dot(a, b)

    if MODE == 0:
        tid = tl.load(tid_ptr + rows_c)
        if SCALE:
            p = tl.load(p_ptr + rows_c)
            acc = acc * p[:, None]
        smask = rmask[:, None] & (offs_n >= pid_n * BN)[None, :]
        tl.atomic_add(buf_ptr + tid[:, None] * N + offs_n[None, :], acc, mask=smask, sem="relaxed")
    else:
        # MODE 1: 只把未缩放的 dh~ (N == I) 以 bf16 写入 buf_ptr ([T, N])
        tl.store(buf_ptr + rows[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


# ----------------------------------------------------------------------------
# 核 Q: 逐元素 SwiGLU 反向 (单次遍历):
#   h = silu(g) * u, dprobs[r] = <h, dh~>, dh = p * dh~, (dg | du) -> dgu
# ----------------------------------------------------------------------------
@triton.jit
def _k_swiglu_bwd(gu_ptr, d_ptr, p_ptr, dgu_ptr, h_ptr, dp_ptr, T, I,
                  BR: tl.constexpr, BC: tl.constexpr, MASKED: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < T
    I2 = 2 * I
    p = tl.load(p_ptr + rows, mask=rm, other=0.0)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for c in range(0, tl.cdiv(I, BC)):
        cols = c * BC + tl.arange(0, BC)
        off1 = rows[:, None] * I + cols[None, :]
        off2 = rows[:, None] * I2 + cols[None, :]
        if MASKED:
            m = rm[:, None] & (cols < I)[None, :]
            g = tl.load(gu_ptr + off2, mask=m, other=0.0).to(tl.float32)
            u = tl.load(gu_ptr + off2 + I, mask=m, other=0.0).to(tl.float32)
            d = tl.load(d_ptr + off1, mask=m, other=0.0).to(tl.float32)
        else:
            g = tl.load(gu_ptr + off2).to(tl.float32)
            u = tl.load(gu_ptr + off2 + I).to(tl.float32)
            d = tl.load(d_ptr + off1).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        sg = g * sig
        h = sg * u
        acc += tl.sum(h * d, axis=1)
        dh = d * p[:, None]
        du = dh * sg
        dg = dh * u * sig * (1.0 + g * (1.0 - sig))
        if MASKED:
            tl.store(dgu_ptr + off2, dg.to(tl.bfloat16), mask=m)
            tl.store(dgu_ptr + off2 + I, du.to(tl.bfloat16), mask=m)
            tl.store(h_ptr + off1, h.to(tl.bfloat16), mask=m)
        else:
            tl.store(dgu_ptr + off2, dg.to(tl.bfloat16))
            tl.store(dgu_ptr + off2 + I, du.to(tl.bfloat16))
            tl.store(h_ptr + off1, h.to(tl.bfloat16))
    tl.store(dp_ptr + rows, acc, mask=rm)


# ----------------------------------------------------------------------------
# 核 D/E: K 分组权重梯度 out[e] = A[rows_e]^T @ B[rows_e]   ([M, N])
#   GA / GB: 对应操作数的行是否经 token_ids gather
# ----------------------------------------------------------------------------
@triton.jit
def _k_wgrad(
    a_ptr, b_ptr, tid_ptr, out_ptr, m_ptr, mend_ptr,
    sa, sb, M, N, nM, nN,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GA: tl.constexpr, GB: tl.constexpr, MFAST: tl.constexpr,
    MSHIFT: tl.constexpr, NSHIFT: tl.constexpr,
):
    pid = tl.program_id(0)
    e = tl.program_id(1)
    if MFAST:
        pid_m = pid % nM
        pid_n = pid // nM
    else:
        pid_n = pid % nN
        pid_m = pid // nN
    m_e = tl.load(m_ptr + e)
    r_end = tl.load(mend_ptr + e)
    r0 = r_end - m_e
    if MSHIFT:
        m0 = tl.minimum(pid_m * BM, M - BM)
    else:
        m0 = pid_m * BM
    if NSHIFT:
        n0 = tl.minimum(pid_n * BN, N - BN)
    else:
        n0 = pid_n * BN
    offs_m = m0 + tl.arange(0, BM)
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(m_e, BK)):
        rows = r0 + k * BK + offs_k
        km = rows < r_end
        rows_c = tl.minimum(rows, r_end - 1)
        t = tl.load(tid_ptr + rows_c)
        if GA:
            ra = t
        else:
            ra = rows_c
        if GB:
            rb = t
        else:
            rb = rows_c
        a = tl.load(a_ptr + ra[:, None] * sa + offs_m[None, :], mask=km[:, None], other=0.0)
        b = tl.load(b_ptr + rb[:, None] * sb + offs_n[None, :])
        acc += tl.dot(tl.trans(a), b)

    op = out_ptr + e.to(tl.int64) * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(op, acc.to(tl.bfloat16))


# ----------------------------------------------------------------------------
# Python 侧调度
# ----------------------------------------------------------------------------
def _cdiv(a, b):
    return (a + b - 1) // b


def _pow2(n):
    p = 16
    while p < n:
        p *= 2
    return p


def _cfg(T, E):
    BM = 128 if T >= 192 * E else 64
    # 每组 8 个 M tile 共享同一列条带 (L2 分组),对偏斜 / 单专家分布尤为重要
    G = 8
    return BM, G


def _prep(m_sizes, BM):
    mend = torch.cumsum(m_sizes, 0, dtype=torch.int32)
    tiles = torch.div(m_sizes + (BM - 1), BM, rounding_mode="floor")
    tend = torch.cumsum(tiles, 0, dtype=torch.int32)
    return mend, tend


def _run_gu(x, wgu, token_ids, m_sizes, mend, tend, T, E, D, I, BM, G):
    gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=x.device)
    h = torch.empty((T, I), dtype=torch.bfloat16, device=x.device)
    BN = 64 if BM == 128 else 128
    BK = _BK
    nM = T // BM + E
    nN = _cdiv(I, BN)
    _k_fwd_gu[(nM * nN,)](
        x, token_ids, wgu, gu, h, m_sizes, mend, tend,
        E, D, I, nM, nN,
        BM=BM, BN=BN, BK=BK, EP=_pow2(E), G=G,
        KMASK=(D % BK != 0), NSHIFT=(I % BN != 0),
        num_warps=(8 if BM == 128 else 4), num_stages=_ST,
    )
    return gu, h


def _run_scatter(a, w, buf, probs, token_ids, m_sizes, mend, tend, T, E, K, N, BM, G, transb, scale,
                 mode=0):
    BN = 128
    BK = _BK
    nM = T // BM + E
    nN = _cdiv(N, BN)
    _k_scatter[(nM * nN,)](
        a, w, buf, probs, token_ids, m_sizes, mend, tend,
        E, K, N, nM, nN,
        BM=BM, BN=BN, BK=BK, EP=_pow2(E), G=G,
        TRANSB=transb, SCALE=scale, MODE=mode,
        KMASK=(K % BK != 0), NSHIFT=(N % BN != 0),
        num_warps=(8 if BM == 128 else 4), num_stages=_ST,
    )


def _run_wgrad(a, b, token_ids, out, m_sizes, mend, sa, sb, M, N, E, ga, gb):
    BM = 128
    BN = 128
    BK = _WBK
    nM = _cdiv(M, BM)
    nN = _cdiv(N, BN)
    _k_wgrad[(nM * nN, E)](
        a, b, token_ids, out, m_sizes, mend,
        sa, sb, M, N, nM, nN,
        BM=BM, BN=BN, BK=BK, GA=ga, GB=gb, MFAST=(M <= N),
        MSHIFT=(M % BM != 0), NSHIFT=(N % BN != 0),
        num_warps=_WW, num_stages=_WST,
    )


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        N_tok, D = x.shape
        E = w_gate_up.shape[0]
        I = w_down.shape[1]
        T = probs.shape[0]
        BM, G = _cfg(T, E)
        mend, tend = _prep(m_sizes, BM)

        _tick('start')
        gu, h = _run_gu(x, w_gate_up, token_ids, m_sizes, mend, tend, T, E, D, I, BM, G)
        _tick('A_gu')
        buf = torch.zeros((N_tok, D), dtype=torch.float32, device=x.device)
        _run_scatter(h, w_down, buf, probs, token_ids, m_sizes, mend, tend,
                     T, E, I, D, BM, G, False, True)
        out = buf.to(torch.bfloat16)
        _tick('B_z')

        ctx.save_for_backward(x, w_gate_up, w_down, probs)
        ctx.token_ids = token_ids
        ctx.m_sizes = m_sizes
        ctx.mend = mend
        ctx.tend = tend
        if T * 2 * I * 2 <= _SAVE_CAP_BYTES:
            ctx.gu = gu
        else:
            ctx.gu = None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_gate_up, w_down, probs = ctx.saved_tensors
        token_ids = ctx.token_ids
        m_sizes = ctx.m_sizes
        mend = ctx.mend
        tend = ctx.tend
        N_tok, D = x.shape
        E = w_gate_up.shape[0]
        I = w_down.shape[1]
        T = probs.shape[0]
        BM, G = _cfg(T, E)
        dev = x.device
        _tick('bstart')
        go = grad_out.contiguous()
        if go.dtype != torch.bfloat16:
            go = go.to(torch.bfloat16)

        gu = ctx.gu
        if gu is None:
            gu, hp = _run_gu(x, w_gate_up, token_ids, m_sizes, mend, tend, T, E, D, I, BM, G)
        else:
            hp = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        dht = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        go_perm = go.index_select(0, token_ids)
        _run_scatter(go_perm, w_down, dht, probs, token_ids, m_sizes, mend, tend,
                     T, E, D, I, BM, G, True, False, 1)
        _tick('C_dh')
        dprobs = torch.empty((T,), dtype=torch.float32, device=dev)
        BR = 16
        BC = 256
        _k_swiglu_bwd[(_cdiv(T, BR),)](
            gu, dht, probs, dgu, hp, dprobs, T, I,
            BR=BR, BC=BC, MASKED=(T % BR != 0 or I % BC != 0), num_warps=4,
        )
        del gu
        del dht
        _tick('P_dp')

        need = ctx.needs_input_grad
        dwd = None
        if need[2]:
            gp = (go_perm * probs.detach().unsqueeze(1)).to(torch.bfloat16)
            dwd = torch.empty_like(w_down)
            _run_wgrad(hp, gp, token_ids, dwd, m_sizes, mend, I, D, I, D, E, False, False)
            del gp
        del go_perm
        _tick('D_dwd')
        del hp
        dwgu = None
        if need[1]:
            dwgu = torch.empty_like(w_gate_up)
            _run_wgrad(x, dgu, token_ids, dwgu, m_sizes, mend, D, 2 * I, D, 2 * I, E, True, False)
        _tick('E_dwgu')
        dx = None
        if need[0]:
            dxbuf = torch.zeros((N_tok, D), dtype=torch.float32, device=dev)
            _run_scatter(dgu, w_gate_up, dxbuf, probs, token_ids, m_sizes, mend, tend,
                         T, E, 2 * I, D, BM, G, True, False)
            dx = dxbuf.to(torch.bfloat16)
        _tick('F_dx')
        if not need[3]:
            dprobs = None
        return dx, dwgu, dwd, dprobs, None, None


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    if probs.shape[0] == 0:
        return x * 0.0
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)


