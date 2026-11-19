"""Top-K routed SwiGLU MoE FFN: Triton 实现的前向 + 反向(grouped GEMM,张量核 bf16 -> fp32 累加)。

设计要点:
  * 所有 expert 的 GEMM 合并在少量 kernel launch 中(按 (expert, 行块) tile 表调度)。
  * 前向: k1 (x·W_gate / x·W_up + SiLU 融合, 输出 h*p) -> k2 (h·W_down) -> index_add_ 汇聚。
  * 反向: k1 (dh = go·W_down^T, 融合激活导数与 dprobs) -> k3 (两个权重梯度, K 维 = 路由行)
           -> k2 (dx = dgu·W_gu^T) -> index_add_。
  * 图上只保留 (g,u) (不超过约 100MB 时), 否则在反向中重算。
"""

import torch
import triton
import triton.language as tl

_SAVE_LIMIT_BYTES = 110 * 1024 * 1024


@triton.jit
def _k1(
    x_ptr, go_ptr, wgu_ptr, wd_ptr, tid_ptr, p_ptr, tiles_ptr,
    gu_ptr, h_ptr, dgu_ptr, dp_ptr,
    D, I,
    MODE: tl.constexpr, SAVE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # MODE 0: 前向 (算 g,u -> 存 h*p, 可选存 g,u)
    # MODE 1: 反向且重算 g,u, 同时算 dhr = go·Wd^T
    # MODE 2: 反向, g,u 从保存的缓冲读取, 只算 dhr
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tiles_ptr + pt * 3).to(tl.int64)
    r0 = tl.load(tiles_ptr + pt * 3 + 1)
    r1 = tl.load(tiles_ptr + pt * 3 + 2)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    tid = tl.load(tid_ptr + rows, mask=rmask, other=0)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < I
    I2 = 2 * I
    omask = rmask[:, None] & cmask[None, :]
    rows64 = rows.to(tl.int64)

    if MODE != 2:
        g = tl.zeros((BM, BN), dtype=tl.float32)
        u = tl.zeros((BM, BN), dtype=tl.float32)
    if MODE != 0:
        dhr = tl.zeros((BM, BN), dtype=tl.float32)

    wbase = wgu_ptr + e * D * I2
    dbase = wd_ptr + e * I * D
    for k0 in range(0, D, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < D
        amask = rmask[:, None] & kmask[None, :]
        aoff = tid[:, None] * D + ks[None, :]
        bmask = kmask[:, None] & cmask[None, :]
        if MODE != 2:
            a = tl.load(x_ptr + aoff, mask=amask, other=0.0)
            boff = ks[:, None] * I2 + cols[None, :]
            bg = tl.load(wbase + boff, mask=bmask, other=0.0)
            bu = tl.load(wbase + boff + I, mask=bmask, other=0.0)
            g = tl.dot(a, bg, g)
            u = tl.dot(a, bu, u)
        if MODE != 0:
            ag = tl.load(go_ptr + aoff, mask=amask, other=0.0)
            bd = tl.load(dbase + cols[None, :] * D + ks[:, None], mask=bmask, other=0.0)
            dhr = tl.dot(ag, bd, dhr)

    guoff = rows64[:, None] * I2 + cols[None, :]
    if MODE == 2:
        g = tl.load(gu_ptr + guoff, mask=omask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + guoff + I, mask=omask, other=0.0).to(tl.float32)

    p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
    sig = tl.sigmoid(g)
    sg = g * sig
    h = sg * u
    hoff = rows64[:, None] * I + cols[None, :]
    tl.store(h_ptr + hoff, (h * p[:, None]).to(tl.bfloat16), mask=omask)
    if MODE == 0:
        if SAVE:
            tl.store(gu_ptr + guoff, g.to(tl.bfloat16), mask=omask)
            tl.store(gu_ptr + guoff + I, u.to(tl.bfloat16), mask=omask)
    else:
        dpv = tl.sum(h * dhr, axis=1)
        tl.atomic_add(dp_ptr + rows, dpv, mask=rmask)
        dh = dhr * p[:, None]
        du = dh * sg
        dg = dh * u * sig * (1.0 + g * (1.0 - sig))
        tl.store(dgu_ptr + guoff, dg.to(tl.bfloat16), mask=omask)
        tl.store(dgu_ptr + guoff + I, du.to(tl.bfloat16), mask=omask)


@triton.jit
def _k2(
    a_ptr, b_ptr, tiles_ptr, o_ptr,
    K, N, sk, sn, estride,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # out[r, n] = sum_k A[r, k] * B[e, k*sk + n*sn]
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tiles_ptr + pt * 3).to(tl.int64)
    r0 = tl.load(tiles_ptr + pt * 3 + 1)
    r1 = tl.load(tiles_ptr + pt * 3 + 2)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < r1
    rows64 = rows.to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    bbase = b_ptr + e * estride
    abase = a_ptr + rows64[:, None] * K
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < K
        a = tl.load(abase + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(bbase + ks[:, None] * sk + cols[None, :] * sn,
                    mask=kmask[:, None] & cmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(o_ptr + rows64[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
             mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _k3(
    p_ptr, q_ptr, tid_ptr, starts_ptr, o_ptr,
    NA, NB, ldp, ldq,
    GP: tl.constexpr, GQ: tl.constexpr,
    BA: tl.constexpr, BB: tl.constexpr, BK: tl.constexpr,
):
    # out[e, a, b] = sum_{r in expert e} P[prow(r), a] * Q[qrow(r), b]
    e = tl.program_id(0)
    pa = tl.program_id(1)
    pb = tl.program_id(2)
    r0 = tl.load(starts_ptr + e)
    r1 = tl.load(starts_ptr + e + 1)
    ia = pa * BA + tl.arange(0, BA)
    ib = pb * BB + tl.arange(0, BB)
    amask = ia < NA
    bmask = ib < NB
    acc = tl.zeros((BA, BB), dtype=tl.float32)
    nk = tl.cdiv(r1 - r0, BK)
    for i in range(0, nk):
        rs = r0 + i * BK + tl.arange(0, BK)
        kmask = rs < r1
        tid = tl.load(tid_ptr + rs, mask=kmask, other=0).to(tl.int64)
        rs64 = rs.to(tl.int64)
        if GP:
            prow = tid
        else:
            prow = rs64
        if GQ:
            qrow = tid
        else:
            qrow = rs64
        pt = tl.load(p_ptr + prow[None, :] * ldp + ia[:, None],
                     mask=amask[:, None] & kmask[None, :], other=0.0)
        qt = tl.load(q_ptr + qrow[:, None] * ldq + ib[None, :],
                     mask=kmask[:, None] & bmask[None, :], other=0.0)
        acc = tl.dot(pt, qt, acc)
    ooff = e.to(tl.int64) * NA * NB + ia[:, None].to(tl.int64) * NB + ib[None, :]
    tl.store(o_ptr + ooff, acc.to(tl.bfloat16), mask=amask[:, None] & bmask[None, :])


@triton.jit
def _k4(
    z_ptr, order_ptr, offs_ptr, o_ptr,
    N, D,
    BT: tl.constexpr, BC: tl.constexpr,
):
    # out[n, :] = sum_{r in rows(n)} z[r, :]  (CSR: token -> 路由行)
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    toks = pt * BT + tl.arange(0, BT)
    tmask = toks < N
    o0 = tl.load(offs_ptr + toks, mask=tmask, other=0)
    o1 = tl.load(offs_ptr + toks + 1, mask=tmask, other=0)
    cnt = o1 - o0
    kmax = tl.max(cnt, axis=0)
    cols = pc * BC + tl.arange(0, BC)
    cmask = cols < D
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for j in range(0, kmax):
        m = j < cnt
        row = tl.load(order_ptr + o0 + j, mask=m, other=0)
        v = tl.load(z_ptr + row[:, None] * D + cols[None, :],
                    mask=m[:, None] & cmask[None, :], other=0.0)
        acc += v.to(tl.float32)
    tl.store(o_ptr + toks[:, None] * D + cols[None, :], acc.to(tl.bfloat16),
             mask=tmask[:, None] & cmask[None, :])


def _cfg(T, E):
    avg = T / max(E, 1)
    if avg >= 128:
        return 128, 8
    if avg >= 48:
        return 64, 4
    return 32, 4


def _cdiv(a, b):
    return (a + b - 1) // b


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        N_tok, D = x.shape
        E, I, _ = w_down.shape
        T = probs.shape[0]
        dev = x.device
        BM, NW = _cfg(T, E)

        # 每次调用都从当前 m_sizes 重新构建调度表(不跨调用缓存)
        ms = m_sizes.tolist()
        starts = [0]
        flat = []
        s = 0
        need_grad = any(ctx.needs_input_grad[:4])
        rmax = (_SAVE_LIMIT_BYTES // (4 * I)) if need_grad else 0
        n_pre = 0
        r_pre = 0
        for e in range(E):
            m = ms[e]
            end = s + m
            r = s
            while r < end:
                flat.append(e)
                flat.append(r)
                r += BM
                if r > end:
                    r = end
                flat.append(r)
                if r <= rmax:
                    n_pre += 1
                    r_pre = r
            s = end
            starts.append(s)
        ntiles = len(flat) // 3
        tiles = torch.tensor(flat if ntiles > 0 else [0, 0, 0], dtype=torch.int32, device=dev)
        starts_t = torch.tensor(starts, dtype=torch.int32, device=dev)

        x = x.contiguous()
        w_gate_up = w_gate_up.contiguous()
        w_down = w_down.contiguous()
        probs = probs.contiguous()
        token_ids = token_ids.contiguous()

        gu = None
        order = None
        offs = None
        if ntiles == 0:
            out = torch.zeros((N_tok, D), dtype=torch.bfloat16, device=dev)
        else:
            # CSR: token -> 路由行 (每次调用在 GPU 上重建)
            svals, order = torch.sort(token_ids.long())
            offs = torch.searchsorted(svals, torch.arange(N_tok + 1, device=dev, dtype=torch.int64))
            del svals
            out = torch.empty((N_tok, D), dtype=torch.bfloat16, device=dev)
            hp = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
            if n_pre > 0:
                # 前缀 tile: 保存 (g,u) 供反向使用
                gu = torch.empty((r_pre, 2 * I), dtype=torch.bfloat16, device=dev)
                _k1[(n_pre, _cdiv(I, 64))](
                    x, x, w_gate_up, w_down, token_ids, probs, tiles,
                    gu, hp, hp, probs,
                    D, I,
                    MODE=0, SAVE=True, BM=BM, BN=64, BK=32,
                    num_warps=NW, num_stages=3,
                )
            if ntiles > n_pre:
                # 其余 tile: 超出内存预算, 反向时重算
                _k1[(ntiles - n_pre, _cdiv(I, 64))](
                    x, x, w_gate_up, w_down, token_ids, probs, tiles[3 * n_pre:],
                    hp, hp, hp, probs,
                    D, I,
                    MODE=0, SAVE=False, BM=BM, BN=64, BK=32,
                    num_warps=NW, num_stages=3,
                )
            z = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
            _k2[(ntiles, _cdiv(D, 128))](
                hp, w_down, tiles, z,
                I, D, D, 1, I * D,
                BM=BM, BN=128, BK=32,
                num_warps=NW, num_stages=3,
            )
            del hp
            _k4[(_cdiv(N_tok, 8), _cdiv(D, 512))](
                z, order, offs, out, N_tok, D, BT=8, BC=512, num_warps=4,
            )
            del z

        ctx.save_for_backward(x, w_gate_up, w_down, probs)
        ctx.aux = (token_ids, tiles, starts_t, ntiles, n_pre, gu, BM, NW, order, offs)
        ctx.mark_non_differentiable()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_gate_up, w_down, probs = ctx.saved_tensors
        token_ids, tiles, starts_t, ntiles, n_pre, gu, BM, NW, order, offs = ctx.aux
        N_tok, D = x.shape
        E, I, _ = w_down.shape
        T = probs.shape[0]
        dev = x.device
        go = grad_out.to(torch.bfloat16).contiguous()

        dp = torch.zeros((T,), dtype=torch.float32, device=dev)
        if ntiles == 0:
            dx = torch.zeros((N_tok, D), dtype=torch.bfloat16, device=dev)
            return (dx, torch.zeros_like(w_gate_up), torch.zeros_like(w_down), dp, None, None)

        hp = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        if n_pre > 0:
            _k1[(n_pre, _cdiv(I, 64))](
                x, go, w_gate_up, w_down, token_ids, probs, tiles,
                gu, hp, dgu, dp,
                D, I,
                MODE=2, SAVE=False, BM=BM, BN=64, BK=(64 if BM == 128 else 32),
                num_warps=NW, num_stages=3,
            )
        if ntiles > n_pre:
            _k1[(ntiles - n_pre, _cdiv(I, 64))](
                x, go, w_gate_up, w_down, token_ids, probs, tiles[3 * n_pre:],
                dgu, hp, dgu, dp,
                D, I,
                MODE=1, SAVE=False, BM=BM, BN=64, BK=32,
                num_warps=NW, num_stages=3,
            )

        need = ctx.needs_input_grad
        dwd = None
        dwgu = None
        if need[2]:
            # dWd[e] = (h*p)^T · go[tid]
            dwd = torch.empty((E, I, D), dtype=torch.bfloat16, device=dev)
            _k3[(E, _cdiv(I, 128), _cdiv(D, 128))](
                hp, go, token_ids, starts_t, dwd,
                I, D, I, D,
                GP=False, GQ=True, BA=128, BB=128, BK=64,
                num_warps=4, num_stages=3,
            )
        del hp
        if need[1]:
            # dWgu[e] = x[tid]^T · dgu
            dwgu = torch.empty((E, D, 2 * I), dtype=torch.bfloat16, device=dev)
            _k3[(E, _cdiv(D, 128), _cdiv(2 * I, 128))](
                x, dgu, token_ids, starts_t, dwgu,
                D, 2 * I, D, 2 * I,
                GP=True, GQ=False, BA=128, BB=128, BK=64,
                num_warps=4, num_stages=3,
            )
        if need[0]:
            dxp = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
            _k2[(ntiles, _cdiv(D, 128))](
                dgu, w_gate_up, tiles, dxp,
                2 * I, D, 1, 2 * I, D * 2 * I,
                BM=BM, BN=128, BK=(64 if BM == 128 else 32),
                num_warps=NW, num_stages=3,
            )
            dx = torch.empty((N_tok, D), dtype=torch.bfloat16, device=dev)
            _k4[(_cdiv(N_tok, 8), _cdiv(D, 512))](
                dxp, order, offs, dx, N_tok, D, BT=8, BC=512, num_warps=4,
            )
        else:
            dx = None
        return (dx, dwgu, dwd, dp if need[3] else None, None, None)


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)
