import math

import torch
import triton
import triton.language as tl


@triton.jit
def _load_tile(blocks_ptr, cur_ptr, t, d, dmask, s_idx, T, D, B):
    # [S_PAD, BLOCK] tile of depth sources for token t (rows < B from blocks, row B from cur)
    row = s_idx[:, None]
    xb = tl.load(blocks_ptr + row * T * D + t * D + d[None, :],
                 mask=(row < B) & dmask[None, :], other=0.0).to(tl.float32)
    xc = tl.load(cur_ptr + t * D + d[None, :] + row * 0,
                 mask=(row == B) & dmask[None, :], other=0.0).to(tl.float32)
    return xb + xc


@triton.jit
def _fwd_stats_kernel(blocks_ptr, cur_ptr, q_ptr, r_ptr, l_ptr, w_ptr,
                      T, D, B, eps, scale,
                      M: tl.constexpr, M_PAD: tl.constexpr, S_PAD: tl.constexpr,
                      BLOCK: tl.constexpr):
    t = tl.program_id(0)
    s_idx = tl.arange(0, S_PAD)
    m_idx = tl.arange(0, M_PAD)
    acc_s = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc0 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc1 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc3 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK)
        dmask = d < D
        x = _load_tile(blocks_ptr, cur_ptr, t, d, dmask, s_idx, T, D, B)
        acc_s += x * x
        q0 = tl.load(q_ptr + 0 * D + d, mask=dmask, other=0.0)
        acc0 += x * q0[None, :]
        if M > 1:
            q1 = tl.load(q_ptr + 1 * D + d, mask=dmask, other=0.0)
            acc1 += x * q1[None, :]
        if M > 2:
            q2 = tl.load(q_ptr + 2 * D + d, mask=dmask, other=0.0)
            acc2 += x * q2[None, :]
        if M > 3:
            q3 = tl.load(q_ptr + 3 * D + d, mask=dmask, other=0.0)
            acc3 += x * q3[None, :]
    sumsq = tl.sum(acc_s, axis=1)
    dots = tl.where(m_idx[:, None] == 0, tl.sum(acc0, axis=1)[None, :], 0.0)
    if M > 1:
        dots += tl.where(m_idx[:, None] == 1, tl.sum(acc1, axis=1)[None, :], 0.0)
    if M > 2:
        dots += tl.where(m_idx[:, None] == 2, tl.sum(acc2, axis=1)[None, :], 0.0)
    if M > 3:
        dots += tl.where(m_idx[:, None] == 3, tl.sum(acc3, axis=1)[None, :], 0.0)
    r = 1.0 / tl.sqrt(sumsq / D + eps)
    valid = (s_idx < B + 1)[None, :]
    l = dots * r[None, :] * scale
    lm = tl.where(valid, l, float("-inf"))
    mx = tl.max(lm, axis=1)
    p = tl.exp(lm - mx[:, None])
    a = p / tl.sum(p, axis=1)[:, None]
    w = a * r[None, :]
    tl.store(r_ptr + t * S_PAD + s_idx, r)
    off = t * S_PAD * M_PAD + s_idx[None, :] * M_PAD + m_idx[:, None]
    tl.store(l_ptr + off, l)
    tl.store(w_ptr + off, w)


@triton.jit
def _fwd_out_kernel(blocks_ptr, cur_ptr, w_ptr, out_ptr, T, D, B,
                    M: tl.constexpr, M_PAD: tl.constexpr, S_PAD: tl.constexpr,
                    BD: tl.constexpr):
    t = tl.program_id(0)
    pd = tl.program_id(1)
    d = pd * BD + tl.arange(0, BD)
    dmask = d < D
    wb = w_ptr + t * S_PAD * M_PAD
    x = tl.load(cur_ptr + t * D + d, mask=dmask, other=0.0).to(tl.float32)
    o0 = tl.load(wb + B * M_PAD + 0) * x
    o1 = o0
    o2 = o0
    o3 = o0
    if M > 1:
        o1 = tl.load(wb + B * M_PAD + 1) * x
    if M > 2:
        o2 = tl.load(wb + B * M_PAD + 2) * x
    if M > 3:
        o3 = tl.load(wb + B * M_PAD + 3) * x
    for s in range(B):
        x = tl.load(blocks_ptr + s * T * D + t * D + d, mask=dmask, other=0.0).to(tl.float32)
        o0 += tl.load(wb + s * M_PAD + 0) * x
        if M > 1:
            o1 += tl.load(wb + s * M_PAD + 1) * x
        if M > 2:
            o2 += tl.load(wb + s * M_PAD + 2) * x
        if M > 3:
            o3 += tl.load(wb + s * M_PAD + 3) * x
    ob = out_ptr + t * D + d
    tl.store(ob, o0.to(tl.bfloat16), mask=dmask)
    if M > 1:
        tl.store(ob + 1 * T * D, o1.to(tl.bfloat16), mask=dmask)
    if M > 2:
        tl.store(ob + 2 * T * D, o2.to(tl.bfloat16), mask=dmask)
    if M > 3:
        tl.store(ob + 3 * T * D, o3.to(tl.bfloat16), mask=dmask)


@triton.jit
def _bwd_stats_kernel(blocks_ptr, cur_ptr, go_ptr, r_ptr, l_ptr,
                      ab_ptr, c_ptr,
                      T, D, B, scale,
                      M: tl.constexpr, M_PAD: tl.constexpr, S_PAD: tl.constexpr,
                      BLOCK: tl.constexpr):
    t = tl.program_id(0)
    s_idx = tl.arange(0, S_PAD)
    m_idx = tl.arange(0, M_PAD)
    acc0 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc1 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    acc3 = tl.zeros([S_PAD, BLOCK], dtype=tl.float32)
    gb = go_ptr + t * D
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK)
        dmask = d < D
        x = _load_tile(blocks_ptr, cur_ptr, t, d, dmask, s_idx, T, D, B)
        g0 = tl.load(gb + d, mask=dmask, other=0.0).to(tl.float32)
        acc0 += x * g0[None, :]
        if M > 1:
            g1 = tl.load(gb + 1 * T * D + d, mask=dmask, other=0.0).to(tl.float32)
            acc1 += x * g1[None, :]
        if M > 2:
            g2 = tl.load(gb + 2 * T * D + d, mask=dmask, other=0.0).to(tl.float32)
            acc2 += x * g2[None, :]
        if M > 3:
            g3 = tl.load(gb + 3 * T * D + d, mask=dmask, other=0.0).to(tl.float32)
            acc3 += x * g3[None, :]
    P = tl.where(m_idx[:, None] == 0, tl.sum(acc0, axis=1)[None, :], 0.0)
    if M > 1:
        P += tl.where(m_idx[:, None] == 1, tl.sum(acc1, axis=1)[None, :], 0.0)
    if M > 2:
        P += tl.where(m_idx[:, None] == 2, tl.sum(acc2, axis=1)[None, :], 0.0)
    if M > 3:
        P += tl.where(m_idx[:, None] == 3, tl.sum(acc3, axis=1)[None, :], 0.0)
    r = tl.load(r_ptr + t * S_PAD + s_idx)
    off = t * S_PAD * M_PAD + s_idx[None, :] * M_PAD + m_idx[:, None]
    l = tl.load(l_ptr + off)
    valid = (s_idx < B + 1)[None, :]
    lm = tl.where(valid, l, float("-inf"))
    mx = tl.max(lm, axis=1)
    p = tl.exp(lm - mx[:, None])
    a = p / tl.sum(p, axis=1)[:, None]
    l0 = tl.where(valid, l, 0.0)
    da = r[None, :] * P
    gdo = tl.sum(a * da, axis=1)
    dl = a * (da - gdo[:, None])
    k = tl.sum(a * da + dl * l0, axis=0)
    off8 = t * S_PAD * 8 + s_idx[None, :] * 8 + m_idx[:, None]
    tl.store(ab_ptr + off8, a * r[None, :])
    tl.store(ab_ptr + off8 + 4, dl * (r * scale)[None, :])
    tl.store(c_ptr + t * S_PAD + s_idx, -(r * r) * k / D)


@triton.jit
def _bwd_row(x, cb, cs, g0, g1, g2, g3, q0, q1, q2, q3, acc0, acc1, acc2, acc3,
             M: tl.constexpr):
    dx = cs * x
    b0 = tl.load(cb + 4 + 0)
    dx += tl.load(cb + 0) * g0 + b0 * q0
    acc0 += b0 * x
    if M > 1:
        b1 = tl.load(cb + 4 + 1)
        dx += tl.load(cb + 1) * g1 + b1 * q1
        acc1 += b1 * x
    if M > 2:
        b2 = tl.load(cb + 4 + 2)
        dx += tl.load(cb + 2) * g2 + b2 * q2
        acc2 += b2 * x
    if M > 3:
        b3 = tl.load(cb + 4 + 3)
        dx += tl.load(cb + 3) * g3 + b3 * q3
        acc3 += b3 * x
    return dx, acc0, acc1, acc2, acc3


@triton.jit
def _bwd_dx_kernel(blocks_ptr, cur_ptr, go_ptr, q_ptr, ab_ptr, c_ptr,
                   db_ptr, dc_ptr, dqp_ptr,
                   T, D, B, TB,
                   M: tl.constexpr, S_PAD: tl.constexpr, BD: tl.constexpr):
    # ab layout: [T, S_PAD, 8] with [0:4] = A (coef of grad_out), [4:8] = Bc (coef of queries)
    pt = tl.program_id(0)
    pd = tl.program_id(1)
    d = pd * BD + tl.arange(0, BD)
    dmask = d < D
    q0 = tl.load(q_ptr + 0 * D + d, mask=dmask, other=0.0)
    q1 = q0
    q2 = q0
    q3 = q0
    if M > 1:
        q1 = tl.load(q_ptr + 1 * D + d, mask=dmask, other=0.0)
    if M > 2:
        q2 = tl.load(q_ptr + 2 * D + d, mask=dmask, other=0.0)
    if M > 3:
        q3 = tl.load(q_ptr + 3 * D + d, mask=dmask, other=0.0)
    acc0 = tl.zeros([BD], dtype=tl.float32)
    acc1 = tl.zeros([BD], dtype=tl.float32)
    acc2 = tl.zeros([BD], dtype=tl.float32)
    acc3 = tl.zeros([BD], dtype=tl.float32)
    t0 = pt * TB
    t1 = tl.minimum(t0 + TB, T)
    for t in range(t0, t1):
        gb = go_ptr + t * D + d
        g0 = tl.load(gb, mask=dmask, other=0.0).to(tl.float32)
        g1 = g0
        g2 = g0
        g3 = g0
        if M > 1:
            g1 = tl.load(gb + 1 * T * D, mask=dmask, other=0.0).to(tl.float32)
        if M > 2:
            g2 = tl.load(gb + 2 * T * D, mask=dmask, other=0.0).to(tl.float32)
        if M > 3:
            g3 = tl.load(gb + 3 * T * D, mask=dmask, other=0.0).to(tl.float32)
        abb = ab_ptr + t * S_PAD * 8
        x = tl.load(cur_ptr + t * D + d, mask=dmask, other=0.0).to(tl.float32)
        dx, acc0, acc1, acc2, acc3 = _bwd_row(x, abb + B * 8, tl.load(c_ptr + t * S_PAD + B),
                                              g0, g1, g2, g3, q0, q1, q2, q3,
                                              acc0, acc1, acc2, acc3, M)
        tl.store(dc_ptr + t * D + d, dx.to(tl.bfloat16), mask=dmask)
        for s in range(B):
            x = tl.load(blocks_ptr + s * T * D + t * D + d, mask=dmask, other=0.0).to(tl.float32)
            dx, acc0, acc1, acc2, acc3 = _bwd_row(x, abb + s * 8, tl.load(c_ptr + t * S_PAD + s),
                                                  g0, g1, g2, g3, q0, q1, q2, q3,
                                                  acc0, acc1, acc2, acc3, M)
            tl.store(db_ptr + s * T * D + t * D + d, dx.to(tl.bfloat16), mask=dmask)
    pbase = dqp_ptr + pt * M * D + d
    tl.store(pbase + 0 * D, acc0, mask=dmask)
    if M > 1:
        tl.store(pbase + 1 * D, acc1, mask=dmask)
    if M > 2:
        tl.store(pbase + 2 * D, acc2, mask=dmask)
    if M > 3:
        tl.store(pbase + 3 * D, acc3, mask=dmask)


def _pads(B, M):
    S_PAD = triton.next_power_of_2(B + 1)
    M_PAD = max(triton.next_power_of_2(M), 1)
    return S_PAD, M_PAD


class _BlockAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, cur, queries, eps):
        B, T, D = blocks.shape
        M = queries.shape[0]
        assert M <= 4
        blocks = blocks.contiguous()
        cur = cur.contiguous()
        queries = queries.contiguous().float()
        S_PAD, M_PAD = _pads(B, M)
        dev = blocks.device
        scale = 1.0 / math.sqrt(D)
        r = torch.empty((T, S_PAD), device=dev, dtype=torch.float32)
        l = torch.empty((T, S_PAD, M_PAD), device=dev, dtype=torch.float32)
        w = torch.empty((T, S_PAD, M_PAD), device=dev, dtype=torch.float32)
        SBLK = 1024 // S_PAD
        _fwd_stats_kernel[(T,)](blocks, cur, queries, r, l, w, T, D, B, float(eps), scale,
                                M=M, M_PAD=M_PAD, S_PAD=S_PAD, BLOCK=SBLK, num_warps=4)
        out = torch.empty((M, T, D), device=dev, dtype=torch.bfloat16)
        BD = 1024
        _fwd_out_kernel[(T, triton.cdiv(D, BD))](blocks, cur, w, out, T, D, B,
                                                 M=M, M_PAD=M_PAD, S_PAD=S_PAD, BD=BD,
                                                 num_warps=4)
        ctx.save_for_backward(blocks, cur, queries, r, l)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        blocks, cur, queries, r, l = ctx.saved_tensors
        B, T, D = blocks.shape
        M = queries.shape[0]
        S_PAD, M_PAD = _pads(B, M)
        dev = blocks.device
        scale = 1.0 / math.sqrt(D)
        grad_out = grad_out.contiguous()
        AB = torch.empty((T, S_PAD, 8), device=dev, dtype=torch.float32)
        C = torch.empty((T, S_PAD), device=dev, dtype=torch.float32)
        SBLK = 1024 // S_PAD
        _bwd_stats_kernel[(T,)](blocks, cur, grad_out, r, l, AB, C, T, D, B, scale,
                                M=M, M_PAD=M_PAD, S_PAD=S_PAD, BLOCK=SBLK, num_warps=4)
        BD = 1024
        nD = triton.cdiv(D, BD)
        TB = max(8, min(64, (T * nD) // 864))
        nT = triton.cdiv(T, TB)
        dblocks = torch.empty_like(blocks)
        dcur = torch.empty_like(cur)
        dqp = torch.empty((nT, M, D), device=dev, dtype=torch.float32)
        _bwd_dx_kernel[(nT, nD)](blocks, cur, grad_out, queries, AB, C,
                                 dblocks, dcur, dqp, T, D, B, TB,
                                 M=M, S_PAD=S_PAD, BD=BD, num_warps=4)
        dq = dqp.sum(dim=0)
        return dblocks, dcur, dq, None


def block_attnres(
    blocks: torch.Tensor,
    cur: torch.Tensor,
    queries: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return _BlockAttnRes.apply(blocks, cur, queries, eps)
