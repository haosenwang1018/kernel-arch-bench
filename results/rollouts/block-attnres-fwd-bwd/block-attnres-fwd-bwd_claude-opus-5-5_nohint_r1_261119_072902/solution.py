import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward: per-token stats (rstd, logits, softmax coefficients)
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_stats_kernel(blocks, cur, q, rstd_out, logit_out, coef_out,
                      T, D, B, S, M, eps, inv_sqrt_d,
                      S_PAD: tl.constexpr, M_PAD: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s_idx = tl.arange(0, S_PAD)
    m_idx = tl.arange(0, M_PAD)
    d_idx = tl.arange(0, BD)
    s_mask = s_idx < S
    m_mask = m_idx < M
    row_off = tl.where(s_idx < B, (s_idx.to(tl.int64) * T + t) * D, 0)
    blk_rows = blocks + row_off
    cur_row = cur + t * D
    is_blk = s_idx < B
    is_cur = s_idx == B

    acc_sq = tl.zeros([S_PAD, BD], tl.float32)
    acc_dot = tl.zeros([S_PAD, M_PAD, BD], tl.float32)
    for d0 in range(0, D, BD):
        dd = d0 + d_idx
        dm = dd < D
        hb = tl.load(blk_rows[:, None] + dd[None, :],
                     mask=is_blk[:, None] & dm[None, :], other=0.0).to(tl.float32)
        hc = tl.load(cur_row + dd, mask=dm, other=0.0).to(tl.float32)
        h = tl.where(is_cur[:, None], hc[None, :], hb)
        qv = tl.load(q + m_idx[:, None] * D + dd[None, :],
                     mask=m_mask[:, None] & dm[None, :], other=0.0)
        acc_sq += h * h
        acc_dot += h[:, None, :] * qv[None, :, :]

    sumsq = tl.sum(acc_sq, axis=1)
    dots = tl.sum(acc_dot, axis=2)
    r = tl.rsqrt(sumsq / D + eps)
    l = dots * (r * inv_sqrt_d)[:, None]
    valid = s_mask[:, None] & m_mask[None, :]
    lm = tl.where(s_mask[:, None], l, float("-inf"))
    mx = tl.max(lm, axis=0)
    e = tl.exp(lm - mx[None, :])
    den = tl.sum(e, axis=0)
    a = e / den[None, :]

    tl.store(rstd_out + t * S + s_idx, r, mask=s_mask)
    off = (t * S + s_idx[:, None]) * M + m_idx[None, :]
    tl.store(logit_out + off, l, mask=valid)
    tl.store(coef_out + off, a * r[:, None], mask=valid)


# ---------------------------------------------------------------------------
# Forward: weighted depth read
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_out_kernel(blocks, cur, coef, out, T, D, B, S, M,
                    M_PAD: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    d = pid_d * BD + tl.arange(0, BD)
    m_idx = tl.arange(0, M_PAD)
    tm = t < T
    dm = d < D
    mm = m_idx < M
    t64 = t.to(tl.int64)
    hoff = t64[:, None] * D + d[None, :]
    hmask = tm[:, None] & dm[None, :]
    cmask = tm[:, None] & mm[None, :]
    acc = tl.zeros([BT, M_PAD, BD], tl.float32)
    for s in range(S):
        if s < B:
            base = blocks + s.to(tl.int64) * T * D
        else:
            base = cur
        h = tl.load(base + hoff, mask=hmask, other=0.0).to(tl.float32)
        c = tl.load(coef + (t64[:, None] * S + s) * M + m_idx[None, :], mask=cmask, other=0.0)
        acc += c[:, :, None] * h[:, None, :]
    optr = out + (m_idx[None, :, None].to(tl.int64) * T + t64[:, None, None]) * D + d[None, None, :]
    omask = tm[:, None, None] & mm[None, :, None] & dm[None, None, :]
    tl.store(optr, acc.to(tl.bfloat16), mask=omask)


# ---------------------------------------------------------------------------
# Backward: per-token stats -> coefficients
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_stats_kernel(blocks, cur, g, rstd, logits, a_out, b_out, c_out,
                      T, D, B, S, M, inv_sqrt_d, inv_d,
                      S_PAD: tl.constexpr, M_PAD: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s_idx = tl.arange(0, S_PAD)
    m_idx = tl.arange(0, M_PAD)
    d_idx = tl.arange(0, BD)
    s_mask = s_idx < S
    m_mask = m_idx < M
    row_off = tl.where(s_idx < B, (s_idx.to(tl.int64) * T + t) * D, 0)
    blk_rows = blocks + row_off
    cur_row = cur + t * D
    g_rows = g + (m_idx.to(tl.int64) * T + t) * D
    is_blk = s_idx < B
    is_cur = s_idx == B

    acc = tl.zeros([S_PAD, M_PAD, BD], tl.float32)
    for d0 in range(0, D, BD):
        dd = d0 + d_idx
        dm = dd < D
        hb = tl.load(blk_rows[:, None] + dd[None, :],
                     mask=is_blk[:, None] & dm[None, :], other=0.0).to(tl.float32)
        hc = tl.load(cur_row + dd, mask=dm, other=0.0).to(tl.float32)
        h = tl.where(is_cur[:, None], hc[None, :], hb)
        gv = tl.load(g_rows[:, None] + dd[None, :],
                     mask=m_mask[:, None] & dm[None, :], other=0.0).to(tl.float32)
        acc += h[:, None, :] * gv[None, :, :]
    P = tl.sum(acc, axis=2)

    valid = s_mask[:, None] & m_mask[None, :]
    r = tl.load(rstd + t * S + s_idx, mask=s_mask, other=0.0)
    off = (t * S + s_idx[:, None]) * M + m_idx[None, :]
    l = tl.load(logits + off, mask=valid, other=0.0)
    lm = tl.where(s_mask[:, None], l, float("-inf"))
    mx = tl.max(lm, axis=0)
    e = tl.exp(lm - mx[None, :])
    den = tl.sum(e, axis=0)
    alpha = e / den[None, :]
    alpha = tl.where(valid, alpha, 0.0)

    da = P * r[:, None]
    sm = tl.sum(alpha * da, axis=0)
    dl = alpha * (da - sm[None, :])
    K = tl.sum(alpha * da + dl * l, axis=1)

    tl.store(a_out + off, alpha * r[:, None], mask=valid)
    tl.store(b_out + off, dl * (r * inv_sqrt_d)[:, None], mask=valid)
    tl.store(c_out + t * S + s_idx, r * r * K * inv_d, mask=s_mask)


# ---------------------------------------------------------------------------
# Backward: apply coefficients -> dH, dq
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_apply_kernel(blocks, cur, g, q, a_c, b_c, c_c, dblocks, dcur, dq,
                      T, D, B, S, M, NTB, NSPLIT,
                      M_PAD: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    pid_d = tl.program_id(0)
    pid_s = tl.program_id(1)
    d = pid_d * BD + tl.arange(0, BD)
    m_idx = tl.arange(0, M_PAD)
    dm = d < D
    mm = m_idx < M
    qv = tl.load(q + m_idx[:, None] * D + d[None, :], mask=mm[:, None] & dm[None, :], other=0.0)
    dq_acc = tl.zeros([M_PAD, BD], tl.float32)
    for tb in range(pid_s, NTB, NSPLIT):
        t = tb * BT + tl.arange(0, BT)
        tm = t < T
        t64 = t.to(tl.int64)
        gptr = g + (m_idx[None, :, None].to(tl.int64) * T + t64[:, None, None]) * D + d[None, None, :]
        gmask = tm[:, None, None] & mm[None, :, None] & dm[None, None, :]
        G = tl.load(gptr, mask=gmask, other=0.0).to(tl.float32)
        hoff = t64[:, None] * D + d[None, :]
        hmask = tm[:, None] & dm[None, :]
        cmask = tm[:, None] & mm[None, :]
        for s in range(S):
            if s < B:
                soff = s.to(tl.int64) * T * D
                base = blocks + soff
                obase = dblocks + soff
            else:
                base = cur
                obase = dcur
            h = tl.load(base + hoff, mask=hmask, other=0.0).to(tl.float32)
            coff = (t64[:, None] * S + s) * M + m_idx[None, :]
            av = tl.load(a_c + coff, mask=cmask, other=0.0)
            bv = tl.load(b_c + coff, mask=cmask, other=0.0)
            cv = tl.load(c_c + t64 * S + s, mask=tm, other=0.0)
            dh = tl.sum(av[:, :, None] * G + bv[:, :, None] * qv[None, :, :], axis=1) - cv[:, None] * h
            tl.store(obase + hoff, dh.to(tl.bfloat16), mask=hmask)
            dq_acc += tl.sum(bv[:, :, None] * h[:, None, :], axis=0)
    tl.atomic_add(dq + m_idx[:, None] * D + d[None, :], dq_acc, mask=mm[:, None] & dm[None, :])


_SM_COUNT = {}


def _sm_count(dev):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _stats_bd(S_PAD, M_PAD):
    return max(32, min(256, 4096 // (S_PAD * M_PAD)))


class _BlockAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, cur, queries, eps):
        blocks = blocks.contiguous()
        cur = cur.contiguous()
        queries = queries.contiguous().float()
        B, T, D = blocks.shape
        M = queries.shape[0]
        S = B + 1
        S_PAD = triton.next_power_of_2(S)
        M_PAD = triton.next_power_of_2(M)
        dev = blocks.device
        inv_sqrt_d = float(D) ** -0.5

        rstd = torch.empty((T, S), device=dev, dtype=torch.float32)
        logits = torch.empty((T, S, M), device=dev, dtype=torch.float32)
        coef = torch.empty((T, S, M), device=dev, dtype=torch.float32)
        _fwd_stats_kernel[(T,)](
            blocks, cur, queries, rstd, logits, coef,
            T, D, B, S, M, float(eps), inv_sqrt_d,
            S_PAD=S_PAD, M_PAD=M_PAD, BD=_stats_bd(S_PAD, M_PAD), num_warps=4,
        )
        out = torch.empty((M, T, D), device=dev, dtype=torch.bfloat16)
        BT = max(1, 32 // M_PAD)
        BD = 128
        _fwd_out_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BD))](
            blocks, cur, coef, out, T, D, B, S, M,
            M_PAD=M_PAD, BT=BT, BD=BD, num_warps=4,
        )
        del coef
        ctx.save_for_backward(blocks, cur, queries, rstd, logits)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        blocks, cur, queries, rstd, logits = ctx.saved_tensors
        g = grad_out.contiguous()
        if g.dtype != torch.bfloat16:
            g = g.to(torch.bfloat16)
        B, T, D = blocks.shape
        M = queries.shape[0]
        S = B + 1
        S_PAD = triton.next_power_of_2(S)
        M_PAD = triton.next_power_of_2(M)
        dev = blocks.device
        inv_sqrt_d = float(D) ** -0.5

        a_c = torch.empty((T, S, M), device=dev, dtype=torch.float32)
        b_c = torch.empty((T, S, M), device=dev, dtype=torch.float32)
        c_c = torch.empty((T, S), device=dev, dtype=torch.float32)
        _bwd_stats_kernel[(T,)](
            blocks, cur, g, rstd, logits, a_c, b_c, c_c,
            T, D, B, S, M, inv_sqrt_d, 1.0 / D,
            S_PAD=S_PAD, M_PAD=M_PAD, BD=_stats_bd(S_PAD, M_PAD), num_warps=4,
        )

        dblocks = torch.empty_like(blocks)
        dcur = torch.empty_like(cur)
        dq = torch.zeros((M, D), device=dev, dtype=torch.float32)
        BD = 64
        BT = max(1, 32 // M_PAD)
        n_d = triton.cdiv(D, BD)
        NTB = triton.cdiv(T, BT)
        NSPLIT = max(1, min(NTB, (_sm_count(dev) * 8) // n_d))
        _bwd_apply_kernel[(n_d, NSPLIT)](
            blocks, cur, g, queries, a_c, b_c, c_c, dblocks, dcur, dq,
            T, D, B, S, M, NTB, NSPLIT,
            M_PAD=M_PAD, BT=BT, BD=BD, num_warps=4,
        )
        return dblocks, dcur, dq, None


def block_attnres(
    blocks: torch.Tensor,
    cur: torch.Tensor,
    queries: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return _BlockAttnRes.apply(blocks, cur, queries, eps)
