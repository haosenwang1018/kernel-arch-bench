"""Top-K routed SwiGLU MoE FFN: grouped Triton GEMM kernels for forward and backward."""

import torch
import triton
import triton.language as tl

# 保存 gate/up 激活 (bf16) 的上限；超过则在 backward 中重算
_SAVE_LIMIT_BYTES = 100 * 1024 * 1024

# 按行分块 (routed rows) 的 kernel: BM 固定为 64；(BN, BK, num_warps, num_stages)
_BM = 64
_GU_CFG = (128, 32, 4, 4)  # 每个 tile 同时累加 gate / up 两个 accumulator
_ROW_CFG = (128, 64, 8, 3)
# 权重梯度 kernel: (BM, BN, BK, num_warps, num_stages)
_WG_CFG = (128, 128, 32, 4, 5)


@triton.jit
def _fwd_gu_kernel(x_ptr, w_ptr, gu_ptr, h_ptr, tok_ptr, be_ptr, bs_ptr, bn_ptr, D, I,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(be_ptr + pb)
    start = tl.load(bs_ptr + pb)
    end = tl.load(bn_ptr + pb)
    N2 = 2 * I
    rows = start + tl.arange(0, BM)
    rmask = rows < end
    tok = tl.load(tok_ptr + rows, mask=rmask, other=0)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < I
    wbase = w_ptr + e.to(tl.int64) * D * N2
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < D
        a = tl.load(x_ptr + tok[:, None] * D + ks[None, :], mask=kmask[None, :], other=0.0)
        bptr = wbase + ks[:, None] * N2 + cols[None, :]
        bmask = kmask[:, None] & cmask[None, :]
        bg = tl.load(bptr, mask=bmask, other=0.0)
        bu = tl.load(bptr + I, mask=bmask, other=0.0)
        accg = tl.dot(a, bg, accg)
        accu = tl.dot(a, bu, accu)
    gb = accg.to(tl.bfloat16)
    ub = accu.to(tl.bfloat16)
    omask = rmask[:, None] & cmask[None, :]
    optr = gu_ptr + rows[:, None] * N2 + cols[None, :]
    tl.store(optr, gb, mask=omask)
    tl.store(optr + I, ub, mask=omask)
    g = gb.to(tl.float32)
    u = ub.to(tl.float32)
    h = g * tl.sigmoid(g) * u
    tl.store(h_ptr + rows[:, None] * I + cols[None, :], h.to(tl.bfloat16), mask=omask)


@triton.jit
def _fwd_down_kernel(h_ptr, wd_ptr, p_ptr, out_ptr, tok_ptr, be_ptr, bs_ptr, bn_ptr, D, I,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(be_ptr + pb)
    start = tl.load(bs_ptr + pb)
    end = tl.load(bn_ptr + pb)
    if end > start:
        rows = start + tl.arange(0, BM)
        rmask = rows < end
        tok = tl.load(tok_ptr + rows, mask=rmask, other=0)
        p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
        cols = pn * BN + tl.arange(0, BN)
        cmask = cols < D
        wbase = wd_ptr + e.to(tl.int64) * I * D
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, I, BK):
            ks = k0 + tl.arange(0, BK)
            kmask = ks < I
            h = tl.load(h_ptr + rows[:, None] * I + ks[None, :],
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            b = tl.load(wbase + ks[:, None] * D + cols[None, :],
                        mask=kmask[:, None] & cmask[None, :], other=0.0)
            acc = tl.dot(h, b, acc)
        acc = acc * p[:, None]
        tl.atomic_add(out_ptr + tok[:, None] * D + cols[None, :], acc,
                      mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _bwd_dh_kernel(go_ptr, wd_ptr, gu_ptr, p_ptr, dgu_ptr, dp_ptr, h_ptr, tok_ptr,
                   be_ptr, bs_ptr, bn_ptr, D, I,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(be_ptr + pb)
    start = tl.load(bs_ptr + pb)
    end = tl.load(bn_ptr + pb)
    if end > start:
        rows = start + tl.arange(0, BM)
        rmask = rows < end
        tok = tl.load(tok_ptr + rows, mask=rmask, other=0)
        p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
        cols = pn * BN + tl.arange(0, BN)
        cmask = cols < I
        wbase = wd_ptr + e.to(tl.int64) * I * D
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            ks = k0 + tl.arange(0, BK)
            kmask = ks < D
            a = tl.load(go_ptr + tok[:, None] * D + ks[None, :], mask=kmask[None, :], other=0.0)
            bt = tl.load(wbase + cols[:, None] * D + ks[None, :],
                         mask=cmask[:, None] & kmask[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(bt), acc)
        omask = rmask[:, None] & cmask[None, :]
        optr = rows[:, None] * (2 * I) + cols[None, :]
        g = tl.load(gu_ptr + optr, mask=omask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + optr + I, mask=omask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        sg = g * sig
        h = sg * u
        tl.store(h_ptr + rows[:, None] * I + cols[None, :], h.to(tl.bfloat16), mask=omask)
        dp = tl.sum(acc * h, axis=1)
        tl.atomic_add(dp_ptr + rows, dp, mask=rmask)
        dh = acc * p[:, None]
        dg = dh * u * sig * (1.0 + g * (1.0 - sig))
        du = dh * sg
        tl.store(dgu_ptr + optr, dg.to(tl.bfloat16), mask=omask)
        tl.store(dgu_ptr + optr + I, du.to(tl.bfloat16), mask=omask)


@triton.jit
def _bwd_dx_kernel(dgu_ptr, w_ptr, dx_ptr, tok_ptr, be_ptr, bs_ptr, bn_ptr, D, N2,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(be_ptr + pb)
    start = tl.load(bs_ptr + pb)
    end = tl.load(bn_ptr + pb)
    if end > start:
        rows = start + tl.arange(0, BM)
        rmask = rows < end
        tok = tl.load(tok_ptr + rows, mask=rmask, other=0)
        cols = pn * BN + tl.arange(0, BN)
        cmask = cols < D
        wbase = w_ptr + e.to(tl.int64) * D * N2
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, N2, BK):
            ks = k0 + tl.arange(0, BK)
            kmask = ks < N2
            a = tl.load(dgu_ptr + rows[:, None] * N2 + ks[None, :],
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            bt = tl.load(wbase + cols[:, None] * N2 + ks[None, :],
                         mask=cmask[:, None] & kmask[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(bt), acc)
        tl.atomic_add(dx_ptr + tok[:, None] * D + cols[None, :], acc,
                      mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _bwd_dwgu_kernel(x_ptr, dgu_ptr, dw_ptr, tok_ptr, st_ptr, ms_ptr, D, N2,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.program_id(2)
    s = tl.load(st_ptr + e)
    m = tl.load(ms_ptr + e)
    mrows = pm * BM + tl.arange(0, BM)
    mmask = mrows < D
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N2
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, m, BK):
        ks = s + k0 + tl.arange(0, BK)
        kmask = ks < s + m
        tok = tl.load(tok_ptr + ks, mask=kmask, other=0)
        at = tl.load(x_ptr + tok[:, None] * D + mrows[None, :],
                     mask=kmask[:, None] & mmask[None, :], other=0.0)
        b = tl.load(dgu_ptr + ks[:, None] * N2 + cols[None, :],
                    mask=kmask[:, None] & cmask[None, :], other=0.0)
        acc = tl.dot(tl.trans(at), b, acc)
    obase = dw_ptr + e.to(tl.int64) * D * N2
    tl.store(obase + mrows[:, None] * N2 + cols[None, :], acc.to(tl.bfloat16),
             mask=mmask[:, None] & cmask[None, :])


@triton.jit
def _bwd_dwd_kernel(h_ptr, go_ptr, p_ptr, dw_ptr, tok_ptr, st_ptr, ms_ptr, D, I,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.program_id(2)
    s = tl.load(st_ptr + e)
    m = tl.load(ms_ptr + e)
    mrows = pm * BM + tl.arange(0, BM)
    mmask = mrows < I
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, m, BK):
        ks = s + k0 + tl.arange(0, BK)
        kmask = ks < s + m
        tok = tl.load(tok_ptr + ks, mask=kmask, other=0)
        p = tl.load(p_ptr + ks, mask=kmask, other=0.0)
        ht = tl.load(h_ptr + ks[:, None] * I + mrows[None, :],
                     mask=kmask[:, None] & mmask[None, :], other=0.0)
        gz = tl.load(go_ptr + tok[:, None] * D + cols[None, :],
                     mask=kmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        b = (gz * p[:, None]).to(tl.bfloat16)
        acc = tl.dot(tl.trans(ht), b, acc)
    obase = dw_ptr + e.to(tl.int64) * I * D
    tl.store(obase + mrows[:, None] * D + cols[None, :], acc.to(tl.bfloat16),
             mask=mmask[:, None] & cmask[None, :])


def _cdiv(a, b):
    return (a + b - 1) // b


def _schedule(m_sizes, device):
    """每次调用根据当前 m_sizes 生成行分块调度表（不跨调用缓存）。"""
    ms = m_sizes.tolist()
    be, bs, bn, starts = [], [], [], []
    s = 0
    for e, m in enumerate(ms):
        starts.append(s)
        for r in range(s, s + m, _BM):
            be.append(e)
            bs.append(r)
            bn.append(min(r + _BM, s + m))
        s += m
    NB = len(be)
    flat = torch.tensor(be + bs + bn + starts, dtype=torch.int32, device=device)
    return NB, flat[:NB], flat[NB:2 * NB], flat[2 * NB:3 * NB], flat[3 * NB:], (be, bs, bn)


def _run_fwd_gu(x, w_gate_up, gu, h, token_ids, NB, be, bs, bn, D, I):
    BN, BK, nw, ns = _GU_CFG
    _fwd_gu_kernel[(NB, _cdiv(I, BN))](
        x, w_gate_up, gu, h, token_ids, be, bs, bn, D, I,
        BM=_BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        N_tok, D = x.shape
        E, I = w_down.shape[0], w_down.shape[1]
        N2 = 2 * I
        T = token_ids.numel()
        ctx.shapes = (N_tok, D, E, I, T)
        if T == 0:
            ctx.empty = True
            return torch.zeros_like(x)
        ctx.empty = False
        dev = x.device
        NB, be, bs, bn, _, _ = _schedule(m_sizes, dev)
        BN, BK, nw, ns = _ROW_CFG
        gu = torch.empty((T, N2), dtype=torch.bfloat16, device=dev)
        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        _run_fwd_gu(x, w_gate_up, gu, h, token_ids, NB, be, bs, bn, D, I)
        out32 = torch.zeros((N_tok, D), dtype=torch.float32, device=dev)
        _fwd_down_kernel[(NB, _cdiv(D, BN))](
            h, w_down, probs, out32, token_ids, be, bs, bn, D, I,
            BM=_BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        out = out32.to(torch.bfloat16)
        ctx.save_for_backward(x, w_gate_up, w_down, probs)
        ctx.token_ids = token_ids
        ctx.m_sizes = m_sizes
        # 只保留内存预算内的前 R 行 gate/up 激活，其余行在 backward 中重算
        R = min(T, _SAVE_LIMIT_BYTES // (N2 * 2))
        ctx.gu = gu if R >= T else gu[:R].clone()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        N_tok, D, E, I, T = ctx.shapes
        N2 = 2 * I
        need_x, need_wgu, need_wd, need_p = ctx.needs_input_grad[:4]
        if ctx.empty:
            dev = grad_out.device
            return (
                torch.zeros((N_tok, D), dtype=torch.bfloat16, device=dev) if need_x else None,
                torch.zeros((E, D, N2), dtype=torch.bfloat16, device=dev) if need_wgu else None,
                torch.zeros((E, I, D), dtype=torch.bfloat16, device=dev) if need_wd else None,
                torch.zeros((0,), dtype=torch.float32, device=dev) if need_p else None,
                None, None)
        x, w_gate_up, w_down, probs = ctx.saved_tensors
        token_ids, m_sizes = ctx.token_ids, ctx.m_sizes
        go = grad_out.contiguous()
        if go.dtype != torch.bfloat16:
            go = go.to(torch.bfloat16)
        dev = go.device
        NB, be, bs, bn, starts, (lbe, lbs, lbn) = _schedule(m_sizes, dev)
        BN, BK, nw, ns = _ROW_CFG
        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        gu = ctx.gu
        R = gu.shape[0]
        if R < T:
            saved = gu
            gu = torch.empty((T, N2), dtype=torch.bfloat16, device=dev)
            gu[:R] = saved
            idx = [i for i in range(NB) if lbn[i] > R]
            sub = torch.tensor([lbe[i] for i in idx] + [lbs[i] for i in idx] + [lbn[i] for i in idx],
                               dtype=torch.int32, device=dev)
            n = len(idx)
            _run_fwd_gu(x, w_gate_up, gu, h, token_ids, n, sub[:n], sub[n:2 * n], sub[2 * n:], D, I)

        dgu = torch.empty((T, N2), dtype=torch.bfloat16, device=dev)
        dprobs = torch.zeros((T,), dtype=torch.float32, device=dev)
        _bwd_dh_kernel[(NB, _cdiv(I, BN))](
            go, w_down, gu, probs, dgu, dprobs, h, token_ids, be, bs, bn, D, I,
            BM=_BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)

        dx = None
        if need_x:
            dx32 = torch.zeros((N_tok, D), dtype=torch.float32, device=dev)
            _bwd_dx_kernel[(NB, _cdiv(D, BN))](
                dgu, w_gate_up, dx32, token_ids, be, bs, bn, D, N2,
                BM=_BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
            dx = dx32.to(torch.bfloat16)

        WM, WN, WK, wnw, wns = _WG_CFG
        dwgu = None
        if need_wgu:
            dwgu = torch.empty((E, D, N2), dtype=torch.bfloat16, device=dev)
            _bwd_dwgu_kernel[(_cdiv(D, WM), _cdiv(N2, WN), E)](
                x, dgu, dwgu, token_ids, starts, m_sizes, D, N2,
                BM=WM, BN=WN, BK=WK, num_warps=wnw, num_stages=wns)
        dwd = None
        if need_wd:
            dwd = torch.empty((E, I, D), dtype=torch.bfloat16, device=dev)
            _bwd_dwd_kernel[(_cdiv(I, WM), _cdiv(D, WN), E)](
                h, go, probs, dwd, token_ids, starts, m_sizes, D, I,
                BM=WM, BN=WN, BK=WK, num_warps=wnw, num_stages=wns)
        return dx, dwgu, dwd, (dprobs if need_p else None), None, None


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)
