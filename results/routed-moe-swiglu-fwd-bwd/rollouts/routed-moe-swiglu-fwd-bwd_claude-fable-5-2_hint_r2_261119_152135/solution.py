"""Top-K routed SwiGLU MoE FFN: Triton forward + backward (no vendor GEMM)."""

import torch
import triton
import triton.language as tl

_SAVE_CAP_BYTES = 112 * 1024 * 1024
_CFG = {}  # optional overrides (used only for tuning experiments)


# ----------------------------------------------------------------------------
# M-grouped GEMM: C[r, :] = A[r, :] . B_e   (rows r routed to expert e)
#   TRANS_B=False: B_e is [K, N] row-major.  TRANS_B=True: B_e is [N, K].
#   SCATTER=False: store bf16 at C[r, n].
#   SCATTER=True : atomic-add fp32 (optionally scaled by p[r]) at C[tid[r], n].
# ----------------------------------------------------------------------------
@triton.jit
def _g1_kernel(
    a_ptr, b_ptr, c_ptr, tid_ptr, p_ptr, info_ptr,
    K, N, stride_a, stride_be, stride_b, stride_c, row_lo, num_pid_m, num_pid_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
    TRANS_B: tl.constexpr, SCATTER: tl.constexpr, USE_P: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = GROUP_M * num_pid_n
    group_id = pid // npg
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % npg) % group_size_m)
    pid_n = (pid % npg) // group_size_m

    e = tl.load(info_ptr + pid_m * 3)
    row0 = tl.load(info_ptr + pid_m * 3 + 1)
    row1 = tl.load(info_ptr + pid_m * 3 + 2)
    if row1 <= row0:
        return

    rows = row0 + tl.arange(0, BM)
    rmask = rows < row1
    rows_c = tl.minimum(rows, row1 - 1)
    offs_k = tl.arange(0, BK)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N

    a_ptrs = a_ptr + rows_c[:, None] * stride_a + offs_k[None, :]
    b_base = b_ptr + e.to(tl.int64) * stride_be
    if TRANS_B:
        b_ptrs = b_base + offs_n[:, None] * stride_b + offs_k[None, :]
    else:
        b_ptrs = b_base + offs_k[:, None] * stride_b + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        if MASKED:
            kmask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=rmask[:, None] & kmask[None, :], other=0.0)
            if TRANS_B:
                bt = tl.load(b_ptrs, mask=nmask[:, None] & kmask[None, :], other=0.0)
                b = tl.trans(bt)
            else:
                b = tl.load(b_ptrs, mask=kmask[:, None] & nmask[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs)
            if TRANS_B:
                bt = tl.load(b_ptrs)
                b = tl.trans(bt)
            else:
                b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BK
        if TRANS_B:
            b_ptrs += BK
        else:
            b_ptrs += BK * stride_b

    cmask = rmask[:, None] & nmask[None, :]
    if SCATTER:
        crow = tl.load(tid_ptr + rows_c)
        if USE_P:
            p = tl.load(p_ptr + rows_c)
            acc = acc * p[:, None]
        c_ptrs = c_ptr + crow[:, None] * stride_c + offs_n[None, :]
        tl.atomic_add(c_ptrs, acc, mask=cmask)
    else:
        cmask = cmask & (rows >= row_lo)[:, None]
        c_ptrs = c_ptr + rows[:, None] * stride_c + offs_n[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)


# ----------------------------------------------------------------------------
# K-grouped GEMM (weight gradients): C_e[m, n] = sum_{r in expert e} A[r, m] * B[r, n]
# ----------------------------------------------------------------------------
@triton.jit
def _g2_kernel(
    a_ptr, b_ptr, c_ptr, einfo_ptr,
    M, N, stride_a, stride_b, stride_ce,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.program_id(2)
    row0 = tl.load(einfo_ptr + 2 * e)
    row1 = tl.load(einfo_ptr + 2 * e + 1)
    m = row1 - row0

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mmask = offs_m < M
    nmask = offs_n < N

    a_ptrs = a_ptr + (row0 + offs_k)[:, None] * stride_a + offs_m[None, :]
    b_ptrs = b_ptr + (row0 + offs_k)[:, None] * stride_b + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, m, BK):
        kmask = (k + offs_k) < m
        at = tl.load(a_ptrs, mask=kmask[:, None] & mmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc += tl.dot(tl.trans(at), b)
        a_ptrs += BK * stride_a
        b_ptrs += BK * stride_b

    c_ptrs = c_ptr + e.to(tl.int64) * stride_ce + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


# ----------------------------------------------------------------------------
# Elementwise kernels
# ----------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(gu_ptr, h_ptr, I, BC: tl.constexpr):
    pid_c = tl.program_id(0)
    r = tl.program_id(1)
    o = pid_c * BC + tl.arange(0, BC)
    mask = o < I
    base = gu_ptr + r * 2 * I
    g = tl.load(base + o, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + I + o, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    h = g * sig * u
    tl.store(h_ptr + r * I + o, h.to(tl.bfloat16), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(gu_ptr, dht_ptr, p_ptr, dgu_ptr, hp_ptr, dp_ptr, I, BC: tl.constexpr):
    r = tl.program_id(0)
    p = tl.load(p_ptr + r)
    offs = tl.arange(0, BC)
    acc = tl.zeros((BC,), dtype=tl.float32)
    gbase = gu_ptr + r * 2 * I
    dbase = dgu_ptr + r * 2 * I
    for c in range(0, I, BC):
        o = c + offs
        mask = o < I
        g = tl.load(gbase + o, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gbase + I + o, mask=mask, other=0.0).to(tl.float32)
        d = tl.load(dht_ptr + r * I + o, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        s = g * sig
        h = s * u
        acc += h * d
        dh = d * p
        dg = dh * u * sig * (1.0 + g * (1.0 - sig))
        du = dh * s
        tl.store(dbase + o, dg.to(tl.bfloat16), mask=mask)
        tl.store(dbase + I + o, du.to(tl.bfloat16), mask=mask)
        tl.store(hp_ptr + r * I + o, (h * p).to(tl.bfloat16), mask=mask)
    tl.store(dp_ptr + r, tl.sum(acc, axis=0))


# ----------------------------------------------------------------------------
# Host-side helpers (all schedule tensors are built on device, per call)
# ----------------------------------------------------------------------------
def _tile_info(m_sizes, BM, T, E):
    dev = m_sizes.device
    m64 = m_sizes.to(torch.int64)
    row_end = torch.cumsum(m64, 0)
    row_start = row_end - m64
    nt = torch.div(m64 + (BM - 1), BM, rounding_mode="floor")
    tile_end = torch.cumsum(nt, 0)
    tile_start = tile_end - nt
    max_tiles = T // BM + E
    tiles = torch.arange(max_tiles, device=dev, dtype=torch.int64)
    te = torch.searchsorted(tile_end, tiles, right=True)
    valid = te < E
    tec = te.clamp(max=E - 1)
    r0 = row_start[tec] + (tiles - tile_start[tec]) * BM
    r1 = row_end[tec]
    zero = torch.zeros_like(r0)
    r0 = torch.where(valid, r0, zero)
    r1 = torch.where(valid, r1, zero)
    info = torch.stack([tec, r0, r1], dim=1).to(torch.int32).contiguous()
    einfo = torch.stack([row_start, row_end], dim=1).to(torch.int32).contiguous()
    return info, einfo, max_tiles


def _g1_cfg(BM, K, N):
    if "g1" in _CFG:
        BN, BK, warps, stages = _CFG["g1"]
    elif BM >= 128 and N % 256 == 0 and K % 64 == 0:
        BN, BK, warps, stages = 256, 64, 8, 3
    elif BM >= 128:
        BN, BK, warps, stages = 128, 32, 4, 4
    elif N % 256 == 0:
        BN, BK, warps, stages = 256, 32, 4, 4
    else:
        BN, BK, warps, stages = 128, 32, 4, 4
    return BN, BK, warps, stages


def _run_g1(a, b, c, tid, p, info, max_tiles, BM, K, N, stride_a, stride_be, stride_b,
            stride_c, trans_b, scatter, use_p, row_lo=0):
    BN, BK, warps, stages = _g1_cfg(BM, K, N)
    num_pid_n = (N + BN - 1) // BN
    masked = (K % BK != 0) or (N % BN != 0)
    grid = (max_tiles * num_pid_n,)
    _g1_kernel[grid](
        a, b, c, tid, p, info,
        K, N, stride_a, stride_be, stride_b, stride_c, row_lo, max_tiles, num_pid_n,
        BM=BM, BN=BN, BK=BK, GROUP_M=_CFG.get("group_m", 8),
        TRANS_B=trans_b, SCATTER=scatter, USE_P=use_p, MASKED=masked,
        num_warps=warps, num_stages=stages,
    )


def _run_g2(a, b, c, einfo, E, M, N, stride_a, stride_b, small):
    if "g2" in _CFG:
        BM, BN, BK, warps, stages = _CFG["g2"]
    else:
        BM, BN, BK, warps, stages = 128, 128, (16 if small else 32), 4, 3
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, E)
    _g2_kernel[grid](
        a, b, c, einfo,
        M, N, stride_a, stride_b, M * N,
        BM=BM, BN=BN, BK=BK,
        num_warps=warps, num_stages=stages,
    )


def _pick_bm(s):
    if "bm" in _CFG:
        return _CFG["bm"]
    if s >= 256:
        return 128
    if s >= 128:
        return 64
    if s >= 64:
        return 32
    return 16


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        N_tok, D = x.shape
        E = w_gate_up.shape[0]
        I = w_down.shape[1]
        T = token_ids.numel()
        ctx.dims = (N_tok, D, E, I, T)
        ctx.save_for_backward(x, w_gate_up, w_down, probs)
        ctx.token_ids = token_ids
        if T == 0:
            return torch.zeros_like(x)

        # row-weighted mean expert size -> tile height (recomputed every call)
        mf = m_sizes.to(torch.float32)
        s = float((mf * mf).sum()) / T
        BM = _pick_bm(s)
        info, einfo, max_tiles = _tile_info(m_sizes, BM, T, E)

        xp = x.detach().index_select(0, token_ids)
        gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=x.device)
        _run_g1(xp, w_gate_up, gu, token_ids, probs, info, max_tiles, BM,
                K=D, N=2 * I, stride_a=D, stride_be=D * 2 * I, stride_b=2 * I,
                stride_c=2 * I, trans_b=False, scatter=False, use_p=False)
        del xp
        h = torch.empty((T, I), dtype=torch.bfloat16, device=x.device)
        BC = 1024
        _silu_mul_kernel[((I + BC - 1) // BC, T)](gu, h, I, BC=BC, num_warps=4)

        outf = torch.zeros((N_tok, D), dtype=torch.float32, device=x.device)
        _run_g1(h, w_down, outf, token_ids, probs, info, max_tiles, BM,
                K=I, N=D, stride_a=I, stride_be=I * D, stride_b=D,
                stride_c=D, trans_b=False, scatter=True, use_p=True)
        out = outf.to(torch.bfloat16)

        R = min(T, _SAVE_CAP_BYTES // (4 * I))
        if R < T:
            ctx.gu = gu[:R].clone() if R > 0 else None
        else:
            ctx.gu = gu
        ctx.R = R
        ctx.BM = BM
        ctx.small = s < 24
        ctx.info = info
        ctx.einfo = einfo
        ctx.max_tiles = max_tiles
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, w_gate_up, w_down, probs = ctx.saved_tensors
        N_tok, D, E, I, T = ctx.dims
        token_ids = ctx.token_ids
        dev = x.device
        need_x, need_wgu, need_wd, need_p = ctx.needs_input_grad[:4]
        if T == 0:
            return (
                torch.zeros_like(x) if need_x else None,
                torch.zeros_like(w_gate_up) if need_wgu else None,
                torch.zeros_like(w_down) if need_wd else None,
                torch.zeros_like(probs) if need_p else None,
                None, None,
            )
        BM, info, einfo, max_tiles = ctx.BM, ctx.info, ctx.einfo, ctx.max_tiles
        go = grad_out.to(torch.bfloat16).contiguous()
        pd = probs.detach()
        xd = x.detach()

        xp = None
        R = ctx.R
        if R < T:
            # gu did not fit under the saved-memory cap: recompute the tail rows
            gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
            if R > 0:
                gu[:R].copy_(ctx.gu)
            tend = torch.minimum(info[:, 1] + BM, info[:, 2])
            info2 = info.clone()
            info2[:, 2] = torch.where(tend <= R, info[:, 1], info[:, 2])
            xp = xd.index_select(0, token_ids)
            _run_g1(xp, w_gate_up, gu, token_ids, pd, info2, max_tiles, BM,
                    K=D, N=2 * I, stride_a=D, stride_be=D * 2 * I, stride_b=2 * I,
                    stride_c=2 * I, trans_b=False, scatter=False, use_p=False, row_lo=R)
        else:
            gu = ctx.gu

        gop = go.index_select(0, token_ids)
        # dh~ = go[tid] . W_down^T
        dht = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        _run_g1(gop, w_down, dht, token_ids, pd, info, max_tiles, BM,
                K=D, N=I, stride_a=D, stride_be=I * D, stride_b=D,
                stride_c=I, trans_b=True, scatter=False, use_p=False)

        dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        hp = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        dprobs = torch.empty((T,), dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(T,)](gu, dht, pd, dgu, hp, dprobs, I, BC=1024, num_warps=4)
        del dht, gu

        dwd = None
        if need_wd:
            dwd = torch.empty((E, I, D), dtype=torch.bfloat16, device=dev)
            _run_g2(hp, gop, dwd, einfo, E, M=I, N=D, stride_a=I, stride_b=D, small=ctx.small)
        del hp, gop

        dwgu = None
        if need_wgu:
            if xp is None:
                xp = xd.index_select(0, token_ids)
            dwgu = torch.empty((E, D, 2 * I), dtype=torch.bfloat16, device=dev)
            _run_g2(xp, dgu, dwgu, einfo, E, M=D, N=2 * I, stride_a=D, stride_b=2 * I,
                    small=ctx.small)
        del xp

        dx = None
        if need_x:
            dxf = torch.zeros((N_tok, D), dtype=torch.float32, device=dev)
            _run_g1(dgu, w_gate_up, dxf, token_ids, pd, info, max_tiles, BM,
                    K=2 * I, N=D, stride_a=2 * I, stride_be=D * 2 * I, stride_b=2 * I,
                    stride_c=D, trans_b=True, scatter=True, use_p=False)
            dx = dxf.to(torch.bfloat16)

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
