"""Top-K routed SwiGLU MoE FFN: grouped Triton GEMM kernels (forward + backward)."""

import torch
import triton
import triton.language as tl

# 前向与反向之间允许保留的 autograd 显存预算（上限 120 MB，留出余量）
_SAVE_BUDGET = 108 * 1024 * 1024

# (BN, BK, num_warps, num_stages)
_CFG_GU = (128, 32, 8, 3)
_CFG_ROW = (128, 32, 4, 3)
_CFG_W = (128, 128, 32, 4, 3)  # BM, BN, BK, warps, stages


@triton.jit
def _k_gate_up(x_ptr, w_ptr, tid_ptr, tile_ptr, NT, gu_ptr, h_ptr, D, I,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STORE_H: tl.constexpr):
    # g = x Wg, u = x Wu, h = silu(g) * u （按 expert 的行块分 tile）
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tile_ptr + pt).to(tl.int64)
    r0 = tl.load(tile_ptr + NT + pt)
    nr = tl.load(tile_ptr + 2 * NT + pt)
    offs_r = tl.arange(0, BM)
    rmask = offs_r < nr
    rows = r0 + offs_r
    tok = tl.load(tid_ptr + rows, mask=rmask, other=0)
    offs_n = pn * BN + tl.arange(0, BN)
    nmask = offs_n < I
    a_base = x_ptr + tok[:, None] * D
    w_base = w_ptr + e * D * (2 * I)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, D, BK):
        offs_k = k + tl.arange(0, BK)
        kmask = offs_k < D
        a = tl.load(a_base + offs_k[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        wrow = w_base + offs_k[:, None] * (2 * I)
        bmask = kmask[:, None] & nmask[None, :]
        bg = tl.load(wrow + offs_n[None, :], mask=bmask, other=0.0)
        bu = tl.load(wrow + I + offs_n[None, :], mask=bmask, other=0.0)
        accg = tl.dot(a, bg, accg)
        accu = tl.dot(a, bu, accu)
    omask = rmask[:, None] & nmask[None, :]
    gu_row = gu_ptr + rows[:, None] * (2 * I) + offs_n[None, :]
    tl.store(gu_row, accg.to(tl.bfloat16), mask=omask)
    tl.store(gu_row + I, accu.to(tl.bfloat16), mask=omask)
    if STORE_H:
        sig = 1.0 / (1.0 + tl.exp(-accg))
        h = accg * sig * accu
        tl.store(h_ptr + rows[:, None] * I + offs_n[None, :], h.to(tl.bfloat16), mask=omask)


@triton.jit
def _k_down(h_ptr, w_ptr, tid_ptr, p_ptr, tile_ptr, NT, out_ptr, D, I,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # out[tok] += p * (h Wd)
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tile_ptr + pt).to(tl.int64)
    r0 = tl.load(tile_ptr + NT + pt)
    nr = tl.load(tile_ptr + 2 * NT + pt)
    offs_r = tl.arange(0, BM)
    rmask = offs_r < nr
    rows = r0 + offs_r
    tok = tl.load(tid_ptr + rows, mask=rmask, other=0)
    p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
    offs_n = pn * BN + tl.arange(0, BN)
    nmask = offs_n < D
    a_base = h_ptr + rows[:, None] * I
    w_base = w_ptr + e * I * D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, I, BK):
        offs_k = k + tl.arange(0, BK)
        kmask = offs_k < I
        a = tl.load(a_base + offs_k[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(w_base + offs_k[:, None] * D + offs_n[None, :],
                    mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    acc = acc * p[:, None]
    tl.atomic_add(out_ptr + tok[:, None] * D + offs_n[None, :], acc,
                  mask=rmask[:, None] & nmask[None, :])


@triton.jit
def _k_dh(go_ptr, w_ptr, tid_ptr, p_ptr, tile_ptr, NT, gu_ptr, h_ptr, dgu_ptr, dp_ptr, D, I,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # dh' = go[tok] Wd^T; dprobs = <h, dh'>; dg/du = p * dh' * silu 导数项
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tile_ptr + pt).to(tl.int64)
    r0 = tl.load(tile_ptr + NT + pt)
    nr = tl.load(tile_ptr + 2 * NT + pt)
    offs_r = tl.arange(0, BM)
    rmask = offs_r < nr
    rows = r0 + offs_r
    tok = tl.load(tid_ptr + rows, mask=rmask, other=0)
    p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
    offs_n = pn * BN + tl.arange(0, BN)
    nmask = offs_n < I
    a_base = go_ptr + tok[:, None] * D
    w_base = w_ptr + e * I * D + offs_n[None, :] * D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, D, BK):
        offs_k = k + tl.arange(0, BK)
        kmask = offs_k < D
        a = tl.load(a_base + offs_k[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(w_base + offs_k[:, None], mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    omask = rmask[:, None] & nmask[None, :]
    gu_row = gu_ptr + rows[:, None] * (2 * I) + offs_n[None, :]
    g = tl.load(gu_row, mask=omask, other=0.0).to(tl.float32)
    u = tl.load(gu_row + I, mask=omask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    s = g * sig
    h = s * u
    tl.store(h_ptr + rows[:, None] * I + offs_n[None, :], h.to(tl.bfloat16), mask=omask)
    tl.atomic_add(dp_ptr + rows, tl.sum(h * acc, axis=1), mask=rmask)
    dh = acc * p[:, None]
    dg = dh * u * sig * (1.0 + g * (1.0 - sig))
    du = dh * s
    dgu_row = dgu_ptr + rows[:, None] * (2 * I) + offs_n[None, :]
    tl.store(dgu_row, dg.to(tl.bfloat16), mask=omask)
    tl.store(dgu_row + I, du.to(tl.bfloat16), mask=omask)


@triton.jit
def _k_dx(dgu_ptr, w_ptr, tid_ptr, tile_ptr, NT, dx_ptr, D, I2,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # dx[tok] += dgu Wgu^T
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tile_ptr + pt).to(tl.int64)
    r0 = tl.load(tile_ptr + NT + pt)
    nr = tl.load(tile_ptr + 2 * NT + pt)
    offs_r = tl.arange(0, BM)
    rmask = offs_r < nr
    rows = r0 + offs_r
    tok = tl.load(tid_ptr + rows, mask=rmask, other=0)
    offs_n = pn * BN + tl.arange(0, BN)
    nmask = offs_n < D
    a_base = dgu_ptr + rows[:, None] * I2
    w_base = w_ptr + e * D * I2 + offs_n[None, :] * I2
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, I2, BK):
        offs_k = k + tl.arange(0, BK)
        kmask = offs_k < I2
        a = tl.load(a_base + offs_k[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(w_base + offs_k[:, None], mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.atomic_add(dx_ptr + tok[:, None] * D + offs_n[None, :], acc,
                  mask=rmask[:, None] & nmask[None, :])


@triton.jit
def _k_dwd(h_ptr, go_ptr, tid_ptr, p_ptr, off_ptr, m_ptr, dw_ptr, D, I,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # dWd[e] = h_e^T (p * go[tok])
    e = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    off = tl.load(off_ptr + e)
    m = tl.load(m_ptr + e)
    offs_m = pm * BM + tl.arange(0, BM)
    mmask = offs_m < I
    offs_n = pn * BN + tl.arange(0, BN)
    nmask = offs_n < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, m, BK):
        r = k + tl.arange(0, BK)
        kmask = r < m
        rows = off + r
        tok = tl.load(tid_ptr + rows, mask=kmask, other=0)
        p = tl.load(p_ptr + rows, mask=kmask, other=0.0)
        a = tl.load(h_ptr + rows[None, :] * I + offs_m[:, None],
                    mask=kmask[None, :] & mmask[:, None], other=0.0)
        b = tl.load(go_ptr + tok[:, None] * D + offs_n[None, :],
                    mask=kmask[:, None] & nmask[None, :], other=0.0)
        b = (b.to(tl.float32) * p[:, None]).to(tl.bfloat16)
        acc = tl.dot(a, b, acc)
    e64 = e.to(tl.int64)
    tl.store(dw_ptr + e64 * I * D + offs_m[:, None] * D + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _k_dwgu(x_ptr, dgu_ptr, tid_ptr, off_ptr, m_ptr, dw_ptr, D, I2,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # dWgu[e] = x_e^T dgu_e
    e = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    off = tl.load(off_ptr + e)
    m = tl.load(m_ptr + e)
    offs_m = pm * BM + tl.arange(0, BM)
    mmask = offs_m < D
    offs_n = pn * BN + tl.arange(0, BN)
    nmask = offs_n < I2
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, m, BK):
        r = k + tl.arange(0, BK)
        kmask = r < m
        rows = off + r
        tok = tl.load(tid_ptr + rows, mask=kmask, other=0)
        a = tl.load(x_ptr + tok[None, :] * D + offs_m[:, None],
                    mask=kmask[None, :] & mmask[:, None], other=0.0)
        b = tl.load(dgu_ptr + rows[:, None] * I2 + offs_n[None, :],
                    mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    e64 = e.to(tl.int64)
    tl.store(dw_ptr + e64 * D * I2 + offs_m[:, None] * I2 + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


def _cdiv(a, b):
    return (a + b - 1) // b


def _build_tiles(ms, device):
    """每次调用都由当前 m_sizes 重新构建行 tile 表（不跨调用缓存）。"""
    best = None
    for bm in (32, 64, 128):
        nt = sum(_cdiv(m, bm) for m in ms)
        cost = nt * (bm + 12)
        if best is None or cost < best[0]:
            best = (cost, bm)
    bm = best[1]
    te, ts, tn = [], [], []
    start = 0
    for e, m in enumerate(ms):
        r = 0
        while r < m:
            te.append(e)
            ts.append(start + r)
            tn.append(min(bm, m - r))
            r += bm
        start += m
    return bm, te, ts, tn


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        N, D = x.shape
        E, I = w_down.shape[0], w_down.shape[1]
        dev = x.device
        ms = m_sizes.tolist()
        T = sum(ms)
        ctx.dims = (N, D, E, I, T)
        ctx.save_for_backward(x, w_gate_up, w_down, probs)
        if T == 0:
            return torch.zeros_like(x)
        bm, te, ts, tn = _build_tiles(ms, dev)
        NT = len(te)
        tiles = torch.tensor(te + ts + tn, dtype=torch.int32, device=dev)
        offs = [0] * E
        acc = 0
        for e in range(E):
            offs[e] = acc
            acc += ms[e]
        offsets = torch.tensor(offs, dtype=torch.int32, device=dev)
        tids = token_ids.contiguous()
        msz = m_sizes.contiguous()

        gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        BN, BK, nw, ns = _CFG_GU
        _k_gate_up[(NT, _cdiv(I, BN))](x, w_gate_up, tids, tiles, NT, gu, h, D, I,
                                       BM=bm, BN=BN, BK=BK, STORE_H=True,
                                       num_warps=nw, num_stages=ns)
        out32 = torch.zeros((N, D), dtype=torch.float32, device=dev)
        BN, BK, nw, ns = _CFG_ROW
        _k_down[(NT, _cdiv(D, BN))](h, w_down, tids, probs, tiles, NT, out32, D, I,
                                    BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        del h
        out = out32.to(torch.bfloat16)

        # 在显存预算内保留 g/u；超出部分在反向时重算
        row_bytes = 4 * I
        R = min(T, _SAVE_BUDGET // row_bytes)
        tiles_re = None
        if R < T:
            gu = gu[:R].clone() if R > 0 else None
            sel = [i for i in range(NT) if ts[i] + tn[i] > R]
            tiles_re = torch.tensor([te[i] for i in sel] + [ts[i] for i in sel] + [tn[i] for i in sel],
                                    dtype=torch.int32, device=dev)
        ctx.state = (bm, tiles, NT, offsets, tids, msz, gu, R, tiles_re)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_gate_up, w_down, probs = ctx.saved_tensors
        N, D, E, I, T = ctx.dims
        dev = x.device
        if T == 0:
            return (torch.zeros_like(x), torch.zeros_like(w_gate_up), torch.zeros_like(w_down),
                    torch.zeros_like(probs), None, None)
        bm, tiles, NT, offsets, tids, msz, gu_saved, R, tiles_re = ctx.state
        go = grad_out.contiguous()
        if go.dtype != torch.bfloat16:
            go = go.to(torch.bfloat16)

        if R >= T:
            gu = gu_saved
        else:
            gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
            if R > 0:
                gu[:R] = gu_saved
            NTR = tiles_re.numel() // 3
            BN, BK, nw, ns = _CFG_GU
            _k_gate_up[(NTR, _cdiv(I, BN))](x, w_gate_up, tids, tiles_re, NTR, gu, gu, D, I,
                                            BM=bm, BN=BN, BK=BK, STORE_H=False,
                                            num_warps=nw, num_stages=ns)

        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        dprobs = torch.zeros((T,), dtype=torch.float32, device=dev)
        BN, BK, nw, ns = _CFG_ROW
        _k_dh[(NT, _cdiv(I, BN))](go, w_down, tids, probs, tiles, NT, gu, h, dgu, dprobs, D, I,
                                  BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        del gu

        dx32 = torch.zeros((N, D), dtype=torch.float32, device=dev)
        _k_dx[(NT, _cdiv(D, BN))](dgu, w_gate_up, tids, tiles, NT, dx32, D, 2 * I,
                                  BM=bm, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        dx = dx32.to(torch.bfloat16)
        del dx32

        WM, WN, WK, nw, ns = _CFG_W
        dwd = torch.empty((E, I, D), dtype=torch.bfloat16, device=dev)
        _k_dwd[(E, _cdiv(I, WM), _cdiv(D, WN))](h, go, tids, probs, offsets, msz, dwd, D, I,
                                                BM=WM, BN=WN, BK=WK, num_warps=nw, num_stages=ns)
        del h
        dwgu = torch.empty((E, D, 2 * I), dtype=torch.bfloat16, device=dev)
        _k_dwgu[(E, _cdiv(D, WM), _cdiv(2 * I, WN))](x, dgu, tids, offsets, msz, dwgu, D, 2 * I,
                                                     BM=WM, BN=WN, BK=WK, num_warps=nw, num_stages=ns)
        return dx, dwgu, dwd, dprobs, None, None


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)
