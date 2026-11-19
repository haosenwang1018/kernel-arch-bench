"""Top-K routed SwiGLU MoE FFN: fused Triton forward + backward."""

import torch
import triton
import triton.language as tl

_SAVE_BUDGET = 118 * 1024 * 1024
_QBN = 64  # int8 quantization block along columns (== fwd BN)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
@triton.jit
def _mgroup_tile(pid, ms_ptr, E, NUM_N, BM: tl.constexpr, E_POW2: tl.constexpr,
                 GROUP_M: tl.constexpr):
    offs_e = tl.arange(0, E_POW2)
    ms = tl.load(ms_ptr + offs_e, mask=offs_e < E, other=0)
    tiles = (ms + BM - 1) // BM
    cum_t = tl.cumsum(tiles, 0)
    cum_r = tl.cumsum(ms, 0)
    total = tl.sum(tiles, 0)
    valid = (pid < total * NUM_N).to(tl.int32)
    npg = GROUP_M * NUM_N
    gid = pid // npg
    first = gid * GROUP_M
    gsz = tl.maximum(tl.minimum(total - first, GROUP_M), 1)
    tile_m = first + (pid % npg) % gsz
    pid_n = (pid % npg) // gsz
    e = tl.sum((cum_t <= tile_m).to(tl.int32), 0)
    sel = offs_e == e
    m_e = tl.sum(tl.where(sel, ms, 0), 0)
    row_end = tl.sum(tl.where(sel, cum_r, 0), 0)
    t_before = tl.sum(tl.where(sel, cum_t - tiles, 0), 0)
    row0 = row_end - m_e + (tile_m - t_before) * BM
    return valid, e, row0, row_end, pid_n


@triton.jit
def _expert_range(ms_ptr, e, E, E_POW2: tl.constexpr):
    offs_e = tl.arange(0, E_POW2)
    ms = tl.load(ms_ptr + offs_e, mask=offs_e < E, other=0)
    cum_r = tl.cumsum(ms, 0)
    sel = offs_e == e
    m_e = tl.sum(tl.where(sel, ms, 0), 0)
    end = tl.sum(tl.where(sel, cum_r, 0), 0)
    return end - m_e, end


# ----------------------------------------------------------------------------
# forward: gu = x[tok] @ Wgu[e]; h = silu(g) * u
# MODE 0: store gu bf16, MODE 1: store gu int8 + scales, MODE 2: no gu store
# ----------------------------------------------------------------------------
@triton.jit
def _fwd_gu_kernel(x_ptr, w_ptr, tid_ptr, ms_ptr, gu_ptr, s_ptr, h_ptr,
                   E, D, I,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   E_POW2: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
                   MODE: tl.constexpr, STORE_H: tl.constexpr):
    pid = tl.program_id(0)
    NUM_N = tl.cdiv(I, BN)
    valid, e, row0, row_end, pid_n = _mgroup_tile(pid, ms_ptr, E, NUM_N, BM, E_POW2, GROUP_M)
    if valid == 0:
        return
    offs_m = row0 + tl.arange(0, BM)
    rmask = offs_m < row_end
    tok = tl.load(tid_ptr + offs_m, mask=rmask, other=0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < I
    offs_nb = offs_n % I
    offs_k = tl.arange(0, BK)
    N2 = 2 * I
    a_ptrs = x_ptr + tok.to(tl.int64)[:, None] * D + offs_k[None, :]
    b_ptrs = w_ptr + e.to(tl.int64) * D * N2 + offs_k[:, None] * N2 + offs_nb[None, :]
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(D, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            bg = tl.load(b_ptrs)
            bu = tl.load(b_ptrs + I)
        else:
            km = offs_k < D - k * BK
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            bg = tl.load(b_ptrs, mask=km[:, None], other=0.0)
            bu = tl.load(b_ptrs + I, mask=km[:, None], other=0.0)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        a_ptrs += BK
        b_ptrs += BK * N2
    rows64 = offs_m.to(tl.int64)
    omask = rmask[:, None] & nmask[None, :]
    if STORE_H:
        hv = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(h_ptr + rows64[:, None] * I + offs_n[None, :], hv.to(tl.bfloat16), mask=omask)
    if MODE == 0:
        gp = gu_ptr + rows64[:, None] * N2 + offs_n[None, :]
        tl.store(gp, acc_g.to(tl.bfloat16), mask=omask)
        tl.store(gp + I, acc_u.to(tl.bfloat16), mask=omask)
    elif MODE == 1:
        NQ = NUM_N
        ag = tl.where(omask, tl.abs(acc_g), 0.0)
        au = tl.where(omask, tl.abs(acc_u), 0.0)
        sg = (tl.maximum(tl.max(ag, 1), 1e-2) / 127.0).to(tl.float16).to(tl.float32)
        su = (tl.maximum(tl.max(au, 1), 1e-2) / 127.0).to(tl.float16).to(tl.float32)
        qg = tl.floor(acc_g / sg[:, None] + 0.5).to(tl.int8)
        qu = tl.floor(acc_u / su[:, None] + 0.5).to(tl.int8)
        gp = gu_ptr + rows64[:, None] * N2 + offs_n[None, :]
        tl.store(gp, qg, mask=omask)
        tl.store(gp + I, qu, mask=omask)
        sp = s_ptr + rows64 * (2 * NQ) + pid_n
        tl.store(sp, sg.to(tl.float16), mask=rmask)
        tl.store(sp + NQ, su.to(tl.float16), mask=rmask)


# ----------------------------------------------------------------------------
# forward: out[tok] += p * (h @ Wd[e])
# ----------------------------------------------------------------------------
@triton.jit
def _fwd_down_kernel(h_ptr, w_ptr, tid_ptr, p_ptr, ms_ptr, out_ptr,
                     E, D, I,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     E_POW2: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    NUM_N = tl.cdiv(D, BN)
    valid, e, row0, row_end, pid_n = _mgroup_tile(pid, ms_ptr, E, NUM_N, BM, E_POW2, GROUP_M)
    if valid == 0:
        return
    offs_m = row0 + tl.arange(0, BM)
    rmask = offs_m < row_end
    rows_c = tl.where(rmask, offs_m, row0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < D
    offs_nb = offs_n % D
    offs_k = tl.arange(0, BK)
    a_ptrs = h_ptr + rows_c.to(tl.int64)[:, None] * I + offs_k[None, :]
    b_ptrs = w_ptr + e.to(tl.int64) * I * D + offs_k[:, None] * D + offs_nb[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(I, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            km = offs_k < I - k * BK
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * D
    tok = tl.load(tid_ptr + offs_m, mask=rmask, other=0)
    p = tl.load(p_ptr + offs_m, mask=rmask, other=0.0)
    acc = acc * p[:, None]
    tl.atomic_add(out_ptr + tok.to(tl.int64)[:, None] * D + offs_n[None, :], acc,
                  mask=rmask[:, None] & nmask[None, :], sem="relaxed")


# ----------------------------------------------------------------------------
# backward: dh~ = go[tok] @ Wd[e]^T, fused SwiGLU backward + probs grad
# ----------------------------------------------------------------------------
@triton.jit
def _bwd_dh_kernel(go_ptr, wd_ptr, tid_ptr, p_ptr, ms_ptr, gu_ptr, s_ptr,
                   dgu_ptr, hp_ptr, dp_ptr,
                   E, D, I, NQ,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   E_POW2: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
                   MODE: tl.constexpr, QBN: tl.constexpr):
    pid = tl.program_id(0)
    NUM_N = tl.cdiv(I, BN)
    valid, e, row0, row_end, pid_n = _mgroup_tile(pid, ms_ptr, E, NUM_N, BM, E_POW2, GROUP_M)
    if valid == 0:
        return
    offs_m = row0 + tl.arange(0, BM)
    rmask = offs_m < row_end
    tok = tl.load(tid_ptr + offs_m, mask=rmask, other=0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < I
    offs_nb = offs_n % I
    offs_k = tl.arange(0, BK)
    a_ptrs = go_ptr + tok.to(tl.int64)[:, None] * D + offs_k[None, :]
    b_ptrs = wd_ptr + e.to(tl.int64) * I * D + offs_nb[None, :] * D + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(D, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            km = offs_k < D - k * BK
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    rows64 = offs_m.to(tl.int64)
    omask = rmask[:, None] & nmask[None, :]
    N2 = 2 * I
    gp = gu_ptr + rows64[:, None] * N2 + offs_n[None, :]
    g = tl.load(gp, mask=omask, other=0).to(tl.float32)
    u = tl.load(gp + I, mask=omask, other=0).to(tl.float32)
    if MODE == 1:
        sp = s_ptr + rows64[:, None] * (2 * NQ) + (offs_n // QBN)[None, :]
        g = g * tl.load(sp, mask=omask, other=0.0).to(tl.float32)
        u = u * tl.load(sp + NQ, mask=omask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    h = silu * u
    dpart = tl.sum(h * acc, 1)
    tl.atomic_add(dp_ptr + offs_m, dpart, mask=rmask, sem="relaxed")
    p = tl.load(p_ptr + offs_m, mask=rmask, other=0.0)
    dh = acc * p[:, None]
    du = dh * silu
    dg = dh * u * sig * (1.0 + g * (1.0 - sig))
    dp_ptrs = dgu_ptr + rows64[:, None] * N2 + offs_n[None, :]
    tl.store(dp_ptrs, dg.to(tl.bfloat16), mask=omask)
    tl.store(dp_ptrs + I, du.to(tl.bfloat16), mask=omask)
    tl.store(hp_ptr + rows64[:, None] * I + offs_n[None, :], (h * p[:, None]).to(tl.bfloat16),
             mask=omask)


# ----------------------------------------------------------------------------
# backward: dx[tok] += dgu @ Wgu[e]^T
# ----------------------------------------------------------------------------
@triton.jit
def _bwd_dx_kernel(dgu_ptr, w_ptr, tid_ptr, ms_ptr, dx_ptr,
                   E, D, I,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   E_POW2: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    NUM_N = tl.cdiv(D, BN)
    valid, e, row0, row_end, pid_n = _mgroup_tile(pid, ms_ptr, E, NUM_N, BM, E_POW2, GROUP_M)
    if valid == 0:
        return
    offs_m = row0 + tl.arange(0, BM)
    rmask = offs_m < row_end
    rows_c = tl.where(rmask, offs_m, row0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < D
    offs_nb = offs_n % D
    offs_k = tl.arange(0, BK)
    K2 = 2 * I
    a_ptrs = dgu_ptr + rows_c.to(tl.int64)[:, None] * K2 + offs_k[None, :]
    b_ptrs = w_ptr + e.to(tl.int64) * D * K2 + offs_nb[None, :] * K2 + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K2, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            km = offs_k < K2 - k * BK
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    tok = tl.load(tid_ptr + offs_m, mask=rmask, other=0)
    tl.atomic_add(dx_ptr + tok.to(tl.int64)[:, None] * D + offs_n[None, :], acc,
                  mask=rmask[:, None] & nmask[None, :], sem="relaxed")


# ----------------------------------------------------------------------------
# backward weight grads (K-grouped over expert rows)
# dWd[e]  [I, D]  = hp[rows]^T @ go[tok]
# dWgu[e] [D, 2I] = x[tok]^T  @ dgu[rows]
# ----------------------------------------------------------------------------
@triton.jit
def _swz(pid, num_m, num_n, GROUP_M: tl.constexpr):
    npg = GROUP_M * num_n
    gid = pid // npg
    first = gid * GROUP_M
    gsz = tl.minimum(num_m - first, GROUP_M)
    pm = first + (pid % npg) % gsz
    pn = (pid % npg) // gsz
    return pm, pn


@triton.jit
def _bwd_dwd_kernel(hp_ptr, go_ptr, tid_ptr, ms_ptr, dw_ptr,
                    E, D, I,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    E_POW2: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.program_id(1)
    num_m = tl.cdiv(I, BM)
    num_n = tl.cdiv(D, BN)
    pm, pn = _swz(pid, num_m, num_n, GROUP_M)
    start, end = _expert_range(ms_ptr, e, E, E_POW2)
    offs_i = pm * BM + tl.arange(0, BM)
    offs_d = pn * BN + tl.arange(0, BN)
    offs_ic = offs_i % I
    offs_dc = offs_d % D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r0 in range(start, end, BK):
        offs_r = r0 + tl.arange(0, BK)
        rm = offs_r < end
        tok = tl.load(tid_ptr + offs_r, mask=rm, other=0)
        a = tl.load(hp_ptr + offs_r.to(tl.int64)[None, :] * I + offs_ic[:, None],
                    mask=rm[None, :], other=0.0)
        b = tl.load(go_ptr + tok.to(tl.int64)[:, None] * D + offs_dc[None, :])
        acc = tl.dot(a, b, acc)
    optr = dw_ptr + e.to(tl.int64) * I * D + offs_i[:, None] * D + offs_d[None, :]
    tl.store(optr, acc.to(tl.bfloat16), mask=(offs_i < I)[:, None] & (offs_d < D)[None, :])


@triton.jit
def _bwd_dwgu_kernel(x_ptr, dgu_ptr, tid_ptr, ms_ptr, dw_ptr,
                     E, D, I,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     E_POW2: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.program_id(1)
    N2 = 2 * I
    num_m = tl.cdiv(D, BM)
    num_n = tl.cdiv(N2, BN)
    pm, pn = _swz(pid, num_m, num_n, GROUP_M)
    start, end = _expert_range(ms_ptr, e, E, E_POW2)
    offs_d = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_dc = offs_d % D
    offs_nc = offs_n % N2
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r0 in range(start, end, BK):
        offs_r = r0 + tl.arange(0, BK)
        rm = offs_r < end
        tok = tl.load(tid_ptr + offs_r, mask=rm, other=0)
        a = tl.load(x_ptr + tok.to(tl.int64)[None, :] * D + offs_dc[:, None])
        b = tl.load(dgu_ptr + offs_r.to(tl.int64)[:, None] * N2 + offs_nc[None, :],
                    mask=rm[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    optr = dw_ptr + e.to(tl.int64) * D * N2 + offs_d[:, None] * N2 + offs_n[None, :]
    tl.store(optr, acc.to(tl.bfloat16), mask=(offs_d < D)[:, None] & (offs_n < N2)[None, :])


# ----------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------
# per-kernel tile configs (BM, BN, BK, warps, stages); "s" = small rows/expert
_CFG_SMALL = {
    "gu": (64, 64, 64, 4, 4), "fdown": (64, 128, 64, 4, 4), "dh": (64, 128, 64, 4, 3),
    "dx": (64, 128, 64, 4, 4), "wd": (128, 128, 32, 8, 4), "wgu": (128, 128, 32, 8, 4),
}
_CFG = {
    "gu": (64, 128, 64, 4, 3), "fdown": (128, 128, 32, 4, 4), "dh": (64, 128, 64, 4, 3),
    "dx": (128, 256, 32, 8, 4), "wd": (128, 128, 64, 4, 3), "wgu": (128, 128, 64, 4, 3),
}


def _cfg(T, E, name):
    if T < 96 * E:
        return _CFG_SMALL[name]
    return _CFG[name]


def _launch_fwd_gu(x, wgu, tids, ms, gu, scales, h, T, E, D, I, mode, store_h):
    BM, BN, BK, nw, ns = _cfg(T, E, "gu")
    if mode == 1:
        BN = _QBN
    ub = triton.cdiv(T, BM) + E
    _fwd_gu_kernel[(ub * triton.cdiv(I, BN),)](
        x, wgu, tids, ms, gu, scales, h, E, D, I,
        BM=BM, BN=BN, BK=BK, E_POW2=triton.next_power_of_2(E), GROUP_M=8,
        EVEN_K=(D % BK == 0), MODE=mode, STORE_H=store_h,
        num_warps=nw, num_stages=ns)


def _forward(x, wgu, wd, probs, tids, ms):
    N, D = x.shape
    E = wgu.shape[0]
    I = wgu.shape[2] // 2
    T = tids.shape[0]
    dev = x.device
    nq = triton.cdiv(I, _QBN)
    if T * 2 * I * 2 <= _SAVE_BUDGET:
        mode = 0
        gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        scales = gu
    elif T * 2 * I + T * 2 * nq * 2 <= _SAVE_BUDGET:
        mode = 1
        gu = torch.empty((T, 2 * I), dtype=torch.int8, device=dev)
        scales = torch.empty((T, 2 * nq), dtype=torch.float16, device=dev)
    else:
        mode = 2
        gu = None
        scales = None
    h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
    _launch_fwd_gu(x, wgu, tids, ms, gu if gu is not None else h,
                   scales if scales is not None else h, h, T, E, D, I, mode, True)
    out32 = torch.zeros((N, D), dtype=torch.float32, device=dev)
    BM, BN, BK, nw, ns = _cfg(T, E, "fdown")
    ub = triton.cdiv(T, BM) + E
    _fwd_down_kernel[(ub * triton.cdiv(D, BN),)](
        h, wd, tids, probs, ms, out32, E, D, I,
        BM=BM, BN=BN, BK=BK, E_POW2=triton.next_power_of_2(E), GROUP_M=8,
        EVEN_K=(I % BK == 0), num_warps=nw, num_stages=ns)
    del h
    out = out32.to(torch.bfloat16)
    return out, mode, gu, scales


def _backward(x, wgu, wd, probs, tids, ms, mode, gu, scales, go):
    N, D = x.shape
    E = wgu.shape[0]
    I = wgu.shape[2] // 2
    T = tids.shape[0]
    dev = x.device
    nq = triton.cdiv(I, _QBN)
    if mode == 2:
        gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        scales = gu
        _launch_fwd_gu(x, wgu, tids, ms, gu, gu, gu, T, E, D, I, 0, False)
        mode = 0
    E_POW2 = triton.next_power_of_2(E)

    dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
    hp = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
    dprobs = torch.zeros((T,), dtype=torch.float32, device=dev)
    BM, BN, BK, nw, ns = _cfg(T, E, "dh")
    ub = triton.cdiv(T, BM) + E
    _bwd_dh_kernel[(ub * triton.cdiv(I, BN),)](
        go, wd, tids, probs, ms, gu, scales, dgu, hp, dprobs, E, D, I, nq,
        BM=BM, BN=BN, BK=BK, E_POW2=E_POW2, GROUP_M=8, EVEN_K=(D % BK == 0),
        MODE=mode, QBN=_QBN, num_warps=nw, num_stages=ns)
    del gu, scales

    dx32 = torch.zeros((N, D), dtype=torch.float32, device=dev)
    BM, BN, BK, nw, ns = _cfg(T, E, "dx")
    ub = triton.cdiv(T, BM) + E
    _bwd_dx_kernel[(ub * triton.cdiv(D, BN),)](
        dgu, wgu, tids, ms, dx32, E, D, I,
        BM=BM, BN=BN, BK=BK, E_POW2=E_POW2, GROUP_M=8, EVEN_K=((2 * I) % BK == 0),
        num_warps=nw, num_stages=ns)
    dx = dx32.to(torch.bfloat16)
    del dx32

    BM, BN, BK, nw, ns = _cfg(T, E, "wd")
    dwd = torch.empty_like(wd)
    _bwd_dwd_kernel[(triton.cdiv(I, BM) * triton.cdiv(D, BN), E)](
        hp, go, tids, ms, dwd, E, D, I,
        BM=BM, BN=BN, BK=BK, E_POW2=E_POW2, GROUP_M=8, num_warps=nw, num_stages=ns)
    del hp
    BM, BN, BK, nw, ns = _cfg(T, E, "wgu")
    dwgu = torch.empty_like(wgu)
    _bwd_dwgu_kernel[(triton.cdiv(D, BM) * triton.cdiv(2 * I, BN), E)](
        x, dgu, tids, ms, dwgu, E, D, I,
        BM=BM, BN=BN, BK=BK, E_POW2=E_POW2, GROUP_M=8, num_warps=nw, num_stages=ns)
    return dx, dwgu, dwd, dprobs


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wgu, wd, probs, tids, ms):
        ctx.empty = tids.shape[0] == 0
        if ctx.empty:
            ctx.save_for_backward(x, wgu, wd, probs)
            return torch.zeros_like(x)
        out, mode, gu, scales = _forward(x, wgu, wd, probs, tids, ms)
        ctx.mode = mode
        extra = [t for t in (gu, scales) if t is not None]
        if mode == 0:
            extra = [gu]
        ctx.save_for_backward(x, wgu, wd, probs, tids, ms, *extra)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        if ctx.empty:
            x, wgu, wd, probs = saved
            return (torch.zeros_like(x), torch.zeros_like(wgu), torch.zeros_like(wd),
                    torch.zeros_like(probs), None, None)
        x, wgu, wd, probs, tids, ms = saved[:6]
        mode = ctx.mode
        gu = scales = None
        if mode == 0:
            gu = scales = saved[6]
        elif mode == 1:
            gu, scales = saved[6], saved[7]
        dx, dwgu, dwd, dprobs = _backward(x, wgu, wd, probs, tids, ms, mode, gu, scales,
                                          grad_out.contiguous())
        return dx, dwgu, dwd, dprobs, None, None


def routed_moe_swiglu(x, w_gate_up, w_down, probs, token_ids, m_sizes):
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)
