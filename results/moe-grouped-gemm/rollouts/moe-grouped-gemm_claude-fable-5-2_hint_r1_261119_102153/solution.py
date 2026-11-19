"""Grouped GEMM for MoE forward, single Triton kernel launch.

Scheduling is done inside the kernel: every program reads m_sizes, finds
its expert / tile from prefix sums held in registers, and exits if its tile id
is past the number of active tiles.
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _tile(x_ptr, w_ptr, o_ptr, e, row_lo, row_hi, n0, T, K, N,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KALIGN: tl.constexpr, EV: tl.constexpr):
    # Shift the tile back inside the buffers so loads need no mask.
    r0 = tl.minimum(row_lo, T - BM)
    c0 = tl.minimum(n0, N - BN)
    am = tl.arange(0, BM)
    an = tl.arange(0, BN)
    ak = tl.arange(0, BK)
    a_ptrs = x_ptr + r0.to(tl.int64) * K + (am[:, None] * K + ak[None, :])
    b_ptrs = w_ptr + (e.to(tl.int64) * K * N + c0) + (ak[:, None] * N + an[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        if EV == 1:
            a = tl.load(a_ptrs, eviction_policy="evict_last")
            b = tl.load(b_ptrs, eviction_policy="evict_first")
        elif EV == 2:
            a = tl.load(a_ptrs, eviction_policy="evict_first")
            b = tl.load(b_ptrs, eviction_policy="evict_last")
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    if not KALIGN:
        krem = K % BK
        a = tl.load(a_ptrs, mask=ak[None, :] < krem, other=0.0)
        b = tl.load(b_ptrs, mask=ak[:, None] < krem, other=0.0)
        acc = tl.dot(a, b, acc)
    rows = r0 + am
    cols = c0 + an
    mask = (rows[:, None] >= row_lo) & (rows[:, None] < row_hi) & (cols[None, :] >= n0)
    o_ptrs = o_ptr + (r0.to(tl.int64) * N + c0) + (am[:, None] * N + an[None, :])
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _gg_kernel(x_ptr, w_ptr, o_ptr, m_ptr, T, K, N, E, NT_N,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr, KALIGN: tl.constexpr, SMALL: tl.constexpr, EV: tl.constexpr):
    pid = tl.program_id(0)
    zero = pid * 0
    tile_acc = zero
    row_acc = zero
    f_e = zero - 1
    f_tile0 = zero
    f_row = zero
    f_m = zero
    f_ntm = zero
    for e in range(E):
        m = tl.load(m_ptr + e)
        ntm = (m + (BM - 1)) // BM
        nt = ntm * NT_N
        hit = (pid >= tile_acc) & (pid < tile_acc + nt)
        f_e = tl.where(hit, e, f_e)
        f_tile0 = tl.where(hit, tile_acc, f_tile0)
        f_row = tl.where(hit, row_acc, f_row)
        f_m = tl.where(hit, m, f_m)
        f_ntm = tl.where(hit, ntm, f_ntm)
        tile_acc += nt
        row_acc += m
    if f_e >= 0:
        # GROUP_M swizzle inside the expert: programs that run together share
        # the same weight columns.
        local = pid - f_tile0
        width = GROUP_M * NT_N
        first_m = (local // width) * GROUP_M
        gsz = tl.minimum(f_ntm - first_m, GROUP_M)
        rem = local % width
        tm = first_m + rem % gsz
        tn = rem // gsz
        row_lo = f_row + tm * BM
        row_hi = f_row + f_m
        n0 = tn * BN
        nrows = row_hi - row_lo
        if SMALL:
            if nrows <= 16:
                _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, 16, BN, BK, KALIGN, EV)
            else:
                if nrows <= 32:
                    _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, 32, BN, BK, KALIGN, EV)
                else:
                    if nrows <= 64:
                        _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, 64, BN, BK, KALIGN, EV)
                    else:
                        if BM > 128:
                            if nrows <= 128:
                                _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, 128, BN, BK, KALIGN, EV)
                            else:
                                _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, BM, BN, BK, KALIGN, EV)
                        else:
                            _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, BM, BN, BK, KALIGN, EV)
        else:
            _tile(x_ptr, w_ptr, o_ptr, f_e, row_lo, row_hi, n0, T, K, N, BM, BN, BK, KALIGN, EV)


# (BM, BN, BK, num_stages, num_warps, GROUP_M, SMALL)
_CANDIDATES = [
    (128, 128, 32, 3, 4, 16, True, 0),
    (128, 128, 64, 3, 4, 16, True, 0),
    (128, 128, 32, 3, 4, 16, True, 1),
    (128, 128, 32, 3, 4, 8, True, 0),
    (64, 256, 32, 3, 4, 16, True, 0),
    (64, 128, 32, 3, 4, 16, True, 0),
]
_FALLBACK = (16, 16, 16, 2, 4, 8, False, 0)

_BEST = {}
_KERNELS = {}
_FLUSH = [None]
_FAST = [True]
_STREAM = [None]


def _launch(cfg, x, w, out, m_sizes, T, K, N, E):
    BM, BN, BK, stages, warps, gm, small, ev = cfg
    nt_n = (N + BN - 1) // BN
    g = (T // BM + E) * nt_n
    kalign = K % BK == 0
    if _FAST[0]:
        # Direct launcher call: skips Triton's per-call binder / cache-key work.
        kk = (cfg, kalign, T == 1, K == 1, N == 1, E == 1, nt_n == 1)
        kern = _KERNELS.get(kk)
        if (kern is not None and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0
                and m_sizes.data_ptr() % 16 == 0):
            try:
                kern.run(g, 1, 1, _STREAM[0], kern.function, kern.packed_metadata, None, None, None,
                         x, w, out, m_sizes, T, K, N, E, nt_n, BM, BN, BK, gm, kalign, small, ev)
                return
            except TypeError:
                _FAST[0] = False
    kern = _gg_kernel[(g,)](x, w, out, m_sizes, T, K, N, E, nt_n,
                            BM=BM, BN=BN, BK=BK, GROUP_M=gm, KALIGN=kalign, SMALL=small, EV=ev,
                            num_stages=stages, num_warps=warps)
    if _FAST[0] and kern is not None:
        try:
            from triton.runtime.driver import driver
            dev = driver.active.get_current_device()
            if dev == 0 and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 \
                    and m_sizes.data_ptr() % 16 == 0:
                _STREAM[0] = driver.active.get_current_stream(dev)
                _KERNELS[kk] = kern
        except Exception:  # noqa: BLE001
            _FAST[0] = False


def _tune(key, x, w, out, m_sizes):
    """Runs once per shape during warmup: pick the fastest tile config."""
    T, E, K, N = key
    cands = [c for c in _CANDIDATES
             if c[0] <= T and c[1] <= N and (not c[6] or T >= 128)]
    if not cands:
        _BEST[key] = _FALLBACK
        return _FALLBACK
    if _FLUSH[0] is None:
        _FLUSH[0] = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=x.device)
    flush = _FLUSH[0]
    sync = torch.cuda.synchronize
    ok = []
    for cfg in cands:
        try:
            _launch(cfg, x, w, out, m_sizes, T, K, N, E)
            _launch(cfg, x, w, out, m_sizes, T, K, N, E)
            sync()
            ok.append(cfg)
        except Exception:  # noqa: BLE001
            continue
    if not ok:
        _BEST[key] = _FALLBACK
        return _FALLBACK
    # Round-robin over the candidates and keep the minimum, so slow drift of
    # the GPU clock does not favor whichever config was timed first.
    best_t = [float("inf")] * len(ok)
    for _ in range(4):
        for i, cfg in enumerate(ok):
            for _ in range(3):
                flush.zero_()
                sync()
                t0 = time.perf_counter()
                _launch(cfg, x, w, out, m_sizes, T, K, N, E)
                sync()
                t = time.perf_counter() - t0
                if t < best_t[i]:
                    best_t[i] = t
    best = ok[best_t.index(min(best_t))]
    _BEST[key] = best
    return best


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=x.dtype, device=x.device)
    if T == 0 or N == 0:
        return out
    if m_sizes.dtype != torch.int32:
        m_sizes = m_sizes.to(torch.int32)
    key = (T, E, K, N)
    cfg = _BEST.get(key)
    if cfg is None:
        cfg = _tune(key, x, w, out, m_sizes)
    _launch(cfg, x, w, out, m_sizes, T, K, N, E)
    return out
