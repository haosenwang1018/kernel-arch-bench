import time

import torch
import triton
import triton.language as tl


@triton.jit
def _gg_raw(x_ptr, w_ptr, o_ptr, m_ptr, T, N, K, E,
            EBLOCK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    num_m_virt = tl.cdiv(T, BM) + E
    num_in_group = GROUP_M * num_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsz = min(num_m_virt - first_m, GROUP_M)
    r = pid % num_in_group
    tm = first_m + r % gsz
    tn = r // gsz

    eoffs = tl.arange(0, EBLOCK)
    ms = tl.load(m_ptr + eoffs, mask=eoffs < E, other=0)
    tiles = (ms + BM - 1) // BM
    ctiles = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    if tm >= total:
        return
    e = tl.sum((ctiles <= tm).to(tl.int32), 0)
    before = eoffs < e
    tile_start = tl.sum(tl.where(before, tiles, 0), 0)
    row_start = tl.sum(tl.where(before, ms, 0), 0)
    m_e = tl.sum(tl.where(eoffs == e, ms, 0), 0)

    local = tm - tile_start
    rows = local * BM + tl.arange(0, BM)
    row_mask = rows < m_e
    lrow = tl.where(row_mask, rows, 0)
    cols = tn * BN + tl.arange(0, BN)
    col_mask = cols < N
    cols_c = tl.max_contiguous(tl.multiple_of(cols % N, BN), BN)
    ks = tl.arange(0, BK)

    x_base = x_ptr + row_start.to(tl.int64) * K
    w_base = w_ptr + e.to(tl.int64) * K * N
    x_ptrs = x_base + lrow[:, None] * K + ks[None, :]
    w_ptrs = w_base + ks[:, None] * N + cols_c[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            a = tl.load(x_ptrs)
            b = tl.load(w_ptrs)
        else:
            kmask = ks < K - k0 * BK
            a = tl.load(x_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(w_ptrs, mask=kmask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BK
        w_ptrs += BK * N

    o_ptrs = o_ptr + row_start.to(tl.int64) * N + lrow[:, None] * N + cols[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :])


_CANDIDATES = [
    # BM, BN, BK, stages, warps
    (128, 256, 64, 3, 8),
    (256, 128, 64, 3, 8),
    (128, 128, 64, 3, 4),
    (128, 128, 32, 4, 4),
    (64, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (64, 128, 64, 4, 4),
    (64, 64, 64, 4, 4),
    (32, 128, 64, 4, 4),
    (32, 256, 64, 3, 4),
]
_BEST = {}
_FLUSH = None


def _launch(cfg, x, w, out, m_sizes, T, N, K, E, eblock):
    bm, bn, bk, st, nw = cfg
    grid = ((triton.cdiv(T, bm) + E) * triton.cdiv(N, bn),)
    _gg_raw[grid](x, w, out, m_sizes, T, N, K, E, EBLOCK=eblock, BM=bm, BN=bn, BK=bk,
                  GROUP_M=8, EVEN_K=(K % bk == 0), num_stages=st, num_warps=nw)


def _tune(args):
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    cands = []
    for cfg in _CANDIDATES:
        try:
            _launch(cfg, *args)
            _launch(cfg, *args)
            cands.append(cfg)
        except Exception:
            pass
    torch.cuda.synchronize()
    # Measure under sustained load, like back-to-back calls: the PCIe A100 is
    # power-limited, so short bursts favor different tiles than steady state.
    times = {c: [] for c in cands}
    for _ in range(2):
        for c in cands:
            for i in range(5):
                _FLUSH.zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _launch(c, *args)
                torch.cuda.synchronize()
                if i >= 2:
                    times[c].append(time.perf_counter() - t0)
    med = {c: sorted(times[c])[len(times[c]) // 2] for c in cands}
    best = min(med.values())
    # Candidates are ordered by preference; take the first within 2% of the best.
    for c in cands:
        if med[c] <= best * 1.02:
            return c


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    eblock = max(16, triton.next_power_of_2(E))
    args = (x, w, out, m_sizes, T, N, K, E, eblock)
    avg = max(T // E, 1)
    key = (T, E, K, N, avg.bit_length())
    cfg = _BEST.get(key)
    if cfg is None:
        cfg = _BEST[key] = _tune(args)
    _launch(cfg, *args)
    return out
