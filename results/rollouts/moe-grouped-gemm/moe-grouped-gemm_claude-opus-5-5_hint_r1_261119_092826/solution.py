import torch
import triton
import triton.language as tl


@triton.jit
def _tile(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, T, N, K, e, row0, row_end,
          pid_n, unit, split, ns, k_iters,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, WS_TILE: tl.constexpr,
          EVEN_K: tl.constexpr, SPLIT: tl.constexpr):
    per = tl.cdiv(k_iters, ns)
    k_begin = split * per
    k_end = tl.minimum(k_begin + per, k_iters)

    rm = row0 + tl.arange(0, BM)
    rm_ld = tl.minimum(rm, T - 1)
    rn = pid_n * BN + tl.arange(0, BN)
    rn_ld = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
    rk = tl.arange(0, BK)
    a_ptrs = x_ptr + rm_ld[:, None] * K + (k_begin * BK + rk)[None, :]
    b_ptrs = w_ptr + e.to(tl.int64) * K * N + (k_begin * BK + rk)[:, None] * N + rn_ld[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k_begin, k_end):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            kmask = rk < K - k * BK
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=kmask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    cmask = (rm < row_end)[:, None] & (rn < N)[None, :]
    c_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
    if SPLIT:
        if ns > 1:
            # split-K tail tile: park the fp32 partial, last arriver reduces
            w_off = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
            ws_base = ws_ptr + (unit * ns).to(tl.int64) * WS_TILE
            tl.store(ws_base + split * WS_TILE + w_off, acc)
            tl.debug_barrier()
            old = tl.atomic_add(cnt_ptr + unit, 1, sem="acq_rel", scope="gpu")
            if old == ns - 1:
                tl.debug_barrier()
                for j in range(0, ns):
                    if j != split:
                        acc += tl.load(ws_base + j * WS_TILE + w_off, cache_modifier=".cg")
                tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)
                tl.store(cnt_ptr + unit, 0)
        else:
            tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)
    else:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)


@triton.jit
def _num_splits(total, k_iters, SLOTS, S_MAX: tl.constexpr, MIN_KI: tl.constexpr,
                WS_CAP: tl.constexpr):
    # full waves run unsplit; tiles of the last partial wave are split along K
    n_main = (total // SLOTS) * SLOTS
    tail = total - n_main
    ns = 1
    best = 1.0
    for s in tl.static_range(2, S_MAX + 1):
        waves = (tail * s + SLOTS - 1) // SLOTS
        cost = waves.to(tl.float32) / s + 0.04 * (s - 1)
        ok = (k_iters >= s * MIN_KI) & (cost < best) & (tail > 0) & (tail * s <= WS_CAP)
        ns = tl.where(ok, s, ns)
        best = tl.where(ok, cost, best)
    return ns, n_main, tail


@triton.jit
def _gg_kernel(x_ptr, w_ptr, out_ptr, m_ptr, ws_ptr, cnt_ptr, T, N, K, E, SLOTS,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr, E_P: tl.constexpr, EVEN_K: tl.constexpr,
               S_MAX: tl.constexpr, MIN_KI: tl.constexpr, WS_CAP: tl.constexpr,
               TAIL: tl.constexpr):
    pid = tl.program_id(0)
    # per-expert tile prefix sums, computed in registers from m_sizes
    eidx = tl.arange(0, E_P)
    m = tl.load(m_ptr + eidx, mask=eidx < E, other=0)
    n_tiles = tl.cdiv(N, BN)
    tiles = ((m + BM - 1) // BM) * n_tiles
    cum = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    k_iters = tl.cdiv(K, BK)

    if TAIL:
        # split-K pass over the tiles of the last partial wave
        ns, n_main, tail = _num_splits(total, k_iters, SLOTS, S_MAX, MIN_KI, WS_CAP)
        if ns == 1:
            return
        if pid >= tail * ns:
            return
        unit = pid // ns
        split = pid % ns
        tile = n_main + unit
    else:
        ns = 1
        split = 0
        unit = 0
        tile = pid
        if S_MAX > 1:
            nsp, n_main, tail = _num_splits(total, k_iters, SLOTS, S_MAX, MIN_KI, WS_CAP)
            limit = tl.where(nsp > 1, n_main, total)
        else:
            limit = total
        if tile >= limit:
            return
    e = tl.sum((cum <= tile).to(tl.int32), 0)
    sel = eidx == e
    tile_start = tl.sum(tl.where(sel, cum - tiles, 0), 0)
    m_e = tl.sum(tl.where(sel, m, 0), 0)
    row_start = tl.sum(tl.where(eidx < e, m, 0), 0)
    mt_e = (m_e + BM - 1) // BM

    # GROUP_M swizzle inside the expert so concurrent CTAs share weight columns
    local = tile - tile_start
    num_in_group = GROUP_M * n_tiles
    group_id = local // num_in_group
    first_m = group_id * GROUP_M
    gsz = tl.minimum(mt_e - first_m, GROUP_M)
    pid_m = first_m + (local % num_in_group) % gsz
    pid_n = (local % num_in_group) // gsz

    row0 = row_start + pid_m * BM
    row_end = row_start + m_e
    rem = row_end - row0
    # expert remainders use thinner M tiles to cut padding work
    if rem <= BM // 4:
        _tile(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, T, N, K, e, row0, row_end,
              pid_n, unit, split, ns, k_iters, BM // 4, BN, BK, BM * BN, EVEN_K, TAIL)
    elif rem <= BM // 2:
        _tile(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, T, N, K, e, row0, row_end,
              pid_n, unit, split, ns, k_iters, BM // 2, BN, BK, BM * BN, EVEN_K, TAIL)
    else:
        _tile(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, T, N, K, e, row0, row_end,
              pid_n, unit, split, ns, k_iters, BM, BN, BK, BM * BN, EVEN_K, TAIL)


_WS_MULT = 4
_WS = {}
_NSM = {}


def _num_sms(dev):
    n = _NSM.get(dev)
    if n is None:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _NSM[dev] = n
    return n


def _workspace(dev, units, tile_elems):
    # split-K scratch + self-resetting arrival counters (no per-call memset)
    key = (dev, units, tile_elems)
    buf = _WS.get(key)
    if buf is None:
        buf = (torch.empty(units * tile_elems, dtype=torch.float32, device=dev),
               torch.zeros(units, dtype=torch.int32, device=dev))
        _WS[key] = buf
    return buf


# BM, BN, BK, warps, stages, GROUP_M, max K splits, CTAs per SM, min K iters per split
_CFG = (128, 128, 32, 4, 4, 8, 8, 2, 16)


def grouped_gemm(x, w, m_sizes):
    BM, BN, BK, nw, nst, GM, smax, cps, min_ki = _CFG
    T, K = x.shape
    E, _, N = w.shape
    dev = x.device
    out = torch.empty((T, N), dtype=x.dtype, device=dev)
    if T == 0:
        return out
    k_iters = triton.cdiv(K, BK)
    if k_iters < 2 * min_ki:
        smax = 1
    slots = _num_sms(dev) * cps
    cap = _WS_MULT * slots
    if smax > 1:
        ws, cnt = _workspace(dev, cap, BM * BN)
    else:
        ws, cnt = out, m_sizes
    ub = (T // BM + E) * triton.cdiv(N, BN)
    args = (x, w, out, m_sizes, ws, cnt, T, N, K, E, slots)
    kw = dict(BM=BM, BN=BN, BK=BK, GROUP_M=GM, E_P=max(triton.next_power_of_2(E), 2),
              EVEN_K=(K % BK == 0), S_MAX=smax, MIN_KI=min_ki, WS_CAP=cap,
              num_warps=nw, num_stages=nst)
    _gg_kernel[(ub,)](*args, TAIL=False, **kw)
    if smax > 1:
        _gg_kernel[(cap,)](*args, TAIL=True, **kw)
    return out
