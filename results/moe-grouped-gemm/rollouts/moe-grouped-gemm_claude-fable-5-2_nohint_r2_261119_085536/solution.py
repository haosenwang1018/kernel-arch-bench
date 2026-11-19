"""Grouped GEMM for MoE forward, written in Triton (tensor-core tl.dot, fp32 accumulate).

One kernel launch covers all experts: each program maps its row-tile index to an
expert by scanning m_sizes on the GPU.  Tile shapes are picked per
(shape, distribution-signature) by a small self-tuner during the first call;
every candidate is validated on sampled rows before it can be selected.
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _gg_kernel(
    x_ptr, w_ptr, o_ptr, m_ptr, E, K, N, nt, ntn, n_shift,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    MODE: tl.constexpr, BIG: tl.constexpr, G: tl.constexpr,
    VAR: tl.constexpr, MASK_K: tl.constexpr,
):
    # MODE 0: all rows of every expert, tiles of BM rows (last tile shifted back).
    # MODE 1: only the full BM-row tiles of every expert.
    # MODE 2: only the remainder rows (m % BIG) of every expert, tiles of BM rows.
    # Programs are ordered in groups of G row-tiles for L2 locality.
    pid = tl.program_id(0)
    npg = G * ntn
    gid = pid // npg
    first = gid * G
    gsz = tl.minimum(nt - first, G)
    r = pid - gid * npg
    tile = first + r % gsz
    pid_n = r // gsz

    # Branch-free scan of m_sizes: find the expert that owns this row-tile.
    zero = tile * 0
    cum = zero
    es = zero
    f_e1 = zero
    f_rs = zero
    f_m = zero
    f_ee = zero
    for e in range(E):
        m = tl.load(m_ptr + e)
        if MODE == 2:
            skip = (m // BIG) * BIG
        else:
            skip = m * 0
        if MODE == 1:
            nt_e = m // BM
        else:
            nt_e = (m - skip + BM - 1) // BM
        hit = ((tile >= cum) & (tile < cum + nt_e)).to(tl.int32)
        f_e1 += hit * (e + 1)
        f_rs += hit * (es + skip + (tile - cum) * BM)
        f_m += hit * m
        f_ee += hit * (es + m)
        cum += nt_e
        es += m
    f_e = f_e1 - 1
    f_es = f_ee - f_m

    if f_e >= 0:
        # shift the last tile of an expert back so it stays inside the expert
        over = tl.maximum(f_rs + BM - f_ee, 0)
        if VAR == 0:
            rs = f_rs - tl.minimum(over, f_rs - f_es)
        else:
            rs = f_rs
        nrows = f_ee - rs
        lm = tl.arange(0, BM)
        mask_m = lm < nrows
        offs_m = rs + lm
        if VAR == 0:
            offs_ml = rs + tl.minimum(lm, nrows - 1)
        else:
            offs_ml = offs_m
        ns = pid_n * BN - (pid_n == ntn - 1).to(tl.int32) * n_shift
        offs_n = ns + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = x_ptr + offs_ml[:, None].to(tl.int64) * K + offs_k[None, :]
        b_ptrs = w_ptr + f_e.to(tl.int64) * K * N + offs_k[:, None] * N + offs_n[None, :]

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            if MASK_K:
                km = offs_k < K - k * BK
                a = tl.load(a_ptrs, mask=km[None, :] & mask_m[:, None], other=0.0)
                b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
            else:
                if VAR == 0:
                    a = tl.load(a_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * N

        c = acc.to(tl.bfloat16)
        o_ptrs = o_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(o_ptrs, c, mask=mask_m[:, None])


# A config is a tuple of passes; a pass is
# (BM, BN, BK, num_warps, num_stages, MODE, BIG, G, VAR).
def _single(bm, bn, bk, warps, stages, g=8, var=0):
    return ((bm, bn, bk, warps, stages, 0, 0, g, var),)


def _hybrid(big, small, g=8):
    return (big + (1, 0, g, 0), small + (2, big[0], g, 0))


_CANDIDATES = [
    _single(128, 256, 64, 8, 3),
    _single(128, 128, 32, 4, 4),
    _single(128, 128, 64, 4, 3),
    _single(256, 128, 64, 8, 3),
    _single(256, 128, 32, 8, 4),
    _single(128, 256, 32, 8, 4),
    _single(64, 256, 32, 4, 4),
    _single(64, 256, 64, 4, 3),
    _single(64, 128, 32, 4, 4),
    _single(32, 256, 32, 4, 4),
    _single(32, 128, 64, 4, 3),
    _single(128, 128, 32, 4, 4, var=1),
    _single(32, 256, 32, 4, 4, var=1),
    _hybrid((128, 128, 32, 4, 4), (32, 256, 32, 4, 4)),
    _hybrid((128, 256, 64, 8, 3), (32, 256, 32, 4, 4)),
]
_GROUPS = (1, 4, 16, 32)

_CHOICE = {}
_FLUSH = [None]
_sync = torch.cuda.synchronize


def _launch(cfg, x, w, out, m_sizes, ms, E, K, N):
    for (BM, BN, BK, warps, stages, MODE, BIG, G, VAR) in cfg:
        if MODE == 0:
            nt = sum([(m + BM - 1) // BM for m in ms])
        elif MODE == 1:
            nt = sum([m // BM for m in ms])
        else:
            nt = sum([(m % BIG + BM - 1) // BM for m in ms])
        if nt == 0:
            continue
        ntn = -(-N // BN)
        _gg_kernel[(nt * ntn,)](
            x, w, out, m_sizes, E, K, N, nt, ntn, ntn * BN - N,
            BM=BM, BN=BN, BK=BK, MODE=MODE, BIG=BIG, G=G, VAR=VAR,
            MASK_K=(K % BK != 0),
            num_warps=warps, num_stages=stages,
        )


def _make_checker(x, w, ms, N):
    """Reference values on sampled rows/cols, via elementwise ops (tuning only)."""
    rows, exps = [], []
    start = 0
    for e, m in enumerate(ms):
        if m > 0:
            for r in sorted(set([start, start + m // 2, start + m - 1])):
                rows.append(r)
                exps.append(e)
        start += m
    dev = x.device
    cols = torch.tensor(sorted(set([0, 1, N // 3, N // 2, N - 2, N - 1])), device=dev)
    rows_t = torch.tensor(rows, device=dev)
    exps_t = torch.tensor(exps, device=dev)
    refs = []
    for i in range(0, len(rows), 32):
        xr = x[rows_t[i:i + 32]].float()
        ws = w[exps_t[i:i + 32][:, None], :, cols[None, :]].float()  # [R, C, K]
        refs.append((ws * xr[:, None, :]).sum(-1))
    ref = torch.cat(refs)

    def check(out):
        got = out[rows_t[:, None], cols[None, :]].float()
        return bool(((got - ref).abs() <= 1e-2 + 1e-2 * ref.abs()).all())

    return check


def _prep(cfg, x, w, m_sizes, ms, E, K, N, check):
    """Compile + run once into a NaN-filled buffer and validate."""
    try:
        out = torch.empty((x.shape[0], N), dtype=x.dtype, device=x.device)
        out.fill_(float("nan"))
        _launch(cfg, x, w, out, m_sizes, ms, E, K, N)
        _sync()
        return check(out)
    except Exception:  # noqa: BLE001
        return False


def _race(fns, rounds):
    """Round-robin timing (keeps the GPU busy so all candidates see the same state)."""
    flush = _FLUSH[0]
    ts = [[] for _ in fns]
    for r in range(rounds + 1):
        for i, fn in enumerate(fns):
            flush.zero_()
            _sync()
            t0 = time.perf_counter()
            o = fn()
            _sync()
            if r > 0:
                ts[i].append(time.perf_counter() - t0)
            del o
    return [sorted(t)[len(t) // 2] for t in ts]


def _tune(x, w, m_sizes, ms, E, K, N, rounds=6, extra=None):
    if _FLUSH[0] is None:
        _FLUSH[0] = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=x.device)
    check = _make_checker(x, w, ms, N)
    T = x.shape[0]

    def runner(cfg):
        def fn():
            out = torch.empty((T, N), dtype=x.dtype, device=x.device)
            _launch(cfg, x, w, out, m_sizes, ms, E, K, N)
            return out
        return fn

    cands = [c for c in _CANDIDATES if max([p[1] for p in c]) <= N]
    good = [c for c in cands if _prep(c, x, w, m_sizes, ms, E, K, N, check)]
    nbad = len(cands) - len(good)
    if not good:
        return [(0.0, cands[-1])], nbad, None
    ex = [extra] if extra is not None else []
    t1 = _race([runner(c) for c in good] + ex, rounds)
    order = sorted(range(len(good)), key=lambda i: t1[i])
    finals = []
    for i in order[:2]:
        finals.append(good[i])
        for g in _GROUPS:
            c = tuple([p[:7] + (g, p[8]) for p in good[i]])
            if _prep(c, x, w, m_sizes, ms, E, K, N, check):
                finals.append(c)
    for i in order[2:4]:
        finals.append(good[i])
    t2 = _race([runner(c) for c in finals] + ex, rounds)
    table = sorted(zip(t2, finals), key=lambda r: r[0])
    return table, nbad, (t2[-1] if ex else None)


def _run(x, w, m_sizes):
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=x.dtype, device=x.device)
    if T == 0 or N == 0:
        return out
    ms = m_sizes.tolist()
    key = (T, E, K, N, max(ms).bit_length(), sum([1 for m in ms if m > 0]).bit_length())
    cfg = _CHOICE.get(key)
    if cfg is None:
        cfg = _tune(x, w, m_sizes, ms, E, K, N)[0][0][1]
        _CHOICE[key] = cfg
    _launch(cfg, x, w, out, m_sizes, ms, E, K, N)
    return out


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    return _run(x, w, m_sizes)
