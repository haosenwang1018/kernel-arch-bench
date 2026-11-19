"""Grouped GEMM for MoE forward pass: one Triton kernel launch for all experts."""

import time

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "E", "NM", "NN"])
def _gg_kernel(
    x_ptr, w_ptr, m_ptr, out_ptr,
    T, K, N, E, NM, NN,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EP: tl.constexpr,
    KMASK: tl.constexpr, NMASK: tl.constexpr, G: tl.constexpr,
):
    # 分组的 tile 顺序，提高 L2 复用
    pid = tl.program_id(0)
    per_group = G * NN
    group_id = pid // per_group
    first_m = group_id * G
    gsize = tl.minimum(NM - first_m, G)
    rem = pid % per_group
    pid_m = first_m + rem % gsize
    pid_n = rem // gsize

    # 找到该 tile 属于哪个 expert（全部在 kernel 内完成，无需 host 同步）
    idx = tl.arange(0, EP)
    m = tl.load(m_ptr + idx, mask=idx < E, other=0)
    tiles = (m + (BM - 1)) // BM
    tile_end = tl.cumsum(tiles, 0)
    e = tl.sum((tile_end <= pid_m).to(tl.int32), 0)
    before = idx < e
    row_start = tl.sum(tl.where(before, m, 0), 0)
    tile_start = tl.sum(tl.where(before, tiles, 0), 0)
    m_e = tl.sum(tl.where(idx == e, m, 0), 0)
    valid = e < E
    e_c = tl.minimum(e, E - 1)
    row0 = row_start + (pid_m - tile_start) * BM
    row_end = row_start + m_e

    offs_m = row0 + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < row_end
    rows = tl.minimum(offs_m, T - 1)

    a_ptrs = x_ptr + rows[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + e_c.to(tl.int64) * K * N + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kiters = tl.where(valid, tl.cdiv(K, BK), 0)
    for k in range(0, kiters):
        if KMASK:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        else:
            a = tl.load(a_ptrs)
        if KMASK or NMASK:
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_n[None, :] < N), other=0.0)
        else:
            b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    o_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & (offs_n[None, :] < N))


_CK = {}
_FAST = [True]


def _launch(cfg, x, w, m_sizes, out, T, K, N, E, fast=True):
    BM, BN, BK, nw, ns, G = cfg
    nm = min(T // BM + E, (T + E * (BM - 1)) // BM)
    nn = -(-N // BN)
    EP = 128
    while EP < E:
        EP *= 2
    km = K % BK != 0
    nmask = N % BN != 0
    aligned = (x.data_ptr() | w.data_ptr() | m_sizes.data_ptr() | out.data_ptr()) % 16 == 0
    key = (cfg, EP, km, nmask, K % 16 == 0, N % 16 == 0)
    if fast and aligned and _FAST[0]:
        ck = _CK.get(key)
        if ck is not None:
            try:
                ck.run(nm * nn, 1, 1, torch.cuda.current_stream().cuda_stream, ck.function, ck.packed_metadata,
                       None, None, None,
                       x, w, m_sizes, out, T, K, N, E, nm, nn, BM, BN, BK, EP, km, nmask, G)
                return
            except Exception:  # noqa: BLE001
                _FAST[0] = False
    ck = _gg_kernel[(nm * nn,)](
        x, w, m_sizes, out, T, K, N, E, nm, nn,
        BM=BM, BN=BN, BK=BK, EP=EP, KMASK=km, NMASK=nmask, G=G,
        num_warps=nw, num_stages=ns,
    )
    if aligned:
        _CK[key] = ck


# 候选配置：(BM, BN, BK, num_warps, num_stages, G)
_CANDIDATES = [
    (128, 128, 32, 4, 4, 8),
    (128, 128, 64, 4, 3, 8),
    (128, 128, 32, 4, 3, 8),
    (256, 128, 64, 8, 3, 8),
    (128, 256, 64, 8, 3, 8),
    (64, 128, 64, 4, 3, 16),
    (64, 128, 32, 2, 3, 16),
    (32, 256, 64, 4, 3, 32),
]
_DEFAULT = (128, 128, 64, 4, 3, 8)
_BEST = {}
_FLUSH = [None]


def _tune(x, w, m_sizes, T, K, N, E):
    """首次遇到某个形状时，用真实输入对候选配置计时，选最快的。"""
    sync = torch.cuda.synchronize
    out = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    ok = []
    for cfg in _CANDIDATES:
        try:
            _launch(cfg, x, w, m_sizes, out, T, K, N, E)
            _launch(cfg, x, w, m_sizes, out, T, K, N, E)
            sync()
            ok.append(cfg)
        except Exception:  # noqa: BLE001
            pass
    if not ok:
        return _DEFAULT
    if _FLUSH[0] is None:
        try:
            _FLUSH[0] = torch.empty(16 * 1024 * 1024, dtype=torch.int32, device=x.device)
        except Exception:  # noqa: BLE001
            _FLUSH[0] = False
    flush = _FLUSH[0]
    best = {cfg: float("inf") for cfg in ok}
    for _ in range(5):
        for cfg in ok:
            if flush is not False:
                flush.zero_()
            sync()
            t0 = time.perf_counter()
            _launch(cfg, x, w, m_sizes, out, T, K, N, E)
            sync()
            dt = time.perf_counter() - t0
            if dt < best[cfg]:
                best[cfg] = dt
    return min(ok, key=lambda c: best[c])


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    T, K = x.shape
    E, _, N = w.shape
    if m_sizes.dtype != torch.int32:
        m_sizes = m_sizes.to(torch.int32)
    if T == 0 or N == 0 or K == 0:
        return torch.zeros((T, N), dtype=torch.bfloat16, device=x.device)
    key = (T, E, K, N)
    cfg = _BEST.get(key)
    if cfg is None:
        cfg = _tune(x, w, m_sizes, T, K, N, E)
        _BEST[key] = cfg
    out = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    _launch(cfg, x, w, m_sizes, out, T, K, N, E)
    return out
