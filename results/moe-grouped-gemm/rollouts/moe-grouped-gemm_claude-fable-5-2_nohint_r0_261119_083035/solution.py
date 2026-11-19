"""Grouped GEMM for MoE forward pass, written in Triton.

设计:
- 单次 kernel launch 覆盖所有 expert。每个 program 处理一个 (M-tile, N-tile)。
- 不做 host 同步: program 在 kernel 内部对 m_sizes 做前缀和, 用向量化比较定位
  自己所属的 expert。grid 使用上界 (T//BM + E) 个 M-tile, 多余的 program 直接退出。
- tile 配置 (BM/BN/BK/warps/stages/遍历顺序) 在每个形状的首次调用 (warmup) 时
  实测自动调优, 只缓存配置与已编译的 kernel 句柄, 不缓存任何输出。
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _gg_kernel(
    x_ptr, w_ptr, out_ptr, meta_ptr,
    T, K, N, E, MT, G2,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EP: tl.constexpr, INK: tl.constexpr, MASK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # 分组遍历顺序: 每组 G 个 M-tile, 组内 M 方向变化最快 (L2 复用)
    G = G2 // 2
    NT = (N + BN - 1) // BN
    gsz = G * NT
    first_m = (pid // gsz) * G
    gm = tl.minimum(MT - first_m, G)
    r = pid % gsz
    pid_m = first_m + r % gm
    pid_n = r // gm

    idx = tl.arange(0, EP)
    emask = idx < E
    if INK:
        # meta = m_sizes, 前缀和在 kernel 内完成
        m = tl.load(meta_ptr + idx, mask=emask, other=0)
        tend = tl.cumsum((m + (BM - 1)) // BM, 0)
        rend = tl.cumsum(m, 0)
    else:
        # meta[0, :] = cumsum(tiles), meta[1, :] = cumsum(m_sizes)
        tend = tl.load(meta_ptr + idx, mask=emask, other=2147483647)
        rend = tl.load(meta_ptr + E + idx, mask=emask, other=0)
    before = tend <= pid_m
    e = tl.sum(before.to(tl.int32))
    if e < E:
        tile_beg = tl.max(tl.where(before, tend, 0))
        row_beg = tl.max(tl.where(before, rend, 0))
        if INK:
            row_end = row_beg + tl.load(meta_ptr + e)
        else:
            row_end = tl.load(meta_ptr + E + e)
        row0 = row_beg + (pid_m - tile_beg) * BM

        rows = row0 + tl.arange(0, BM)
        rows_c = tl.minimum(rows, T - 1)
        # 最后一个 N-tile 向前平移, 与前一个 tile 重叠, 避免列方向的 mask
        col0 = tl.minimum(pid_n * BN, N - BN)
        cols = col0 + tl.arange(0, BN)
        ks = tl.arange(0, BK)

        a_ptrs = x_ptr + rows_c[:, None].to(tl.int64) * K + ks[None, :]
        w_base = w_ptr + e.to(tl.int64) * (K * N)
        b_ptrs = w_base + ks[:, None].to(tl.int64) * N + cols[None, :]

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            if MASK_K:
                kk = k * BK + ks
                a = tl.load(a_ptrs, mask=kk[None, :] < K, other=0.0)
                b = tl.load(b_ptrs, mask=kk[:, None] < K, other=0.0)
            else:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * N

        o_ptrs = out_ptr + rows[:, None].to(tl.int64) * N + cols[None, :]
        o_mask = (rows[:, None] < row_end) & (cols[None, :] >= 0)
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


_ST = {"ink": None, "fast": None, "variant": 0, "stream": 0}
_FAST = {}


def _try_fast(ck, grid, args, cargs, out):
    """验证绕过 Triton Python 调度层的直接 launch; 结果必须与常规路径逐位一致。"""
    if _ST["fast"] is False:
        return False
    try:
        run, fn, pm = ck.run, ck.function, ck.packed_metadata
        stream = torch.cuda.current_stream().cuda_stream
        variants = (1, 2) if _ST["variant"] == 0 else (_ST["variant"],)
        for v in variants:
            try:
                out2 = torch.zeros_like(out)
                a = list(args)
                a[2] = out2
                if v == 1:
                    a = a + list(cargs)
                run(grid, 1, 1, stream, fn, pm, None, None, None, *a)
                torch.cuda.synchronize()
                if bool(torch.equal(out2, out)):
                    _ST["variant"] = v
                    _ST["stream"] = stream
                    _ST["fast"] = True
                    return (run, fn, pm)
            except Exception:
                pass
    except Exception:
        pass
    _ST["fast"] = False
    return False


def _run(x, w, m_sizes, cfg):
    BM, BN, BK, nw, ns, G = cfg
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    MT = T // BM + E
    NT = -(-N // BN)
    if G <= 0 or G > MT:
        G = MT
    EP = 128
    while EP < E:
        EP *= 2
    ink = _ST["ink"]
    mk = K % BK != 0
    if m_sizes.dtype != torch.int32:
        m_sizes = m_sizes.to(torch.int32)
    if ink:
        meta = m_sizes
    else:
        tiles = (m_sizes + (BM - 1)) // BM
        meta = torch.stack((tiles, m_sizes)).cumsum(1, dtype=torch.int32)
    grid = MT * NT
    fk = (BM, BN, BK, nw, ns, EP, mk, ink)
    ent = _FAST.get(fk)
    if ent and ((x.data_ptr() | w.data_ptr() | meta.data_ptr()) & 15) == 0:
        if _ST["variant"] == 1:
            ent[0](grid, 1, 1, _ST["stream"], ent[1], ent[2], None, None, None,
                   x, w, out, meta, T, K, N, E, MT, 2 * G, BM, BN, BK, EP, ink, mk)
        else:
            ent[0](grid, 1, 1, _ST["stream"], ent[1], ent[2], None, None, None,
                   x, w, out, meta, T, K, N, E, MT, 2 * G)
        return out
    ck = _gg_kernel[(grid,)](
        x, w, out, meta, T, K, N, E, MT, 2 * G,
        BM=BM, BN=BN, BK=BK, EP=EP, INK=ink, MASK_K=mk,
        num_warps=nw, num_stages=ns,
    )
    if ent is None:
        torch.cuda.synchronize()
        _FAST[fk] = _try_fast(ck, grid, [x, w, out, meta, T, K, N, E, MT, 2 * G],
                              [BM, BN, BK, EP, ink, mk], out)
    return out


# (BM, BN, BK, num_warps, num_stages)
_BASE = [
    (128, 128, 32, 4, 3),
    (128, 128, 32, 4, 4),
    (128, 128, 32, 4, 2),
    (128, 128, 32, 4, 5),
    (128, 128, 32, 8, 3),
    (128, 128, 32, 8, 4),
    (128, 128, 16, 4, 4),
    (128, 128, 16, 4, 3),
    (128, 128, 16, 4, 2),
    (128, 256, 32, 8, 3),
    (128, 256, 32, 8, 4),
    (128, 256, 32, 8, 2),
    (128, 256, 16, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 32, 8, 3),
    (128, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 128, 32, 4, 4),
    (64, 128, 32, 4, 5),
    (64, 128, 64, 4, 4),
    (64, 256, 64, 4, 4),
    (32, 128, 64, 4, 4),
    (32, 256, 64, 4, 4),
    (16, 128, 64, 2, 4),
]
_DEFAULT = (64, 128, 64, 4, 4, 0)
_CFG = {}
_BAD = set()
_LOG = {}
_FLUSH = [None]


def _init_mode(x, w, m_sizes):
    """确认 kernel 内前缀和路径可用且与 torch 前缀和路径结果一致, 否则回退。"""
    _ST["ink"] = False
    ref = _run(x, w, m_sizes, _DEFAULT)
    torch.cuda.synchronize()
    try:
        _ST["ink"] = True
        o = _run(x, w, m_sizes, _DEFAULT)
        torch.cuda.synchronize()
        if not bool(torch.equal(o, ref)):
            _ST["ink"] = False
    except Exception:
        _ST["ink"] = False


def _time_cfg(x, w, m_sizes, cfg, flush):
    sync = torch.cuda.synchronize
    _run(x, w, m_sizes, cfg)
    sync()
    _run(x, w, m_sizes, cfg)
    sync()
    ts = []
    n = 5
    i = 0
    while i < n:
        if flush is not None:
            flush.zero_()
        sync()
        t0 = time.perf_counter()
        o = _run(x, w, m_sizes, cfg)
        sync()
        t = time.perf_counter() - t0
        ts.append(t)
        del o
        if i == 0 and t < 1e-3:
            n = 9
        i += 1
    ts.sort()
    return ts[len(ts) // 2]


def _tune(x, w, m_sizes, key):
    T, K = x.shape
    E, _, N = w.shape
    if _ST["ink"] is None:
        _init_mode(x, w, m_sizes)
    avg = max(1, T // max(1, E))
    try:
        if _FLUSH[0] is None:
            _FLUSH[0] = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=x.device)
        flush = _FLUSH[0]
    except Exception:
        flush = None
    log = []

    def trial(cfg):
        if cfg in _BAD:
            return
        try:
            t = _time_cfg(x, w, m_sizes, cfg, flush)
        except Exception:
            _BAD.add(cfg)
            return
        log.append((t, cfg))

    for base in _BASE:
        BM, BN = base[0], base[1]
        if BN > N or BM > T:
            continue
        # 剪枝: tile 高度远大于/远小于平均 expert 大小的配置没有意义
        if BM > 8 * avg or (BM * 16 < avg and BM < 64):
            continue
        trial(base + (0,))
        trial(base + (1,))
    if log:
        top = []
        for t, cfg in sorted(log):
            if cfg[:5] not in top:
                top.append(cfg[:5])
            if len(top) == 3:
                break
        for base in top:
            for g in (2, 4, 8, 16):
                trial(base + (g,))
        # 复测前几名, 降低噪声造成的误选
        final = []
        for t, cfg in sorted(log)[:4]:
            try:
                final.append((min(t, _time_cfg(x, w, m_sizes, cfg, flush)), cfg))
            except Exception:
                pass
        best = min(final)[1] if final else min(log)[1]
    else:
        best = _DEFAULT
    _CFG[key] = best
    _LOG[key] = sorted(log)
    return best


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    key = (x.shape[0], x.shape[1], w.shape[0], w.shape[2])
    cfg = _CFG.get(key)
    if cfg is None:
        cfg = _tune(x, w, m_sizes, key)
    return _run(x, w, m_sizes, cfg)
