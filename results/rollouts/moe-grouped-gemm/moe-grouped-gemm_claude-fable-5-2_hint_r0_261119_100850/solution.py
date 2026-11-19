"""Grouped GEMM for MoE forward, written in Triton.

设计要点:
- 零预处理: m_sizes 在 kernel 内部读取, 每个 program 自己扫描 expert 前缀和,
  找到自己负责的 (expert, m-tile, n-tile); 多余的 program 直接空转退出。
- expert 内部对 tile 做 GROUP_M swizzle, 让并发的 SM 共享同一批权重列 (L2 复用)。
- 每个 expert 最后一个不完整的 M tile 通过标量平移保持在合法内存范围内,
  加载完全无 mask; N 方向最后一个 tile 向左平移重叠 (写入相同的值)。
- 第一次遇到新的 (形状, m_sizes buffer) 时在 warmup 里实测挑选 tile 配置和
  GROUP_M, 之后热路径只有一次 dict 查询 + 一次 kernel 启动。
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _gg_kernel(x_ptr, w_ptr, o_ptr, m_ptr, T, E, K, N, num_n, GROUP_M,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               FULL_ONLY: tl.constexpr, REM_OF: tl.constexpr,
               KTAIL: tl.constexpr, EVX: tl.constexpr, EVW: tl.constexpr):
    pid = tl.program_id(0)
    zero = pid * 0
    p_acc = zero
    row_acc = zero
    f_e = zero
    f_row = zero
    f_m = zero
    f_nt = zero
    f_local = zero
    # 在 kernel 内扫描 m_sizes, 定位本 program 所属 expert
    for e in range(E):
        m = tl.load(m_ptr + e)
        if REM_OF > 0:
            base = (m // REM_OF) * REM_OF
            mm = m - base
            nt = (mm + BM - 1) // BM
        else:
            if FULL_ONLY:
                base = zero
                nt = m // BM
                mm = nt * BM
            else:
                base = zero
                mm = m
                nt = (m + BM - 1) // BM
        np_ = nt * num_n
        hit = ((pid >= p_acc) & (pid < p_acc + np_)).to(tl.int32)
        f_e += hit * e
        f_row += hit * (row_acc + base)
        f_m += hit * mm
        f_nt += hit * nt
        f_local += hit * (pid - p_acc)
        p_acc += np_
        row_acc += m

    # expert 内部的 tile swizzle
    nt1 = tl.maximum(f_nt, 1)
    gsz = GROUP_M * num_n
    group_id = f_local // gsz
    first_m = group_id * GROUP_M
    group_size = tl.minimum(nt1 - first_m, GROUP_M)
    in_g = f_local % gsz
    tm = first_m + in_g % group_size
    tn = in_g // group_size

    r0 = f_row + tm * BM
    r_end = f_row + f_m
    r_lo = tl.minimum(r0, T - BM)
    rows = r_lo + tl.arange(0, BM)
    c0 = tl.minimum(tn * BN, N - BN)
    cols = c0 + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
    w_ptrs = w_ptr + f_e.to(tl.int64) * K * N + ks[:, None] * N + cols[None, :]

    kiters = ((K + BK - 1) // BK) * tl.minimum(f_nt, 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, kiters):
        if KTAIL:
            km = (k * BK + ks) < K
            a = tl.load(x_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(w_ptrs, mask=km[:, None], other=0.0)
        else:
            a = tl.load(x_ptrs, eviction_policy=EVX)
            b = tl.load(w_ptrs, eviction_policy=EVW)
        acc = tl.dot(a, b, acc)
        x_ptrs += BK
        w_ptrs += BK * N

    o_ptrs = o_ptr + rows[:, None] * N + cols[None, :]
    o_mask = ((rows >= r0) & (rows < r_end))[:, None] & (cols < N)[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


def _launch(x, w, m_sizes, out, T, E, K, N, cfg, group_m, full_only, rem_of, evx, evw):
    BM, BN, BK, nw, ns = cfg
    num_n = (N + BN - 1) // BN
    if rem_of:
        bound = min(E * ((rem_of - 1 + BM - 1) // BM), (T + E * (BM - 1)) // BM)
    elif full_only:
        bound = T // BM
    else:
        bound = (T + E * (BM - 1)) // BM
    if bound <= 0:
        return
    _gg_kernel[(bound * num_n,)](
        x, w, out, m_sizes, T, E, K, N, num_n, group_m,
        BM=BM, BN=BN, BK=BK, FULL_ONLY=full_only, REM_OF=rem_of,
        KTAIL=(K % BK != 0), EVX=evx, EVW=evw,
        num_warps=nw, num_stages=ns)


# plan = (cfg, group_m), cfg = (BM, BN, BK, num_warps, num_stages)
def _run(plan, x, w, m_sizes, out, T, E, K, N):
    _launch(x, w, m_sizes, out, T, E, K, N, plan[0], plan[1], False, 0, "", "")


_SINGLES = [
    (128, 128, 32, 4, 4),
    (128, 128, 64, 4, 3),
    (64, 256, 32, 4, 4),
    (64, 256, 32, 4, 3),
    (64, 128, 32, 4, 4),
    (64, 128, 64, 4, 4),
    (32, 256, 32, 4, 4),
    (32, 128, 64, 4, 4),
    (16, 256, 32, 4, 4),
]
_ALL = 4096

_PLANS = {}        # (T,E,K,N,m_ptr) -> plan
_SHAPE_PLANS = {}  # (T,E,K,N) -> [plan, 调优次数]
_MAX_TUNES_PER_SHAPE = 4
_FLUSH = [None]


def _time_once(plan, args):
    sync = torch.cuda.synchronize
    _FLUSH[0].zero_()
    sync()
    t0 = time.perf_counter()
    _run(plan, *args)
    sync()
    return time.perf_counter() - t0


def _rounds(plans, best, args, n_rounds):
    # 各候选交替测量并取最小值, 降低 GPU 状态漂移对比较的影响
    for _ in range(n_rounds):
        for p in plans:
            try:
                t = _time_once(p, args)
            except Exception:
                t = float("inf")
            if t < best.get(p, float("inf")):
                best[p] = t


def _tune(x, w, m_sizes, out, T, E, K, N):
    if _FLUSH[0] is None:
        _FLUSH[0] = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=x.device)
    args = (x, w, m_sizes, out, T, E, K, N)
    # 先全部编译 / 预热
    plans = []
    for cfg in _SINGLES:
        if cfg[0] > T or cfg[1] > N:
            continue
        p = (cfg, _ALL)
        try:
            _run(p, *args)
            torch.cuda.synchronize()
            plans.append(p)
        except Exception:
            pass
    if not plans:
        return None
    best = {}
    _rounds(plans, best, args, 2)
    top = sorted(plans, key=lambda p: best[p])[:3]
    cand = list(top)
    for p in top:
        for grows in (256, 1024):
            q = (p[0], max(1, grows // p[0][0]))
            if q not in cand:
                cand.append(q)
    _rounds(cand, best, args, 3)
    plan = min(cand, key=lambda p: best[p])
    # 用选中的方案重新计算, 保证 out 内容正确
    _run(plan, *args)
    return plan


def _fallback(x, w, m_sizes, out):
    # 极小形状 (T 或 N 小于最小 tile) 的保底路径: 逐行广播乘加
    start = 0
    for e, m in enumerate(m_sizes.tolist()):
        we = w[e].float()
        for r in range(start, start + m):
            out[r] = (x[r].float()[:, None] * we).sum(0).to(out.dtype)
        start += m
    return out


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=x.dtype, device=x.device)
    if T == 0 or N == 0:
        return out
    plan = _PLANS.get((T, E, K, N, m_sizes.data_ptr()))
    if plan is None:
        skey = (T, E, K, N)
        ent = _SHAPE_PLANS.get(skey)
        if ent is not None and ent[1] >= _MAX_TUNES_PER_SHAPE:
            plan = ent[0]
        else:
            plan = _tune(x, w, m_sizes, out, T, E, K, N)
            if plan is None:
                return _fallback(x, w, m_sizes, out)
            _PLANS[(T, E, K, N, m_sizes.data_ptr())] = plan
            if ent is None:
                _SHAPE_PLANS[skey] = [plan, 1]
            else:
                ent[0] = plan
                ent[1] += 1
            return out
    _run(plan, x, w, m_sizes, out, T, E, K, N)
    return out
