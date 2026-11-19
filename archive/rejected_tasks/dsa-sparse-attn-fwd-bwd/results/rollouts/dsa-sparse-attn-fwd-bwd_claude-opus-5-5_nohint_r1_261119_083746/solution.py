import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_fwd_kernel(
    Q, K, V, IDX, O, LSE,
    T, KSEL, scale_log2,
    G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    bh = pid // T

    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, BS)
    gmask = offs_g < G

    q_off = ((bh * G + offs_g) * T + t) * D
    q = tl.load(Q + q_off[:, None] + offs_d[None, :], mask=gmask[:, None], other=0.0)

    kv_base = bh * T * D
    idx_base = IDX + (bh * T + t) * KSEL

    m_i = tl.full([GP], -1e30, dtype=tl.float32)
    l_i = tl.zeros([GP], dtype=tl.float32)
    acc = tl.zeros([GP, D], dtype=tl.float32)

    for s0 in range(0, KSEL, BS):
        sm = (s0 + offs_s) < KSEL
        idx = tl.load(idx_base + s0 + offs_s, mask=sm, other=-1)
        valid = (idx >= 0) & (idx <= t)
        idx = tl.where(valid, idx, 0)
        kv_off = kv_base + idx * D
        k = tl.load(K + kv_off[:, None] + offs_d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(V + kv_off[:, None] + offs_d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(O + q_off[:, None] + offs_d[None, :], o.to(O.dtype.element_ty), mask=gmask[:, None])
    lse = m_i + tl.log2(l_i)
    tl.store(LSE + (bh * G + offs_g) * T + t, lse, mask=gmask)


@triton.jit
def _dsa_bwd_kernel(
    Q, K, V, IDX, O, DO, LSE, DQ, SCR,
    T, KSEL, bh0, scale, scale_log2,
    G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
):
    # dQ for one (b, h_kv, t); per-pair dV/dK partials go to SCR (bf16, no atomics).
    lpid = tl.program_id(0)
    t = lpid % T
    bh = bh0 + lpid // T

    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, BS)
    gmask = offs_g < G

    q_off = ((bh * G + offs_g) * T + t) * D
    ptr2 = q_off[:, None] + offs_d[None, :]
    q = tl.load(Q + ptr2, mask=gmask[:, None], other=0.0)
    do = tl.load(DO + ptr2, mask=gmask[:, None], other=0.0)
    o = tl.load(O + ptr2, mask=gmask[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    lse = tl.load(LSE + (bh * G + offs_g) * T + t, mask=gmask, other=0.0)

    kv_base = bh * T * D
    idx_base = IDX + (bh * T + t) * KSEL
    scr_base = SCR + lpid.to(tl.int64) * KSEL * (2 * D)
    qs = (q.to(tl.float32) * scale).to(q.dtype)

    dq = tl.zeros([GP, D], dtype=tl.float32)
    for s0 in range(0, KSEL, BS):
        sm = (s0 + offs_s) < KSEL
        idx = tl.load(idx_base + s0 + offs_s, mask=sm, other=-1)
        valid = (idx >= 0) & (idx <= t)
        idx = tl.where(valid, idx, 0)
        kv_off = kv_base + idx * D
        kv_ptr = kv_off[:, None] + offs_d[None, :]
        k = tl.load(K + kv_ptr, mask=valid[:, None], other=0.0)
        v = tl.load(V + kv_ptr, mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        p = tl.exp2(s - lse[:, None])
        p = tl.where(valid[None, :] & gmask[:, None], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dsb = ds.to(k.dtype)
        dq += tl.dot(dsb, k)
        dv_part = tl.dot(tl.trans(p.to(do.dtype)), do)
        dk_part = tl.dot(tl.trans(dsb), qs)
        sp = scr_base + ((s0 + offs_s) * (2 * D))[:, None] + offs_d[None, :]
        tl.store(sp, dv_part.to(SCR.dtype.element_ty), mask=valid[:, None])
        tl.store(sp + D, dk_part.to(SCR.dtype.element_ty), mask=valid[:, None])

    dq = dq * scale
    tl.store(DQ + ptr2, dq.to(DQ.dtype.element_ty), mask=gmask[:, None])


@triton.jit
def _count_kernel(IDX, CNT, NP, T, KSEL, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pm = p < NP
    idx = tl.load(IDX + p, mask=pm, other=-1)
    row = p // KSEL
    t = row % T
    valid = pm & (idx >= 0) & (idx <= t)
    key = (row - t) + idx
    tl.atomic_add(CNT + key, tl.full([BLOCK], 1, tl.int32), mask=valid, sem="relaxed")


@triton.jit
def _fill_kernel(IDX, CUR, LST, NP, T, KSEL, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pm = p < NP
    idx = tl.load(IDX + p, mask=pm, other=-1)
    row = p // KSEL
    t = row % T
    valid = pm & (idx >= 0) & (idx <= t)
    key = (row - t) + idx
    pos = tl.atomic_add(CUR + key, tl.full([BLOCK], 1, tl.int32), mask=valid, sem="relaxed")
    tl.store(LST + pos, p, mask=valid)


@triton.jit
def _reduce_kernel(SCR, LST, OFF, CNT, DK, DV, key0, pair0, D: tl.constexpr, BL: tl.constexpr):
    key = key0 + tl.program_id(0)
    start = tl.load(OFF + key)
    n = tl.load(CNT + key)
    offs_l = tl.arange(0, BL)
    offs_c = tl.arange(0, 2 * D)
    acc = tl.zeros([BL, 2 * D], dtype=tl.float32)
    for i in range(0, n, BL):
        lm = (i + offs_l) < n
        pr = tl.load(LST + start + i + offs_l, mask=lm, other=pair0)
        row = (pr - pair0).to(tl.int64) * (2 * D)
        x = tl.load(SCR + row[:, None] + offs_c[None, :], mask=lm[:, None], other=0.0)
        acc += x.to(tl.float32)
    r = tl.sum(acc, 0)
    is_v = offs_c < D
    base = key.to(tl.int64) * D
    tl.store(DV + base + offs_c, r.to(DV.dtype.element_ty), mask=is_v)
    tl.store(DK + base + offs_c - D, r.to(DK.dtype.element_ty), mask=offs_c >= D)


_FWD_CONFIGS = [
    (32, 4, 1), (32, 4, 2), (64, 4, 1), (64, 4, 2), (64, 8, 2), (128, 8, 2), (64, 4, 3),
]
_BWD_CONFIGS = [
    (16, 4, 2), (32, 4, 1), (32, 4, 2), (64, 4, 2), (64, 8, 2), (32, 8, 2),
]
_TUNE_CACHE = {}


def _time_it(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / reps


def _pick(key, configs, launch, reset=None):
    cfg = _TUNE_CACHE.get(key)
    if cfg is not None:
        return cfg
    best, best_t = None, float("inf")
    for c in configs:
        def run(c=c):
            if reset is not None:
                reset()
            launch(c)
        try:
            ms = _time_it(run)
        except Exception:  # noqa: BLE001
            continue
        if ms < best_t:
            best, best_t = c, ms
    _TUNE_CACHE[key] = best
    if reset is not None:
        reset()
    return best


def _shape_key(q, k, topk_idx):
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    G = H_q // H_kv
    return (B, H_kv, G, T, D, topk_idx.shape[-1])


def _fwd(q, k, v, topk_idx):
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    G = H_q // H_kv
    GP = max(16, triton.next_power_of_2(G))
    KSEL = topk_idx.shape[-1]
    o = torch.empty_like(q)
    lse = torch.empty((B, H_q, T), dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(D)
    grid = (B * H_kv * T,)

    def launch(c):
        bs, w, st = c
        _dsa_fwd_kernel[grid](
            q, k, v, topk_idx, o, lse, T, KSEL, scale * 1.4426950408889634,
            G=G, GP=GP, D=D, BS=bs, num_warps=w, num_stages=st,
        )

    cfg = _pick(("fwd",) + _shape_key(q, k, topk_idx), _FWD_CONFIGS, launch)
    launch(cfg)
    return o, lse


_SCR_CAP = 3 << 30


def _bwd(q, k, v, topk_idx, o, lse, do):
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    G = H_q // H_kv
    GP = max(16, triton.next_power_of_2(G))
    KSEL = topk_idx.shape[-1]
    dev = q.device
    scale = 1.0 / math.sqrt(D)
    NBH = B * H_kv
    NK = NBH * T
    NP = NK * KSEL

    # inverse index: for every key, the list of pair ids selecting it (counting sort)
    BLK = 1024
    cnt = torch.zeros((NK,), dtype=torch.int32, device=dev)
    _count_kernel[(triton.cdiv(NP, BLK),)](topk_idx, cnt, NP, T, KSEL, BLOCK=BLK, num_warps=4)
    off = torch.cumsum(cnt, 0, dtype=torch.int32) - cnt
    cur = off.clone()
    lst = torch.empty((NP,), dtype=torch.int32, device=dev)
    _fill_kernel[(triton.cdiv(NP, BLK),)](topk_idx, cur, lst, NP, T, KSEL, BLOCK=BLK, num_warps=4)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    per_bh = T * KSEL * 2 * D * 2
    nb = max(1, min(NBH, _SCR_CAP // per_bh))
    scr = torch.empty((nb * T * KSEL * 2 * D,), dtype=torch.bfloat16, device=dev)
    rbl = 32 if D == 128 else 64

    for bh0 in range(0, NBH, nb):
        n = min(nb, NBH - bh0)
        grid = (n * T,)

        def launch(c, bh0=bh0, grid=grid):
            bs, w, st = c
            _dsa_bwd_kernel[grid](
                q, k, v, topk_idx, o, do, lse, dq, scr,
                T, KSEL, bh0, scale, scale * 1.4426950408889634,
                G=G, GP=GP, D=D, BS=bs, num_warps=w, num_stages=st,
            )

        cfg = _pick(("bwd",) + _shape_key(q, k, topk_idx), _BWD_CONFIGS, launch)
        launch(cfg)
        _reduce_kernel[(n * T,)](
            scr, lst, off, cnt, dk, dv, bh0 * T, bh0 * T * KSEL,
            D=D, BL=rbl, num_warps=4,
        )
    return dq, dk, dv


class _DSASparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk_idx):
        o, lse = _fwd(q, k, v, topk_idx)
        ctx.save_for_backward(q, k, v, topk_idx, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, topk_idx, o, lse = ctx.saved_tensors
        do = do.contiguous()
        dq, dk, dv = _bwd(q, k, v, topk_idx, o, lse, do)
        return dq, dk, dv, None


def dsa_sparse_attn(q, k, v, topk_idx):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    topk_idx = topk_idx.contiguous()
    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)
    return _DSASparseAttn.apply(q, k, v, topk_idx)
