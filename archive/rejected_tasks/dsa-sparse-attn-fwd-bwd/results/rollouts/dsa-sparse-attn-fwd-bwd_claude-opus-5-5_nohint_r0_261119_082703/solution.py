import math

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def _fwd_kernel(Q, K, V, IDX, O, LSE, T, qk_scale,
                KSEL: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
                D: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    bh = tl.program_id(1)
    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    rows = (bh * G + offs_g).to(tl.int64) * T + t
    q = tl.load(Q + rows[:, None] * D + offs_d[None, :], mask=gmask[:, None], other=0.0)
    kv_base = bh.to(tl.int64) * T * D
    idx_base = (bh.to(tl.int64) * T + t) * KSEL
    m_i = tl.full([GP], -1e30, dtype=tl.float32)
    l_i = tl.zeros([GP], dtype=tl.float32)
    acc = tl.zeros([GP, D], dtype=tl.float32)
    for start in range(0, KSEL, BN):
        offs_n = start + tl.arange(0, BN)
        idx = tl.load(IDX + idx_base + offs_n, mask=offs_n < KSEL, other=-1)
        valid = (idx >= 0) & (idx <= t)
        idx_s = tl.where(valid, idx, 0)
        kptr = kv_base + idx_s[:, None] * D + offs_d[None, :]
        k = tl.load(K + kptr, mask=valid[:, None], other=0.0)
        v = tl.load(V + kptr, mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(O + rows[:, None] * D + offs_d[None, :], out.to(O.dtype.element_ty), mask=gmask[:, None])
    tl.store(LSE + rows, m_i + tl.log2(l_i), mask=gmask)


@triton.jit(do_not_specialize=["sd"])
def _bwd_kernel(Q, K, V, IDX, O, DO, LSE, DQ, DK, DV, T, qk_scale, sm_scale, sd,
                KSEL: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
                D: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    bh = tl.program_id(1)
    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    rows = (bh * G + offs_g).to(tl.int64) * T + t
    rp = rows[:, None] * D + offs_d[None, :]
    q = tl.load(Q + rp, mask=gmask[:, None], other=0.0)
    do = tl.load(DO + rp, mask=gmask[:, None], other=0.0)
    o = tl.load(O + rp, mask=gmask[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    lse = tl.load(LSE + rows, mask=gmask, other=float("inf"))
    kv_base = bh.to(tl.int64) * T * D
    idx_base = (bh.to(tl.int64) * T + t) * KSEL
    dq = tl.zeros([GP, D], dtype=tl.float32)
    for start in range(0, KSEL, BN):
        offs_n = start + tl.arange(0, BN)
        idx = tl.load(IDX + idx_base + offs_n, mask=offs_n < KSEL, other=-1)
        valid = (idx >= 0) & (idx <= t)
        idx_s = tl.where(valid, idx, 0)
        kptr = kv_base + idx_s[:, None] * D + offs_d[None, :]
        k = tl.load(K + kptr, mask=valid[:, None], other=0.0)
        v = tl.load(V + kptr, mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.exp2(s - lse[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dsb = ds.to(k.dtype)
        dq += tl.dot(dsb, k)
        dv_c = tl.dot(tl.trans(p.to(do.dtype)), do)
        dk_c = tl.dot(tl.trans(dsb), q) * sm_scale
        # sd == 1 at runtime; hiding it from the compiler's contiguity analysis makes the
        # atomics use a row-coalesced layout (one warp covers whole sectors), ~3x faster.
        aptr = kv_base + idx_s[:, None] * D + offs_d[None, :] * sd
        tl.atomic_add(DV + aptr, dv_c, mask=valid[:, None], sem="relaxed")
        tl.atomic_add(DK + aptr, dk_c, mask=valid[:, None], sem="relaxed")
    tl.store(DQ + rp, (dq * sm_scale).to(DQ.dtype.element_ty), mask=gmask[:, None])


_FWD_CFG = {"BN": 64, "num_warps": 4, "num_stages": 2}
_BWD_CFG = {"BN": 32, "num_warps": 4, "num_stages": 2}


def _fwd(q, k, v, idx, cfg=None):
    cfg = cfg or _FWD_CFG
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    KSEL = idx.shape[-1]
    G = Hq // Hkv
    GP = max(16, triton.next_power_of_2(G))
    o = torch.empty_like(q)
    lse = torch.empty((B, Hq, T), device=q.device, dtype=torch.float32)
    BN = min(cfg["BN"], KSEL)
    _fwd_kernel[(T, B * Hkv)](
        q, k, v, idx, o, lse, T, _LOG2E / math.sqrt(D),
        KSEL=KSEL, G=G, GP=GP, D=D, BN=BN,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return o, lse


def _bwd(q, k, v, idx, o, lse, do, cfg=None):
    cfg = cfg or _BWD_CFG
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    KSEL = idx.shape[-1]
    G = Hq // Hkv
    GP = max(16, triton.next_power_of_2(G))
    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, device=q.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=q.device, dtype=torch.float32)
    BN = min(cfg["BN"], KSEL)
    _bwd_kernel[(T, B * Hkv)](
        q, k, v, idx, o, do, lse, dq, dk, dv, T, _LOG2E / math.sqrt(D), 1.0 / math.sqrt(D), 1,
        KSEL=KSEL, G=G, GP=GP, D=D, BN=BN,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return dq, dk.to(k.dtype), dv.to(v.dtype)


class _DSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, idx):
        o, lse = _fwd(q, k, v, idx)
        ctx.save_for_backward(q, k, v, idx, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, idx, o, lse = ctx.saved_tensors
        do = do.contiguous()
        dq, dk, dv = _bwd(q, k, v, idx, o, lse, do)
        return dq, dk, dv, None


def dsa_sparse_attn(q, k, v, topk_idx):
    return _DSA.apply(q.contiguous(), k.contiguous(), v.contiguous(), topk_idx.contiguous())
