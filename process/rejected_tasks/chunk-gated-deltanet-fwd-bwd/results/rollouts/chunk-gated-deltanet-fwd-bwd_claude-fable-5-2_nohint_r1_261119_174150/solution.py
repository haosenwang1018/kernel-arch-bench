"""Grouped-value packed varlen Gated DeltaNet: chunkwise WY forward/backward in Triton.

前向: prep(每 chunk 的 T=(I+M)^-1 与 A) -> 按序列串行的 chunk 递推。
反向: 重算每个 chunk 的起始状态 S0 与 u (临时缓冲, 不跨 fwd/bwd 保留) -> 逆序传播 dS -> 并行汇总 dq/dk/dbeta。
"""

import torch
import triton
import triton.language as tl

BT = 64
RETAIN_LIMIT = 140 * 1024 * 1024


@triton.jit
def _prep_kernel(q, k, g, beta, cst, cln, Tm, Am, H_k, H_v, G, K: tl.constexpr, BT: tl.constexpr):
    i_c = tl.program_id(0).to(tl.int64)
    i_hv = tl.program_id(1).to(tl.int64)
    i_hk = i_hv // G
    s = tl.load(cst + i_c).to(tl.int64)
    L = tl.load(cln + i_c)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    m_i = o_i < L
    row = s + o_i
    m_ik = m_i[:, None] & (o_k[None, :] >= 0)
    p_k = (row[:, None] * H_k + i_hk) * K + o_k[None, :]
    b_k = tl.load(k + p_k, mask=m_ik, other=0.0).to(tl.float32)
    b_q = tl.load(q + p_k, mask=m_ik, other=0.0).to(tl.float32)
    b_g = tl.load(g + row * H_v + i_hv, mask=m_i, other=0.0)
    b_b = tl.load(beta + row * H_v + i_hv, mask=m_i, other=0.0).to(tl.float32)
    m_ge = o_i[:, None] >= o_i[None, :]
    m_gt = o_i[:, None] > o_i[None, :]
    gam = tl.sum(tl.where(m_ge, b_g[None, :], 0.0), axis=1)
    b_kt = tl.trans(b_k)
    kk = tl.dot(b_k, b_kt)
    qk = tl.dot(b_q, b_kt)
    dec = tl.exp(tl.minimum(gam[:, None] - gam[None, :], 0.0))
    b_A = tl.where(m_ge, qk * dec, 0.0)
    b_N = tl.where(m_gt, -(b_b[:, None] * dec * kk), 0.0)
    b_P = b_N + tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0)
    for _ in range(5):
        b_N = tl.dot(b_N, b_N)
        b_P = b_P + tl.dot(b_P, b_N)
    p_o = (i_c * H_v + i_hv) * BT * BT + o_i[:, None] * BT + o_i[None, :]
    tl.store(Tm + p_o, b_P.to(tl.bfloat16))
    tl.store(Am + p_o, b_A.to(tl.bfloat16))


@triton.jit
def _fwd_kernel(q, k, v, g, beta, Tm, Am, cu, coff, out, Sst, Ust, H_k, H_v, G, V,
                K: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr, SAVE: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H_v
    i_hv = (i_bh % H_v).to(tl.int64)
    i_hk = i_hv // G
    bos = tl.load(cu + i_b).to(tl.int64)
    eos = tl.load(cu + i_b + 1).to(tl.int64)
    c0 = tl.load(coff + i_b).to(tl.int64)
    NT = ((eos - bos + BT - 1) // BT).to(tl.int32)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_ge = o_i[:, None] >= o_i[None, :]
    b_S = tl.zeros([K, BV], dtype=tl.float32)
    for i_t in range(NT):
        s = bos + i_t * BT
        row = s + o_i
        m_i = row < eos
        m_ik = m_i[:, None] & (o_k[None, :] >= 0)
        m_iv = m_i[:, None] & (o_v[None, :] >= 0)
        c = c0 + i_t
        p_k = (row[:, None] * H_k + i_hk) * K + o_k[None, :]
        p_v = (row[:, None] * H_v + i_hv) * V + o_v[None, :]
        p_m = (c * H_v + i_hv) * BT * BT + o_i[:, None] * BT + o_i[None, :]
        b_k = tl.load(k + p_k, mask=m_ik, other=0.0).to(tl.float32)
        b_v = tl.load(v + p_v, mask=m_iv, other=0.0).to(tl.float32)
        b_g = tl.load(g + row * H_v + i_hv, mask=m_i, other=0.0)
        b_b = tl.load(beta + row * H_v + i_hv, mask=m_i, other=0.0).to(tl.float32)
        gam = tl.sum(tl.where(m_ge, b_g[None, :], 0.0), axis=1)
        gl = tl.sum(b_g)
        eg = tl.exp(gam)
        b_T = tl.load(Tm + p_m).to(tl.float32)
        if SAVE:
            p_s = ((c * H_v + i_hv) * K + o_k[:, None]) * V + o_v[None, :]
            tl.store(Sst + p_s, b_S.to(tl.bfloat16))
        kS = tl.dot(b_k, b_S)
        b_r = b_b[:, None] * (b_v - eg[:, None] * kS)
        b_u = tl.dot(b_T, b_r)
        if SAVE:
            tl.store(Ust + p_v, b_u.to(tl.bfloat16), mask=m_iv)
        else:
            b_q = tl.load(q + p_k, mask=m_ik, other=0.0).to(tl.float32)
            b_A = tl.load(Am + p_m).to(tl.float32)
            b_o = eg[:, None] * tl.dot(b_q, b_S) + tl.dot(b_A, b_u)
            tl.store(out + p_v, b_o.to(tl.bfloat16), mask=m_iv)
        kbar = b_k * tl.exp(gl - gam)[:, None]
        b_S = b_S * tl.exp(gl) + tl.dot(tl.trans(kbar), b_u)


@triton.jit
def _bwd_kernel(q, k, v, g, beta, Tm, Am, cu, coff, out, dout, dv, Sst, dSst, Cst, H_k, H_v, G, V, NV,
                K: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H_v
    i_hv = (i_bh % H_v).to(tl.int64)
    i_hk = i_hv // G
    bos = tl.load(cu + i_b).to(tl.int64)
    eos = tl.load(cu + i_b + 1).to(tl.int64)
    c0 = tl.load(coff + i_b).to(tl.int64)
    NT = ((eos - bos + BT - 1) // BT).to(tl.int32)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_ge = o_i[:, None] >= o_i[None, :]
    b_dS = tl.zeros([K, BV], dtype=tl.float32)
    for i_tt in range(NT):
        i_t = NT - 1 - i_tt
        s = bos + i_t * BT
        row = s + o_i
        m_i = row < eos
        m_ik = m_i[:, None] & (o_k[None, :] >= 0)
        m_iv = m_i[:, None] & (o_v[None, :] >= 0)
        c = c0 + i_t
        cn = c0 + tl.minimum(i_t + 1, NT - 1)
        p_k = (row[:, None] * H_k + i_hk) * K + o_k[None, :]
        p_v = (row[:, None] * H_v + i_hv) * V + o_v[None, :]
        p_m = (c * H_v + i_hv) * BT * BT + o_i[:, None] * BT + o_i[None, :]
        p_s = ((c * H_v + i_hv) * K + o_k[:, None]) * V + o_v[None, :]
        p_sn = ((cn * H_v + i_hv) * K + o_k[:, None]) * V + o_v[None, :]
        b_k = tl.load(k + p_k, mask=m_ik, other=0.0).to(tl.float32)
        b_q = tl.load(q + p_k, mask=m_ik, other=0.0).to(tl.float32)
        b_v = tl.load(v + p_v, mask=m_iv, other=0.0).to(tl.float32)
        b_o = tl.load(out + p_v, mask=m_iv, other=0.0).to(tl.float32)
        b_do = tl.load(dout + p_v, mask=m_iv, other=0.0).to(tl.float32)
        b_g = tl.load(g + row * H_v + i_hv, mask=m_i, other=0.0)
        b_b = tl.load(beta + row * H_v + i_hv, mask=m_i, other=0.0).to(tl.float32)
        gam = tl.sum(tl.where(m_ge, b_g[None, :], 0.0), axis=1)
        gl = tl.sum(b_g)
        eg = tl.exp(gam)
        p_mt = (c * H_v + i_hv) * BT * BT + o_i[None, :] * BT + o_i[:, None]
        b_Tt = tl.load(Tm + p_mt).to(tl.float32)
        b_At = tl.load(Am + p_mt).to(tl.float32)
        b_Sn = tl.load(Sst + p_sn).to(tl.float32)
        tail = tl.sum(b_Sn * b_dS)
        tl.store(dSst + p_s, b_dS.to(tl.bfloat16))
        kbar = b_k * tl.exp(gl - gam)[:, None]
        b_du = tl.dot(b_At, b_do) + tl.dot(kbar, b_dS)
        b_dr = tl.dot(b_Tt, b_du)
        b_dv = b_b[:, None] * b_dr
        tl.store(dv + p_v, b_dv.to(tl.bfloat16), mask=m_iv)
        crow = tl.sum(b_o * b_do - b_v * b_dv, axis=1)
        dgp = tl.sum(tl.where(m_ge, crow[:, None], 0.0), axis=0) + tail
        tl.store(Cst + (row * H_v + i_hv) * NV + i_v, dgp, mask=m_i)
        b_dS = b_dS * tl.exp(gl) + tl.dot(tl.trans(b_q), eg[:, None] * b_do) \
            - tl.dot(tl.trans(b_k), eg[:, None] * b_dv)


@triton.jit
def _fin_kernel(q, k, g, beta, cst, cln, dout, dv, Ust, Sst, dSst, dq, dk, dbeta, H_k, H_v, G, V, NVF,
                K: tl.constexpr, BVF: tl.constexpr, BT: tl.constexpr):
    i_c = tl.program_id(0).to(tl.int64)
    i_hk = tl.program_id(1).to(tl.int64)
    s = tl.load(cst + i_c).to(tl.int64)
    L = tl.load(cln + i_c)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    m_i = o_i < L
    row = s + o_i
    m_ik = m_i[:, None] & (o_k[None, :] >= 0)
    m_ge = o_i[:, None] >= o_i[None, :]
    m_gt = o_i[:, None] > o_i[None, :]
    m_eq = o_i[:, None] == o_i[None, :]
    p_k = (row[:, None] * H_k + i_hk) * K + o_k[None, :]
    accq = tl.zeros([BT, K], dtype=tl.float32)
    acck = tl.zeros([BT, K], dtype=tl.float32)
    for ig in range(G):
        i_hv = i_hk * G + ig
        b_g = tl.load(g + row * H_v + i_hv, mask=m_i, other=0.0)
        b_b = tl.load(beta + row * H_v + i_hv, mask=m_i, other=0.0).to(tl.float32)
        gam = tl.sum(tl.where(m_ge, b_g[None, :], 0.0), axis=1)
        gl = tl.sum(b_g)
        eg = tl.exp(gam)
        ebar = tl.exp(gl - gam)
        b_DA = tl.zeros([BT, BT], dtype=tl.float32)
        b_DM = tl.zeros([BT, BT], dtype=tl.float32)
        for iv in range(NVF):
            o_v = iv * BVF + tl.arange(0, BVF)
            m_iv = m_i[:, None] & (o_v[None, :] >= 0)
            p_v = (row[:, None] * H_v + i_hv) * V + o_v[None, :]
            p_st = ((i_c * H_v + i_hv) * K + o_k[None, :]) * V + o_v[:, None]
            b_do = tl.load(dout + p_v, mask=m_iv, other=0.0).to(tl.float32)
            b_dv = tl.load(dv + p_v, mask=m_iv, other=0.0).to(tl.float32)
            b_u = tl.load(Ust + p_v, mask=m_iv, other=0.0).to(tl.float32)
            b_St = tl.load(Sst + p_st).to(tl.float32)
            b_dSt = tl.load(dSst + p_st).to(tl.float32)
            b_ut = tl.trans(b_u)
            b_DA += tl.dot(b_do, b_ut)
            b_DM += tl.dot(b_dv, b_ut)
            accq += tl.dot(eg[:, None] * b_do, b_St)
            acck += tl.dot(ebar[:, None] * b_u, b_dSt) - tl.dot(eg[:, None] * b_dv, b_St)
        dec = tl.exp(tl.minimum(gam[:, None] - gam[None, :], 0.0))
        b_At = tl.where(m_ge, b_DA * dec, 0.0)
        b_Mt = tl.where(m_gt, b_DM * dec, 0.0)
        b_k = tl.load(k + p_k, mask=m_ik, other=0.0).to(tl.float32)
        b_q = tl.load(q + p_k, mask=m_ik, other=0.0).to(tl.float32)
        accq += tl.dot(b_At, b_k)
        acck += tl.dot(tl.trans(b_At), b_q) - tl.dot(b_Mt + tl.trans(b_Mt), b_k)
        dgl = tl.sum(tl.where(m_eq, b_DM, 0.0), axis=1)
        db = tl.where(b_b > 0, dgl / (b_b * b_b), 0.0)
        tl.store(dbeta + row * H_v + i_hv, db.to(tl.bfloat16), mask=m_i)
    tl.store(dq + p_k, accq.to(tl.bfloat16), mask=m_ik)
    tl.store(dk + p_k, acck.to(tl.bfloat16), mask=m_ik)


def _tables(cu_seqlens):
    cu = cu_seqlens.tolist()
    coff, cst, cln = [], [], []
    for b in range(len(cu) - 1):
        s, e = cu[b], cu[b + 1]
        coff.append(len(cst))
        for t in range(s, e, BT):
            cst.append(t)
            cln.append(min(BT, e - t))
    nb, nc = len(coff), len(cst)
    tab = torch.tensor(coff + cst + cln, dtype=torch.int32, device=cu_seqlens.device)
    return nb, nc, tab[:nb], tab[nb:nb + nc], tab[nb + nc:]


def _pick_bv(B, H_v, V):
    return 32


class _GatedDeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens):
        T, H_k, K = q.shape
        H_v, V = v.shape[1], v.shape[2]
        G = H_v // H_k
        dev = q.device
        B, NC, coff, cst, cln = _tables(cu_seqlens)
        Tm = torch.empty((NC, H_v, BT, BT), dtype=torch.bfloat16, device=dev)
        Am = torch.empty((NC, H_v, BT, BT), dtype=torch.bfloat16, device=dev)
        _prep_kernel[(NC, H_v)](q, k, g, beta, cst, cln, Tm, Am, H_k, H_v, G, K=K, BT=BT, num_warps=4)
        out = torch.empty((T, H_v, V), dtype=torch.bfloat16, device=dev)
        BV = _pick_bv(B, H_v, V)
        _fwd_kernel[(V // BV, B * H_v)](q, k, v, g, beta, Tm, Am, cu_seqlens, coff, out, out, out,
                                        H_k, H_v, G, V, K=K, BV=BV, BT=BT, SAVE=False, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, out)
        ctx.cu = cu_seqlens
        ctx.tabs = (B, NC, coff, cst, cln)
        if 2 * Tm.numel() * 2 <= RETAIN_LIMIT:
            ctx.ta = (Tm, Am)
        else:
            ctx.ta = None
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, out = ctx.saved_tensors
        cu_seqlens = ctx.cu
        B, NC, coff, cst, cln = ctx.tabs
        T, H_k, K = q.shape
        H_v, V = v.shape[1], v.shape[2]
        G = H_v // H_k
        dev = q.device
        dout = dout.contiguous()
        if dout.dtype != torch.bfloat16:
            dout = dout.to(torch.bfloat16)
        if ctx.ta is not None:
            Tm, Am = ctx.ta
        else:
            Tm = torch.empty((NC, H_v, BT, BT), dtype=torch.bfloat16, device=dev)
            Am = torch.empty((NC, H_v, BT, BT), dtype=torch.bfloat16, device=dev)
            _prep_kernel[(NC, H_v)](q, k, g, beta, cst, cln, Tm, Am, H_k, H_v, G, K=K, BT=BT, num_warps=4)
        BV = _pick_bv(B, H_v, V)
        NV = V // BV
        Sst = torch.empty((NC, H_v, K, V), dtype=torch.bfloat16, device=dev)
        dSst = torch.empty((NC, H_v, K, V), dtype=torch.bfloat16, device=dev)
        Ust = torch.empty((T, H_v, V), dtype=torch.bfloat16, device=dev)
        _fwd_kernel[(NV, B * H_v)](q, k, v, g, beta, Tm, Am, cu_seqlens, coff, out, Sst, Ust,
                                   H_k, H_v, G, V, K=K, BV=BV, BT=BT, SAVE=True, num_warps=4)
        dv = torch.empty((T, H_v, V), dtype=torch.bfloat16, device=dev)
        Cst = torch.empty((T, H_v, NV), dtype=torch.float32, device=dev)
        _bwd_kernel[(NV, B * H_v)](q, k, v, g, beta, Tm, Am, cu_seqlens, coff, out, dout, dv, Sst, dSst, Cst,
                                   H_k, H_v, G, V, NV, K=K, BV=BV, BT=BT, num_warps=8)
        dq = torch.empty((T, H_k, K), dtype=torch.bfloat16, device=dev)
        dk = torch.empty((T, H_k, K), dtype=torch.bfloat16, device=dev)
        dbeta = torch.empty((T, H_v), dtype=torch.bfloat16, device=dev)
        BVF = 32
        _fin_kernel[(NC, H_k)](q, k, g, beta, cst, cln, dout, dv, Ust, Sst, dSst, dq, dk, dbeta,
                               H_k, H_v, G, V, V // BVF, K=K, BVF=BVF, BT=BT, num_warps=8)
        dg = Cst.sum(-1)
        ctx.ta = None
        return dq, dk, dv, dg, dbeta, None


def chunk_gated_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return _GatedDeltaFn.apply(q, k, v, g, beta, cu_seqlens)
