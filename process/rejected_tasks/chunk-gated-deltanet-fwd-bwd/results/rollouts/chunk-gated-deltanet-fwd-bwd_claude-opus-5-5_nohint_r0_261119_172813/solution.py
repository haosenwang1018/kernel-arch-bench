"""Chunked (BT=64) WY-form Gated DeltaNet forward + backward in Triton, varlen + grouped value heads."""

import time
import torch
import triton
import triton.language as tl

BT = 64
PROFILE = False
_EV = None
_VAR = []


def _tick(name):
    if _EV is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        _EV.append((name, e, time.perf_counter()))


@triton.jit
def _table_kernel(cu, info, seqoff, B, NTM, BLOCK: tl.constexpr, BB: tl.constexpr):
    b = tl.program_id(0)
    ob = tl.arange(0, BB)
    lo = tl.load(cu + ob, mask=ob < B, other=0)
    hi = tl.load(cu + ob + 1, mask=ob < B, other=0)
    nall = (hi - lo + 63) // 64
    off = tl.sum(tl.where(ob < b, nall, 0), 0)
    n = tl.sum(tl.where(ob == b, nall, 0), 0)
    tot = tl.sum(nall, 0)
    tl.store(seqoff + b, off)
    o = tl.arange(0, BLOCK)
    for s in range(0, n, BLOCK):
        idx = s + o
        m = idx < n
        tl.store(info + (off + idx) * 2, tl.zeros([BLOCK], dtype=tl.int32) + b, mask=m)
        tl.store(info + (off + idx) * 2 + 1, idx, mask=m)
    if b == B - 1:
        for s in range(tot, NTM, BLOCK):
            idx = s + o
            tl.store(info + idx * 2, tl.zeros([BLOCK], dtype=tl.int32) - 1, mask=idx < NTM)


@triton.jit
def _solve_tril(Ab, BT: tl.constexpr):
    # Ab: 64x64 fp32 row-major strictly-lower A (upper part zero). Overwritten with (I + A)^-1.
    o = tl.arange(0, 16)
    lo = o[:, None] > o[None, :]
    eye = (o[:, None] == o[None, :]).to(tl.float32)
    p = Ab + o[:, None] * BT + o[None, :]
    A11 = tl.load(p)
    A22 = tl.load(p + 16 * BT + 16)
    A33 = tl.load(p + 32 * BT + 32)
    A44 = tl.load(p + 48 * BT + 48)
    A21 = tl.load(p + 16 * BT)
    A31 = tl.load(p + 32 * BT)
    A32 = tl.load(p + 32 * BT + 16)
    A41 = tl.load(p + 48 * BT)
    A42 = tl.load(p + 48 * BT + 16)
    A43 = tl.load(p + 48 * BT + 32)
    X1 = -tl.where(lo, A11, 0.)
    X2 = -tl.where(lo, A22, 0.)
    X3 = -tl.where(lo, A33, 0.)
    X4 = -tl.where(lo, A44, 0.)
    for i in tl.static_range(2, 16):
        sel = (o == i)[:, None]
        a1 = tl.sum(tl.where(sel, X1, 0.), 0)
        a2 = tl.sum(tl.where(sel, X2, 0.), 0)
        a3 = tl.sum(tl.where(sel, X3, 0.), 0)
        a4 = tl.sum(tl.where(sel, X4, 0.), 0)
        a1 = a1 + tl.sum(a1[:, None] * X1, 0)
        a2 = a2 + tl.sum(a2[:, None] * X2, 0)
        a3 = a3 + tl.sum(a3[:, None] * X3, 0)
        a4 = a4 + tl.sum(a4[:, None] * X4, 0)
        X1 = tl.where(sel, a1[None, :], X1)
        X2 = tl.where(sel, a2[None, :], X2)
        X3 = tl.where(sel, a3[None, :], X3)
        X4 = tl.where(sel, a4[None, :], X4)
    X1 += eye
    X2 += eye
    X3 += eye
    X4 += eye
    T21 = -tl.dot(tl.dot(X2, A21, input_precision="ieee"), X1, input_precision="ieee")
    T32 = -tl.dot(tl.dot(X3, A32, input_precision="ieee"), X2, input_precision="ieee")
    T43 = -tl.dot(tl.dot(X4, A43, input_precision="ieee"), X3, input_precision="ieee")
    T31 = -tl.dot(X3, tl.dot(A31, X1, input_precision="ieee") + tl.dot(A32, T21, input_precision="ieee"),
                  input_precision="ieee")
    T42 = -tl.dot(X4, tl.dot(A42, X2, input_precision="ieee") + tl.dot(A43, T32, input_precision="ieee"),
                  input_precision="ieee")
    T41 = -tl.dot(X4, tl.dot(A41, X1, input_precision="ieee") + tl.dot(A42, T21, input_precision="ieee")
                  + tl.dot(A43, T31, input_precision="ieee"), input_precision="ieee")
    tl.debug_barrier()
    tl.store(p, X1)
    tl.store(p + 16 * BT + 16, X2)
    tl.store(p + 32 * BT + 32, X3)
    tl.store(p + 48 * BT + 48, X4)
    tl.store(p + 16 * BT, T21)
    tl.store(p + 32 * BT, T31)
    tl.store(p + 32 * BT + 16, T32)
    tl.store(p + 48 * BT, T41)
    tl.store(p + 48 * BT + 16, T42)
    tl.store(p + 48 * BT + 32, T43)
    tl.debug_barrier()


@triton.jit
def _prep_kernel(k, v, g, beta, cu, info, w, u, Abuf,
                 H: tl.constexpr, HK: tl.constexpr, G: tl.constexpr,
                 K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_c = tl.program_id(0)
    i_h = tl.program_id(1)
    b = tl.load(info + 2 * i_c)
    if b < 0:
        return
    c = tl.load(info + 2 * i_c + 1)
    bos = tl.load(cu + b)
    eos = tl.load(cu + b + 1)
    o_i = tl.arange(0, BT)
    o_t = bos + c * BT + o_i
    m_t = o_t < eos
    i_hk = i_h // G
    o_k = tl.arange(0, K)
    b_k = tl.load(k + (o_t[:, None] * HK + i_hk) * K + o_k[None, :], mask=m_t[:, None], other=0.)
    b_g = tl.load(g + o_t * H + i_h, mask=m_t, other=0.)
    b_gc = tl.cumsum(b_g, 0)
    b_beta = tl.load(beta + o_t * H + i_h, mask=m_t, other=0.).to(tl.float32)
    b_kk = tl.dot(b_k, tl.trans(b_k))
    strict = o_i[:, None] > o_i[None, :]
    b_A = tl.where(strict, b_kk * b_beta[:, None] * tl.exp(b_gc[:, None] - b_gc[None, :]), 0.)
    Ab = Abuf + (i_c * H + i_h).to(tl.int64) * (BT * BT)
    pA = Ab + o_i[:, None] * BT + o_i[None, :]
    tl.store(pA, b_A)
    tl.debug_barrier()
    _solve_tril(Ab, BT)
    b_T = tl.load(pA).to(tl.bfloat16)
    b_k = tl.load(k + (o_t[:, None] * HK + i_hk) * K + o_k[None, :], mask=m_t[:, None], other=0.)
    b_X = (b_k.to(tl.float32) * (b_beta * tl.exp(b_gc))[:, None]).to(tl.bfloat16)
    b_w = tl.dot(b_T, b_X)
    tl.store(w + (o_t[:, None] * H + i_h) * K + o_k[None, :], b_w.to(tl.bfloat16), mask=m_t[:, None])
    for iv in range(V // BV):
        o_v = iv * BV + tl.arange(0, BV)
        b_v = tl.load(v + (o_t[:, None] * H + i_h) * V + o_v[None, :], mask=m_t[:, None], other=0.)
        b_Y = (b_v.to(tl.float32) * b_beta[:, None]).to(tl.bfloat16)
        b_u = tl.dot(b_T, b_Y)
        tl.store(u + (o_t[:, None] * H + i_h) * V + o_v[None, :], b_u.to(tl.bfloat16), mask=m_t[:, None])


@triton.jit
def _fwd_h_kernel(q, k, w, u, g, cu, seqoff, o_ptr, h_ptr, vn_ptr,
                  H: tl.constexpr, HK: tl.constexpr, G: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
                  STORE_O: tl.constexpr, STORE_H: tl.constexpr):
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_h = tl.program_id(2)
    bos = tl.load(cu + i_b)
    eos = tl.load(cu + i_b + 1)
    nt = (eos - bos + BT - 1) // BT
    boff = tl.load(seqoff + i_b)
    i_hk = i_h // G
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    incl = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([K, BV], dtype=tl.float32)
    for c in range(nt):
        o_t = bos + c * BT + o_i
        m_t = o_t < eos
        if STORE_H:
            hb = h_ptr + ((boff + c) * H + i_h).to(tl.int64) * (K * V)
            tl.store(hb + o_k[:, None] * V + o_v[None, :], b_h.to(tl.bfloat16))
        b_w = tl.load(w + (o_t[:, None] * H + i_h) * K + o_k[None, :], mask=m_t[:, None], other=0.)
        b_u = tl.load(u + (o_t[:, None] * H + i_h) * V + o_v[None, :], mask=m_t[:, None], other=0.)
        b_k = tl.load(k + (o_t[:, None] * HK + i_hk) * K + o_k[None, :], mask=m_t[:, None], other=0.)
        b_g = tl.load(g + o_t * H + i_h, mask=m_t, other=0.)
        b_gc = tl.cumsum(b_g, 0)
        g_last = tl.sum(tl.where(o_i == BT - 1, b_gc, 0.), 0)
        b_hb = b_h.to(tl.bfloat16)
        b_vn = b_u.to(tl.float32) - tl.dot(b_w, b_hb)
        b_vnb = b_vn.to(tl.bfloat16)
        if STORE_H:
            tl.store(vn_ptr + (o_t[:, None] * H + i_h) * V + o_v[None, :], b_vnb, mask=m_t[:, None])
        if STORE_O:
            b_q = tl.load(q + (o_t[:, None] * HK + i_hk) * K + o_k[None, :], mask=m_t[:, None], other=0.)
            b_o = tl.dot(b_q, b_hb) * tl.exp(b_gc)[:, None]
            b_qk = tl.dot(b_q, tl.trans(b_k))
            b_qk = tl.where(incl, b_qk * tl.exp(b_gc[:, None] - b_gc[None, :]), 0.)
            b_o += tl.dot(b_qk.to(tl.bfloat16), b_vnb)
            tl.store(o_ptr + (o_t[:, None] * H + i_h) * V + o_v[None, :], b_o.to(tl.bfloat16),
                     mask=m_t[:, None])
        b_kd = (b_k.to(tl.float32) * tl.exp(g_last - b_gc)[:, None]).to(tl.bfloat16)
        b_h = b_h * tl.exp(g_last) + tl.dot(tl.trans(b_kd), b_vnb)


@triton.jit
def _bwd_dh_kernel(q, k, w, g, do, cu, seqoff, dh_ptr, dvn_ptr,
                   H: tl.constexpr, HK: tl.constexpr, G: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_h = tl.program_id(2)
    bos = tl.load(cu + i_b)
    eos = tl.load(cu + i_b + 1)
    nt = (eos - bos + BT - 1) // BT
    boff = tl.load(seqoff + i_b)
    i_hk = i_h // G
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_up = o_i[:, None] <= o_i[None, :]
    b_dh = tl.zeros([K, BV], dtype=tl.float32)
    for cc in range(nt):
        c = nt - 1 - cc
        o_t = bos + c * BT + o_i
        m_t = o_t < eos
        hb = dh_ptr + ((boff + c) * H + i_h).to(tl.int64) * (K * V)
        b_dhb = b_dh.to(tl.bfloat16)
        tl.store(hb + o_k[:, None] * V + o_v[None, :], b_dhb)
        b_q = tl.load(q + (o_t[:, None] * HK + i_hk) * K + o_k[None, :], mask=m_t[:, None], other=0.)
        b_k = tl.load(k + (o_t[:, None] * HK + i_hk) * K + o_k[None, :], mask=m_t[:, None], other=0.)
        b_w = tl.load(w + (o_t[:, None] * H + i_h) * K + o_k[None, :], mask=m_t[:, None], other=0.)
        b_do = tl.load(do + (o_t[:, None] * H + i_h) * V + o_v[None, :], mask=m_t[:, None], other=0.)
        b_g = tl.load(g + o_t * H + i_h, mask=m_t, other=0.)
        b_gc = tl.cumsum(b_g, 0)
        g_last = tl.sum(tl.where(o_i == BT - 1, b_gc, 0.), 0)
        b_kq = tl.dot(b_k, tl.trans(b_q))  # [j, i]
        b_kq = tl.where(m_up, b_kq * tl.exp(b_gc[None, :] - b_gc[:, None]), 0.)
        b_kd = (b_k.to(tl.float32) * tl.exp(g_last - b_gc)[:, None]).to(tl.bfloat16)
        b_dv = tl.dot(b_kq.to(tl.bfloat16), b_do) + tl.dot(b_kd, b_dhb)
        b_dvb = b_dv.to(tl.bfloat16)
        tl.store(dvn_ptr + (o_t[:, None] * H + i_h) * V + o_v[None, :], b_dvb, mask=m_t[:, None])
        b_qd = (b_q.to(tl.float32) * tl.exp(b_gc)[:, None]).to(tl.bfloat16)
        b_dh = b_dh * tl.exp(g_last) + tl.dot(tl.trans(b_qd), b_do) - tl.dot(tl.trans(b_w), b_dvb)


@triton.jit
def _bwd_dqk_kernel(q, k, g, do, cu, info, h_ptr, dh_ptr, vn_ptr, dvn_ptr, dw_ptr, dq, dk, dgp, T,
                    H: tl.constexpr, HK: tl.constexpr, G: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr):
    i_c = tl.program_id(0)
    i_h = tl.program_id(1)
    i_kb = tl.program_id(2)
    b = tl.load(info + 2 * i_c)
    if b < 0:
        return
    c = tl.load(info + 2 * i_c + 1)
    bos = tl.load(cu + b)
    eos = tl.load(cu + b + 1)
    o_i = tl.arange(0, BT)
    o_t = bos + c * BT + o_i
    m_t = o_t < eos
    i_hk = i_h // G
    o_k = i_kb * BK + tl.arange(0, BK)
    b_g = tl.load(g + o_t * H + i_h, mask=m_t, other=0.)
    b_gc = tl.cumsum(b_g, 0)
    g_last = tl.sum(tl.where(o_i == BT - 1, b_gc, 0.), 0)

    b_dq1 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_D = tl.zeros([BT, BT], dtype=tl.float32)
    b_sdh = tl.zeros([BK], dtype=tl.float32)
    hb = h_ptr + (i_c * H + i_h).to(tl.int64) * (K * V)
    dhb = dh_ptr + (i_c * H + i_h).to(tl.int64) * (K * V)
    for iv in range(V // BV):
        o_v = iv * BV + tl.arange(0, BV)
        vo = (o_t[:, None] * H + i_h) * V + o_v[None, :]
        b_hh = tl.load(hb + o_k[:, None] * V + o_v[None, :])
        b_dh = tl.load(dhb + o_k[:, None] * V + o_v[None, :])
        b_vn = tl.load(vn_ptr + vo, mask=m_t[:, None], other=0.)
        b_dvn = tl.load(dvn_ptr + vo, mask=m_t[:, None], other=0.)
        b_do = tl.load(do + vo, mask=m_t[:, None], other=0.)
        b_dq1 += tl.dot(b_do, tl.trans(b_hh))
        b_dk2 += tl.dot(b_vn, tl.trans(b_dh))
        b_dw += tl.dot(b_dvn, tl.trans(b_hh))
        b_D += tl.dot(b_do, tl.trans(b_vn))
        b_sdh += tl.sum(b_hh.to(tl.float32) * b_dh.to(tl.float32), 1)
    tl.store(dw_ptr + (o_t[:, None] * H + i_h) * K + o_k[None, :], (-b_dw).to(tl.bfloat16),
             mask=m_t[:, None])

    qk_off = (o_t[:, None] * HK + i_hk) * K + o_k[None, :]
    b_q = tl.load(q + qk_off, mask=m_t[:, None], other=0.)
    b_k = tl.load(k + qk_off, mask=m_t[:, None], other=0.)
    egc = tl.exp(b_gc)
    incl = o_i[:, None] >= o_i[None, :]
    b_Dm = tl.where(incl, b_D * tl.exp(b_gc[:, None] - b_gc[None, :]), 0.)
    b_Dmb = b_Dm.to(tl.bfloat16)
    b_dqv = b_dq1 * egc[:, None] + tl.dot(b_Dmb, b_k)
    b_dgam = tl.sum(b_dqv * b_q.to(tl.float32), 1)
    tl.atomic_add(dq + qk_off, b_dqv, mask=m_t[:, None], sem="relaxed")
    b_kf = b_k.to(tl.float32)
    b_dk2d = b_dk2 * tl.exp(g_last - b_gc)[:, None]
    b_dkv = b_dk2d + tl.dot(tl.trans(b_Dmb), b_q)
    b_dgam -= tl.sum(b_dkv * b_kf, 1)
    last = tl.exp(g_last) * tl.sum(b_sdh, 0) + tl.sum(tl.sum(b_dk2d * b_kf, 1), 0)
    b_dgam += tl.where(o_i == BT - 1, last, 0.)
    tl.atomic_add(dk + qk_off, b_dkv, mask=m_t[:, None], sem="relaxed")
    tl.store(dgp + (i_kb * T + o_t) * H + i_h, b_dgam, mask=m_t)


@triton.jit
def _bwd_wy_kernel(k, v, g, beta, cu, info, Abuf, dw_ptr, dvn_ptr, dgp, dk, dv, dg, dbeta, T,
                   H: tl.constexpr, HK: tl.constexpr, G: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
                   BV: tl.constexpr):
    i_c = tl.program_id(0)
    i_h = tl.program_id(1)
    b = tl.load(info + 2 * i_c)
    if b < 0:
        return
    c = tl.load(info + 2 * i_c + 1)
    bos = tl.load(cu + b)
    eos = tl.load(cu + b + 1)
    o_i = tl.arange(0, BT)
    o_t = bos + c * BT + o_i
    m_t = o_t < eos
    i_hk = i_h // G
    b_g = tl.load(g + o_t * H + i_h, mask=m_t, other=0.)
    b_gc = tl.cumsum(b_g, 0)
    egc = tl.exp(b_gc)
    b_beta = tl.load(beta + o_t * H + i_h, mask=m_t, other=0.).to(tl.float32)
    Ab = Abuf + (i_c * H + i_h).to(tl.int64) * (BT * BT)
    b_T = tl.load(Ab + o_i[:, None] * BT + o_i[None, :])
    b_TTb = tl.trans(b_T.to(tl.bfloat16))

    b_dT = tl.zeros([BT, BT], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dgam = tl.zeros([BT], dtype=tl.float32)
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    for iv in range(V // BV):
        o_v = iv * BV + tl.arange(0, BV)
        vo = (o_t[:, None] * H + i_h) * V + o_v[None, :]
        b_dvn = tl.load(dvn_ptr + vo, mask=m_t[:, None], other=0.)
        b_v = tl.load(v + vo, mask=m_t[:, None], other=0.).to(tl.float32)
        b_Y = (b_v * b_beta[:, None]).to(tl.bfloat16)
        b_dT += tl.dot(b_dvn, tl.trans(b_Y))
        b_dY = tl.dot(b_TTb, b_dvn)
        tl.store(dv + vo, (b_dY * b_beta[:, None]).to(tl.bfloat16), mask=m_t[:, None])
        b_dbeta += tl.sum(b_dY * b_v, 1)
    bg = b_beta * egc
    for ik in range(K // BK):
        o_k = ik * BK + tl.arange(0, BK)
        qk_off = (o_t[:, None] * HK + i_hk) * K + o_k[None, :]
        b_k = tl.load(k + qk_off, mask=m_t[:, None], other=0.)
        b_dw = tl.load(dw_ptr + (o_t[:, None] * H + i_h) * K + o_k[None, :], mask=m_t[:, None], other=0.)
        b_kk += tl.dot(b_k, tl.trans(b_k))
        b_kf = b_k.to(tl.float32)
        b_X = b_kf * bg[:, None]
        b_dX = tl.dot(b_TTb, b_dw)
        b_dgam += tl.sum(b_dX * b_X, 1)
        b_dbeta += tl.sum(b_dX * b_kf, 1) * egc
        b_dT += tl.dot(b_dw, tl.trans(b_X.to(tl.bfloat16)))
        tl.atomic_add(dk + qk_off, b_dX * bg[:, None], mask=m_t[:, None], sem="relaxed")
    b_dA = -tl.dot(tl.trans(b_T), tl.dot(b_dT, tl.trans(b_T)))
    strict = o_i[:, None] > o_i[None, :]
    b_E = tl.where(strict, b_dA * tl.exp(b_gc[:, None] - b_gc[None, :]), 0.)
    b_dbeta += tl.sum(b_E * b_kk, 1)
    b_Eb = b_E * b_beta[:, None]
    b_Ebb = b_Eb.to(tl.bfloat16)
    b_Q = b_Eb * b_kk
    b_dgam += tl.sum(b_Q, 1) - tl.sum(b_Q, 0)
    for ik in range(K // BK):
        o_k = ik * BK + tl.arange(0, BK)
        qk_off = (o_t[:, None] * HK + i_hk) * K + o_k[None, :]
        b_k = tl.load(k + qk_off, mask=m_t[:, None], other=0.)
        b_dkE = tl.dot(b_Ebb, b_k) + tl.dot(tl.trans(b_Ebb), b_k)
        tl.atomic_add(dk + qk_off, b_dkE, mask=m_t[:, None], sem="relaxed")
    for ik in range(K // BK):
        b_dgam += tl.load(dgp + (ik * T + o_t) * H + i_h, mask=m_t, other=0.)
    b_dg = tl.sum(tl.where(o_i[None, :] >= o_i[:, None], b_dgam[None, :], 0.), 1)
    tl.store(dg + o_t * H + i_h, b_dg, mask=m_t)
    tl.store(dbeta + o_t * H + i_h, b_dbeta.to(tl.bfloat16), mask=m_t)


def _pick_bv(B, H, V):
    if (V // 32) * B * H >= 108:
        return 32
    return 16


def _table(cu, B, NTM):
    info = torch.empty((NTM, 2), device=cu.device, dtype=torch.int32)
    seqoff = torch.empty((B,), device=cu.device, dtype=torch.int32)
    _table_kernel[(B,)](cu, info, seqoff, B, NTM, BLOCK=256, BB=max(16, triton.next_power_of_2(B)), num_warps=4)
    return info, seqoff


def _prep(k, v, g, beta, cu, info, NTM, H, HK, G, K, V):
    T = k.shape[0]
    w = torch.empty((T, H, K), device=k.device, dtype=torch.bfloat16)
    u = torch.empty((T, H, V), device=k.device, dtype=torch.bfloat16)
    Abuf = torch.empty((NTM, H, BT, BT), device=k.device, dtype=torch.float32)
    _prep_kernel[(NTM, H)](k, v, g, beta, cu, info, w, u, Abuf,
                           H=H, HK=HK, G=G, K=K, V=V, BT=BT, BV=64, num_warps=4, num_stages=1)
    return w, u, Abuf


class _ChunkGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu):
        T, HK, K = q.shape
        H, V = v.shape[1], v.shape[2]
        G = H // HK
        B = cu.numel() - 1
        NTM = T // BT + B
        _tick('start')
        info, seqoff = _table(cu, B, NTM)
        _tick('table')
        w, u, Abuf = _prep(k, v, g, beta, cu, info, NTM, H, HK, G, K, V)
        _tick('prep_f')
        del Abuf
        o = torch.empty((T, H, V), device=q.device, dtype=torch.bfloat16)
        BV = _pick_bv(B, H, V)
        _fwd_h_kernel[(V // BV, B, H)](q, k, w, u, g, cu, seqoff, o, o, o,
                                       H=H, HK=HK, G=G, K=K, V=V, BT=BT, BV=BV,
                                       STORE_O=True, STORE_H=False, num_warps=8, num_stages=2)
        _tick('fwd_h_o')
        ctx.save_for_backward(q, k, v, g, beta, cu)
        ctx.tab = (info, seqoff)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, cu = ctx.saved_tensors
        info, seqoff = ctx.tab
        do = do.contiguous()
        T, HK, K = q.shape
        H, V = v.shape[1], v.shape[2]
        G = H // HK
        B = cu.numel() - 1
        NTM = T // BT + B
        w, u, Abuf = _prep(k, v, g, beta, cu, info, NTM, H, HK, G, K, V)
        _tick('prep_b')
        BV = _pick_bv(B, H, V)
        h = torch.empty((NTM, H, K, V), device=q.device, dtype=torch.bfloat16)
        vn = torch.empty((T, H, V), device=q.device, dtype=torch.bfloat16)
        _fwd_h_kernel[(V // BV, B, H)](q, k, w, u, g, cu, seqoff, vn, h, vn,
                                       H=H, HK=HK, G=G, K=K, V=V, BT=BT, BV=BV,
                                       STORE_O=False, STORE_H=True, num_warps=4, num_stages=2)
        _tick('fwd_h_store')
        del u
        dh = torch.empty((NTM, H, K, V), device=q.device, dtype=torch.bfloat16)
        dvn = torch.empty((T, H, V), device=q.device, dtype=torch.bfloat16)
        _bwd_dh_kernel[(V // BV, B, H)](q, k, w, g, do, cu, seqoff, dh, dvn,
                                        H=H, HK=HK, G=G, K=K, V=V, BT=BT, BV=BV,
                                        num_warps=8, num_stages=2)
        _tick('dh')
        _tick('dh')
        dqk = torch.zeros((2, T, HK, K), device=q.device, dtype=torch.float32)
        dgp = torch.empty((K // 64, T, H), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        dbeta = torch.empty_like(beta)
        dw = w
        _bwd_dqk_kernel[(NTM, H, K // 64)](q, k, g, do, cu, info, h, dh, vn, dvn, dw, dqk[0], dqk[1], dgp, T,
                                           H=H, HK=HK, G=G, K=K, V=V, BT=BT, BK=64, BV=64,
                                           num_warps=4, num_stages=1)
        _tick('dqk')
        del h, dh, vn
        _bwd_wy_kernel[(NTM, H)](k, v, g, beta, cu, info, Abuf, dw, dvn, dgp, dqk[1], dv, dg, dbeta, T,
                                 H=H, HK=HK, G=G, K=K, V=V, BT=BT, BK=64, BV=64,
                                 num_warps=4, num_stages=1)
        _tick('wy')
        dqk = dqk.to(torch.bfloat16)
        _tick('cast')
        return dqk[0], dqk[1], dv, dg, dbeta, None


def _profile(q, k, v, g, beta, cu):
    global _EV
    rep = []
    args = [t.detach().clone().requires_grad_(t.dtype != torch.int32) for t in (q, k, v, g, beta)]
    go = torch.randn(v.shape, device=v.device).bfloat16()
    for it in range(4):
        _EV = []
        torch.cuda.synchronize()
        o = _ChunkGDN.apply(*args, cu)
        o.backward(go)
        torch.cuda.synchronize()
        ev = _EV
        _EV = None
    for i in range(1, len(ev)):
        rep.append("%s:g%.3f/c%.3f" % (ev[i][0], ev[i - 1][1].elapsed_time(ev[i][1]), (ev[i][2] - ev[i - 1][2]) * 1e3))
    torch.cuda.synchronize()
    for nm, e0, e1 in _VAR[-8:]:
        rep.append("%s:%.3f" % (nm, e0.elapsed_time(e1) if e0 is not None else -1))
    t0 = time.perf_counter()
    for it in range(5):
        o = _ChunkGDN.apply(*args, cu)
        o.backward(go)
    torch.cuda.synchronize()
    rep.append("total %.3f ms" % ((time.perf_counter() - t0) / 5 * 1e3))
    t0 = time.perf_counter()
    for it in range(5):
        o = _ChunkGDN.apply(*args, cu)
        o.backward(go)
    rep.append("cpu-only %.3f ms" % ((time.perf_counter() - t0) / 5 * 1e3))
    torch.cuda.synchronize()
    raise RuntimeError(" ".join(rep))


def chunk_gated_deltanet(q, k, v, g, beta, cu_seqlens):
    if PROFILE:
        _profile(q, k, v, g, beta, cu_seqlens)
    return _ChunkGDN.apply(q, k, v, g, beta, cu_seqlens)
