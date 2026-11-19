import torch
import triton
import triton.language as tl

BT = 64
F32 = tl.float32
BF16 = tl.bfloat16


# ----------------------------------------------------------------------------- helpers
@triton.jit
def _chunk_info(cs, cl, cu):
    i_c = tl.program_id(0)
    i_n = tl.load(cs + i_c)
    bos = tl.load(cu + tl.maximum(i_n, 0))
    eos = tl.load(cu + tl.maximum(i_n, 0) + 1)
    t0 = bos + tl.load(cl + i_c) * 64
    return i_c, i_n, t0, eos


@triton.jit
def _inv16(L):
    r = tl.arange(0, 16)
    Y = tl.zeros([16, 16], dtype=tl.float32)
    for i in tl.static_range(1, 16):
        a = -tl.sum(tl.where(r[:, None] == i, L, 0.), 0)
        a = a + tl.sum(a[:, None] * Y, 0)
        Y = tl.where(r[:, None] == i, a[None, :], Y)
    return Y + (r[:, None] == r[None, :]).to(tl.float32)


@triton.jit
def _ld16(k, beta, gc, t, eos, i_h, i_hk, H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr):
    m = t < eos
    ok = tl.arange(0, K)
    kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    b = tl.load(beta + t * H + i_h, mask=m, other=0.).to(tl.float32)
    g = tl.load(gc + t * H + i_h, mask=m, other=0.)
    return kk, b, g, m


@triton.jit
def _lblk(ka, kb, ba, ga, gb, ma, mb, DIAG: tl.constexpr):
    r = tl.arange(0, 16)
    s = tl.dot(ka, tl.trans(kb))
    msk = ma[:, None] & mb[None, :]
    if DIAG:
        msk = msk & (r[:, None] > r[None, :])
    return tl.where(msk, s * ba[:, None] * tl.exp(tl.where(msk, ga[:, None] - gb[None, :], 0.)), 0.)


@triton.jit
def _st16(Ai, X, t, m, i_h, col: tl.constexpr, H: tl.constexpr):
    r = tl.arange(0, 16)
    tl.store(Ai + (t[:, None] * H + i_h) * 64 + col * 16 + r[None, :], X.to(tl.bfloat16), mask=m[:, None])


# ----------------------------------------------------------------------------- forward kernels
@triton.jit
def _gcum_kernel(g, gc, cs, cl, cu, H: tl.constexpr, BH: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    t = t0 + tl.arange(0, 64)
    oh = tl.arange(0, BH)
    m = (t < eos)[:, None] & (oh < H)[None, :]
    p = t[:, None] * H + oh[None, :]
    x = tl.load(g + p, mask=m, other=0.)
    tl.store(gc + p, tl.cumsum(x, 0), mask=m)


@triton.jit
def _kkt_solve_kernel(k, beta, gc, Ai, cs, cl, cu,
                      H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    i_h = tl.program_id(1)
    i_hk = i_h // (H // HK)
    r = tl.arange(0, 16)
    t_0 = t0 + r
    t_1 = t0 + 16 + r
    t_2 = t0 + 32 + r
    t_3 = t0 + 48 + r
    k0, b0, g0, m0 = _ld16(k, beta, gc, t_0, eos, i_h, i_hk, H, HK, K)
    k1, b1, g1, m1 = _ld16(k, beta, gc, t_1, eos, i_h, i_hk, H, HK, K)
    k2, b2, g2, m2 = _ld16(k, beta, gc, t_2, eos, i_h, i_hk, H, HK, K)
    k3, b3, g3, m3 = _ld16(k, beta, gc, t_3, eos, i_h, i_hk, H, HK, K)
    X00 = _inv16(_lblk(k0, k0, b0, g0, g0, m0, m0, True))
    X11 = _inv16(_lblk(k1, k1, b1, g1, g1, m1, m1, True))
    X22 = _inv16(_lblk(k2, k2, b2, g2, g2, m2, m2, True))
    X33 = _inv16(_lblk(k3, k3, b3, g3, g3, m3, m3, True))
    L10 = _lblk(k1, k0, b1, g1, g0, m1, m0, False)
    L20 = _lblk(k2, k0, b2, g2, g0, m2, m0, False)
    L21 = _lblk(k2, k1, b2, g2, g1, m2, m1, False)
    L30 = _lblk(k3, k0, b3, g3, g0, m3, m0, False)
    L31 = _lblk(k3, k1, b3, g3, g1, m3, m1, False)
    L32 = _lblk(k3, k2, b3, g3, g2, m3, m2, False)
    X10 = -tl.dot(X11, tl.dot(L10, X00))
    X21 = -tl.dot(X22, tl.dot(L21, X11))
    X32 = -tl.dot(X33, tl.dot(L32, X22))
    X20 = -tl.dot(X22, tl.dot(L20, X00) + tl.dot(L21, X10))
    X31 = -tl.dot(X33, tl.dot(L31, X11) + tl.dot(L32, X21))
    X30 = -tl.dot(X33, tl.dot(L30, X00) + tl.dot(L31, X10) + tl.dot(L32, X20))
    Z = tl.zeros([16, 16], dtype=tl.float32)
    _st16(Ai, X00, t_0, m0, i_h, 0, H)
    _st16(Ai, Z, t_0, m0, i_h, 1, H)
    _st16(Ai, Z, t_0, m0, i_h, 2, H)
    _st16(Ai, Z, t_0, m0, i_h, 3, H)
    _st16(Ai, X10, t_1, m1, i_h, 0, H)
    _st16(Ai, X11, t_1, m1, i_h, 1, H)
    _st16(Ai, Z, t_1, m1, i_h, 2, H)
    _st16(Ai, Z, t_1, m1, i_h, 3, H)
    _st16(Ai, X20, t_2, m2, i_h, 0, H)
    _st16(Ai, X21, t_2, m2, i_h, 1, H)
    _st16(Ai, X22, t_2, m2, i_h, 2, H)
    _st16(Ai, Z, t_2, m2, i_h, 3, H)
    _st16(Ai, X30, t_3, m3, i_h, 0, H)
    _st16(Ai, X31, t_3, m3, i_h, 1, H)
    _st16(Ai, X32, t_3, m3, i_h, 2, H)
    _st16(Ai, X33, t_3, m3, i_h, 3, H)


@triton.jit
def _wu_kernel(k, v, beta, gc, Ai, w, u, cs, cl, cu,
               H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    i_h = tl.program_id(1)
    i_hk = i_h // (H // HK)
    o64 = tl.arange(0, 64)
    ok = tl.arange(0, K)
    t = t0 + o64
    m = t < eos
    A = tl.load(Ai + (t[:, None] * H + i_h) * 64 + o64[None, :], mask=m[:, None], other=0.)
    b = tl.load(beta + t * H + i_h, mask=m, other=0.).to(F32)
    gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
    kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    kb = (kk.to(F32) * (b * tl.exp(gg))[:, None]).to(BF16)
    ww = tl.dot(A, kb)
    tl.store(w + (t[:, None] * H + i_h) * K + ok[None, :], ww.to(BF16), mask=m[:, None])
    for iv in tl.static_range(V // BV):
        ov = iv * BV + tl.arange(0, BV)
        vv = tl.load(v + (t[:, None] * H + i_h) * V + ov[None, :], mask=m[:, None], other=0.)
        uu = tl.dot(A, (vv.to(F32) * b[:, None]).to(BF16))
        tl.store(u + (t[:, None] * H + i_h) * V + ov[None, :], uu.to(BF16), mask=m[:, None])


@triton.jit
def _h_fwd_kernel(k, w, u, gc, h, vn, cu, coff,
                  H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H
    i_hk = i_h // (H // HK)
    bos = tl.load(cu + i_n)
    eos = tl.load(cu + i_n + 1)
    boff = tl.load(coff + i_n)
    NTs = tl.cdiv(eos - bos, 64)
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    o64 = tl.arange(0, 64)
    bh = tl.zeros([K, BV], dtype=F32)
    for i_t in range(NTs):
        tl.store(h + ((boff + i_t) * H + i_h).to(tl.int64) * (K * V) + ok[:, None] * V + ov[None, :], bh.to(BF16))
        t = bos + i_t * 64 + o64
        m = t < eos
        ww = tl.load(w + (t[:, None] * H + i_h) * K + ok[None, :], mask=m[:, None], other=0.)
        uu = tl.load(u + (t[:, None] * H + i_h) * V + ov[None, :], mask=m[:, None], other=0.)
        vnew = uu.to(F32) - tl.dot(ww, bh.to(BF16))
        tl.store(vn + (t[:, None] * H + i_h) * V + ov[None, :], vnew.to(BF16), mask=m[:, None])
        tlast = tl.minimum(bos + i_t * 64 + 64, eos) - 1
        gl = tl.load(gc + tlast * H + i_h)
        gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
        kT = tl.load(k + (t[None, :] * HK + i_hk) * K + ok[:, None], mask=m[None, :], other=0.)
        vs = (vnew * tl.exp(gl - gg)[:, None]).to(BF16)
        bh = bh * tl.exp(gl) + tl.dot(kT, vs)


@triton.jit
def _o_kernel(q, k, vn, gc, h, o, cs, cl, cu,
              H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    i_h = tl.program_id(1)
    i_v = tl.program_id(2)
    i_hk = i_h // (H // HK)
    o64 = tl.arange(0, 64)
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    t = t0 + o64
    m = t < eos
    qq = tl.load(q + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
    hh = tl.load(h + (i_c * H + i_h).to(tl.int64) * (K * V) + ok[:, None] * V + ov[None, :])
    oo = tl.dot(qq, hh) * tl.exp(gg)[:, None]
    s = tl.dot(qq, tl.trans(kk))
    msk = (o64[:, None] >= o64[None, :]) & m[:, None] & m[None, :]
    s = tl.where(msk, s * tl.exp(tl.where(msk, gg[:, None] - gg[None, :], 0.)), 0.)
    vv = tl.load(vn + (t[:, None] * H + i_h) * V + ov[None, :], mask=m[:, None], other=0.)
    oo += tl.dot(s.to(BF16), vv)
    tl.store(o + (t[:, None] * H + i_h) * V + ov[None, :], oo.to(BF16), mask=m[:, None])


# ----------------------------------------------------------------------------- backward kernels
@triton.jit
def _dvl_kernel(q, k, do, gc, dvl, cs, cl, cu,
                H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    i_h = tl.program_id(1)
    i_v = tl.program_id(2)
    i_hk = i_h // (H // HK)
    o64 = tl.arange(0, 64)
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    t = t0 + o64
    m = t < eos
    qq = tl.load(q + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
    s = tl.dot(kk, tl.trans(qq))
    msk = (o64[:, None] <= o64[None, :]) & m[:, None] & m[None, :]
    s = tl.where(msk, s * tl.exp(tl.where(msk, gg[None, :] - gg[:, None], 0.)), 0.)
    dd = tl.load(do + (t[:, None] * H + i_h) * V + ov[None, :], mask=m[:, None], other=0.)
    dv = tl.dot(s.to(BF16), dd)
    tl.store(dvl + (t[:, None] * H + i_h) * V + ov[None, :], dv, mask=m[:, None])


@triton.jit
def _dh_kernel(q, k, w, do, dvl, gc, dh, dv2, cu, coff,
               H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H
    i_hk = i_h // (H // HK)
    bos = tl.load(cu + i_n)
    eos = tl.load(cu + i_n + 1)
    boff = tl.load(coff + i_n)
    NTs = tl.cdiv(eos - bos, 64)
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    o64 = tl.arange(0, 64)
    bdh = tl.zeros([K, BV], dtype=F32)
    for ii in range(NTs):
        i_t = NTs - 1 - ii
        tl.store(dh + ((boff + i_t) * H + i_h).to(tl.int64) * (K * V) + ok[:, None] * V + ov[None, :], bdh.to(BF16))
        t = bos + i_t * 64 + o64
        m = t < eos
        tlast = tl.minimum(bos + i_t * 64 + 64, eos) - 1
        gl = tl.load(gc + tlast * H + i_h)
        gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
        kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
        kbar = (kk.to(F32) * tl.exp(gl - gg)[:, None]).to(BF16)
        dv = tl.load(dvl + (t[:, None] * H + i_h) * V + ov[None, :], mask=m[:, None], other=0.)
        dv = dv + tl.dot(kbar, bdh.to(BF16))
        dvb = dv.to(BF16)
        tl.store(dv2 + (t[:, None] * H + i_h) * V + ov[None, :], dvb, mask=m[:, None])
        qT = tl.load(q + (t[None, :] * HK + i_hk) * K + ok[:, None], mask=m[None, :], other=0.)
        qT = (qT.to(F32) * tl.exp(gg)[None, :]).to(BF16)
        wT = tl.load(w + (t[None, :] * H + i_h) * K + ok[:, None], mask=m[None, :], other=0.)
        dd = tl.load(do + (t[:, None] * H + i_h) * V + ov[None, :], mask=m[:, None], other=0.)
        bdh = bdh * tl.exp(gl) + tl.dot(qT, dd) - tl.dot(wT, dvb)


@triton.jit
def _dqkw_kernel(q, k, vn, do, dv2, gc, h, dh, dw, dqb, dkb, dG, cs, cl, cu,
                 H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    i_h = tl.program_id(1)
    i_hk = i_h // (H // HK)
    o64 = tl.arange(0, 64)
    ok = tl.arange(0, K)
    t = t0 + o64
    m = t < eos
    dq1 = tl.zeros([64, K], dtype=F32)
    dk2 = tl.zeros([64, K], dtype=F32)
    dwa = tl.zeros([64, K], dtype=F32)
    dA = tl.zeros([64, 64], dtype=F32)
    dglv = tl.zeros([K], dtype=F32)
    hoff = (i_c * H + i_h).to(tl.int64) * (K * V)
    for iv in tl.static_range(V // BV):
        ov = iv * BV + tl.arange(0, BV)
        pv = (t[:, None] * H + i_h) * V + ov[None, :]
        dd = tl.load(do + pv, mask=m[:, None], other=0.)
        vv = tl.load(vn + pv, mask=m[:, None], other=0.)
        d2 = tl.load(dv2 + pv, mask=m[:, None], other=0.)
        hT = tl.load(h + hoff + ok[None, :] * V + ov[:, None])
        dhT = tl.load(dh + hoff + ok[None, :] * V + ov[:, None])
        dq1 += tl.dot(dd, hT)
        dk2 += tl.dot(vv, dhT)
        dwa += tl.dot(d2, hT)
        dA += tl.dot(dd, tl.trans(vv))
        dglv += tl.sum(hT.to(F32) * dhT.to(F32), 0)
    pk = (t[:, None] * H + i_h) * K + ok[None, :]
    tl.store(dw + pk, -dwa, mask=m[:, None])
    qq = tl.load(q + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
    last = tl.minimum(eos - t0, 64) - 1
    gl = tl.load(gc + (t0 + last) * H + i_h)
    dq = dq1 * tl.exp(gg)[:, None]
    dGv = tl.sum(dq * qq.to(F32), 1)
    msk = (o64[:, None] >= o64[None, :]) & m[:, None] & m[None, :]
    s = tl.dot(qq, tl.trans(kk))
    Gm = tl.exp(tl.where(msk, gg[:, None] - gg[None, :], 0.))
    dAg = tl.where(msk, dA * Gm, 0.)
    P = dAg * s
    dGv += tl.sum(P, 1) - tl.sum(P, 0)
    dAgb = dAg.to(BF16)
    dq += tl.dot(dAgb, kk)
    dk = tl.dot(tl.trans(dAgb), qq)
    dk2 = dk2 * tl.exp(gl - gg)[:, None]
    dk += dk2
    t2 = tl.sum(dk2 * kk.to(F32), 1)
    dGv -= t2
    dGv += tl.where(o64 == last, tl.sum(t2) + tl.sum(dglv) * tl.exp(gl), 0.)
    tl.store(dqb + pk, dq, mask=m[:, None])
    tl.store(dkb + pk, dk, mask=m[:, None])
    tl.store(dG + t * H + i_h, dGv, mask=m)


@triton.jit
def _wy_bwd_kernel(k, v, beta, gc, Ai, dw, dv2, dkb, dG, dv, db, dg, cs, cl, cu,
                   H: tl.constexpr, HK: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_c, i_n, t0, eos = _chunk_info(cs, cl, cu)
    if i_n < 0:
        return
    i_h = tl.program_id(1)
    i_hk = i_h // (H // HK)
    o64 = tl.arange(0, 64)
    ok = tl.arange(0, K)
    t = t0 + o64
    m = t < eos
    A = tl.load(Ai + (t[:, None] * H + i_h) * 64 + o64[None, :], mask=m[:, None], other=0.)
    Af = A.to(F32)
    bb = tl.load(beta + t * H + i_h, mask=m, other=0.).to(F32)
    gg = tl.load(gc + t * H + i_h, mask=m, other=0.)
    eg = tl.exp(gg)
    kk = tl.load(k + (t[:, None] * HK + i_hk) * K + ok[None, :], mask=m[:, None], other=0.)
    kf = kk.to(F32)
    kb = (kf * (bb * eg)[:, None]).to(BF16)
    pk = (t[:, None] * H + i_h) * K + ok[None, :]
    dwb = tl.load(dw + pk, mask=m[:, None], other=0.).to(BF16)
    dAi = tl.dot(dwb, tl.trans(kb))
    X = tl.dot(tl.trans(A), dwb)
    dk = X * (bb * eg)[:, None]
    kx = tl.sum(kf * X, 1)
    dbv = eg * kx
    dGv = bb * eg * kx
    for iv in tl.static_range(V // BV):
        ov = iv * BV + tl.arange(0, BV)
        pv = (t[:, None] * H + i_h) * V + ov[None, :]
        du = tl.load(dv2 + pv, mask=m[:, None], other=0.)
        vv = tl.load(v + pv, mask=m[:, None], other=0.).to(F32)
        vb = (vv * bb[:, None]).to(BF16)
        dAi += tl.dot(du, tl.trans(vb))
        Y = tl.dot(tl.trans(A), du)
        tl.store(dv + pv, (Y * bb[:, None]).to(BF16), mask=m[:, None])
        dbv += tl.sum(vv * Y, 1)
    tmp = tl.dot(dAi, tl.trans(Af))
    dL = -tl.dot(tl.trans(Af), tmp)
    msk = (o64[:, None] > o64[None, :]) & m[:, None] & m[None, :]
    Gm = tl.exp(tl.where(msk, gg[:, None] - gg[None, :], 0.))
    kkT = tl.dot(kk, tl.trans(kk))
    dLG = tl.where(msk, dL * Gm, 0.)
    dbv += tl.sum(dLG * kkT, 1)
    Dk = dLG * bb[:, None]
    P2 = Dk * kkT
    dGv += tl.sum(P2, 1) - tl.sum(P2, 0)
    Dkb = Dk.to(BF16)
    dk += tl.dot(Dkb, kk) + tl.dot(tl.trans(Dkb), kk)
    dk += tl.load(dkb + pk, mask=m[:, None], other=0.)
    dGv += tl.load(dG + t * H + i_h, mask=m, other=0.)
    dGv = tl.where(m, dGv, 0.)
    dgv = tl.sum(tl.where(o64[:, None] <= o64[None, :], dGv[None, :], 0.), 1)
    tl.store(dkb + pk, dk, mask=m[:, None])
    tl.store(db + t * H + i_h, dbv, mask=m)
    tl.store(dg + t * H + i_h, dgv, mask=m)


# ----------------------------------------------------------------------------- python glue
def _meta(cu, T, B):
    dev = cu.device
    NTm = T // BT + B
    lens = cu[1:] - cu[:-1]
    nc = (lens + (BT - 1)) // BT
    coff = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    coff[1:] = torch.cumsum(nc, 0)
    idx = torch.arange(NTm, dtype=torch.int32, device=dev)
    seq = torch.searchsorted(coff[1:].contiguous(), idx, right=True, out_int32=True)
    valid = seq < B
    seqc = seq.clamp(max=B - 1)
    loc = (idx - coff.index_select(0, seqc)).to(torch.int32)
    cs = torch.where(valid, seqc, torch.full_like(seqc, -1)).to(torch.int32)
    return NTm, cs.contiguous(), loc.contiguous(), coff


def _rec_bv(B, H, V):
    return 64 if B * H * (V // 64) >= 216 else 32


def _fwd_prep(k, v, g, beta, cu, NTm, cs, cl):
    T, HK, K = k.shape
    H, V = v.shape[1], v.shape[2]
    gc = torch.empty_like(g)
    _gcum_kernel[(NTm,)](g, gc, cs, cl, cu, H=H, BH=triton.next_power_of_2(H), num_warps=4)
    Ai = torch.empty((T, H, BT), dtype=torch.bfloat16, device=k.device)
    _kkt_solve_kernel[(NTm, H)](k, beta, gc, Ai, cs, cl, cu, H=H, HK=HK, K=K, num_warps=4)
    return gc, Ai


def _fwd_wuh(k, v, beta, gc, Ai, cu, coff, NTm, cs, cl, B):
    T, HK, K = k.shape
    H, V = v.shape[1], v.shape[2]
    dev = k.device
    w = torch.empty((T, H, K), dtype=torch.bfloat16, device=dev)
    u = torch.empty((T, H, V), dtype=torch.bfloat16, device=dev)
    _wu_kernel[(NTm, H)](k, v, beta, gc, Ai, w, u, cs, cl, cu, H=H, HK=HK, K=K, V=V, BV=64, num_warps=4)
    h = torch.empty((NTm, H, K, V), dtype=torch.bfloat16, device=dev)
    vn = torch.empty((T, H, V), dtype=torch.bfloat16, device=dev)
    BV = _rec_bv(B, H, V)
    _h_fwd_kernel[(V // BV, B * H)](k, w, u, gc, h, vn, cu, coff, H=H, HK=HK, K=K, V=V, BV=BV, num_warps=4)
    return w, h, vn


class _GDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu):
        T, HK, K = q.shape
        H, V = v.shape[1], v.shape[2]
        B = cu.numel() - 1
        NTm, cs, cl, coff = _meta(cu, T, B)
        gc, Ai = _fwd_prep(k, v, g, beta, cu, NTm, cs, cl)
        w, h, vn = _fwd_wuh(k, v, beta, gc, Ai, cu, coff, NTm, cs, cl, B)
        del w
        o = torch.empty((T, H, V), dtype=torch.bfloat16, device=q.device)
        _o_kernel[(NTm, H, V // 64)](q, k, vn, gc, h, o, cs, cl, cu, H=H, HK=HK, K=K, V=V, BV=64, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, cu, gc, Ai, cs, cl, coff)
        ctx.NTm = NTm
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, cu, gc, Ai, cs, cl, coff = ctx.saved_tensors
        NTm = ctx.NTm
        do = do.contiguous()
        T, HK, K = q.shape
        H, V = v.shape[1], v.shape[2]
        G = H // HK
        B = cu.numel() - 1
        dev = q.device
        w, h, vn = _fwd_wuh(k, v, beta, gc, Ai, cu, coff, NTm, cs, cl, B)
        dvl = torch.empty((T, H, V), dtype=torch.float32, device=dev)
        _dvl_kernel[(NTm, H, V // 64)](q, k, do, gc, dvl, cs, cl, cu, H=H, HK=HK, K=K, V=V, BV=64, num_warps=4)
        dh = torch.empty((NTm, H, K, V), dtype=torch.bfloat16, device=dev)
        dv2 = torch.empty((T, H, V), dtype=torch.bfloat16, device=dev)
        BV = _rec_bv(B, H, V)
        _dh_kernel[(V // BV, B * H)](q, k, w, do, dvl, gc, dh, dv2, cu, coff,
                                     H=H, HK=HK, K=K, V=V, BV=BV, num_warps=4)
        del dvl
        dw = torch.empty((T, H, K), dtype=torch.float32, device=dev)
        dqb = torch.empty((T, H, K), dtype=torch.float32, device=dev)
        dkb = torch.empty((T, H, K), dtype=torch.float32, device=dev)
        dG = torch.empty((T, H), dtype=torch.float32, device=dev)
        _dqkw_kernel[(NTm, H)](q, k, vn, do, dv2, gc, h, dh, dw, dqb, dkb, dG, cs, cl, cu,
                               H=H, HK=HK, K=K, V=V, BV=64, num_warps=8)
        del h, dh, vn
        dv = torch.empty((T, H, V), dtype=torch.bfloat16, device=dev)
        db = torch.empty((T, H), dtype=torch.float32, device=dev)
        dg = torch.empty((T, H), dtype=torch.float32, device=dev)
        _wy_bwd_kernel[(NTm, H)](k, v, beta, gc, Ai, dw, dv2, dkb, dG, dv, db, dg, cs, cl, cu,
                                 H=H, HK=HK, K=K, V=V, BV=64, num_warps=8)
        if G == 1:
            dq = dqb.to(torch.bfloat16)
            dk = dkb.to(torch.bfloat16)
        else:
            dq = dqb.view(T, HK, G, K).sum(2).to(torch.bfloat16)
            dk = dkb.view(T, HK, G, K).sum(2).to(torch.bfloat16)
        return dq, dk, dv, dg, db.to(torch.bfloat16), None


def chunk_gated_deltanet(q, k, v, g, beta, cu_seqlens):
    return _GDN.apply(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
                      beta.contiguous(), cu_seqlens.contiguous())
