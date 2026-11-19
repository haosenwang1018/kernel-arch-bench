"""Heterogeneous batched Polar-Express Muon + Hyperball step (Triton)."""

import torch
import triton
import triton.language as tl

COEFFS = (
    (4.203190, -12.416490, 9.203400),
    (4.165205, -12.100974, 8.925868),
    (4.006145, -10.838158, 7.822113),
    (3.408849, -6.883363, 4.464614),
    (2.294167, -2.281445, 0.977377),
)

NF = tl.constexpr(24)
F_ST_SYRK = tl.constexpr(0)
F_TPS_SYRK = tl.constexpr(1)
F_ST_SQ = tl.constexpr(2)
F_TPS_SQ = tl.constexpr(3)
F_ST_BX = tl.constexpr(4)
F_TPS_BX = tl.constexpr(5)
F_ST_EW = tl.constexpr(6)
F_TPS_EW = tl.constexpr(7)
F_SLOT = tl.constexpr(8)
F_S = tl.constexpr(9)
F_L = tl.constexpr(10)
F_M = tl.constexpr(11)
F_N = tl.constexpr(12)
F_TRANS = tl.constexpr(13)
F_SPLIT = tl.constexpr(14)
F_XOFF = tl.constexpr(15)
F_AOFF = tl.constexpr(16)
F_BOFF = tl.constexpr(17)
F_WOFF = tl.constexpr(18)
F_MOFF = tl.constexpr(19)
F_GOFF = tl.constexpr(20)
F_NT = tl.constexpr(21)
F_NTN = tl.constexpr(22)
F_NTEN = tl.constexpr(23)
NFI = 24

MIB = 1024 * 1024
BUDGET = 186 * MIB
TM = 128          # square tile for syrk / A^2
BK_SYRK = 64
BX_BM = 128
BX_BN = 256
BX_BK = 64
EB = 64


@triton.jit
def _find(tab, pid, NG, FIELD: tl.constexpr):
    g = pid * 0
    for i in range(1, NG):
        st = tl.load(tab + i * NF + FIELD)
        g = tl.where(pid >= st, i, g)
    return g


@triton.jit
def _tri(tt):
    i = ((tl.sqrt((8 * tt + 1).to(tl.float32)) - 1.0) * 0.5).to(tl.int64)
    i = tl.where((i + 1) * (i + 2) // 2 <= tt, i + 1, i)
    i = tl.where(i * (i + 1) // 2 > tt, i - 1, i)
    j = tt - i * (i + 1) // 2
    return i, j


@triton.jit
def _ew_decode(tab, NG, pid, EVEN: tl.constexpr, EBS: tl.constexpr):
    gi = _find(tab, pid, NG, F_ST_EW)
    row = tab + gi * NF
    local = pid - tl.load(row + F_ST_EW)
    tps = tl.load(row + F_TPS_EW)
    s = local // tps
    t = local % tps
    ntn = tl.load(row + F_NTEN)
    tm = t // ntn
    tn = t % ntn
    M = tl.load(row + F_M)
    N = tl.load(row + F_N)
    if EVEN:
        M = (M // 64) * 64
        N = (N // 64) * 64
    slot = tl.load(row + F_SLOT) + s
    b = tl.load(tab + NG * NF + slot)
    offs_m = tm * EBS + tl.arange(0, EBS)
    offs_n = tn * EBS + tl.arange(0, EBS)
    return row, s, slot, b, M, N, offs_m, offs_n


@triton.jit
def _ew_norm_kernel(tab, NG, m_base, g_base, stats, beta,
                    EVEN: tl.constexpr, EBS: tl.constexpr):
    pid = tl.program_id(0)
    row, s, slot, b, M, N, offs_m, offs_n = _ew_decode(tab, NG, pid, EVEN, EBS)
    moff = tl.load(row + F_MOFF)
    goff = tl.load(row + F_GOFF)
    if EVEN:
        moff = (moff // 8) * 8
        goff = (goff // 8) * 8
    idx = b * M * N + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mo = tl.load(m_base + moff + idx, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(g_base + goff + idx, mask=mask, other=0.0).to(tl.float32)
    mn = beta * mo + (1.0 - beta) * go
    u = beta * mn + (1.0 - beta) * go
    tl.atomic_add(stats + slot * 4, tl.sum(tl.sum(u * u, axis=1), axis=0), sem="relaxed")


@triton.jit
def _ew_x0_kernel(tab, NG, m_base, g_base, xw, stats, beta, eps,
                  EVEN: tl.constexpr, EBS: tl.constexpr):
    pid = tl.program_id(0)
    row, s, slot, b, M, N, offs_m, offs_n = _ew_decode(tab, NG, pid, EVEN, EBS)
    moff = tl.load(row + F_MOFF)
    goff = tl.load(row + F_GOFF)
    xoff = tl.load(row + F_XOFF)
    if EVEN:
        moff = (moff // 8) * 8
        goff = (goff // 8) * 8
        xoff = (xoff // 128) * 128
    loc = offs_m[:, None] * N + offs_n[None, :]
    idx = b * M * N + loc
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mo = tl.load(m_base + moff + idx, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(g_base + goff + idx, mask=mask, other=0.0).to(tl.float32)
    mn = beta * mo + (1.0 - beta) * go
    tl.store(m_base + moff + idx, mn.to(tl.bfloat16), mask=mask)
    u = beta * mn + (1.0 - beta) * go
    usq = tl.load(stats + slot * 4)
    inv = 1.0 / (1.01 * tl.sqrt(usq) + eps)
    x = (u * inv).to(tl.bfloat16)
    xb = xw + xoff + s * M * N
    trans = tl.load(row + F_TRANS)
    if trans != 0:
        tl.store(xb + offs_n[None, :] * M + offs_m[:, None], x, mask=mask)
    else:
        tl.store(xb + loc, x, mask=mask)


@triton.jit
def _ew_reduce_kernel(tab, NG, w_base, xw, xf_reg, stats,
                      EVEN: tl.constexpr, EBS: tl.constexpr):
    pid = tl.program_id(0)
    row, s, slot, b, M, N, offs_m, offs_n = _ew_decode(tab, NG, pid, EVEN, EBS)
    woff = tl.load(row + F_WOFF)
    xoff = tl.load(row + F_XOFF)
    if EVEN:
        woff = (woff // 4) * 4
        xoff = (xoff // 128) * 128
    loc = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    w = tl.load(w_base + woff + b * M * N + loc, mask=mask, other=0.0)
    xb = xw + xf_reg + xoff + s * M * N
    trans = tl.load(row + F_TRANS)
    if trans != 0:
        o = tl.load(xb + offs_n[None, :] * M + offs_m[:, None], mask=mask, other=0.0).to(tl.float32)
    else:
        o = tl.load(xb + loc, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(stats + slot * 4 + 1, tl.sum(tl.sum(w * w, axis=1), axis=0), sem="relaxed")
    tl.atomic_add(stats + slot * 4 + 2, tl.sum(tl.sum(o * o, axis=1), axis=0), sem="relaxed")
    tl.atomic_add(stats + slot * 4 + 3, tl.sum(tl.sum(w * o, axis=1), axis=0), sem="relaxed")


@triton.jit
def _ew_update_kernel(tab, NG, w_base, xw, xf_reg, stats, lr, eps,
                      EVEN: tl.constexpr, EBS: tl.constexpr):
    pid = tl.program_id(0)
    row, s, slot, b, M, N, offs_m, offs_n = _ew_decode(tab, NG, pid, EVEN, EBS)
    woff = tl.load(row + F_WOFF)
    xoff = tl.load(row + F_XOFF)
    if EVEN:
        woff = (woff // 4) * 4
        xoff = (xoff // 128) * 128
    loc = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    wp = w_base + woff + b * M * N + loc
    w = tl.load(wp, mask=mask, other=0.0)
    xb = xw + xf_reg + xoff + s * M * N
    trans = tl.load(row + F_TRANS)
    if trans != 0:
        o = tl.load(xb + offs_n[None, :] * M + offs_m[:, None], mask=mask, other=0.0).to(tl.float32)
    else:
        o = tl.load(xb + loc, mask=mask, other=0.0).to(tl.float32)
    rw2 = tl.load(stats + slot * 4 + 1)
    ro2 = tl.load(stats + slot * 4 + 2)
    wo = tl.load(stats + slot * 4 + 3)
    rw = tl.sqrt(rw2)
    ro = tl.sqrt(ro2)
    k = lr * (rw / (ro + eps))
    wt2 = rw2 - 2.0 * k * wo + k * k * ro2
    sc = rw / (tl.sqrt(tl.maximum(wt2, 0.0)) + eps)
    tl.store(wp, (w - k * o) * sc, mask=mask)


@triton.jit
def _syrk_kernel(tab, NG, xw, a_bf, a_f32, x_reg, a_reg,
                 A_FP32: tl.constexpr, EVEN: tl.constexpr,
                 BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    gi = _find(tab, pid, NG, F_ST_SYRK)
    row = tab + gi * NF
    local = pid - tl.load(row + F_ST_SYRK)
    tps = tl.load(row + F_TPS_SYRK)
    s = local // tps
    t = local % tps
    S = tl.load(row + F_S)
    L = tl.load(row + F_L)
    nt = tl.load(row + F_NT)
    split = tl.load(row + F_SPLIT)
    xoff = tl.load(row + F_XOFF)
    aoff = tl.load(row + F_AOFF)
    if EVEN:
        S = (S // 128) * 128
        L = (L // 128) * 128
        xoff = (xoff // 128) * 128
        aoff = (aoff // 128) * 128
    T = nt * (nt + 1) // 2
    ks = t // T
    tt = t % T
    i, j = _tri(tt)
    kit = tl.cdiv(L, BK)
    per = tl.cdiv(kit, split)
    k0 = ks * per * BK
    k1 = tl.minimum(L, k0 + per * BK)
    xb = xw + x_reg + xoff + s * S * L
    rm = i * BM + tl.arange(0, BM)
    rn = j * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    a_ptrs = xb + rm[:, None] * L + (k0 + rk)[None, :]
    b_ptrs = xb + rn[None, :] * L + (k0 + rk)[:, None]
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k in range(k0, k1, BK):
        if EVEN:
            a = tl.load(a_ptrs)
            bb = tl.load(b_ptrs)
        else:
            kk = k + rk
            a = tl.load(a_ptrs, mask=(rm[:, None] < S) & (kk[None, :] < k1), other=0.0)
            bb = tl.load(b_ptrs, mask=(rn[None, :] < S) & (kk[:, None] < k1), other=0.0)
        acc += tl.dot(a, bb)
        a_ptrs += BK
        b_ptrs += BK
    ab = a_reg + aoff + s * S * S
    p1 = ab + rm[:, None] * S + rn[None, :]
    p2 = ab + rn[None, :] * S + rm[:, None]
    msk = (rm[:, None] < S) & (rn[None, :] < S)
    if A_FP32:
        if split > 1:
            tl.atomic_add(a_f32 + p1, acc, mask=msk, sem="relaxed")
            if i != j:
                tl.atomic_add(a_f32 + p2, acc, mask=msk, sem="relaxed")
        else:
            tl.store(a_f32 + p1, acc, mask=msk)
            if i != j:
                tl.store(a_f32 + p2, acc, mask=msk)
    else:
        c = acc.to(tl.bfloat16)
        tl.store(a_bf + p1, c, mask=msk)
        if i != j:
            tl.store(a_bf + p2, c, mask=msk)


@triton.jit
def _sq_kernel(tab, NG, a_bf, a_f32, b_bf, a_reg, b_reg, coef_b, coef_c,
               A_FP32: tl.constexpr, EVEN: tl.constexpr,
               BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    gi = _find(tab, pid, NG, F_ST_SQ)
    row = tab + gi * NF
    local = pid - tl.load(row + F_ST_SQ)
    tps = tl.load(row + F_TPS_SQ)
    s = local // tps
    tt = local % tps
    S = tl.load(row + F_S)
    aoff = tl.load(row + F_AOFF)
    boff = tl.load(row + F_BOFF)
    if EVEN:
        S = (S // 128) * 128
        aoff = (aoff // 128) * 128
        boff = (boff // 128) * 128
    i, j = _tri(tt)
    ab = a_reg + aoff + s * S * S
    if A_FP32:
        abase = a_f32 + ab
    else:
        abase = a_bf + ab
    rm = i * BM + tl.arange(0, BM)
    rn = j * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    a_ptrs = abase + rm[:, None] * S + rk[None, :]
    b_ptrs = abase + rk[:, None] * S + rn[None, :]
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k in range(0, S, BK):
        if EVEN:
            a = tl.load(a_ptrs)
            bb = tl.load(b_ptrs)
        else:
            kk = k + rk
            a = tl.load(a_ptrs, mask=(rm[:, None] < S) & (kk[None, :] < S), other=0.0)
            bb = tl.load(b_ptrs, mask=(rn[None, :] < S) & (kk[:, None] < S), other=0.0)
        acc += tl.dot(a.to(tl.bfloat16), bb.to(tl.bfloat16))
        a_ptrs += BK
        b_ptrs += BK * S
    msk = (rm[:, None] < S) & (rn[None, :] < S)
    aij = tl.load(abase + rm[:, None] * S + rn[None, :], mask=msk, other=0.0)
    aij = aij.to(tl.bfloat16).to(tl.float32)
    out = (coef_b * aij + coef_c * acc).to(tl.bfloat16)
    bbp = b_bf + b_reg + boff + s * S * S
    tl.store(bbp + rm[:, None] * S + rn[None, :], out, mask=msk)
    if i != j:
        tl.store(bbp + rn[None, :] * S + rm[:, None], out, mask=msk)


@triton.jit
def _bx_kernel(tab, NG, xw, src_reg, dst_reg, b_reg, coef_a, col0, chunk_w,
               CHUNK: tl.constexpr, EVEN: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    gi = _find(tab, pid, NG, F_ST_BX)
    row = tab + gi * NF
    local = pid - tl.load(row + F_ST_BX)
    S = tl.load(row + F_S)
    L = tl.load(row + F_L)
    nt = tl.load(row + F_NT)
    xoff = tl.load(row + F_XOFF)
    boff = tl.load(row + F_BOFF)
    if EVEN:
        S = (S // 128) * 128
        L = (L // 128) * 128
        xoff = (xoff // 128) * 128
        boff = (boff // 128) * 128
    if CHUNK:
        nn = tl.cdiv(chunk_w, BN)
        ncols = chunk_w
        tps = nt * nn
    else:
        nn = tl.load(row + F_NTN)
        ncols = L
        tps = tl.load(row + F_TPS_BX)
    s = local // tps
    t = local % tps
    width = GROUP_M * nn
    group_id = t // width
    first_m = group_id * GROUP_M
    gsz = tl.minimum(nt - first_m, GROUP_M)
    pid_m = first_m + (t % width) % gsz
    pid_n = (t % width) // gsz
    xs = xoff + s * S * L
    if CHUNK:
        xin = xw + src_reg + xs + col0
        outp = xw + dst_reg
        ldo = chunk_w
    else:
        xin = xw + src_reg + xs
        outp = xw + dst_reg + xs
        ldo = L
    bb_base = xw + b_reg + boff + s * S * S
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = bb_base + rm[:, None] * S + rk[None, :]
    x_ptrs = xin + rk[:, None] * L + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, S, BK):
        if EVEN:
            a = tl.load(a_ptrs)
            xv = tl.load(x_ptrs)
        else:
            kk = k + rk
            a = tl.load(a_ptrs, mask=(rm[:, None] < S) & (kk[None, :] < S), other=0.0)
            xv = tl.load(x_ptrs, mask=(kk[:, None] < S) & (rn[None, :] < ncols), other=0.0)
        acc += tl.dot(a, xv)
        a_ptrs += BK
        x_ptrs += BK * L
    msk = (rm[:, None] < S) & (rn[None, :] < ncols)
    xc = tl.load(xin + rm[:, None] * L + rn[None, :], mask=msk, other=0.0).to(tl.float32)
    o = (coef_a * xc + acc).to(tl.bfloat16)
    tl.store(outp + rm[:, None] * ldo + rn[None, :], o, mask=msk)


def _cdiv(a, b):
    return (a + b - 1) // b


def _run_round(rd, ws, ws32, bases, lr, beta, eps, dev):
    """rd: dict with groups list [(ginfo, blist)], chunk flag, a_fp32 flag, splits."""
    chunk = rd["chunk"]
    a_fp32 = rd["a_fp32"]
    groups = rd["groups"]
    w_base, m_base, g_base = bases
    rows = []
    slots = []
    st = [0, 0, 0, 0]
    xo = 0
    ao = 0
    bo = 0
    even = True
    for (gi, blist, split) in groups:
        B, M, N, S, L, trans, woff, moff, goff, al = gi
        n = len(blist)
        nt = _cdiv(S, TM)
        T = nt * (nt + 1) // 2
        tps_syrk = T * split
        tps_sq = T
        ntn = _cdiv(L, BX_BN)
        tps_bx = nt * ntn
        nten = _cdiv(N, EB)
        tps_ew = _cdiv(M, EB) * nten
        if (M % 128) or (N % 128) or (L % BX_BN) or (not al):
            even = False
        rows.append([st[0], tps_syrk, st[1], tps_sq, st[2], tps_bx, st[3], tps_ew,
                     len(slots), S, L, M, N, trans, split, xo, ao, bo,
                     woff, moff, goff, nt, ntn, nten])
        st[0] += n * tps_syrk
        st[1] += n * tps_sq
        st[2] += n * tps_bx
        st[3] += n * tps_ew
        slots.extend(blist)
        xo += n * S * L
        ao += n * S * S
        bo += n * S * S
    NG = len(rows)
    flat = [v for r in rows for v in r] + slots
    tab = torch.tensor(flat, dtype=torch.int64, device=dev)
    nslot = len(slots)
    stats = torch.zeros(nslot * 4, dtype=torch.float32, device=dev)

    totx = xo
    p0 = 0
    if chunk:
        p1 = 0
        b_reg = totx
        a_reg_bf = totx + bo
    else:
        p1 = totx
        b_reg = 2 * totx
        a_reg_bf = 2 * totx + bo
    a_reg_bf = (a_reg_bf + 1) // 2 * 2
    if chunk and (groups[0][0][3] % BX_BN):
        even = False
    a_reg = a_reg_bf // 2 if a_fp32 else a_reg_bf
    any_split = a_fp32 and any(sp > 1 for (_, _, sp) in groups)

    _ew_norm_kernel[(st[3],)](tab, NG, m_base, g_base, stats, beta,
                              EVEN=even, EBS=EB, num_warps=4)
    _ew_x0_kernel[(st[3],)](tab, NG, m_base, g_base, ws, stats, beta, eps,
                            EVEN=even, EBS=EB, num_warps=4)
    src, dst = p0, p1
    bk_sq = 32 if a_fp32 else 64
    for (ca, cb, cc) in COEFFS:
        if any_split:
            ws32[a_reg:a_reg + ao].zero_()
        _syrk_kernel[(st[0],)](tab, NG, ws, ws, ws32, src, a_reg,
                               A_FP32=a_fp32, EVEN=even, BM=TM, BK=BK_SYRK,
                               num_warps=8, num_stages=3)
        _sq_kernel[(st[1],)](tab, NG, ws, ws32, ws, a_reg, b_reg, cb, cc,
                             A_FP32=a_fp32, EVEN=even, BM=TM, BK=bk_sq,
                             num_warps=8, num_stages=3)
        if chunk:
            S = groups[0][0][3]
            L = groups[0][0][4]
            nt = _cdiv(S, BX_BM)
            xv = ws[0:S * L].view(S, L)
            for c0 in range(0, L, S):
                w = min(S, L - c0)
                _bx_kernel[(nt * _cdiv(w, BX_BN),)](
                    tab, NG, ws, src, a_reg_bf, b_reg, ca, c0, w,
                    CHUNK=True, EVEN=even, BM=BX_BM, BN=BX_BN, BK=BX_BK, GROUP_M=8,
                    num_warps=8, num_stages=3)
                xv[:, c0:c0 + w].copy_(ws[a_reg_bf:a_reg_bf + S * w].view(S, w))
        else:
            _bx_kernel[(st[2],)](tab, NG, ws, src, dst, b_reg, ca, 0, 0,
                                 CHUNK=False, EVEN=even, BM=BX_BM, BN=BX_BN, BK=BX_BK,
                                 GROUP_M=8, num_warps=8, num_stages=3)
            src, dst = dst, src
    xf = src
    _ew_reduce_kernel[(st[3],)](tab, NG, w_base, ws, xf, stats,
                                EVEN=even, EBS=EB, num_warps=4)
    _ew_update_kernel[(st[3],)](tab, NG, w_base, ws, xf, stats, lr, eps,
                                EVEN=even, EBS=EB, num_warps=4)


def _step_impl(
    weights: list,
    momentums: list,
    grads: list,
    active_masks: list,
    lr: float,
    beta: float = 0.95,
    eps: float = 1e-7,
) -> None:
    P = len(weights)
    if P == 0:
        return
    dev = weights[0].device
    flat_mask = torch.cat([m.reshape(-1) for m in active_masks]).cpu().tolist()
    w0, m0, g0 = weights[0], momentums[0], grads[0]
    wp0, mp0, gp0 = w0.data_ptr(), m0.data_ptr(), g0.data_ptr()

    items = []  # (ginfo, blist, per_slice_bytes)
    pos = 0
    for p in range(P):
        w = weights[p]
        B, M, N = w.shape
        mk = flat_mask[pos:pos + B]
        pos += B
        blist = [b for b in range(B) if mk[b]]
        if not blist:
            continue
        S, L = (N, M) if M > N else (M, N)
        trans = 1 if M > N else 0
        wp, mp, gp = w.data_ptr(), momentums[p].data_ptr(), grads[p].data_ptr()
        al = ((wp - wp0) % 16 == 0 and (mp - mp0) % 16 == 0 and (gp - gp0) % 16 == 0)
        gi = (B, M, N, S, L, trans, (wp - wp0) // 4, (mp - mp0) // 2, (gp - gp0) // 2, al)
        items.append((gi, blist))
    if not items:
        return

    rounds = []
    cur = []
    cur_bytes = 0
    for gi, blist in items:
        S, L = gi[3], gi[4]
        per = (2 * S * L + 2 * S * S) * 2
        if per > BUDGET:
            for b in blist:
                rounds.append({"chunk": True, "a_fp32": False, "groups": [(gi, [b], 1)],
                               "bytes": (S * L + 2 * S * S) * 2})
            continue
        rem = list(blist)
        while rem:
            cap = (BUDGET - cur_bytes) // per
            if cap <= 0:
                rounds.append({"chunk": False, "groups": cur, "bytes": cur_bytes})
                cur, cur_bytes = [], 0
                continue
            take = rem[:cap]
            rem = rem[cap:]
            cur.append((gi, take))
            cur_bytes += per * len(take)
    if cur:
        rounds.append({"chunk": False, "groups": cur, "bytes": cur_bytes})

    # decide split-K / fp32 A for normal rounds
    for rd in rounds:
        if rd["chunk"]:
            continue
        ttot = 0
        extra = 0
        for gi, take in rd["groups"]:
            nt = _cdiv(gi[3], TM)
            ttot += len(take) * nt * (nt + 1) // 2
            extra += len(take) * gi[3] * gi[3] * 2
        target = 0
        if ttot < 216 and rd["bytes"] + extra <= BUDGET:
            target = _cdiv(432, ttot)
        newg = []
        any_sp = False
        for gi, take in rd["groups"]:
            sp = 1
            if target > 1:
                sp = max(1, min(target, gi[4] // (BK_SYRK * 4)))
            any_sp = any_sp or sp > 1
            newg.append((gi, take, sp))
        rd["groups"] = newg
        rd["a_fp32"] = any_sp
        if any_sp:
            rd["bytes"] += extra

    maxb = max(rd["bytes"] for rd in rounds)
    ws = torch.empty((maxb // 4 + 64) * 2, dtype=torch.bfloat16, device=dev)
    ws32 = ws.view(torch.float32)
    bases = (w0, m0, g0)
    for rd in rounds:
        _run_round(rd, ws, ws32, bases, float(lr), float(beta), float(eps), dev)


def muon_hyperball_step(weights, momentums, grads, active_masks, lr, beta=0.95, eps=1e-7):
    try:
        _step_impl(weights, momentums, grads, active_masks, lr, beta, eps)
    except Exception as e:  # surface full chained error text
        msgs = []
        cur = e
        depth = 0
        while cur is not None and depth < 6:
            msgs.append(type(cur).__name__ + ": " + str(cur))
            cur = cur.__cause__ or cur.__context__
            depth += 1
        raise RuntimeError(" || ".join(msgs)[-6000:])
