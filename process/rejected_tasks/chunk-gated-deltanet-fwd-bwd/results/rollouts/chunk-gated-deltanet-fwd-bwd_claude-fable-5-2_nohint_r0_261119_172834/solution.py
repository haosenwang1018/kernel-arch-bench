"""Grouped-value packed variable-length Gated DeltaNet, forward + backward in Triton.

设计概要
--------
* 状态 S (K x V) 的各个 V 列互相独立, 所以按 (序列, value head, V 切片) 开一个 program,
  S 的切片 (K x BV, fp32) 常驻寄存器, 在 kernel 内沿时间顺序循环 (每步 16 个 token, 用
  WY 表示 + tensor core 的 tl.dot)。
* 16x16 的单位下三角逆 (Neumann 倍增) 在一个按 chunk 完全并行的 prep kernel 里算好。
* 反向: (1) 重算前向, 记录每 64 token 的起始状态与 u = beta * e;
        (2) 逆序 kernel 得到 dv, 并记录每 64 token 末尾的 dS;
        (3) 按 64-token chunk 完全并行的 kernel 计算 dq / dk;
        (4) dbeta, dg 由逐 token 标量恒等式得到
            dbeta_t = <u_t, dv_t> / beta_t^2,
            dg_t    = sum_{s>=t, 同序列} (<o_s, do_s> - <v_s, dv_s>).
* 前向与反向之间只保留 prep 结果 (T * H_v * 32 个 fp16, 最多约 33 MB)。
"""

import torch
import triton
import triton.language as tl

C_SUB = 16      # 顺序 kernel 的子块大小
N_SUB = 4       # 每个 64-token 大块包含的子块数
C_BIG = C_SUB * N_SUB
BV_SEQ = 32     # 顺序 kernel 的 V 切片宽度
BV_POST = 64    # 参数梯度 kernel 的 V 切片宽度
KB_POST = 64    # 参数梯度 kernel 的 K 切片宽度


@triton.jit
def _mul3(a, b):
    return tl.sum(a[:, :, None] * b[None, :, :], axis=1)


@triton.jit
def _prep_kernel(
    q, k, g, beta, cst, cen, TB, AA,
    H_k: tl.constexpr, H_v: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr,
):
    n = tl.program_id(0)
    hv = tl.program_id(1)
    hk = hv // G
    cs = tl.load(cst + n)
    ce = tl.load(cen + n)
    ar = tl.arange(0, C)
    idx = cs + ar
    valid = idx < ce
    ok = tl.arange(0, K)
    mk = valid[:, None] & (ok[None, :] < K)
    kp = (idx[:, None] * H_k + hk) * K + ok[None, :]
    Kc = tl.load(k + kp, mask=mk, other=0.0).to(tl.float32).to(tl.float16)
    Qc = tl.load(q + kp, mask=mk, other=0.0).to(tl.float32).to(tl.float16)
    KcT = tl.trans(Kc)
    kkt = tl.dot(Kc, KcT)
    qkt = tl.dot(Qc, KcT)
    gv = tl.load(g + idx * H_v + hv, mask=valid, other=0.0).to(tl.float32)
    bv = tl.load(beta + idx * H_v + hv, mask=valid, other=0.0).to(tl.float32)
    lower_inc = ar[:, None] >= ar[None, :]
    strict = ar[:, None] > ar[None, :]
    gl = tl.sum(tl.where(lower_inc, gv[None, :], 0.0), axis=1)
    dec = tl.exp(tl.where(lower_inc, gl[:, None] - gl[None, :], 0.0))
    N1 = tl.where(strict, -(bv[:, None] * dec * kkt), 0.0)
    eye = (ar[:, None] == ar[None, :]).to(tl.float32)
    N2 = _mul3(N1, N1)
    P = _mul3(eye + N1, eye + N2)
    N4 = _mul3(N2, N2)
    P = _mul3(P, eye + N4)
    N8 = _mul3(N4, N4)
    P = _mul3(P, eye + N8)
    Tb = P * bv[None, :]
    A = tl.where(lower_inc, dec * qkt, 0.0)
    mc = valid[:, None] & (ar[None, :] < C)
    op = (idx[:, None] * H_v + hv) * C + ar[None, :]
    tl.store(TB + op, Tb.to(tl.float16), mask=mc)
    tl.store(AA + op, A.to(tl.float16), mask=mc)


@triton.jit
def _fwd_kernel(
    q, k, v, g, cu, TB, AA, out, U, SB, coff,
    H_k: tl.constexpr, H_v: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    C: tl.constexpr, NSUB: tl.constexpr, STORE: tl.constexpr,
):
    b = tl.program_id(0)
    hv = tl.program_id(1)
    vs = tl.program_id(2)
    hk = hv // G
    bos = tl.load(cu + b)
    eos = tl.load(cu + b + 1)
    nsup = (eos - bos + C * NSUB - 1) // (C * NSUB)
    ar = tl.arange(0, C)
    ok = tl.arange(0, K)
    ov = vs * BV + tl.arange(0, BV)
    lower_inc = ar[:, None] >= ar[None, :]
    S = tl.zeros([K, BV], dtype=tl.float32)
    if STORE:
        n0 = tl.load(coff + b)
    for j in range(nsup):
        if STORE:
            sp = (((n0 + j) * H_v + hv) * V + ov[:, None]) * K + ok[None, :]
            tl.store(SB + sp, tl.trans(S.to(tl.float16)))
        for i in range(NSUB):
            cs = bos + (j * NSUB + i) * C
            idx = cs + ar
            valid = idx < eos
            mk = valid[:, None] & (ok[None, :] < K)
            kp = (idx[:, None] * H_k + hk) * K + ok[None, :]
            Kf = tl.load(k + kp, mask=mk, other=0.0).to(tl.float32)
            mv = valid[:, None] & (ov[None, :] < V)
            vp = (idx[:, None] * H_v + hv) * V + ov[None, :]
            vv = tl.load(v + vp, mask=mv, other=0.0).to(tl.float32)
            gv = tl.load(g + idx * H_v + hv, mask=valid, other=0.0).to(tl.float32)
            gl = tl.sum(tl.where(lower_inc, gv[None, :], 0.0), axis=1)
            glast = tl.sum(gv, axis=0)
            gam = tl.exp(gl)
            gback = tl.exp(glast - gl)
            gC = tl.exp(glast)
            mc = valid[:, None] & (ar[None, :] < C)
            tp = (idx[:, None] * H_v + hv) * C + ar[None, :]
            Tb = tl.load(TB + tp, mask=mc, other=0.0)
            S16 = S.to(tl.float16)
            KS = tl.dot(Kf.to(tl.float16), S16)
            r = vv - gam[:, None] * KS
            u = tl.dot(Tb, r.to(tl.float16))
            u16 = u.to(tl.float16)
            if STORE:
                tl.store(U + vp, u16, mask=mv)
            else:
                Qc = tl.load(q + kp, mask=mk, other=0.0).to(tl.float32).to(tl.float16)
                A = tl.load(AA + tp, mask=mc, other=0.0)
                QS = tl.dot(Qc, S16)
                o = gam[:, None] * QS + tl.dot(A, u16)
                tl.store(out + vp, o.to(tl.bfloat16), mask=mv)
            Kbar = (Kf * gback[:, None]).to(tl.float16)
            S = gC * S + tl.dot(tl.trans(Kbar), u16)


@triton.jit
def _bwd_kernel(
    q, k, v, g, cu, TB, AA, out, U, do, dv, GB, PB, PG, coff,
    H_k: tl.constexpr, H_v: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    C: tl.constexpr, NSUB: tl.constexpr, NV: tl.constexpr,
):
    b = tl.program_id(0)
    hv = tl.program_id(1)
    vs = tl.program_id(2)
    hk = hv // G
    bos = tl.load(cu + b)
    eos = tl.load(cu + b + 1)
    nsup = (eos - bos + C * NSUB - 1) // (C * NSUB)
    n0 = tl.load(coff + b)
    ar = tl.arange(0, C)
    ok = tl.arange(0, K)
    ov = vs * BV + tl.arange(0, BV)
    lower_inc = ar[:, None] >= ar[None, :]
    Gs = tl.zeros([K, BV], dtype=tl.float32)
    for jj in range(nsup):
        j = nsup - 1 - jj
        sp = (((n0 + j) * H_v + hv) * V + ov[:, None]) * K + ok[None, :]
        tl.store(GB + sp, tl.trans(Gs.to(tl.float16)))
        for ii in range(NSUB):
            i = NSUB - 1 - ii
            cs = bos + (j * NSUB + i) * C
            idx = cs + ar
            valid = idx < eos
            mk = valid[:, None] & (ok[None, :] < K)
            kp = (idx[:, None] * H_k + hk) * K + ok[None, :]
            Kf = tl.load(k + kp, mask=mk, other=0.0).to(tl.float32)
            Qf = tl.load(q + kp, mask=mk, other=0.0).to(tl.float32)
            mv = valid[:, None] & (ov[None, :] < V)
            vp = (idx[:, None] * H_v + hv) * V + ov[None, :]
            dof = tl.load(do + vp, mask=mv, other=0.0).to(tl.float32)
            gv = tl.load(g + idx * H_v + hv, mask=valid, other=0.0).to(tl.float32)
            gl = tl.sum(tl.where(lower_inc, gv[None, :], 0.0), axis=1)
            glast = tl.sum(gv, axis=0)
            gam = tl.exp(gl)
            gback = tl.exp(glast - gl)
            gC = tl.exp(glast)
            mc = valid[:, None] & (ar[None, :] < C)
            tp = (idx[:, None] * H_v + hv) * C + ar[None, :]
            Tb = tl.load(TB + tp, mask=mc, other=0.0)
            A = tl.load(AA + tp, mask=mc, other=0.0)
            G16 = Gs.to(tl.float16)
            do16 = dof.to(tl.float16)
            KG = tl.dot(Kf.to(tl.float16), G16)
            rhs = gback[:, None] * KG + tl.dot(tl.trans(A), do16)
            de = tl.dot(tl.trans(Tb), rhs.to(tl.float16))
            tl.store(dv + vp, de.to(tl.bfloat16), mask=mv)
            de16 = de.to(tl.float16)
            uu = tl.load(U + vp, mask=mv, other=0.0).to(tl.float32)
            oo = tl.load(out + vp, mask=mv, other=0.0).to(tl.float32)
            vv = tl.load(v + vp, mask=mv, other=0.0).to(tl.float32)
            pb = tl.sum(uu * de, axis=1)
            pg = tl.sum(oo * dof - vv * de, axis=1)
            pp = (idx * H_v + hv) * NV + vs
            tl.store(PB + pp, pb, mask=valid)
            tl.store(PG + pp, pg, mask=valid)
            Qg = (Qf * gam[:, None]).to(tl.float16)
            Kg = (Kf * gam[:, None]).to(tl.float16)
            Gs = gC * Gs + tl.dot(tl.trans(Qg), do16) - tl.dot(tl.trans(Kg), de16)


@triton.jit
def _post_kernel(
    q, k, g, do, dv, U, SB, GB, cst, cen, dq, dk,
    H_k: tl.constexpr, H_v: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    KB: tl.constexpr, C: tl.constexpr,
):
    n = tl.program_id(0)
    hk = tl.program_id(1)
    kb = tl.program_id(2)
    cs = tl.load(cst + n)
    ce = tl.load(cen + n)
    ar = tl.arange(0, C)
    idx = cs + ar
    valid = idx < ce
    okb = kb * KB + tl.arange(0, KB)
    mk = valid[:, None] & (okb[None, :] < K)
    kp = (idx[:, None] * H_k + hk) * K + okb[None, :]
    Kb = tl.load(k + kp, mask=mk, other=0.0).to(tl.float32).to(tl.float16)
    Qb = tl.load(q + kp, mask=mk, other=0.0).to(tl.float32).to(tl.float16)
    lower_inc = ar[:, None] >= ar[None, :]
    strict = ar[:, None] > ar[None, :]
    aq = tl.zeros([C, KB], dtype=tl.float32)
    ak = tl.zeros([C, KB], dtype=tl.float32)
    for gi in range(G):
        hv = hk * G + gi
        gv = tl.load(g + idx * H_v + hv, mask=valid, other=0.0).to(tl.float32)
        gl = tl.sum(tl.where(lower_inc, gv[None, :], 0.0), axis=1)
        glast = tl.sum(gv, axis=0)
        gam = tl.exp(gl)
        gback = tl.exp(glast - gl)
        dec = tl.exp(tl.where(lower_inc, gl[:, None] - gl[None, :], 0.0))
        X = tl.zeros([C, C], dtype=tl.float32)
        Zt = tl.zeros([C, C], dtype=tl.float32)
        for vs in range(V // BV):
            ov = vs * BV + tl.arange(0, BV)
            mv = valid[:, None] & (ov[None, :] < V)
            vp = (idx[:, None] * H_v + hv) * V + ov[None, :]
            dof = tl.load(do + vp, mask=mv, other=0.0).to(tl.float32)
            def_ = tl.load(dv + vp, mask=mv, other=0.0).to(tl.float32)
            u16 = tl.load(U + vp, mask=mv, other=0.0)
            uf = u16.to(tl.float32)
            uT = tl.trans(u16)
            X += tl.dot(dof.to(tl.float16), uT)
            Zt += tl.dot(def_.to(tl.float16), uT)
            sp = ((n * H_v + hv) * V + ov[:, None]) * K + okb[None, :]
            S0T = tl.load(SB + sp)
            GT = tl.load(GB + sp)
            aq += tl.dot((dof * gam[:, None]).to(tl.float16), S0T)
            ak += tl.dot((uf * gback[:, None]).to(tl.float16), GT)
            ak -= tl.dot((def_ * gam[:, None]).to(tl.float16), S0T)
        X = tl.where(lower_inc, X * dec, 0.0)
        Zt = tl.where(strict, Zt * dec, 0.0)
        X16 = X.to(tl.float16)
        ZZ16 = (Zt + tl.trans(Zt)).to(tl.float16)
        aq += tl.dot(X16, Kb)
        ak += tl.dot(tl.trans(X16), Qb)
        ak -= tl.dot(ZZ16, Kb)
    tl.store(dq + kp, aq.to(tl.bfloat16), mask=mk)
    tl.store(dk + kp, ak.to(tl.bfloat16), mask=mk)


@triton.jit
def _tail_kernel(
    PB, PG, beta, cu, dbeta, dg,
    H_v: tl.constexpr, NV: tl.constexpr, C: tl.constexpr,
):
    b = tl.program_id(0)
    hv = tl.program_id(1)
    bos = tl.load(cu + b)
    eos = tl.load(cu + b + 1)
    nch = (eos - bos + C - 1) // C
    ar = tl.arange(0, C)
    on = tl.arange(0, NV)
    upper_inc = ar[:, None] <= ar[None, :]
    carry = tl.sum(tl.zeros([C], dtype=tl.float32), axis=0)
    for jj in range(nch):
        j = nch - 1 - jj
        idx = bos + j * C + ar
        valid = idx < eos
        m = valid[:, None] & (on[None, :] < NV)
        pp = (idx[:, None] * H_v + hv) * NV + on[None, :]
        pb = tl.sum(tl.load(PB + pp, mask=m, other=0.0), axis=1)
        pg = tl.sum(tl.load(PG + pp, mask=m, other=0.0), axis=1)
        bt = tl.load(beta + idx * H_v + hv, mask=valid, other=1.0).to(tl.float32)
        tl.store(dbeta + idx * H_v + hv, (pb / (bt * bt)).to(tl.bfloat16), mask=valid)
        rc = tl.sum(tl.where(upper_inc, pg[None, :], 0.0), axis=1) + carry
        tl.store(dg + idx * H_v + hv, rc, mask=valid)
        carry = carry + tl.sum(pg, axis=0)


def _chunk_bounds(cul, step, dev):
    st = []
    en = []
    cnt = []
    for b in range(len(cul) - 1):
        r = range(cul[b], cul[b + 1], step)
        st.extend(r)
        en.extend([cul[b + 1]] * len(r))
        cnt.append(len(r))
    t = torch.tensor([st, en], dtype=torch.int32, device=dev)
    return t[0].contiguous(), t[1].contiguous(), cnt


class _GatedDeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens):
        T, H_k, K = q.shape
        H_v, V = v.shape[1], v.shape[2]
        G = H_v // H_k
        dev = q.device
        cul = cu_seqlens.tolist()
        B = len(cul) - 1
        st, en, _ = _chunk_bounds(cul, C_SUB, dev)
        N16 = st.numel()
        TB = torch.empty((T, H_v, C_SUB), dtype=torch.float16, device=dev)
        AA = torch.empty((T, H_v, C_SUB), dtype=torch.float16, device=dev)
        _prep_kernel[(N16, H_v)](
            q, k, g, beta, st, en, TB, AA,
            H_k=H_k, H_v=H_v, G=G, K=K, C=C_SUB, num_warps=4,
        )
        out = torch.empty((T, H_v, V), dtype=torch.bfloat16, device=dev)
        _fwd_kernel[(B, H_v, V // BV_SEQ)](
            q, k, v, g, cu_seqlens, TB, AA, out, out, TB, cu_seqlens,
            H_k=H_k, H_v=H_v, G=G, K=K, V=V, BV=BV_SEQ,
            C=C_SUB, NSUB=N_SUB, STORE=False, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, g, beta, out)
        ctx.cu = cu_seqlens
        ctx.cul = cul
        ctx.TB = TB
        ctx.AA = AA
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, out = ctx.saved_tensors
        cu = ctx.cu
        cul = ctx.cul
        TB, AA = ctx.TB, ctx.AA
        T, H_k, K = q.shape
        H_v, V = v.shape[1], v.shape[2]
        G = H_v // H_k
        dev = q.device
        B = len(cul) - 1
        do = do.to(torch.bfloat16).contiguous()

        st, en, cnt = _chunk_bounds(cul, C_BIG, dev)
        N64 = st.numel()
        offs = [0] * B
        acc = 0
        for b in range(B):
            offs[b] = acc
            acc += cnt[b]
        coff = torch.tensor(offs, dtype=torch.int32, device=dev)

        NV = V // BV_SEQ
        SB = torch.empty((N64, H_v, V, K), dtype=torch.float16, device=dev)
        GB = torch.empty((N64, H_v, V, K), dtype=torch.float16, device=dev)
        U = torch.empty((T, H_v, V), dtype=torch.float16, device=dev)
        dv = torch.empty((T, H_v, V), dtype=torch.bfloat16, device=dev)
        PB = torch.empty((T, H_v, NV), dtype=torch.float32, device=dev)
        PG = torch.empty((T, H_v, NV), dtype=torch.float32, device=dev)
        dq = torch.empty((T, H_k, K), dtype=torch.bfloat16, device=dev)
        dk = torch.empty((T, H_k, K), dtype=torch.bfloat16, device=dev)

        _fwd_kernel[(B, H_v, NV)](
            q, k, v, g, cu, TB, AA, out, U, SB, coff,
            H_k=H_k, H_v=H_v, G=G, K=K, V=V, BV=BV_SEQ,
            C=C_SUB, NSUB=N_SUB, STORE=True, num_warps=4,
        )
        _bwd_kernel[(B, H_v, NV)](
            q, k, v, g, cu, TB, AA, out, U, do, dv, GB, PB, PG, coff,
            H_k=H_k, H_v=H_v, G=G, K=K, V=V, BV=BV_SEQ,
            C=C_SUB, NSUB=N_SUB, NV=NV, num_warps=4,
        )
        _post_kernel[(N64, H_k, K // KB_POST)](
            q, k, g, do, dv, U, SB, GB, st, en, dq, dk,
            H_k=H_k, H_v=H_v, G=G, K=K, V=V, BV=BV_POST,
            KB=KB_POST, C=C_BIG, num_warps=8,
        )

        dbeta = torch.empty((T, H_v), dtype=torch.bfloat16, device=dev)
        dg = torch.empty((T, H_v), dtype=torch.float32, device=dev)
        _tail_kernel[(B, H_v)](
            PB, PG, beta, cu, dbeta, dg,
            H_v=H_v, NV=NV, C=C_BIG, num_warps=1,
        )
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
