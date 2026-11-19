"""Top-K routed SwiGLU MoE FFN: 前向/反向均由 Triton 分组 GEMM kernel 实现。

设计要点
- 所有 expert 在一次 kernel launch 内完成（按行块表调度），token gather 融合进 GEMM 的读取。
- 前向只保留 g/u（bf16，受 ~108MiB 预算限制，超出部分的行在反向重算）。
- 权重梯度直接写入最终张量（空 expert 自然写 0），不做 stack/拷贝。
- scatter（按 token 聚合）用 CSR + fp32 累加完成，无原子操作竞争。
"""

import torch
import triton
import triton.language as tl

_GU_BUDGET_BYTES = 108 * 1024 * 1024
_PROF = None


def _tick(name):
    if _PROF is not None:
        import time
        torch.cuda.synchronize()
        t = time.perf_counter()
        _PROF.append((name, t))


# ----------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------
@triton.jit
def _gu_kernel(x_ptr, tok_ptr, w_ptr, blk_ptr, h_ptr, gu_ptr, gu_limit, D, I,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               KM: tl.constexpr, NM: tl.constexpr,
               STORE_H: tl.constexpr, STORE_GU: tl.constexpr):
    # g = x[tok] W_gate, u = x[tok] W_up ; h = silu(g) * u
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(blk_ptr + 3 * pb).to(tl.int64)
    lo = tl.load(blk_ptr + 3 * pb + 1)
    hi = tl.load(blk_ptr + 3 * pb + 2)
    rows = lo + tl.arange(0, BM)
    rmask = rows < hi
    tok = tl.load(tok_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < I
    ks = tl.arange(0, BK)
    a_ptrs = x_ptr + tok[:, None] * D + ks[None, :]
    b_ptrs = w_ptr + e * D * (2 * I) + ks[:, None].to(tl.int64) * (2 * I) + cols[None, :]
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, D, BK):
        if KM:
            kmask = (k + ks) < D
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            bg = tl.load(b_ptrs, mask=kmask[:, None] & cmask[None, :], other=0.0)
            bu = tl.load(b_ptrs + I, mask=kmask[:, None] & cmask[None, :], other=0.0)
        elif NM:
            a = tl.load(a_ptrs)
            bg = tl.load(b_ptrs, mask=cmask[None, :], other=0.0)
            bu = tl.load(b_ptrs + I, mask=cmask[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs)
            bg = tl.load(b_ptrs)
            bu = tl.load(b_ptrs + I)
        accg = tl.dot(a, bg, accg)
        accu = tl.dot(a, bu, accu)
        a_ptrs += BK
        b_ptrs += BK * 2 * I
    g = accg.to(tl.bfloat16)
    u = accu.to(tl.bfloat16)
    rows64 = rows.to(tl.int64)
    smask = rmask[:, None] & cmask[None, :]
    if STORE_GU:
        gmask = smask & (rows[:, None] < gu_limit)
        gp = gu_ptr + rows64[:, None] * (2 * I) + cols[None, :]
        tl.store(gp, g, mask=gmask)
        tl.store(gp + I, u, mask=gmask)
    if STORE_H:
        gf = g.to(tl.float32)
        uf = u.to(tl.float32)
        h = gf * tl.sigmoid(gf) * uf
        tl.store(h_ptr + rows64[:, None] * I + cols[None, :], h.to(tl.bfloat16), mask=smask)


@triton.jit
def _down_kernel(h_ptr, w_ptr, blk_ptr, p_ptr, z_ptr, D, I,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 KM: tl.constexpr, NM: tl.constexpr):
    # z[r, :] = probs[r] * (h[r, :] W_down[e])
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(blk_ptr + 3 * pb).to(tl.int64)
    lo = tl.load(blk_ptr + 3 * pb + 1)
    hi = tl.load(blk_ptr + 3 * pb + 2)
    rows = lo + tl.arange(0, BM)
    rmask = rows < hi
    rows_c = tl.minimum(rows, hi - 1).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < D
    ks = tl.arange(0, BK)
    a_ptrs = h_ptr + rows_c[:, None] * I + ks[None, :]
    b_ptrs = w_ptr + e * I * D + ks[:, None].to(tl.int64) * D + cols[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, I, BK):
        if KM:
            kmask = (k + ks) < I
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=kmask[:, None] & cmask[None, :], other=0.0)
        elif NM:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, mask=cmask[None, :], other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * D
    p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
    z = acc * p[:, None]
    tl.store(z_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], z.to(tl.bfloat16),
             mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _dh_kernel(go_ptr, tok_ptr, w_ptr, blk_ptr, out_ptr, D, I,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               KM: tl.constexpr, NM: tl.constexpr):
    # dh'[r, i] = sum_d go[tok[r], d] * W_down[e, i, d]
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(blk_ptr + 3 * pb).to(tl.int64)
    lo = tl.load(blk_ptr + 3 * pb + 1)
    hi = tl.load(blk_ptr + 3 * pb + 2)
    rows = lo + tl.arange(0, BM)
    rmask = rows < hi
    tok = tl.load(tok_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < I
    ks = tl.arange(0, BK)
    a_ptrs = go_ptr + tok[:, None] * D + ks[None, :]
    b_ptrs = w_ptr + e * I * D + cols[:, None].to(tl.int64) * D + ks[None, :]  # [BN, BK]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, D, BK):
        if KM:
            kmask = (k + ks) < D
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=kmask[None, :] & cmask[:, None], other=0.0)
        elif NM:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, mask=cmask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BK
        b_ptrs += BK
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * I + cols[None, :], acc.to(tl.bfloat16),
             mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _act_bwd_kernel(hp_ptr, gu_ptr, p_ptr, dgu_ptr, dp_ptr, T, I,
                    BR: tl.constexpr, BC: tl.constexpr):
    # 输入 hp 中存 dh'；输出 dg/du、dprobs，并把 hp 原地改写为 h*p
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < T
    rows64 = rows.to(tl.int64)
    p = tl.load(p_ptr + rows, mask=rm, other=0.0)
    dp = tl.zeros((BR,), dtype=tl.float32)
    for c0 in range(0, I, BC):
        cols = c0 + tl.arange(0, BC)
        m = rm[:, None] & (cols < I)[None, :]
        hpp = hp_ptr + rows64[:, None] * I + cols[None, :]
        gp = gu_ptr + rows64[:, None] * (2 * I) + cols[None, :]
        dh = tl.load(hpp, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gp, mask=m, other=0.0).to(tl.float32)
        u = tl.load(gp + I, mask=m, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        sg = g * sig
        h = sg * u
        dp += tl.sum(h * dh, axis=1)
        dh = dh * p[:, None]
        dgp = dgu_ptr + rows64[:, None] * (2 * I) + cols[None, :]
        tl.store(dgp, (dh * u * sig * (1.0 + g * (1.0 - sig))).to(tl.bfloat16), mask=m)
        tl.store(dgp + I, (dh * sg).to(tl.bfloat16), mask=m)
        tl.store(hpp, (h * p[:, None]).to(tl.bfloat16), mask=m)
    tl.store(dp_ptr + rows, dp, mask=rm)


@triton.jit
def _wgrad_kernel(a_ptr, tok_ptr, b_ptr, se_ptr, out_ptr, SA, NA, SB, NB,
                  GATHER_A: tl.constexpr,
                  BA: tl.constexpr, BB: tl.constexpr, BK: tl.constexpr,
                  AM: tl.constexpr, BMK: tl.constexpr):
    # out[e, a, b] = sum_{r in expert e} A[rowA(r), a] * B[rowB(r), b]
    # GATHER_A=True : A 按 token gather（x），B 按路由行直接索引（dgu）  -> dW_gate_up
    # GATHER_A=False: A 按路由行直接索引（h*p），B 按 token gather（go）  -> dW_down
    pa = tl.program_id(0)
    pbb = tl.program_id(1)
    e = tl.program_id(2)
    lo = tl.load(se_ptr + 2 * e)
    hi = tl.load(se_ptr + 2 * e + 1)
    acols = pa * BA + tl.arange(0, BA)
    bcols = pbb * BB + tl.arange(0, BB)
    amask = acols < NA
    bmask = bcols < NB
    kr = tl.arange(0, BK)
    acc = tl.zeros((BA, BB), dtype=tl.float32)
    for r0 in range(lo, hi, BK):
        rows = r0 + kr
        rm = rows < hi
        rows64 = rows.to(tl.int64)
        tok = tl.load(tok_ptr + rows, mask=rm, other=0).to(tl.int64)
        if GATHER_A:
            ap = a_ptr + tok[:, None] * SA + acols[None, :]
            bp = b_ptr + rows64[:, None] * SB + bcols[None, :]
            if AM:
                a = tl.load(ap, mask=amask[None, :], other=0.0)
            else:
                a = tl.load(ap)
            if BMK:
                b = tl.load(bp, mask=rm[:, None] & bmask[None, :], other=0.0)
            else:
                b = tl.load(bp, mask=rm[:, None], other=0.0)
        else:
            ap = a_ptr + rows64[:, None] * SA + acols[None, :]
            bp = b_ptr + tok[:, None] * SB + bcols[None, :]
            if AM:
                a = tl.load(ap, mask=rm[:, None] & amask[None, :], other=0.0)
            else:
                a = tl.load(ap, mask=rm[:, None], other=0.0)
            if BMK:
                b = tl.load(bp, mask=bmask[None, :], other=0.0)
            else:
                b = tl.load(bp)
        acc = tl.dot(tl.trans(a), b, acc)
    e64 = e.to(tl.int64)
    op = out_ptr + e64 * NA * NB + acols[:, None].to(tl.int64) * NB + bcols[None, :]
    tl.store(op, acc.to(tl.bfloat16), mask=amask[:, None] & bmask[None, :])


@triton.jit
def _dx_kernel(dgu_ptr, w_ptr, blk_ptr, out_ptr, D, I2,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               KM: tl.constexpr, NM: tl.constexpr):
    # dx_perm[r, d] = sum_j dgu[r, j] * W_gate_up[e, d, j]
    pb = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(blk_ptr + 3 * pb).to(tl.int64)
    lo = tl.load(blk_ptr + 3 * pb + 1)
    hi = tl.load(blk_ptr + 3 * pb + 2)
    rows = lo + tl.arange(0, BM)
    rmask = rows < hi
    rows_c = tl.minimum(rows, hi - 1).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < D
    ks = tl.arange(0, BK)
    a_ptrs = dgu_ptr + rows_c[:, None] * I2 + ks[None, :]
    b_ptrs = w_ptr + e * D * I2 + cols[:, None].to(tl.int64) * I2 + ks[None, :]  # [BN, BK]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, I2, BK):
        if KM:
            kmask = (k + ks) < I2
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=kmask[None, :] & cmask[:, None], other=0.0)
        elif NM:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, mask=cmask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BK
        b_ptrs += BK
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], acc.to(tl.bfloat16),
             mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _scatter_kernel(src_ptr, order_ptr, ptr_ptr, out_ptr, N, D,
                    BT: tl.constexpr, BD: tl.constexpr, DM: tl.constexpr):
    # out[n, :] = sum_{r: tok[r]==n} src[r, :]   (CSR: order / ptr)
    pt = tl.program_id(0)
    pd = tl.program_id(1)
    n = pt * BT + tl.arange(0, BT)
    nm = n < N
    st = tl.load(ptr_ptr + n, mask=nm, other=0)
    en = tl.load(ptr_ptr + n + 1, mask=nm, other=0)
    cnt = tl.max(en - st, axis=0)
    cols = pd * BD + tl.arange(0, BD)
    cm = cols < D
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for j in range(0, cnt):
        idx = st + j
        m = idx < en
        r = tl.load(order_ptr + idx, mask=m, other=0)
        if DM:
            v = tl.load(src_ptr + r[:, None] * D + cols[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        else:
            v = tl.load(src_ptr + r[:, None] * D + cols[None, :], mask=m[:, None], other=0.0)
        acc += v.to(tl.float32)
    tl.store(out_ptr + n.to(tl.int64)[:, None] * D + cols[None, :], acc.to(tl.bfloat16),
             mask=nm[:, None] & cm[None, :])


# ----------------------------------------------------------------------------
# 调度配置
# ----------------------------------------------------------------------------
# 每阶段配置: (BM/BA, BN/BB, BK, num_warps, num_stages)
_CFG = {
    "gu": (64, 128, 32, 4, 3),
    "down": (64, 256, 32, 4, 4),
    "dh": (64, 256, 32, 4, 4),
    "dx": (128, 256, 64, 8, 3),
    "dwd": (128, 128, 32, 4, 3),
    "dwgu": (128, 128, 32, 4, 3),
}
_CFG_SC = (16, 256, 4)
_CFG_ACT = (8, 256, 4)


def _cdiv(a, b):
    return (a + b - 1) // b


def _pick_cfg(ms):
    return _CFG


def _build_blocks(ms, BM, row_from=0):
    blk = []
    s = 0
    for e, m in enumerate(ms):
        end = s + m
        lo0 = s if s > row_from else row_from
        for lo in range(lo0, end, BM):
            blk.append(e)
            blk.append(lo)
            blk.append(lo + BM if lo + BM < end else end)
        s = end
    return blk


def _run_gu(cfg, x, tok, w, blk_t, nblk, h, gu, gu_limit, D, I, store_h, store_gu):
    BM, BN, BK, nw, ns = cfg
    _gu_kernel[(nblk, _cdiv(I, BN))](
        x, tok, w, blk_t, h, gu, gu_limit, D, I,
        BM=BM, BN=BN, BK=BK, KM=(D % BK != 0), NM=(I % BN != 0),
        STORE_H=store_h, STORE_GU=store_gu, num_warps=nw, num_stages=ns)


def _run_scatter(src, order, ptr, out, N, D):
    BT, BD, nw = _CFG_SC
    _scatter_kernel[(_cdiv(N, BT), _cdiv(D, BD))](
        src, order, ptr, out, N, D, BT=BT, BD=BD, DM=(D % BD != 0), num_warps=nw)


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        N, D = x.shape
        E = w_gate_up.shape[0]
        I = w_down.shape[1]
        ms = m_sizes.tolist()
        T = token_ids.shape[0]
        dev = x.device
        out = torch.empty((N, D), dtype=torch.bfloat16, device=dev)
        ctx.dims = (N, D, E, I, T)
        if T == 0:
            out.zero_()
            return out
        _tick('start')
        cfg = _pick_cfg(ms)
        # 一次上传: [start,end]*E + 各 BM 的行块表
        lst = []
        s = 0
        for m in ms:
            lst.append(s)
            s += m
            lst.append(s)
        offs = {}
        for key in ("gu", "down", "dh", "dx"):
            bm = cfg[key][0]
            if bm not in offs:
                blk = _build_blocks(ms, bm)
                offs[bm] = (len(lst), len(blk) // 3)
                lst.extend(blk)
        tab = torch.tensor(lst, dtype=torch.int32, device=dev)
        tabs = {bm: (tab[o:], n) for bm, (o, n) in offs.items()}

        # CSR: token -> 路由行
        sorted_tok, order = torch.sort(token_ids)
        ptr = torch.searchsorted(sorted_tok, torch.arange(N + 1, dtype=torch.int32, device=dev))
        _tick('prep')

        # g/u 保存预算
        R = min(T, _GU_BUDGET_BYTES // (4 * I))
        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        blk_t, nblk = tabs[cfg["gu"][0]]
        if R > 0:
            gu = torch.empty((R, 2 * I), dtype=torch.bfloat16, device=dev)
            _run_gu(cfg["gu"], x, token_ids, w_gate_up, blk_t, nblk, h, gu, R, D, I, True, True)
        else:
            gu = None
            _run_gu(cfg["gu"], x, token_ids, w_gate_up, blk_t, nblk, h, h, 0, D, I, True, False)
        _tick('gu')

        z = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
        BM, BN, BK, nw, ns = cfg["down"]
        blk_t, nblk = tabs[BM]
        _down_kernel[(nblk, _cdiv(D, BN))](
            h, w_down, blk_t, probs, z, D, I,
            BM=BM, BN=BN, BK=BK, KM=(I % BK != 0), NM=(D % BN != 0),
            num_warps=nw, num_stages=ns)
        _tick('down')
        _run_scatter(z, order, ptr, out, N, D)
        _tick('scat')

        ctx.save_for_backward(x, w_gate_up, w_down, probs)
        ctx.tok = token_ids
        ctx.ms = ms
        ctx.cfg = cfg
        ctx.tab = tab
        ctx.tabs = tabs
        ctx.order = order
        ctx.ptr = ptr
        ctx.gu = gu
        ctx.R = R
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, w_gate_up, w_down, probs = ctx.saved_tensors
        N, D, E, I, T = ctx.dims
        dev = x.device
        need_x, need_wgu, need_wd, need_p = ctx.needs_input_grad[:4]
        if T == 0:
            return (
                torch.zeros_like(x) if need_x else None,
                torch.zeros_like(w_gate_up) if need_wgu else None,
                torch.zeros_like(w_down) if need_wd else None,
                torch.zeros_like(probs) if need_p else None,
                None, None,
            )
        go = grad_out
        if go.dtype != torch.bfloat16:
            go = go.to(torch.bfloat16)
        go = go.contiguous()
        tok = ctx.tok
        ms = ctx.ms
        cfg = ctx.cfg
        tab = ctx.tab
        tabs = ctx.tabs
        R = ctx.R
        _tick('bstart')

        # 取回 / 重算 g,u
        if R >= T:
            gu = ctx.gu
        else:
            gu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
            if R > 0:
                gu[:R].copy_(ctx.gu)
            blk2 = _build_blocks(ms, cfg["gu"][0], row_from=R)
            blk2_t = torch.tensor(blk2, dtype=torch.int32, device=dev)
            _run_gu(cfg["gu"], x, tok, w_gate_up, blk2_t, len(blk2) // 3, gu, gu, T, D, I, False, True)
        ctx.gu = None
        _tick('regu')

        hp = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        BM, BN, BK, nw, ns = cfg["dh"]
        blk_t, nblk = tabs[BM]
        _dh_kernel[(nblk, _cdiv(I, BN))](
            go, tok, w_down, blk_t, hp, D, I,
            BM=BM, BN=BN, BK=BK, KM=(D % BK != 0), NM=(I % BN != 0),
            num_warps=nw, num_stages=ns)
        _tick('dh')
        dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        dprobs = torch.empty((T,), dtype=torch.float32, device=dev)
        BR, BC, nwa = _CFG_ACT
        _act_bwd_kernel[(_cdiv(T, BR),)](hp, gu, probs, dgu, dprobs, T, I, BR=BR, BC=BC, num_warps=nwa)
        del gu
        _tick('act')

        dwd = None
        if need_wd:
            BA, BB, BKW, nww, nsw = cfg["dwd"]
            dwd = torch.empty((E, I, D), dtype=torch.bfloat16, device=dev)
            _wgrad_kernel[(_cdiv(I, BA), _cdiv(D, BB), E)](
                hp, tok, go, tab, dwd, I, I, D, D,
                GATHER_A=False, BA=BA, BB=BB, BK=BKW,
                AM=(I % BA != 0), BMK=(D % BB != 0), num_warps=nww, num_stages=nsw)
        _tick('dwd')
        dwgu = None
        if need_wgu:
            BA, BB, BKW, nww, nsw = cfg["dwgu"]
            dwgu = torch.empty((E, D, 2 * I), dtype=torch.bfloat16, device=dev)
            _wgrad_kernel[(_cdiv(D, BA), _cdiv(2 * I, BB), E)](
                x, tok, dgu, tab, dwgu, D, D, 2 * I, 2 * I,
                GATHER_A=True, BA=BA, BB=BB, BK=BKW,
                AM=(D % BA != 0), BMK=((2 * I) % BB != 0), num_warps=nww, num_stages=nsw)
        _tick('dwgu')
        dx = None
        if need_x:
            BM, BN, BK, nw, ns = cfg["dx"]
            blk_t, nblk = tabs[BM]
            dxp = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
            _dx_kernel[(nblk, _cdiv(D, BN))](
                dgu, w_gate_up, blk_t, dxp, D, 2 * I,
                BM=BM, BN=BN, BK=BK, KM=((2 * I) % BK != 0), NM=(D % BN != 0),
                num_warps=nw, num_stages=ns)
            _tick('dx')
            dx = torch.empty((N, D), dtype=torch.bfloat16, device=dev)
            _run_scatter(dxp, ctx.order, ctx.ptr, dx, N, D)
        _tick('scat2')
        return dx, dwgu, dwd, (dprobs if need_p else None), None, None


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)
