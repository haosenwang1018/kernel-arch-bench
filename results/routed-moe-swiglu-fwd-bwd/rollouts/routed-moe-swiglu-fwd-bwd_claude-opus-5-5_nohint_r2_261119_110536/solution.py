"""Top-K routed SwiGLU MoE FFN: forward + backward in Triton."""

import torch
import triton
import triton.language as tl

# bytes of pre-activations (g, u) kept on the autograd graph; remaining rows are recomputed
SAVE_LIMIT_BYTES = 112 * 1024 * 1024


def _cfgs(lst):
    return [triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP": 8}, num_warps=w, num_stages=s)
            for (bm, bn, bk, w, s) in lst]


GU_CFGS = _cfgs([(128, 64, 32, 4, 4), (128, 128, 32, 8, 4), (64, 64, 64, 4, 4), (64, 128, 32, 4, 4),
                 (64, 128, 64, 4, 3), (64, 128, 32, 4, 5), (64, 256, 32, 8, 3), (128, 128, 64, 8, 3)])
ROWS_CFGS = _cfgs([(128, 128, 32, 4, 4), (128, 128, 64, 4, 3), (128, 256, 64, 8, 3), (128, 256, 32, 8, 4),
                   (256, 128, 64, 8, 3), (64, 128, 32, 4, 4), (64, 128, 64, 4, 3), (64, 256, 32, 4, 4)])
BH_CFGS = _cfgs([(128, 64, 64, 8, 3), (128, 64, 64, 4, 3), (128, 128, 64, 8, 3), (64, 128, 32, 4, 4),
                 (64, 128, 64, 4, 3), (64, 64, 64, 4, 3), (64, 256, 32, 8, 3), (64, 64, 32, 4, 4)])
WG_CFGS = _cfgs([(128, 128, 32, 4, 4), (128, 128, 64, 4, 3), (128, 128, 64, 4, 4), (128, 256, 32, 8, 3),
                 (128, 256, 64, 8, 3), (256, 128, 64, 8, 3), (64, 128, 32, 4, 4)])


@triton.jit
def _locate(pid, ms_ptr, E, num_n, EP: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr):
    # map a linear program id to (expert, expert row start, expert rows, local m tile, n tile)
    offs = tl.arange(0, EP)
    m = tl.load(ms_ptr + offs, mask=offs < E, other=0)
    nt = (m + BM - 1) // BM
    cum = tl.cumsum(nt, axis=0) * num_n
    e = tl.sum((cum <= pid).to(tl.int32), axis=0)
    sel = offs == e
    m_e = tl.sum(tl.where(sel, m, 0), axis=0)
    cum_e = tl.sum(tl.where(sel, cum, 0), axis=0)
    nt_e = tl.sum(tl.where(sel, nt, 0), axis=0)
    row0 = tl.sum(tl.where(offs < e, m, 0), axis=0)
    lin = pid - (cum_e - nt_e * num_n)
    in_group = GROUP * num_n
    first_m = (lin // in_group) * GROUP
    gsz = tl.minimum(nt_e - first_m, GROUP)
    gsz = tl.maximum(gsz, 1)
    local = first_m + (lin % in_group) % gsz
    pid_n = (lin % in_group) // gsz
    return e, row0, m_e, local, pid_n


@triton.jit
def _swizzle(pid, num_m, num_n, GROUP: tl.constexpr):
    in_group = GROUP * num_n
    first_m = (pid // in_group) * GROUP
    gsz = tl.minimum(num_m - first_m, GROUP)
    pid_m = first_m + (pid % in_group) % gsz
    pid_n = (pid % in_group) // gsz
    return pid_m, pid_n


@triton.autotune(configs=GU_CFGS, key=["D", "I", "MB"])
@triton.heuristics({"EVEN_K": lambda a: a["D"] % a["BK"] == 0})
@triton.jit
def _fwd_gu_kernel(x_ptr, w_ptr, tid_ptr, ms_ptr, h_ptr, gu_ptr,
                   E, D, I, MB, R_save, R_skip,
                   EP: tl.constexpr, STORE_H: tl.constexpr, SAVE_GU: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                   EVEN_K: tl.constexpr):
    e, row0, m_e, local, pid_n = _locate(tl.program_id(0), ms_ptr, E, tl.cdiv(I, BN), EP, BM, GROUP)
    if e >= E:
        return
    if row0 + local * BM + BM <= R_skip:
        return
    rl = local * BM + tl.arange(0, BM)
    rmask = rl < m_e
    rows = row0 + rl
    tok = tl.load(tid_ptr + rows, mask=rmask, other=0)
    cols = pid_n * BN + tl.arange(0, BN)
    cmask = cols < I
    ks = tl.arange(0, BK)
    I2 = 2 * I
    a_ptrs = x_ptr + tok[:, None] * D + ks[None, :]
    b_ptrs = w_ptr + e.to(tl.int64) * D * I2 + ks[:, None] * I2 + cols[None, :]
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, D, BK):
        if EVEN_K:
            a = tl.load(a_ptrs)
            bg = tl.load(b_ptrs, mask=cmask[None, :], other=0.0)
            bu = tl.load(b_ptrs + I, mask=cmask[None, :], other=0.0)
        else:
            km = (k + ks) < D
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            bg = tl.load(b_ptrs, mask=km[:, None] & cmask[None, :], other=0.0)
            bu = tl.load(b_ptrs + I, mask=km[:, None] & cmask[None, :], other=0.0)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        a_ptrs += BK
        b_ptrs += BK * I2
    m2 = rmask[:, None] & cmask[None, :]
    if SAVE_GU:
        gp = gu_ptr + rows[:, None] * I2 + cols[None, :]
        sm = m2 & (rows < R_save)[:, None]
        tl.store(gp, acc_g.to(tl.bfloat16), mask=sm)
        tl.store(gp + I, acc_u.to(tl.bfloat16), mask=sm)
    if STORE_H:
        h = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(h_ptr + rows[:, None] * I + cols[None, :], h.to(tl.bfloat16), mask=m2)


@triton.autotune(configs=ROWS_CFGS, key=["N", "K", "MB"])
@triton.heuristics({"EVEN_K": lambda a: a["K"] % a["BK"] == 0})
@triton.jit
def _rows_gemm_kernel(a_ptr, b_ptr, c_ptr, p_ptr, ms_ptr,
                      E, N, K, MB, lda, stride_be, stride_bk, stride_bn, ldc,
                      EP: tl.constexpr, SCALE: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                      EVEN_K: tl.constexpr):
    # C[rows of expert e] = A[rows] @ B[e]  (optionally scaled by p[row])
    e, row0, m_e, local, pid_n = _locate(tl.program_id(0), ms_ptr, E, tl.cdiv(N, BN), EP, BM, GROUP)
    if e >= E:
        return
    rl = local * BM + tl.arange(0, BM)
    rmask = rl < m_e
    rows = row0 + rl
    rows_c = tl.where(rmask, rows, row0)
    cols = pid_n * BN + tl.arange(0, BN)
    cmask = cols < N
    ks = tl.arange(0, BK)
    a_ptrs = a_ptr + rows_c[:, None] * lda + ks[None, :]
    b_ptrs = b_ptr + e.to(tl.int64) * stride_be + ks[:, None] * stride_bk + cols[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, mask=cmask[None, :], other=0.0)
        else:
            km = (k + ks) < K
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=km[:, None] & cmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * stride_bk
    if SCALE:
        p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
        acc = acc * p[:, None]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], acc.to(tl.bfloat16),
             mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=BH_CFGS, key=["D", "I", "MB"], reset_to_zero=["dp_ptr"])
@triton.heuristics({"EVEN_K": lambda a: a["D"] % a["BK"] == 0})
@triton.jit
def _bwd_h_kernel(go_ptr, wd_ptr, tid_ptr, p_ptr, ms_ptr, gu_ptr,
                  dgu_ptr, h_ptr, dp_ptr,
                  E, D, I, MB,
                  EP: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                  EVEN_K: tl.constexpr):
    # dh_raw = go[tok] @ Wd[e]^T ; dprobs, dg, du, h from saved/recomputed (g, u)
    e, row0, m_e, local, pid_n = _locate(tl.program_id(0), ms_ptr, E, tl.cdiv(I, BN), EP, BM, GROUP)
    if e >= E:
        return
    rl = local * BM + tl.arange(0, BM)
    rmask = rl < m_e
    rows = row0 + rl
    tok = tl.load(tid_ptr + rows, mask=rmask, other=0)
    cols = pid_n * BN + tl.arange(0, BN)
    cmask = cols < I
    ks = tl.arange(0, BK)
    I2 = 2 * I
    go_ptrs = go_ptr + tok[:, None] * D + ks[None, :]
    wd_ptrs = wd_ptr + e.to(tl.int64) * I * D + cols[None, :] * D + ks[:, None]
    acc_dh = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, D, BK):
        if EVEN_K:
            a = tl.load(go_ptrs)
            b = tl.load(wd_ptrs, mask=cmask[None, :], other=0.0)
        else:
            km = (k + ks) < D
            a = tl.load(go_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(wd_ptrs, mask=km[:, None] & cmask[None, :], other=0.0)
        acc_dh = tl.dot(a, b, acc_dh)
        go_ptrs += BK
        wd_ptrs += BK
    m2 = rmask[:, None] & cmask[None, :]
    gp = gu_ptr + rows[:, None] * I2 + cols[None, :]
    g = tl.load(gp, mask=m2, other=0.0).to(tl.float32)
    u = tl.load(gp + I, mask=m2, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    h = silu * u
    dpp = tl.sum(acc_dh * h, axis=1)
    tl.atomic_add(dp_ptr + rows, dpp, mask=rmask)
    p = tl.load(p_ptr + rows, mask=rmask, other=0.0)
    dh = acc_dh * p[:, None]
    du = dh * silu
    dg = dh * u * sig * (1.0 + g * (1.0 - sig))
    op = dgu_ptr + rows[:, None] * I2 + cols[None, :]
    tl.store(op, dg.to(tl.bfloat16), mask=m2)
    tl.store(op + I, du.to(tl.bfloat16), mask=m2)
    tl.store(h_ptr + rows[:, None] * I + cols[None, :], h.to(tl.bfloat16), mask=m2)


@triton.jit
def _wg_step(a_ptr, b_ptr, p_ptr, tid_ptr, row0, k0, m_e, lda, ldb, rm, rn, mmask, nmask, acc,
             BK: tl.constexpr, A_GATHER: tl.constexpr, B_GATHER: tl.constexpr,
             B_SCALE: tl.constexpr, TAIL: tl.constexpr):
    r = k0 + tl.arange(0, BK)
    rows = row0 + r
    if TAIL:
        kmask = r < m_e
        rows = tl.where(kmask, rows, row0)
    if A_GATHER or B_GATHER:
        tok = tl.load(tid_ptr + rows)
    if A_GATHER:
        ar = tok
    else:
        ar = rows
    if B_GATHER:
        br = tok
    else:
        br = rows
    if TAIL:
        a = tl.load(a_ptr + ar[None, :] * lda + rm[:, None], mask=kmask[None, :] & mmask[:, None], other=0.0)
    else:
        a = tl.load(a_ptr + ar[None, :] * lda + rm[:, None], mask=mmask[:, None], other=0.0)
    b = tl.load(b_ptr + br[:, None] * ldb + rn[None, :], mask=nmask[None, :], other=0.0)
    if B_SCALE:
        pr = tl.load(p_ptr + rows)
        b = (b.to(tl.float32) * pr[:, None]).to(tl.bfloat16)
    return tl.dot(a, b, acc)


@triton.autotune(configs=WG_CFGS, key=["M", "N", "MB"])
@triton.jit
def _wgrad_kernel(a_ptr, b_ptr, c_ptr, tid_ptr, p_ptr, ms_ptr,
                  E, M, N, MB, lda, ldb,
                  EP: tl.constexpr, A_GATHER: tl.constexpr, B_GATHER: tl.constexpr, B_SCALE: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    # C[e] (M x N) = sum_{r in expert e} A_row(r)^T B_row(r)
    e = tl.program_id(1)
    pid_m, pid_n = _swizzle(tl.program_id(0), tl.cdiv(M, BM), tl.cdiv(N, BN), GROUP)
    offs = tl.arange(0, EP)
    m = tl.load(ms_ptr + offs, mask=offs < E, other=0)
    m_e = tl.sum(tl.where(offs == e, m, 0), axis=0)
    row0 = tl.sum(tl.where(offs < e, m, 0), axis=0)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_full = (m_e // BK) * BK
    for k0 in range(0, k_full, BK):
        acc = _wg_step(a_ptr, b_ptr, p_ptr, tid_ptr, row0, k0, m_e, lda, ldb, rm, rn, mmask, nmask, acc,
                       BK, A_GATHER, B_GATHER, B_SCALE, False)
    if k_full < m_e:
        acc = _wg_step(a_ptr, b_ptr, p_ptr, tid_ptr, row0, k_full, m_e, lda, ldb, rm, rn, mmask, nmask, acc,
                       BK, A_GATHER, B_GATHER, B_SCALE, True)
    cp = c_ptr + e.to(tl.int64) * M * N + rm[:, None] * N + rn[None, :]
    tl.store(cp, acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _combine_kernel(src_ptr, out_ptr, order_ptr, offs_ptr, D, BD: tl.constexpr):
    n = tl.program_id(0)
    pd = tl.program_id(1)
    start = tl.load(offs_ptr + n)
    end = tl.load(offs_ptr + n + 1)
    cols = pd * BD + tl.arange(0, BD)
    cm = cols < D
    acc = tl.zeros((BD,), dtype=tl.float32)
    for j in range(start, end):
        r = tl.load(order_ptr + j)
        acc += tl.load(src_ptr + r * D + cols, mask=cm, other=0.0).to(tl.float32)
    tl.store(out_ptr + n.to(tl.int64) * D + cols, acc.to(tl.bfloat16), mask=cm)


def _ep(E):
    return max(16, triton.next_power_of_2(E))


def _mb(T, E):
    # bucket of average rows per expert (autotune key)
    return min(4096, triton.next_power_of_2(max(1, T // E)))


def _row_grid(T, E, N):
    return lambda META: ((triton.cdiv(T, META["BM"]) + E) * triton.cdiv(N, META["BN"]),)


def _combine(src, order, offs, N_tok, D):
    out = torch.empty((N_tok, D), dtype=torch.bfloat16, device=src.device)
    BD = 512
    _combine_kernel[(N_tok, triton.cdiv(D, BD))](src, out, order, offs, D, BD=BD, num_warps=4)
    return out


def _launch_gu(x, w_gate_up, token_ids, m_sizes, h, gu, store_h, save_gu, r_save, r_skip):
    T = token_ids.shape[0]
    D = x.shape[1]
    E = w_gate_up.shape[0]
    I = w_gate_up.shape[2] // 2
    _fwd_gu_kernel[_row_grid(T, E, I)](
        x, w_gate_up, token_ids, m_sizes, h, gu, E, D, I, _mb(T, E), r_save, r_skip,
        EP=_ep(E), STORE_H=store_h, SAVE_GU=save_gu)


def _launch_rows(a, w, c, probs, m_sizes, N, K, stride_be, stride_bk, stride_bn, scale):
    T = a.shape[0]
    E = m_sizes.shape[0]
    _rows_gemm_kernel[_row_grid(T, E, N)](
        a, w, c, probs, m_sizes, E, N, K, _mb(T, E), a.stride(0), stride_be, stride_bk, stride_bn, c.stride(0),
        EP=_ep(E), SCALE=scale)


def _launch_bh(go, w_down, token_ids, probs, m_sizes, gu, dgu, h, dprobs):
    T = token_ids.shape[0]
    D = go.shape[1]
    E = w_down.shape[0]
    I = w_down.shape[1]
    _bwd_h_kernel[_row_grid(T, E, I)](
        go, w_down, token_ids, probs, m_sizes, gu, dgu, h, dprobs, E, D, I, _mb(T, E), EP=_ep(E))


def _launch_wg(a, b, c, token_ids, probs, m_sizes, M, N, a_gather, b_gather, b_scale):
    E = m_sizes.shape[0]
    T = token_ids.shape[0]
    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]), E)
    _wgrad_kernel[grid](
        a, b, c, token_ids, probs, m_sizes, E, M, N, _mb(T, E), a.stride(0), b.stride(0),
        EP=_ep(E), A_GATHER=a_gather, B_GATHER=b_gather, B_SCALE=b_scale)


class _RoutedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate_up, w_down, probs, token_ids, m_sizes):
        x = x.contiguous()
        N_tok, D = x.shape
        I = w_down.shape[1]
        T = token_ids.shape[0]
        dev = x.device

        sorted_tok, order = torch.sort(token_ids, stable=True)
        offs = torch.searchsorted(sorted_tok, torch.arange(N_tok + 1, device=dev, dtype=torch.int32)).to(torch.int32)
        del sorted_tok

        r_save = min(T, SAVE_LIMIT_BYTES // (4 * I))
        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        gu = torch.empty((r_save, 2 * I) if r_save > 0 else (1,), dtype=torch.bfloat16, device=dev)
        zw = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
        if T > 0:
            _launch_gu(x, w_gate_up, token_ids, m_sizes, h, gu, True, r_save > 0, r_save, 0)
            _launch_rows(h, w_down, zw, probs, m_sizes, D, I, I * D, D, 1, True)
        del h
        out = _combine(zw, order, offs, N_tok, D)
        del zw
        ctx.r_save = r_save
        ctx.save_for_backward(x, w_gate_up, w_down, probs, token_ids, m_sizes, order, offs, gu)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_gate_up, w_down, probs, token_ids, m_sizes, order, offs, gu = ctx.saved_tensors
        r_save = ctx.r_save
        go = grad_out.contiguous()
        N_tok, D = x.shape
        I = w_down.shape[1]
        T = token_ids.shape[0]
        dev = x.device

        dprobs = torch.zeros((T,), dtype=torch.float32, device=dev)
        dgu = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
        h = torch.empty((T, I), dtype=torch.bfloat16, device=dev)
        if T > 0:
            if r_save < T:
                gu_full = torch.empty((T, 2 * I), dtype=torch.bfloat16, device=dev)
                if r_save > 0:
                    gu_full[:r_save].copy_(gu)
                _launch_gu(x, w_gate_up, token_ids, m_sizes, gu_full, gu_full, False, True, T, r_save)
            else:
                gu_full = gu
            _launch_bh(go, w_down, token_ids, probs, m_sizes, gu_full, dgu, h, dprobs)
            del gu_full
        # dW_down[e] = h_e^T @ (p * go[tok])
        dwd = torch.empty_like(w_down)
        _launch_wg(h, go, dwd, token_ids, probs, m_sizes, I, D, False, True, True)
        del h
        # dW_gate_up[e] = x[tok]^T @ dgu_e
        dwgu = torch.empty_like(w_gate_up)
        _launch_wg(x, dgu, dwgu, token_ids, probs, m_sizes, D, 2 * I, True, False, False)
        # dx_perm = dgu @ Wgu[e]^T
        dxp = torch.empty((T, D), dtype=torch.bfloat16, device=dev)
        if T > 0:
            _launch_rows(dgu, w_gate_up, dxp, probs, m_sizes, D, 2 * I, D * 2 * I, 1, 2 * I, False)
        del dgu
        dx = _combine(dxp, order, offs, N_tok, D)
        return dx, dwgu, dwd, dprobs, None, None


def routed_moe_swiglu(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    return _RoutedMoE.apply(x, w_gate_up, w_down, probs, token_ids, m_sizes)


def _prof():
    import sys as _s
    import reference as ref
    from triton.testing import do_bench
    P = lambda *a: print(*a, file=_s.stderr, flush=True)
    for sh in [(4096, 8, 8, 4096, 7168, "uniform"), (4096, 2, 8, 4096, 7168, "uniform"), (4096, 8, 16, 1536, 4096, "zipf")]:
        x, wgu, wd, probs, tids, ms, go = ref.make_inputs(*sh, seed=1)
        def run(fn):
            x.grad = wgu.grad = wd.grad = probs.grad = None
            fn(x, wgu, wd, probs, tids, ms).backward(go)
        run(routed_moe_swiglu)
        to = do_bench(lambda: run(routed_moe_swiglu), warmup=2, rep=20)
        tb = do_bench(lambda: run(ref.routed_moe_swiglu_baseline), warmup=2, rep=20)
        P("E2E", sh, "ours %.3f base %.3f sp %.2f" % (to, tb, tb / to))
        for fn, nm in [(routed_moe_swiglu, "O"), (ref.routed_moe_swiglu_baseline, "B")]:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                run(fn)
                torch.cuda.synchronize()
            evs = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
            P(nm, " | ".join("%s:%d:%.2f" % (e.key[:28], e.count, e.self_device_time_total / 1e3) for e in evs[:9]))
        del x, wgu, wd, probs, tids, ms, go
        torch.cuda.empty_cache()
    for nm, k in [("gu", _fwd_gu_kernel), ("rows", _rows_gemm_kernel), ("bh", _bwd_h_kernel), ("wg", _wgrad_kernel)]:
        for key, c in k.cache.items():
            P(nm, key[:3], c.kwargs["BM"], c.kwargs["BN"], c.kwargs["BK"], c.num_warps, c.num_stages)


import sys as _sys
if len(_sys.argv) > 2 and _sys.argv[2] == "guard":
    _prof()
    _sys.exit(1)
