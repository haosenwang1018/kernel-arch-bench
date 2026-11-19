import torch
import triton
import triton.language as tl


@triton.jit
def _tile(x_ptr, wb_ptr, y_ptr, row_start, m_end, n_start, K, N, T,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KMASK: tl.constexpr, EV: tl.constexpr, AFF: tl.constexpr):
    # 行下标 clamp 到本 expert 的最后一行，避免 masked load；写回时再 mask。
    if AFF:
        # 仿射行下标：整块向前平移保证不越界，多读的行在写回时被 mask 掉
        offs_m = tl.minimum(row_start, T - BM) + tl.arange(0, BM)
        offs_mc = offs_m
    else:
        offs_m = row_start + tl.arange(0, BM)
        offs_mc = tl.minimum(offs_m, m_end - 1)
    offs_n = n_start + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = x_ptr + offs_mc[:, None] * K + offs_k[None, :]
    b_ptrs = wb_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if KMASK:
        for k in range(0, tl.cdiv(K, BK)):
            km = offs_k < K - k * BK
            a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * N
    else:
        for k in range(0, K // BK):
            if EV == 1:
                a = tl.load(a_ptrs, eviction_policy="evict_last")
                b = tl.load(b_ptrs)
            elif EV == 2:
                a = tl.load(a_ptrs, eviction_policy="evict_last")
                b = tl.load(b_ptrs, eviction_policy="evict_first")
            elif EV == 3:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, eviction_policy="evict_last")
            elif EV == 4:
                a = tl.load(a_ptrs, eviction_policy="evict_last")
                b = tl.load(b_ptrs, eviction_policy="evict_last")
            else:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * N
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] >= row_start) & (offs_m[:, None] < m_end) & (offs_n[None, :] < N))


@triton.jit
def _gg_kernel(x_ptr, w_ptr, y_ptr, m_ptr, E, K, N, T, NPROG, GM,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               BMS: tl.constexpr, BNS: tl.constexpr, THR: tl.constexpr,
               KMASK: tl.constexpr, EV: tl.constexpr, AFF: tl.constexpr):
    # persistent kernel：每个 program 处理 tile 序号 pid, pid+NPROG, ...
    tidx = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    row0 = 0
    tile0 = 0
    for e in range(E):
        m = tl.load(m_ptr + e)
        if m > 0:
            m_end = row0 + m
            wb_ptr = w_ptr + (e.to(tl.int64) * K) * N
            if BMS > 0:
                nb = m // BM
                r = m - nb * BM
                ex = (r > THR).to(tl.int32)
                nb = nb + ex
                r = r * (1 - ex)
                ns = (r + BMS - 1) // BMS
            else:
                nb = (m + BM - 1) // BM
            # 大 tile：GM 个 M-tile 一组，组内按 N 列优先，保证并发的 SM 共享同一批权重列
            ntb = nb * num_n
            per_group = GM * num_n
            while tidx < tile0 + ntb:
                l = tidx - tile0
                gid = l // per_group
                first_m = gid * GM
                gsz = tl.minimum(nb - first_m, GM)
                li = l - gid * per_group
                mi = first_m + li % gsz
                ni = li // gsz
                n_start = tl.minimum(ni * BN, N - BN)
                _tile(x_ptr, wb_ptr, y_ptr, row0 + mi * BM, m_end, n_start, K, N, T,
                      BM, BN, BK, KMASK, EV, AFF)
                tidx += NPROG
            tile0 += ntb
            if BMS > 0:
                num_ns = tl.cdiv(N, BNS)
                nts = ns * num_ns
                while tidx < tile0 + nts:
                    l = tidx - tile0
                    mi = l % ns
                    ni = l // ns
                    n_start = tl.minimum(ni * BNS, N - BNS)
                    _tile(x_ptr, wb_ptr, y_ptr, row0 + nb * BM + mi * BMS, m_end, n_start,
                          K, N, T, BMS, BNS, BK, KMASK, EV, AFF)
                    tidx += NPROG
                tile0 += nts
        row0 += m


_NUM_SMS = 108

# (BM, BN, BK, warps, stages, BMS, BNS, THR, GM, MULT, EV, AFF)
_CFG_SMALL = (64, 128, 64, 4, 3, 0, 128, 0, 8, 2, 1, 0)      # expert 平均行数少：细 M 粒度
_CFG_DENSE = (128, 128, 64, 4, 3, 0, 128, 0, 16, 3, 1, 1)    # 每个 expert 行数多
_CFG_WIDE = (128, 256, 64, 8, 3, 0, 128, 0, 8, 1, 1, 1)      # 行数多且权重面板很大


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, cfg=None) -> torch.Tensor:
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=x.dtype, device=x.device)
    if T == 0 or E == 0:
        return out
    if cfg is None:
        if T < 192 * E:
            cfg = _CFG_SMALL
        elif K * N > 30000000:
            cfg = _CFG_WIDE
        else:
            cfg = _CFG_DENSE
    BM, BN, BK, W, S, BMS, BNS, THR, GM, MULT, EV, AFF = cfg
    kmask = False
    if K % BK != 0:
        if K % 32 == 0:
            BK = 32
        else:
            kmask = True
    if T < BM or (BMS > 0 and T < BMS):
        AFF = 0
    nprog = _NUM_SMS * MULT
    key = (BM, BN, BK, W, S, BMS, BNS, THR, MULT, EV, AFF, kmask)
    ent = _fast.get(key)
    if ent:
        # 快速路径：直接调用已编译 kernel 的 launcher，跳过 Triton 的 Python 分发开销
        ent[0](nprog, 1, 1, torch._C._cuda_getCurrentRawStream(x.device.index or 0), ent[1], ent[2],
               None, None, None, x, w, out, m_sizes, E, K, N, T, nprog, GM,
               BM, BN, BK, BMS, BNS, THR, kmask, EV, AFF)
        return out
    k = _gg_kernel[(nprog,)](x, w, out, m_sizes, E, K, N, T, nprog, GM,
                             BM=BM, BN=BN, BK=BK, BMS=BMS, BNS=BNS, THR=THR, KMASK=kmask, EV=EV, AFF=AFF,
                             num_warps=W, num_stages=S)
    if ent is None:
        ok = False
        try:
            run, fn, md = k.run, k.function, k.packed_metadata
            out2 = torch.empty_like(out)
            run(nprog, 1, 1, torch._C._cuda_getCurrentRawStream(x.device.index or 0), fn, md,
                None, None, None, x, w, out2, m_sizes, E, K, N, T, nprog, GM,
                BM, BN, BK, BMS, BNS, THR, kmask, EV, AFF)
            torch.cuda.synchronize()
            ok = bool(torch.equal(out, out2))
        except Exception as exc:  # noqa: BLE001
            _fast_err.append(str(exc)[-200:])
        _fast[key] = (run, fn, md) if ok else False
    return out


_fast = {}
_fast_err = []
