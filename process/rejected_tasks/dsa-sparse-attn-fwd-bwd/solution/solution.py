"""Reference solution: GQA Tensor-Core Contraction + Key-Inverted CSR Triton DSA (Fwd + Bwd).

Key mechanisms:
1. GQA Tensor-Core packing: all `G = H_q // H_kv` query heads at `(b, h_kv, t)`
   share `topk_idx[b, h_kv, t, :]`. Packing those `G` heads into `[BLOCK_G >= 16, D]`
   turns token-sparse vector gathers into `tl.dot` Tensor Core tiles while sharing
   every `K`/`V` gather `G`-ways across HBM.
2. Early block skipping and compact `LSE` state: skips `BLOCK_KS = 64` index blocks
   that are entirely `-1` padding on early causal positions (`t < K_sel`), saving only
   `out` and per-`(b, h_q, t)` `LSE` (`<= 1 MB`) for backward recomputation.
3. Two-stage Tensor-Core contraction + Key-Inverted CSR reduction in backward:
   - Query-parallel Tensor Core pass computes `dQ` directly and contracts `dK_edge`,
     `dV_edge` across all `G` query heads in registers (`dS^T @ Q_grp`, `P^T @ dO_grp`),
     writing contiguous edge rows with zero elementwise atomics.
   - Inverted key-to-edge CSR schedule gathers and reduces all attending edge rows per
     key `(b, h_kv, s)` in registers, avoiding billions of scalar L2 atomic adds and
     eliminating contention on shared prefix sink and anchor keys.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    out_ptr,
    lse_ptr,
    H_q,
    H_kv,
    G,
    T,
    K_sel,
    sm_scale,
    BLOCK_G: tl.constexpr,
    BLOCK_KS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bhk = tl.program_id(1)
    b = pid_bhk // H_kv
    hk = pid_bhk % H_kv

    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    mask_g = offs_g < G

    q_head = b * H_q + hk * G + offs_g
    q_ptrs = q_ptr + (q_head[:, None].to(tl.int64) * T + pid_t) * BLOCK_D + offs_d[None, :]
    q_grp = tl.load(q_ptrs, mask=mask_g[:, None], other=0.0)

    kv_base = ((b * H_kv + hk).to(tl.int64) * T) * BLOCK_D
    k_bh_ptr = k_ptr + kv_base
    v_bh_ptr = v_ptr + kv_base
    idx_base = idx_ptr + ((b * H_kv + hk).to(tl.int64) * T + pid_t) * K_sel

    m_i = tl.full((BLOCK_G,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)

    for ks0 in range(0, K_sel, BLOCK_KS):
        offs_ks = ks0 + tl.arange(0, BLOCK_KS)
        mask_ks = offs_ks < K_sel
        idx = tl.load(idx_base + offs_ks, mask=mask_ks, other=-1)
        valid = mask_ks & (idx >= 0) & (idx <= pid_t)

        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            safe_idx = tl.where(valid, idx, 0).to(tl.int64)
            kv_offs = safe_idx[:, None] * BLOCK_D + offs_d[None, :]
            k_tile = tl.load(k_bh_ptr + kv_offs, mask=valid[:, None], other=0.0)
            v_tile = tl.load(v_bh_ptr + kv_offs, mask=valid[:, None], other=0.0)

            s_ij = tl.dot(q_grp, tl.trans(k_tile)) * sm_scale
            s_ij = tl.where(valid[None, :], s_ij, -float("inf"))

            m_ij = tl.max(s_ij, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_new_safe))
            p = tl.where(valid[None, :], tl.exp(s_ij - m_new_safe[:, None]), 0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out_grp = acc / tl.where(l_i[:, None] > 0.0, l_i[:, None], 1.0)
    lse = m_i + tl.log(tl.where(l_i > 0.0, l_i, 1.0))

    out_ptrs = out_ptr + (q_head[:, None].to(tl.int64) * T + pid_t) * BLOCK_D + offs_d[None, :]
    tl.store(out_ptrs, out_grp.to(tl.bfloat16), mask=mask_g[:, None])
    tl.store(lse_ptr + q_head.to(tl.int64) * T + pid_t, lse, mask=mask_g)


@triton.jit
def _dsa_bwd_delta_kernel(
    out_ptr,
    do_ptr,
    delta_ptr,
    total_rows,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= total_rows:
        return
    offs_d = tl.arange(0, BLOCK_D)
    base = row.to(tl.int64) * BLOCK_D + offs_d
    o = tl.load(out_ptr + base).to(tl.float32)
    do = tl.load(do_ptr + base).to(tl.float32)
    tl.store(delta_ptr + row, tl.sum(o * do, axis=0))


@triton.jit
def _build_csr_counts_kernel(
    idx_ptr,
    counts_ptr,
    slots_ptr,
    bhk_start,
    T,
    K_sel,
    BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    tk = T * K_sel
    pos = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = pos < tk

    bhk = (bhk_start + pid_c).to(tl.int64)
    c64 = pid_c.to(tl.int64)
    t = pos // K_sel

    s = tl.load(idx_ptr + bhk * tk + pos, mask=mask_p, other=-1)
    valid = mask_p & (s >= 0) & (s <= t)
    safe_s1 = tl.where(valid, s + 1, 0)

    slot = tl.atomic_add(counts_ptr + c64 * (T + 1) + safe_s1, 1, mask=valid)
    tl.store(slots_ptr + c64 * tk + pos, tl.where(valid, slot, -1), mask=mask_p)


@triton.jit
def _scatter_csr_edges_kernel(
    idx_ptr,
    offsets_ptr,
    slots_ptr,
    csr_edges_ptr,
    bhk_start,
    T,
    K_sel,
    BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    tk = T * K_sel
    pos = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = pos < tk

    bhk = (bhk_start + pid_c).to(tl.int64)
    c64 = pid_c.to(tl.int64)

    slot = tl.load(slots_ptr + c64 * tk + pos, mask=mask_p, other=-1)
    valid = mask_p & (slot >= 0)
    s = tl.load(idx_ptr + bhk * tk + pos, mask=valid, other=0)
    base = tl.load(offsets_ptr + c64 * (T + 1) + s, mask=valid, other=0)
    dst = c64 * tk + (base + slot).to(tl.int64)
    tl.store(csr_edges_ptr + dst, pos, mask=valid)


@triton.jit
def _dsa_bwd_dq_tiles_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    idx_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_edges_ptr,
    dv_edges_ptr,
    bhk_start,
    H_q,
    H_kv,
    G,
    T,
    K_sel,
    sm_scale,
    BLOCK_G: tl.constexpr,
    BLOCK_KS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    bhk = bhk_start + pid_c
    b = bhk // H_kv
    hk = bhk % H_kv

    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    mask_g = offs_g < G

    q_head = b * H_q + hk * G + offs_g
    qh_t_offs = (q_head[:, None].to(tl.int64) * T + pid_t) * BLOCK_D + offs_d[None, :]

    q_grp = tl.load(q_ptr + qh_t_offs, mask=mask_g[:, None], other=0.0)
    do_grp = tl.load(do_ptr + qh_t_offs, mask=mask_g[:, None], other=0.0)
    lse = tl.load(lse_ptr + q_head.to(tl.int64) * T + pid_t, mask=mask_g, other=0.0)
    delta = tl.load(delta_ptr + q_head.to(tl.int64) * T + pid_t, mask=mask_g, other=0.0)

    kv_base = ((b * H_kv + hk).to(tl.int64) * T) * BLOCK_D
    k_bh_ptr = k_ptr + kv_base
    v_bh_ptr = v_ptr + kv_base
    idx_base = idx_ptr + ((b * H_kv + hk).to(tl.int64) * T + pid_t) * K_sel
    edge_row_base = pid_c.to(tl.int64) * (T * K_sel) + pid_t.to(tl.int64) * K_sel

    dq_acc = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)

    for ks0 in range(0, K_sel, BLOCK_KS):
        offs_ks = ks0 + tl.arange(0, BLOCK_KS)
        mask_ks = offs_ks < K_sel
        idx = tl.load(idx_base + offs_ks, mask=mask_ks, other=-1)
        valid = mask_ks & (idx >= 0) & (idx <= pid_t)

        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            safe_idx = tl.where(valid, idx, 0).to(tl.int64)
            kv_offs = safe_idx[:, None] * BLOCK_D + offs_d[None, :]
            k_tile = tl.load(k_bh_ptr + kv_offs, mask=valid[:, None], other=0.0)
            v_tile = tl.load(v_bh_ptr + kv_offs, mask=valid[:, None], other=0.0)

            s_ij = tl.dot(q_grp, tl.trans(k_tile)) * sm_scale
            s_ij_safe = tl.where(valid[None, :] & mask_g[:, None], s_ij - lse[:, None], -float("inf"))
            p = tl.where(valid[None, :] & mask_g[:, None], tl.exp(s_ij_safe), 0.0)

            dp = tl.dot(do_grp, tl.trans(v_tile))
            ds = tl.where(valid[None, :] & mask_g[:, None], p * (dp - delta[:, None]) * sm_scale, 0.0)

            p_bf16 = p.to(tl.bfloat16)
            ds_bf16 = ds.to(tl.bfloat16)

            dq_acc += tl.dot(ds_bf16, k_tile)
            dv_tile = tl.dot(tl.trans(p_bf16), do_grp)
            dk_tile = tl.dot(tl.trans(ds_bf16), q_grp)

            edge_offs = (edge_row_base + offs_ks.to(tl.int64))[:, None] * BLOCK_D + offs_d[None, :]
            tl.store(dk_edges_ptr + edge_offs, dk_tile.to(tl.bfloat16), mask=valid[:, None])
            tl.store(dv_edges_ptr + edge_offs, dv_tile.to(tl.bfloat16), mask=valid[:, None])

    tl.store(dq_ptr + qh_t_offs, dq_acc.to(tl.bfloat16), mask=mask_g[:, None])


@triton.jit
def _dsa_bwd_csr_reduce_kernel(
    dk_edges_ptr,
    dv_edges_ptr,
    offsets_ptr,
    csr_edges_ptr,
    dk_ptr,
    dv_ptr,
    bhk_start,
    T,
    K_sel,
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    c64 = pid_c.to(tl.int64)
    bhk = (bhk_start + pid_c).to(tl.int64)
    tk = T * K_sel

    start = tl.load(offsets_ptr + c64 * (T + 1) + pid_s)
    end = tl.load(offsets_ptr + c64 * (T + 1) + pid_s + 1)

    offs_d = tl.arange(0, BLOCK_D)
    dk_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    c_edge_base = c64 * tk

    for e0 in range(start, end, BLOCK_E):
        offs_e = e0 + tl.arange(0, BLOCK_E)
        mask_e = offs_e < end
        edge_pos = tl.load(csr_edges_ptr + c_edge_base + offs_e.to(tl.int64), mask=mask_e, other=0).to(tl.int64)
        row_offs = (c_edge_base + edge_pos[:, None]) * BLOCK_D + offs_d[None, :]
        dk_rows = tl.load(dk_edges_ptr + row_offs, mask=mask_e[:, None], other=0.0).to(tl.float32)
        dv_rows = tl.load(dv_edges_ptr + row_offs, mask=mask_e[:, None], other=0.0).to(tl.float32)
        dk_acc += tl.sum(dk_rows, axis=0)
        dv_acc += tl.sum(dv_rows, axis=0)

    out_offs = (bhk * T + pid_s.to(tl.int64)) * BLOCK_D + offs_d
    tl.store(dk_ptr + out_offs, dk_acc.to(tl.bfloat16))
    tl.store(dv_ptr + out_offs, dv_acc.to(tl.bfloat16))


class _DSASparseAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk_idx: torch.Tensor):
        B, H_q, T, D = q.shape
        H_kv = k.shape[1]
        K_sel = topk_idx.shape[-1]
        G = H_q // H_kv
        block_g = max(16, 1 << (G - 1).bit_length())
        sm_scale = 1.0 / math.sqrt(D)

        out = torch.empty_like(q)
        lse = torch.empty((B, H_q, T), dtype=torch.float32, device=q.device)

        _dsa_fwd_kernel[(T, B * H_kv)](
            q, k, v, topk_idx, out, lse,
            H_q, H_kv, G, T, K_sel, sm_scale,
            BLOCK_G=block_g, BLOCK_KS=64, BLOCK_D=D,
            num_warps=4, num_stages=2,
        )

        ctx.save_for_backward(q, k, v, topk_idx, out, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, topk_idx, out, lse = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        B, H_q, T, D = q.shape
        H_kv = k.shape[1]
        K_sel = topk_idx.shape[-1]
        G = H_q // H_kv
        block_g = max(16, 1 << (G - 1).bit_length())
        sm_scale = 1.0 / math.sqrt(D)

        total_q_rows = B * H_q * T
        delta = torch.empty((B, H_q, T), dtype=torch.float32, device=q.device)
        _dsa_bwd_delta_kernel[(total_q_rows,)](
            out, grad_out, delta, total_q_rows,
            BLOCK_D=D, num_warps=2,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        total_bhk = B * H_kv
        tk = T * K_sel
        bytes_per_slice = 2 * tk * D * 2
        max_work_bytes = 2 * 1024 * 1024 * 1024
        max_c = min(total_bhk, max(1, max_work_bytes // max(1, bytes_per_slice)))

        dk_edges = torch.empty((max_c, tk, D), dtype=torch.bfloat16, device=q.device)
        dv_edges = torch.empty((max_c, tk, D), dtype=torch.bfloat16, device=q.device)
        counts = torch.empty((max_c, T + 1), dtype=torch.int32, device=q.device)
        slots = torch.empty((max_c, tk), dtype=torch.int32, device=q.device)
        csr_edges = torch.empty((max_c, tk), dtype=torch.int32, device=q.device)

        grid_p = triton.cdiv(tk, 1024)
        for bhk_start in range(0, total_bhk, max_c):
            cur_c = min(max_c, total_bhk - bhk_start)
            counts[:cur_c].zero_()
            _build_csr_counts_kernel[(grid_p, cur_c)](
                topk_idx, counts, slots, bhk_start, T, K_sel,
                BLOCK_P=1024, num_warps=4,
            )
            offsets = torch.cumsum(counts[:cur_c], dim=-1, dtype=torch.int32)
            _scatter_csr_edges_kernel[(grid_p, cur_c)](
                topk_idx, offsets, slots, csr_edges, bhk_start, T, K_sel,
                BLOCK_P=1024, num_warps=4,
            )
            _dsa_bwd_dq_tiles_kernel[(T, cur_c)](
                q, k, v, grad_out, topk_idx, lse, delta,
                dq, dk_edges, dv_edges,
                bhk_start, H_q, H_kv, G, T, K_sel, sm_scale,
                BLOCK_G=block_g, BLOCK_KS=64, BLOCK_D=D,
                num_warps=4, num_stages=2,
            )
            _dsa_bwd_csr_reduce_kernel[(T, cur_c)](
                dk_edges, dv_edges, offsets, csr_edges,
                dk, dv, bhk_start, T, K_sel,
                BLOCK_E=16, BLOCK_D=D,
                num_warps=4,
            )

        return dq, dk, dv, None


def dsa_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
) -> torch.Tensor:
    return _DSASparseAttnFn.apply(q, k, v, topk_idx)
