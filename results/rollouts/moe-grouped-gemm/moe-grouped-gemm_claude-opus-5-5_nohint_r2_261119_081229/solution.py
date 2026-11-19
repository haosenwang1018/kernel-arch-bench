import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bm, bn, bk, s, w in [
        (128, 128, 32, 4, 4),
        (128, 128, 64, 3, 8),
        (128, 128, 32, 5, 8),
        (128, 256, 32, 3, 8),
        (128, 256, 64, 3, 8),
        (256, 128, 32, 3, 8),
        (64, 128, 64, 4, 4),
        (64, 128, 32, 5, 4),
        (64, 256, 32, 4, 8),
        (64, 64, 64, 4, 4),
        (128, 64, 64, 4, 4),
        (32, 128, 64, 4, 4),
    ]:
        cfgs.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8},
                                  num_stages=s, num_warps=w))
    return cfgs


@triton.autotune(configs=_configs(), key=["T", "N", "K", "E", "M_BUCKET"])
@triton.heuristics({"EVEN_K": lambda a: a["K"] % a["BK"] == 0})
@triton.jit
def _grouped_gemm_kernel(
    x_ptr, w_ptr, out_ptr, m_ptr,
    T, N, K, E, M_BUCKET,
    E_PAD: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_m = tl.cdiv(T, BM) + E
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = tl.minimum(num_pid_m - first_m, GROUP_M)
    tile_m = first_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize

    # Map virtual tile index to (expert, tile within expert) on device.
    eidx = tl.arange(0, E_PAD)
    msz = tl.load(m_ptr + eidx, mask=eidx < E, other=0)
    tiles = (msz + BM - 1) // BM
    cum_tiles = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    if tile_m >= total:
        return
    expert = tl.sum((cum_tiles <= tile_m).to(tl.int32), 0)
    before = eidx < expert
    row_start = tl.sum(tl.where(before, msz, 0), 0)
    tile_start = tl.sum(tl.where(before, tiles, 0), 0)
    m_e = tl.sum(tl.where(eidx == expert, msz, 0), 0)
    local_m = (tile_m - tile_start) * BM
    row_end = row_start + m_e

    rm = row_start + local_m + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = rm < row_end
    n_mask = rn < N
    rm_safe = tl.where(m_mask, rm, row_start)
    rn_safe = tl.max_contiguous(tl.multiple_of(tl.where(n_mask, rn, 0), BN), BN)

    a_ptrs = x_ptr + rm_safe[:, None].to(tl.int64) * K + rk[None, :]
    b_base = w_ptr + expert.to(tl.int64) * K * N
    b_ptrs = b_base + rk[:, None].to(tl.int64) * N + rn_safe[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            kmask = rk < K - k * BK
            a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=kmask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    c = acc.to(tl.bfloat16)
    o_ptrs = out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :]
    tl.store(o_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def _triton_gg(x, w, m_sizes):
    T, K = x.shape
    E, _, N = w.shape
    out = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    if m_sizes.dtype != torch.int32:
        m_sizes = m_sizes.to(torch.int32)
    E_PAD = max(16, triton.next_power_of_2(E))
    # Bucket of average tokens per expert, so autotune can adapt to it.
    m_bucket = triton.next_power_of_2(max(1, T // E))
    grid = lambda meta: ((triton.cdiv(T, meta["BM"]) + E) * triton.cdiv(N, meta["BN"]),)
    _grouped_gemm_kernel[grid](x, w, out, m_sizes, T, N, K, E, m_bucket, E_PAD=E_PAD)
    return out

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldm_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldm_x4_t(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int BM, int BN, int WMW, int WNW, int STAGES, int PIPE>
__global__ void __launch_bounds__(WMW * WNW * 32)
gg_kernel(const bf16* __restrict__ X, const bf16* __restrict__ W, bf16* __restrict__ O,
          const int* __restrict__ msz, int T, int N, int K, int E, int group_m) {
    constexpr int BK = 32;
    constexpr int NT = WMW * WNW * 32;
    constexpr int WTM = BM / WMW, WTN = BN / WNW;
    constexpr int MI = WTM / 16, NI = WTN / 8;
    constexpr int A_PER = BM * BK / 8 / NT;
    constexpr int B_PER = BK * BN / 8 / NT;
    constexpr int BNC = BN / 8;
    extern __shared__ __align__(128) uint8_t smem[];
    bf16* sA = (bf16*)smem;
    bf16* sB = sA + STAGES * BM * BK;
    __shared__ int s_info[3];

    const int pid = blockIdx.x;
    const int num_pid_n = N / BN;
    const int num_pid_m = (T + BM - 1) / BM + E;
    const int num_in_group = group_m * num_pid_n;
    const int group_id = pid / num_in_group;
    const int first_m = group_id * group_m;
    const int gsize = min(num_pid_m - first_m, group_m);
    const int tile_m = first_m + (pid % num_in_group) % gsize;
    const int pid_n = (pid % num_in_group) / gsize;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (warp == 0) {
        int mm[4], tt[4];
        int lsum = 0, rsum = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int e = lane * 4 + i;
            mm[i] = e < E ? msz[e] : 0;
            tt[i] = (mm[i] + BM - 1) / BM;
            lsum += tt[i];
            rsum += mm[i];
        }
        int it = lsum, ir = rsum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int a = __shfl_up_sync(0xffffffffu, it, o);
            int b = __shfl_up_sync(0xffffffffu, ir, o);
            if (lane >= o) { it += a; ir += b; }
        }
        int total = __shfl_sync(0xffffffffu, it, 31);
        int ct = it - lsum, cr = ir - rsum;
        if (tile_m >= ct && tile_m < it) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (tile_m >= ct && tile_m < ct + tt[i]) {
                    s_info[0] = lane * 4 + i;
                    s_info[1] = cr + (tile_m - ct) * BM;
                    s_info[2] = cr + mm[i];
                }
                ct += tt[i];
                cr += mm[i];
            }
        }
        if (lane == 0 && tile_m >= total) s_info[0] = -1;
    }
    __syncthreads();
    const int expert = s_info[0];
    if (expert < 0) return;
    const int row0 = s_info[1], row_end = s_info[2];

    const bf16* a_src[A_PER];
    uint32_t a_dst[A_PER];
#pragma unroll
    for (int i = 0; i < A_PER; i++) {
        int c = tid + i * NT;
        int r = c >> 2, kc = c & 3;
        int rg = min(row0 + r, row_end - 1);
        a_src[i] = X + (int64_t)rg * K + kc * 8;
        a_dst[i] = (uint32_t)((r * BK + ((kc ^ ((r >> 1) & 3)) * 8)) * 2);
    }
    const bf16* gB = W + (int64_t)expert * K * N + (int64_t)pid_n * BN;
    const bf16* b_src[B_PER];
    uint32_t b_dst[B_PER];
#pragma unroll
    for (int i = 0; i < B_PER; i++) {
        int c = tid + i * NT;
        int kr = c / BNC, nc = c % BNC;
        b_src[i] = gB + (int64_t)kr * N + nc * 8;
        b_dst[i] = (uint32_t)((kr * BN + ((nc ^ (kr & 7)) * 8)) * 2);
    }
    const uint32_t sA0 = smem_u32(sA), sB0 = smem_u32(sB);
    const int KT = K / BK;
    const int64_t b_step = (int64_t)BK * N;

    auto load_stage = [&](int stage, int kt) {
        uint32_t ab = sA0 + stage * (BM * BK * 2);
        uint32_t bb = sB0 + stage * (BK * BN * 2);
#pragma unroll
        for (int i = 0; i < A_PER; i++) cp_async16(ab + a_dst[i], a_src[i] + kt * BK);
#pragma unroll
        for (int i = 0; i < B_PER; i++) cp_async16(bb + b_dst[i], b_src[i] + kt * b_step);
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;

    const int warp_m = warp / WNW, warp_n = warp % WNW;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load_stage(s, s);
        cp_commit();
    }

    auto load_frags = [&](uint32_t (&af)[MI][4], uint32_t (&bfr)[NI][2], int st, int kk) {
        const uint32_t ab = sA0 + st * (BM * BK * 2);
        const uint32_t bb = sB0 + st * (BK * BN * 2);
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            int r = warp_m * WTM + mi * 16 + (lane & 15);
            int kc = kk * 2 + (lane >> 4);
            ldm_x4(af[mi], ab + (uint32_t)((r * BK + ((kc ^ ((r >> 1) & 3)) * 8)) * 2));
        }
#pragma unroll
        for (int nj = 0; nj < NI / 2; nj++) {
            int kr = kk * 16 + (lane & 15);
            int nc = (warp_n * WTN + nj * 16) / 8 + (lane >> 4);
            uint32_t r4[4];
            ldm_x4_t(r4, bb + (uint32_t)((kr * BN + ((nc ^ (kr & 7)) * 8)) * 2));
            bfr[2 * nj][0] = r4[0];
            bfr[2 * nj][1] = r4[1];
            bfr[2 * nj + 1][0] = r4[2];
            bfr[2 * nj + 1][1] = r4[3];
        }
    };
    auto do_mma = [&](uint32_t (&af)[MI][4], uint32_t (&bfr)[NI][2]) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af[mi], bfr[ni]);
    };

    if (PIPE == 0) {
        for (int kt = 0; kt < KT; kt++) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            {
                int nk = kt + STAGES - 1;
                if (nk < KT) load_stage(nk % STAGES, nk);
                cp_commit();
            }
            const int st = kt % STAGES;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                uint32_t af[MI][4];
                uint32_t bfr[NI][2];
                load_frags(af, bfr, st, kk);
                do_mma(af, bfr);
            }
        }
    } else {
        uint32_t af0[MI][4], bf0[NI][2], af1[MI][4], bf1[NI][2];
        cp_wait<STAGES - 2>();
        __syncthreads();
        int rs = 0, ws = STAGES - 1;
        load_frags(af0, bf0, 0, 0);
        for (int kt = 0; kt < KT; kt++) {
            load_frags(af1, bf1, rs, 1);
            {
                int nk = kt + STAGES - 1;
                if (nk < KT) load_stage(ws, nk);
                cp_commit();
            }
            do_mma(af0, bf0);
            cp_wait<STAGES - 2>();
            __syncthreads();
            rs = (rs + 1 == STAGES) ? 0 : rs + 1;
            ws = (ws + 1 == STAGES) ? 0 : ws + 1;
            load_frags(af0, bf0, rs, 0);
            do_mma(af1, bf1);
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
        int r = row0 + warp_m * WTM + mi * 16 + (lane >> 2);
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
            int c = pid_n * BN + warp_n * WTN + ni * 8 + (lane & 3) * 2;
            if (r < row_end)
                *(__nv_bfloat162*)(O + (int64_t)r * N + c) = __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
            if (r + 8 < row_end)
                *(__nv_bfloat162*)(O + (int64_t)(r + 8) * N + c) = __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

template <int BM, int BN, int WMW, int WNW, int STAGES, int PIPE>
static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out,
                   const torch::Tensor& m, int group_m) {
    int T = x.size(0), K = x.size(1), E = w.size(0), N = w.size(2);
    constexpr int smem = STAGES * (BM * 32 + 32 * BN) * 2;
    auto kern = gg_kernel<BM, BN, WMW, WNW, STAGES, PIPE>;
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    int grid = ((T + BM - 1) / BM + E) * (N / BN);
    auto stream = at::cuda::getCurrentCUDAStream();
    kern<<<grid, WMW * WNW * 32, smem, stream>>>(
        (const bf16*)x.data_ptr(), (const bf16*)w.data_ptr(), (bf16*)out.data_ptr(),
        (const int*)m.data_ptr(), T, N, K, E, group_m);
}

void gg_launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor m,
               int64_t cfg, int64_t group_m) {
    switch (cfg) {
        case 0: launch<128, 128, 2, 2, 4, 0>(x, w, out, m, group_m); break;
        case 1: launch<128, 128, 2, 2, 4, 1>(x, w, out, m, group_m); break;
        case 2: launch<128, 128, 2, 2, 3, 1>(x, w, out, m, group_m); break;
        case 3: launch<128, 128, 2, 2, 5, 1>(x, w, out, m, group_m); break;
        case 4: launch<128, 256, 2, 4, 3, 1>(x, w, out, m, group_m); break;
        case 5: launch<64, 128, 1, 2, 4, 1>(x, w, out, m, group_m); break;
        default: launch<64, 128, 1, 2, 4, 0>(x, w, out, m, group_m); break;
    }
}
"""

_CPP_SRC = "void gg_launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor m, int64_t cfg, int64_t group_m);"

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        import os
        import sys
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        bindir = os.path.dirname(sys.executable)
        if bindir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        from torch.utils.cpp_extension import load_inline
        _ext = load_inline(
            name="moe_gg_cuda_v2",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["gg_launch"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _ext


try:
    _get_ext()
    _HAS_CUDA = True
except Exception:  # fall back to the Triton kernel
    _HAS_CUDA = False

# (cuda cfg id, group_m); None means the Triton kernel.
_CANDIDATES = [(0, 8), (1, 8), (3, 8), (4, 8), (5, 8), (6, 8), None]
_BN = {0: 128, 1: 128, 2: 128, 3: 128, 4: 256, 5: 128, 6: 128}
_choice = {}


def _run(cand, x, w, m_sizes):
    if cand is None:
        return _triton_gg(x, w, m_sizes)
    T, K = x.shape
    N = w.shape[2]
    out = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    _ext.gg_launch(x, w, out, m_sizes, cand[0], cand[1])
    return out


def _tune(key, x, w, m_sizes):
    T, K = x.shape
    E, _, N = w.shape
    cands = [c for c in _CANDIDATES if c is None or N % _BN[c[0]] == 0]
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=x.device)
    ok = []
    for c in cands:
        try:
            _run(c, x, w, m_sizes)
            ok.append(c)
        except Exception:
            pass
    torch.cuda.synchronize()
    times = {i: [] for i in range(len(ok))}
    # Round-robin over candidates so clock drift hits all of them equally.
    for _ in range(7):
        for i, c in enumerate(ok):
            flush.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _run(c, x, w, m_sizes)
            e.record()
            e.synchronize()
            times[i].append(s.elapsed_time(e))
    del flush
    best, best_t = None, float("inf")
    for i, c in enumerate(ok):
        ts = sorted(times[i])
        t = ts[len(ts) // 2]
        if t < best_t:
            best, best_t = c, t
    _choice[key] = best
    return best


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    T, K = x.shape
    E, _, N = w.shape
    if m_sizes.dtype != torch.int32:
        m_sizes = m_sizes.to(torch.int32)
    if not (_HAS_CUDA and E <= 128 and K % 32 == 0 and N % 128 == 0 and T > 0
            and x.is_contiguous() and w.is_contiguous() and m_sizes.is_contiguous()):
        return _triton_gg(x, w, m_sizes)
    key = (T, E, K, N)
    cand = _choice.get(key, 0)
    if cand == 0:
        cand = _tune(key, x, w, m_sizes)
    return _run(cand, x, w, m_sizes)
