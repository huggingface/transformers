"""Batch-1 4-bit affine Metal kernels (MLX packing) for the decode shapes MLX's kernels handle poorly: matvecs with a
short side (hyper-connection mixes 320 x 10240 / 10240 x 320, shared experts 640 x 2560: 1.3-5.7x faster on M3 Max)
and the experts' short-row down projections with their routing sum."""

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Batch-1 4-bit matvec for matrices with a short side: y[n] = sum_k (scale[n, g] * q[n, k] + bias[n, g]) * x[k].
// W: [N, K / 8] uint32 (8 nibbles, low first) read as uint4 blocks of 32 values (GS >= 32: one scale per block).
// LPR lanes per row (16 or 32), 32 / LPR rows per simdgroup; lanes stride over the row's blocks, then shuffle-reduce.
template <typename T, int GS, int LPR>
kernel void qmv4(device const uint4* w, device const T* sc, device const T* bi, device const T* x, device T* y,
                 constant int& N, constant int& K, uint gid [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
    int n = gid / LPR, t = lane % LPR;
    int blocks = K / 32, groups = K / GS;
    float acc = 0.0f;
    if (n < N) {
        device const vec<T, 4>* xv = (device const vec<T, 4>*)x;
        for (int blk = t; blk < blocks; blk += LPR) {
            uint4 q = w[(long)n * blocks + blk];
            float dot = 0.0f, xsum = 0.0f;
            for (int word = 0; word < 4; ++word) {
                float4 x0 = float4(xv[blk * 8 + word * 2]), x1 = float4(xv[blk * 8 + word * 2 + 1]);
                uint qw = q[word];
                float4 q0 = float4(qw & 0xF, (qw >> 4) & 0xF, (qw >> 8) & 0xF, (qw >> 12) & 0xF);
                float4 q1 = float4((qw >> 16) & 0xF, (qw >> 20) & 0xF, (qw >> 24) & 0xF, (qw >> 28) & 0xF);
                dot += metal::dot(q0, x0) + metal::dot(q1, x1);
                xsum += x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
            }
            int g = blk * 32 / GS;
            acc += float(sc[(long)n * groups + g]) * dot + float(bi[(long)n * groups + g]) * xsum;
        }
    }
    for (int off = LPR / 2; off > 0; off /= 2) acc += simd_shuffle_down(acc, off);
    if (t == 0 && n < N) y[n] = T(acc);
}

#define QMV(T, GS, L) \
template [[host_name("qmv4_" #T "_" #GS "_" #L)]] kernel void qmv4<T, GS, L>(device const uint4*, device const T*, \
    device const T*, device const T*, device T*, constant int&, constant int&, uint, uint);
#define QMV_ALL(L) QMV(float, 32, L) QMV(half, 32, L) QMV(bfloat, 32, L) QMV(float, 64, L) QMV(half, 64, L) QMV(bfloat, 64, L)
QMV_ALL(16) QMV_ALL(32)

// Experts decode, down projection + routing: y[h] = sum_r wt[r] * (down[e_r, h] . act[r]). Two simdgroups (64 lanes)
// per row h stride over the (route, uint4 block of 32 inputs) pairs: the rows are short (I inputs), the routes fill
// lanes (M3 Max: 239 GB/s vs 192 with one simdgroup per row).
template <typename T, int GS>
kernel void experts_down_sum(device const uint4* w, device const T* sc, device const T* bi, device const T* act,
                             device const int* ids, device const float* wt, device T* y, constant int& H,
                             constant int& I, constant int& R, uint gid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int LANES = 64;
    threadgroup float partial[16];
    int h = gid / LANES, t = gid % LANES;
    int blocks = I / 32, groups = I / GS, total = R * blocks;
    float acc = 0.0f;
    if (h < H) {
        for (int p = t; p < total; p += LANES) {
            int r = p / blocks, blk = p % blocks;
            long row = (long)ids[r] * H + h;
            uint4 q = w[row * blocks + blk];
            device const vec<T, 4>* xr = (device const vec<T, 4>*)(act + r * I + blk * 32);
            float dot = 0.0f, xsum = 0.0f;
            for (int word = 0; word < 4; ++word) {
                float4 x0 = float4(xr[word * 2]), x1 = float4(xr[word * 2 + 1]);
                uint qw = q[word];
                float4 q0 = float4(qw & 0xF, (qw >> 4) & 0xF, (qw >> 8) & 0xF, (qw >> 12) & 0xF);
                float4 q1 = float4((qw >> 16) & 0xF, (qw >> 20) & 0xF, (qw >> 24) & 0xF, (qw >> 28) & 0xF);
                dot += metal::dot(q0, x0) + metal::dot(q1, x1);
                xsum += x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
            }
            int g = blk * 32 / GS;
            acc += wt[r] * (float(sc[row * groups + g]) * dot + float(bi[row * groups + g]) * xsum);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0 && h < H) y[h] = T(partial[sg] + partial[sg + 1]);
}

#define DOWN(T, GS) \
template [[host_name("experts_down_sum_" #T "_" #GS)]] kernel void experts_down_sum<T, GS>(device const uint4*, \
    device const T*, device const T*, device const T*, device const int*, device const float*, device T*, \
    constant int&, constant int&, constant int&, uint, uint, uint);
DOWN(float, 32) DOWN(half, 32) DOWN(bfloat, 32) DOWN(float, 64) DOWN(half, 64) DOWN(bfloat, 64)
"""

_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


@torch.library.custom_op("transformers::metal_qmv4", mutates_args=())
def qmv4(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group_size: int
) -> torch.Tensor:
    """`x` [..., K] holding a single row, `weight` [N, K / 8] uint32 -> [..., N]."""
    rows, words = weight.shape
    lanes = 16 if words // 4 <= 16 else 32  # lanes per row: rows of <= 16 blocks (K <= 512) share a simdgroup
    out = torch.empty(*x.shape[:-1], rows, dtype=x.dtype, device=x.device)
    getattr(_lib(), f"qmv4_{_NAMES[x.dtype]}_{group_size}_{lanes}")(
        weight, scales, biases, x.contiguous(), out, rows, words * 8,
        threads=-(-rows * lanes // 256) * 256, group_size=256,
    )  # fmt: skip
    return out


@qmv4.register_fake
def _(x, weight, scales, biases, group_size):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("transformers::metal_experts_down_sum", mutates_args=())
def experts_down_sum(
    act: torch.Tensor, down: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, ids: torch.Tensor,
    weights: torch.Tensor, group_size: int,
) -> torch.Tensor:  # fmt: skip
    """One token's routed down projections and their routing-weighted sum: `act` [R, I] contiguous, `down`
    [E, H, I / 8] uint32, `ids` int32 / `weights` float32 [R] -> [1, H]."""
    hidden, inter, routes = down.shape[-2], act.shape[-1], ids.numel()
    out = torch.empty(1, hidden, dtype=act.dtype, device=act.device)
    getattr(_lib(), f"experts_down_sum_{_NAMES[act.dtype]}_{group_size}")(
        down, scales, biases, act, ids, weights, out,
        hidden, inter, routes, threads=hidden * 64, group_size=64,
    )  # fmt: skip
    return out


@experts_down_sum.register_fake
def _(act, down, scales, biases, ids, weights, group_size):
    return act.new_empty(1, down.shape[-2])
