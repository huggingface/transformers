"""Batch-1 4-bit affine matvec (MLX packing) for the shapes MLX's `qmv` handles poorly: few output rows over a long
input (e.g. 4 x 10240, 320 x 10240), where MLX gives each row too few threads. Here each row gets up to 256 threads
splitting its input, reduced with `simd_sum`."""

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// y[n] = sum_k (scale[n, g] * q[n, k] + bias[n, g]) * x[k], g = k / GS. W: [N, K / 8] uint32, 8 nibbles low first.
// A threadgroup of 256 threads covers 256 / TPR rows; a row's TPR threads stride over its words.
template <typename T, int GS>
kernel void qmv4(device const uint* w, device const T* scales, device const T* biases, device const T* x,
                 device T* y, constant int& N, constant int& K, constant int& TPR,
                 uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    int rows_per_group = 256 / TPR, local_row = tid / TPR, t = tid % TPR;
    int n = group * rows_per_group + local_row;
    int words = K / 8, groups = K / GS;
    float acc = 0.0f;
    if (n < N) {
        device const uint* row = w + (long)n * words;
        device const T* s = scales + (long)n * groups;
        device const T* b = biases + (long)n * groups;
        for (int i = t; i < words; i += TPR) {
            uint q = row[i];
            device const T* xx = x + i * 8;
            float dot = 0.0f, xsum = 0.0f;
            for (int j = 0; j < 8; ++j) {
                float xv = float(xx[j]);
                dot += float((q >> (4 * j)) & 0xF) * xv;
                xsum += xv;
            }
            int g = i * 8 / GS;
            acc += float(s[g]) * dot + float(b[g]) * xsum;
        }
    }
    acc = simd_sum(acc);
    if (TPR <= 32) {  // a row lives in one simdgroup (TPR divides 32): lanes 0, TPR, 2 TPR... hold partial sums
        // simd_sum mixed rows; redo per row with shuffles instead
    }
    if (TPR >= 32) {
        if (lane == 0) partial[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int per_row = TPR / 32;
        if (t == 0 && n < N) {
            float total = 0.0f;
            for (int i = 0; i < per_row; ++i) total += partial[local_row * per_row + i];
            y[n] = T(total);
        }
    }
}

#define QMV(T, GS) \\
template [[host_name("qmv4_" #T "_" #GS)]] kernel void qmv4<T, GS>(device const uint*, device const T*, \\
    device const T*, device const T*, device T*, constant int&, constant int&, constant int&, uint, uint, uint, uint);
QMV(float, 32) QMV(half, 32) QMV(bfloat, 32) QMV(float, 64) QMV(half, 64) QMV(bfloat, 64)

// Experts decode, down projection + routing: y[h] = sum_r wt[r] * (down[e_r, h] . act[r]). One simdgroup per row h,
// lanes stride over the (route, uint4 block of 32 inputs) pairs: the rows are short (I inputs), the routes fill lanes.
template <typename T, int GS>
kernel void experts_down_sum(device const uint4* w, device const T* sc, device const T* bi, device const T* act,
                             device const int* ids, device const float* wt, device T* y, constant int& H,
                             constant int& I, constant int& R, uint gid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    int h = gid / 32;
    if (h >= H) return;
    int blocks = I / 32, groups = I / GS, total = R * blocks;
    float acc = 0.0f;
    for (int p = lane; p < total; p += 32) {
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
    acc = simd_sum(acc);
    if (lane == 0) y[h] = T(acc);
}

#define DOWN(T, GS) \
template [[host_name("experts_down_sum_" #T "_" #GS)]] kernel void experts_down_sum<T, GS>(device const uint4*, \
    device const T*, device const T*, device const T*, device const int*, device const float*, device T*, \
    constant int&, constant int&, constant int&, uint, uint);
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
    # ponytail: threads per row >= 32 only (one or more whole simdgroups per row); rows shorter than 256 words waste lanes
    tpr = 256 if words >= 1024 else 128 if words >= 512 else 64 if words >= 256 else 32
    out = torch.empty(*x.shape[:-1], rows, dtype=x.dtype, device=x.device)
    getattr(_lib(), f"qmv4_{_NAMES[x.dtype]}_{group_size}")(
        weight, scales, biases, x.contiguous(), out, rows, words * 8, tpr,
        threads=-(-rows // (256 // tpr)) * 256, group_size=256,
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
    """One token's routed down projections and their routing-weighted sum: `act` [R, I], `down` [E, H, I / 8] uint32,
    `ids` / `weights` [R] -> [1, H]."""
    hidden, inter, routes = down.shape[-2], act.shape[-1], ids.numel()
    out = torch.empty(1, hidden, dtype=act.dtype, device=act.device)
    getattr(_lib(), f"experts_down_sum_{_NAMES[act.dtype]}_{group_size}")(
        down, scales, biases, act.contiguous(), ids.to(torch.int32).contiguous(), weights.float().contiguous(), out,
        hidden, inter, routes, threads=hidden * 32, group_size=128,
    )  # fmt: skip
    return out


@experts_down_sum.register_fake
def _(act, down, scales, biases, ids, weights, group_size):
    return act.new_empty(1, down.shape[-2])
