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
