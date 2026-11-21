"""Hyper-connection (`GatedResidual`) decode in 3 dispatches on Metal instead of ~5: the grouped RMS norm (compiled),
then K1 = down + inject rows (one 4-bit matvec) with their activations applied in-kernel, K2 = up + sigmoid + gated
mean over the hc streams. Batch 1, one token, 4-bit g32 weights."""

import torch


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline float qdot(uint4 q, device const vec<bfloat, 4>* xr, thread float& xsum) {
    float dot = 0.0f;
    for (int word = 0; word < 4; ++word) {
        float4 x0 = float4(xr[word * 2]), x1 = float4(xr[word * 2 + 1]);
        uint qw = q[word];
        float4 q0 = float4(qw & 0xF, (qw >> 4) & 0xF, (qw >> 8) & 0xF, (qw >> 12) & 0xF);
        float4 q1 = float4((qw >> 16) & 0xF, (qw >> 20) & 0xF, (qw >> 24) & 0xF, (qw >> 28) & 0xF);
        dot += metal::dot(q0, x0) + metal::dot(q1, x1);
        xsum += x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
    }
    return dot;
}

// K1: row n of [down (L rows); inject (HC rows)] . x_norm, 32 lanes per row; a[n] = silu(y / HC) for down rows,
// inj[n - L] = 2 sigmoid(y / HC) for inject rows (each rounded through bf16 like the torch module).
kernel void hyper_pre(device const uint4* w, device const bfloat* sc, device const bfloat* bi, device const bfloat* x,
                      device bfloat* a, device bfloat* inj, constant int& ROWS, constant int& K, constant int& L,
                      constant int& HC, uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    int n = gid / 32, blocks = K / 32, groups = K / 32;
    float acc = 0.0f;
    if (n < ROWS) {
        for (int blk = lane; blk < blocks; blk += 32) {
            float xsum = 0.0f;
            float dot = qdot(w[(long)n * blocks + blk], (device const vec<bfloat, 4>*)(x + blk * 32), xsum);
            acc += float(sc[(long)n * groups + blk]) * dot + float(bi[(long)n * groups + blk]) * xsum;
        }
    }
    acc = simd_sum(acc);
    if (lane == 0 && n < ROWS) {
        float z = float(bfloat(float(bfloat(acc)) / HC));
        if (n < L) a[n] = bfloat(z / (1.0f + exp(-z)));
        else inj[n - L] = bfloat(2.0f * float(bfloat(1.0f / (1.0f + exp(-z)))));
    }
}

// K2: one simdgroup per column c: mixed[c] = mean_j bf16(sigmoid(up[j * H + c] . a)) * x_norm[j * H + c].
kernel void hyper_mix(device const uint4* w, device const bfloat* sc, device const bfloat* bi, device const bfloat* a,
                      device const bfloat* x, device bfloat* mixed, constant int& H, constant int& L, constant int& HC,
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    int c = gid / 32;
    if (c >= H) return;
    int blocks = L / 32, groups = L / 32;
    float out = 0.0f;
    for (int j = 0; j < HC; ++j) {
        long row = (long)j * H + c;
        float acc = 0.0f;
        for (int blk = lane; blk < blocks; blk += 32) {
            float xsum = 0.0f;
            float dot = qdot(w[row * blocks + blk], (device const vec<bfloat, 4>*)(a + blk * 32), xsum);
            acc += float(sc[row * groups + blk]) * dot + float(bi[row * groups + blk]) * xsum;
        }
        acc = simd_sum(acc);
        float gate = float(bfloat(1.0f / (1.0f + exp(-float(bfloat(acc))))));
        out += float(bfloat(gate * float(x[row])));
    }
    if (lane == 0) mixed[c] = bfloat(out / HC);
}
"""
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


@torch.library.custom_op("bench::hyper_pre", mutates_args=())
def hyper_pre(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, lowrank: int, hc: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, words = w.shape
    a = torch.empty(lowrank, dtype=x.dtype, device=x.device)
    inj = torch.empty(max(rows - lowrank, 1), dtype=x.dtype, device=x.device)
    _lib().hyper_pre(w, s, b, x, a, inj, rows, words * 8, lowrank, hc, threads=rows * 32, group_size=256)
    return a, inj


@hyper_pre.register_fake
def _(x, w, s, b, lowrank, hc):
    return x.new_empty(lowrank), x.new_empty(max(w.shape[0] - lowrank, 1))


@torch.library.custom_op("bench::hyper_mix", mutates_args=())
def hyper_mix(
    a: torch.Tensor, w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, x: torch.Tensor, hc: int
) -> torch.Tensor:
    hidden = w.shape[0] // hc
    mixed = torch.empty(hidden, dtype=x.dtype, device=x.device)
    _lib().hyper_mix(w, s, b, a, x, mixed, hidden, a.shape[-1], hc, threads=hidden * 32, group_size=256)
    return mixed


@hyper_mix.register_fake
def _(a, w, s, b, x, hc):
    return x.new_empty(w.shape[0] // hc)


def install(model):
    """Batch-1 decode of every `GatedResidual` with 4-bit g32 projections (fused or not) via `hyper_pre` + `hyper_mix`."""
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen4ExpTextGatedResidual":
            continue
        down, up, inj = module.input_mix_weight_down, module.input_mix_weight_up, module.block_inject_weight
        down_linear = getattr(down, "fused", down)  # a `_SiblingSlice` holds the fused down + inject rows
        if getattr(down_linear, "group_size", None) != 32 or getattr(up, "group_size", None) != 32:
            continue
        pre = (down_linear.weight, down_linear.scales, down_linear.qbiases)
        original = module.forward

        def forward(hyper_input, module=module, pre=pre, up=up, original=original, combine=inj is not None):
            if hyper_input.numel() != module.hc_count * module.hidden_size:
                return original(hyper_input)
            normed = module.hc_norm(hyper_input).reshape(-1).contiguous()
            a, inject = hyper_pre(normed, *pre, up.in_features, module.hc_count)
            mixed = hyper_mix(a, up.weight, up.scales, up.qbiases, normed, module.hc_count)
            mixed = mixed.view(*hyper_input.shape[:-1], module.hidden_size)
            if not combine:
                return mixed
            return mixed, hyper_input, inject.view(*hyper_input.shape[:-1], module.hc_count)

        module.forward = forward
        count += 1
    return count
