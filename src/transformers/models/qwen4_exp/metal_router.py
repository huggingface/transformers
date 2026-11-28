"""Fused Qwen4 router: one Metal dispatch for a decode token's 512-expert softmax top-10."""

import torch


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
constant constexpr int MAX_E = 512;
template <typename T, int SPLIT>
kernel void qwen_router(device const T* x, device const float* w, device float* logits,
                        device float* out_w, device long* out_i, device atomic_uint* counter,
                        constant int& E, constant int& H, constant int& K,
                        uint tg [[threadgroup_position_in_grid]], uint n_tg [[threadgroups_per_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr int ROWS = 8 / SPLIT;
    threadgroup float partial[8];
    int local = sg / SPLIT, part = sg % SPLIT;
    int e = tg * ROWS + local;
    float acc = 0.0f;
    if (e < E) {
        device const float4* wr = (device const float4*)(w + (long)e * H);
        device const vec<T, 4>* xr = (device const vec<T, 4>*)x;
        for (int i = part * 32 + lane; i < H / 4; i += 32 * SPLIT) acc += dot(wr[i], float4(xr[i]));
        acc = simd_sum(acc);
    }
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0 && lane == 0 && e < E) {
        float total = 0.0f;
        for (int p = 0; p < SPLIT; ++p) total += partial[local * SPLIT + p];
        atomic_store_explicit((device atomic_float*)&logits[e], float(bfloat(total)), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup bool is_last;
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        is_last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == n_tg - 1;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!is_last || sg != 0) return;
    if (lane == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);

    constexpr int PER = MAX_E / 32;
    float logit[PER];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        int idx = lane + 32 * j;
        logit[j] = idx < E ? atomic_load_explicit((device atomic_float*)&logits[idx], memory_order_relaxed) : -INFINITY;
        mx = max(mx, logit[j]);
    }
    mx = simd_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) if (lane + 32 * j < E) sum += exp(logit[j] - mx);
    sum = simd_sum(sum);

    float picked = 0.0f;
    long picked_i = 0;
    float key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = logit[j];
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY; int best_j = 0;
#pragma unroll
        for (int j = 0; j < PER; ++j) { bool better = key[j] > best; best = better ? key[j] : best; best_j = better ? j : best_j; }
        float top = simd_max(best);
        int candidate = best == top ? int(lane) + 32 * best_j : 0x7fffffff;
        int winner = simd_min(candidate);
        float sc = exp(top - mx);
        bool mine = int(lane) == winner % 32;
#pragma unroll
        for (int j = 0; j < PER; ++j) key[j] = (mine && j == winner / 32) ? -INFINITY : key[j];
        if (int(lane) == k) { picked = sc; picked_i = winner; }
    }
    if (lane < K) {
        out_w[lane] = picked / (sum + 1e-20f);
        out_i[lane] = picked_i;
    }
}

template <typename T>
kernel void qwen_router_single(device const T* x, device const float* w, device float* logits,
                               device float* out_w, device long* out_i,
                               constant int& E, constant int& H, constant int& K,
                               uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float lg[512];
    device const vec<T, 4>* xr = (device const vec<T, 4>*)x;
    for (int e = int(tid); e < E; e += 256) {
        device const float4* wr = (device const float4*)(w + (long)e * H);
        float acc = 0.0f;
        for (int i = 0; i < H / 4; ++i) acc += dot(wr[i], float4(xr[i]));
        lg[e] = float(bfloat(acc));
        logits[e] = lg[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int PER = 512 / 32;
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < PER; ++j) { int idx = int(lane) + 32 * j; if (idx < E) mx = max(mx, lg[idx]); }
    mx = simd_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) { int idx = int(lane) + 32 * j; if (idx < E) sum += exp(lg[idx] - mx); }
    sum = simd_sum(sum);
    threadgroup float tv[32];
    threadgroup long ti[32];
    if (tid == 0) {
        for (int k = 0; k < K; ++k) {
            float best = -INFINITY; int bi = -1;
            for (int e = 0; e < E; ++e) {
                bool used = false;
                for (int p = 0; p < k; ++p) if (ti[p] == e) { used = true; break; }
                if (!used && lg[e] > best) { best = lg[e]; bi = e; }
            }
            tv[k] = best; ti[k] = bi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < K) {
        out_w[tid] = exp(tv[tid] - mx) / (sum + 1e-20f);
        out_i[tid] = ti[tid];
    }
}
template [[host_name("qwen_router_single_bfloat")]] kernel void qwen_router_single<bfloat>(device const bfloat*, device const float*, device float*, device float*, device long*, constant int&, constant int&, constant int&, uint, uint);
template [[host_name("qwen_router_single_half")]] kernel void qwen_router_single<half>(device const half*, device const float*, device float*, device float*, device long*, constant int&, constant int&, constant int&, uint, uint);
template [[host_name("qwen_router_single_float")]] kernel void qwen_router_single<float>(device const float*, device const float*, device float*, device float*, device long*, constant int&, constant int&, constant int&, uint, uint);

template <typename T, int SPLIT>
kernel void qwen_router_logits(device const T* x, device const float* w, device float* logits,
                               constant int& E, constant int& H,
                               uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int ROWS = 8 / SPLIT;
    threadgroup float partial[8];
    int local = int(sg) / SPLIT, part = int(sg) % SPLIT;
    int e = int(tg) * ROWS + local;
    float acc = 0.0f;
    if (e < E) {
        device const float4* wr = (device const float4*)(w + (long)e * H);
        device const vec<T, 4>* xr = (device const vec<T, 4>*)x;
        for (int i = part * 32 + int(lane); i < H / 4; i += 32 * SPLIT) acc += dot(wr[i], float4(xr[i]));
        acc = simd_sum(acc);
    }
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0 && lane == 0 && e < E) {
        float total = 0.0f;
        for (int p = 0; p < SPLIT; ++p) total += partial[local * SPLIT + p];
        logits[e] = float(bfloat(total));
    }
}
template [[host_name("qwen_router_logits_bfloat_8")]] kernel void qwen_router_logits<bfloat, 8>(device const bfloat*, device const float*, device float*, constant int&, constant int&, uint, uint, uint, uint);
template [[host_name("qwen_router_logits_half_8")]] kernel void qwen_router_logits<half, 8>(device const half*, device const float*, device float*, constant int&, constant int&, uint, uint, uint, uint);
template [[host_name("qwen_router_logits_float_8")]] kernel void qwen_router_logits<float, 8>(device const float*, device const float*, device float*, constant int&, constant int&, uint, uint, uint, uint);

kernel void qwen_router_select(device const float* logits, device float* out_w, device long* out_i,
                               constant int& E, constant int& H, constant int& K,
                               uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int PER = 512 / 32;
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < PER; ++j) { int idx = int(lane) + 32 * j; if (idx < E) mx = max(mx, logits[idx]); }
    mx = simd_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) { int idx = int(lane) + 32 * j; if (idx < E) sum += exp(logits[idx] - mx); }
    sum = simd_sum(sum);
    threadgroup float tv[32];
    threadgroup long ti[32];
    if (tid == 0) {
        for (int k = 0; k < K; ++k) {
            float best = -INFINITY; int bi = -1;
            for (int e = 0; e < E; ++e) {
                bool used = false;
                for (int p = 0; p < k; ++p) if (ti[p] == e) { used = true; break; }
                if (!used && logits[e] > best) { best = logits[e]; bi = e; }
            }
            tv[k] = best; ti[k] = bi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < K) {
        out_w[tid] = exp(tv[tid] - mx) / (sum + 1e-20f);
        out_i[tid] = ti[tid];
    }
}
template [[host_name("qwen_router_bfloat_8")]] kernel void qwen_router<bfloat, 8>(device const bfloat*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, uint, uint, uint, uint, uint);
template [[host_name("qwen_router_half_8")]] kernel void qwen_router<half, 8>(device const half*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, uint, uint, uint, uint, uint);
template [[host_name("qwen_router_float_8")]] kernel void qwen_router<float, 8>(device const float*, device const float*, device float*, device float*, device long*, device atomic_uint*, constant int&, constant int&, constant int&, uint, uint, uint, uint, uint);
"""
_NAMES = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}
_LIB = None
_COUNTERS = {}


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


_OPS = torch.library.Library("qwen_router_metal", "DEF")
_OPS.define("router(Tensor x, Tensor weight, int E, int H, int K) -> (Tensor, Tensor, Tensor)")


def _router_impl(x, weight, E, H, K):
    logits = torch.empty(1, E, dtype=torch.float32, device=x.device)
    weights = torch.empty(1, K, dtype=torch.float32, device=x.device)
    indices = torch.empty(1, K, dtype=torch.int64, device=x.device)
    counter = torch.zeros(1, dtype=torch.int32, device=x.device)
    getattr(_lib(), f"qwen_router_{_NAMES[x.dtype]}_8")(
        x.contiguous(),
        weight,
        logits,
        weights,
        indices,
        counter,
        E,
        H,
        K,
        threads=E * 256,
        group_size=256,
    )
    return logits, weights, indices


_OPS.impl("router", _router_impl, "MPS")


@torch.library.register_fake("qwen_router_metal::router")
def _(x, weight, E, H, K):
    return (
        x.new_empty(1, E, dtype=torch.float32),
        x.new_empty(1, K, dtype=torch.float32),
        x.new_empty(1, K, dtype=torch.int64),
    )


router = torch.ops.qwen_router_metal.router.default


def _forward(self, hidden_states):
    flat = hidden_states.reshape(-1, self.hidden_dim)
    if flat.shape[0] != 1 or flat.device.type != "mps" or flat.dtype not in _NAMES or self.num_experts > 512:
        return _orig_forward(self, hidden_states)
    logits, weights, indices = router(flat, self._w32, self.num_experts, self.hidden_dim, self.top_k)
    return logits.to(self.weight.dtype), weights.to(self.weight.dtype), indices


def prepare(model):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    for module in model.modules():
        if isinstance(module, Qwen4ExpTextTopKRouter):
            module._metal_w32 = module.weight.detach().float().contiguous()


def install():
    global _orig_forward
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    if getattr(Qwen4ExpTextTopKRouter, "_metal_router", False):
        return
    _orig_forward = Qwen4ExpTextTopKRouter.forward
    Qwen4ExpTextTopKRouter.forward = _forward
    Qwen4ExpTextTopKRouter._metal_router = True


def router_topk(hidden_states, weight, num_experts, hidden_dim, top_k):
    """Fused Qwen4 decode router: logits (bf16) + full-softmax top-k in one Metal dispatch."""
    x = hidden_states.reshape(-1, hidden_dim)
    logits, weights, indices = router(x, weight, num_experts, hidden_dim, top_k)
    return logits.to(x.dtype), weights.to(x.dtype), indices
