"""Metal kernels for the linear-attention ops (gated delta rule, short causal conv) that otherwise fall back to their
reference PyTorch loops on MPS. `use_kernel_func_from_hub_with_fallback` routes MPS tensors here by function name.

Each op is a `torch.library.custom_op` with a fake impl, so `torch.compile` keeps it in the graph."""

import torch


_SOURCE = """
#include <metal_stdlib>
using namespace metal;

inline float act(float x, int silu) { return silu ? x / (1.0f + exp(-x)) : x; }

// out[b, c, t] = act(sum_j w[c, j] * x[b, c, t - K + 1 + j] + bias[c]), zero padded on the left.
template <typename T>
kernel void causal_conv1d_fwd(device const T* x, device const T* w, device const T* bias, device T* out,
                              constant int& C, constant int& L, constant int& K, constant int& has_bias,
                              constant int& silu, uint3 tid [[thread_position_in_grid]]) {
    int t = tid.x, c = tid.y, b = tid.z;
    if (t >= L) return;
    device const T* row = x + (b * C + c) * L;
    float acc = has_bias ? float(bias[c]) : 0.0f;
    for (int j = 0; j < K; ++j) {
        int s = t - K + 1 + j;
        if (s >= 0) acc += float(w[c * K + j]) * float(row[s]);
    }
    out[(b * C + c) * L + t] = T(act(acc, silu));
}

// One new token: window = last K of [state, x]; state shifts left by one and takes x.
template <typename T, typename S>
kernel void causal_conv1d_step(device const T* x, device S* state, device const T* w, device const T* bias,
                               device T* out, constant int& C, constant int& N, constant int& K,
                               constant int& has_bias, constant int& silu, uint2 tid [[thread_position_in_grid]]) {
    int c = tid.x, b = tid.y;
    if (c >= C) return;
    device S* st = state + (b * C + c) * N;
    float xt = float(x[b * C + c]);
    float acc = has_bias ? float(bias[c]) : 0.0f;
    for (int j = 0; j < K - 1; ++j) acc += float(w[c * K + j]) * float(st[N - K + 1 + j]);
    acc += float(w[c * K + K - 1]) * xt;
    for (int j = 0; j < N - 1; ++j) st[j] = st[j + 1];
    st[N - 1] = S(xt);
    out[b * C + c] = T(act(acc, silu));
}

// Gated delta rule, token by token. One threadgroup per (batch, value head), one thread per value column, each
// holding its column of the [DK, DV] state in registers:
//   S = S * exp(g_t);  S += k_t (x) (v_t - S^T k_t) * beta_t;  o_t = S^T q_t / sqrt(DK)
template <typename T, int DK>
kernel void gated_delta_rule(device const T* q, device const T* k, device const T* v, device const float* g,
                             device const T* beta, device float* state, device T* out,
                             constant int& L, constant int& HK, constant int& HV, constant int& DV,
                             constant int& l2norm, uint2 lane [[thread_position_in_threadgroup]],
                             uint2 group [[threadgroup_position_in_grid]]) {
    // Thread (col, slice) holds rows [slice * R, (slice + 1) * R) of the state's column `col`; the dot products over
    // DK are summed across the SLICES threads of a column through threadgroup memory.
    constexpr int SLICES = 4, R = DK / SLICES;
    int col = lane.x, slice = lane.y, h = group.x, b = group.y, hk = h / (HV / HK);
    threadgroup float qs[DK], ks[DK];
    threadgroup float norms[2];
    threadgroup float red[SLICES][256];
    float s[R];
    device float* st = state + ((b * HV + h) * DK + slice * R) * DV + col;
    for (int i = 0; i < R; ++i) s[i] = st[i * DV];
    int tid = slice * DV + col, nthreads = SLICES * DV;
    for (int t = 0; t < L; ++t) {
        int qk = ((b * L + t) * HK + hk) * DK;
        for (int i = tid; i < DK; i += nthreads) { qs[i] = float(q[qk + i]); ks[i] = float(k[qk + i]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float nq = 0.0f, nk = 0.0f;
            for (int i = 0; i < DK; ++i) { nq += qs[i] * qs[i]; nk += ks[i] * ks[i]; }
            norms[0] = (l2norm ? rsqrt(nq + 1e-6f) : 1.0f) * rsqrt(float(DK));
            norms[1] = l2norm ? rsqrt(nk + 1e-6f) : 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int gh = (b * L + t) * HV + h;
        float decay = exp(g[gh]), kv = 0.0f, nk = norms[1];
        for (int i = 0; i < R; ++i) { s[i] *= decay; kv += s[i] * ks[slice * R + i]; }
        red[slice][col] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kv = 0.0f;
        for (int j = 0; j < SLICES; ++j) kv += red[j][col];
        float delta = (float(v[gh * DV + col]) - kv * nk) * float(beta[gh]) * nk;
        float o = 0.0f;
        for (int i = 0; i < R; ++i) { s[i] += ks[slice * R + i] * delta; o += s[i] * qs[slice * R + i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[slice][col] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slice == 0) {
            o = 0.0f;
            for (int j = 0; j < SLICES; ++j) o += red[j][col];
            out[gh * DV + col] = T(o * norms[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < R; ++i) st[i * DV] = s[i];
}

#define CONV(T) \\
template [[host_name("causal_conv1d_fwd_" #T)]] kernel void causal_conv1d_fwd<T>(device const T*, device const T*, \\
    device const T*, device T*, constant int&, constant int&, constant int&, constant int&, constant int&, uint3);
#define STEP(T, S) \\
template [[host_name("causal_conv1d_step_" #T "_" #S)]] kernel void causal_conv1d_step<T, S>(device const T*, \\
    device S*, device const T*, device const T*, device T*, constant int&, constant int&, constant int&, \\
    constant int&, constant int&, uint2);
#define GDR(T, D) \\
template [[host_name("gated_delta_rule_" #T "_" #D)]] kernel void gated_delta_rule<T, D>(device const T*, \\
    device const T*, device const T*, device const float*, device const T*, device float*, device T*, \\
    constant int&, constant int&, constant int&, constant int&, constant int&, uint2, uint2);
CONV(float) CONV(half) CONV(bfloat)
STEP(float, float) STEP(half, half) STEP(half, float) STEP(bfloat, bfloat) STEP(bfloat, float)
GDR(float, 64) GDR(half, 64) GDR(bfloat, 64) GDR(float, 128) GDR(half, 128) GDR(bfloat, 128)
GDR(float, 256) GDR(half, 256) GDR(bfloat, 256)
"""

_NAMES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def _silu(activation) -> int:
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Metal causal conv1d supports silu or no activation, got {activation}")
    return int(activation is not None)


@torch.library.custom_op("transformers::metal_causal_conv1d", mutates_args=())
def _conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: int) -> torch.Tensor:
    batch, channels, length = x.shape
    out = torch.empty_like(x)
    bias_ = weight if bias is None else bias.contiguous()
    getattr(_lib(), f"causal_conv1d_fwd_{_NAMES[x.dtype]}")(
        x, weight, bias_, out, channels, length, weight.shape[-1], int(bias is not None), silu,
        threads=(length, channels, batch), group_size=(min(length, 256), 1, 1),
    )  # fmt: skip
    return out


@_conv.register_fake
def _(x, weight, bias, silu):
    return torch.empty_like(x, dtype=weight.dtype)


@torch.library.custom_op("transformers::metal_causal_conv1d_step", mutates_args=("state",))
def _conv_step(
    x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, silu: int
) -> torch.Tensor:
    batch, channels, _ = x.shape
    out = torch.empty_like(x)
    bias_ = weight if bias is None else bias.contiguous()
    getattr(_lib(), f"causal_conv1d_step_{_NAMES[x.dtype]}_{_NAMES[state.dtype]}")(
        x, state, weight, bias_, out, channels, state.shape[-1], weight.shape[-1], int(bias is not None), silu,
        threads=(channels, batch), group_size=(min(channels, 256), 1),
    )  # fmt: skip
    return out


@_conv_step.register_fake
def _(x, state, weight, bias, silu):
    return torch.empty_like(x, dtype=weight.dtype)


@torch.library.custom_op("transformers::metal_gated_delta_rule", mutates_args=("state",))
def _gdr(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
    l2norm: int,
) -> torch.Tensor:  # fmt: skip
    """Advances the float32 contiguous `state` [B, HV, DK, DV] in place; returns the outputs."""
    batch, length, k_heads, k_dim = k.shape
    v_heads, v_dim = v.shape[-2:]
    dtype = v.dtype
    out = torch.empty_like(v)
    getattr(_lib(), f"gated_delta_rule_{_NAMES[dtype]}_{k_dim}")(
        q, k, v, g, beta, state, out, length, k_heads, v_heads, v_dim, l2norm,
        threads=(v_heads * v_dim, batch * 4), group_size=(v_dim, 4),  # 4 = SLICES in the kernel
    )  # fmt: skip
    return out


@_gdr.register_fake
def _(q, k, v, g, beta, state, l2norm):
    return torch.empty_like(v)


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, seq_idx=None, **kwargs):
    if seq_idx is not None:
        raise ValueError("Metal causal conv1d does not keep packed sequences apart (`seq_idx`)")
    x = hidden_states.to(weight.dtype).contiguous()
    return _conv(x, weight.contiguous(), bias, _silu(activation)).to(hidden_states.dtype)


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None, **kwargs):
    if hidden_states.shape[-1] != 1:
        raise ValueError("Metal causal_conv1d_update takes one token at a time")
    x = hidden_states.to(weight.dtype).contiguous()
    return _conv_step(x, conv_state, weight.contiguous(), bias, _silu(activation)).to(hidden_states.dtype)


def gated_delta_rule(
    query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
    cu_seqlens=None, **kwargs,
):  # fmt: skip
    """Same contract as `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` (and FLA's)."""
    if cu_seqlens is not None:
        raise ValueError("Metal gated delta rule does not keep packed sequences apart (`cu_seqlens`)")
    batch, _, _, k_dim = key.shape
    v_heads, v_dim = value.shape[-2:]
    if initial_state is None:
        state = torch.zeros(batch, v_heads, k_dim, v_dim, device=value.device, dtype=torch.float32)
    elif initial_state.dtype == torch.float32 and initial_state.is_contiguous():
        # Advanced in place: a cache's state buffer, which the cache's `copy_` of the result then skips (same tensor)
        state = initial_state
    else:
        state = initial_state.float().contiguous()
    # Contiguous / dtype casts here, where compile fuses them into the producers, not inside the op
    dtype = value.dtype
    query, key = query.to(dtype).contiguous(), key.to(dtype).contiguous()
    value, beta, g = value.contiguous(), beta.to(dtype).contiguous(), g.float().contiguous()
    out = _gdr(query, key, value, g, beta, state, int(use_qk_l2norm_in_kernel))
    return out, state if output_final_state else None


METAL_IMPLEMENTATIONS = {
    "causal_conv1d_fn": causal_conv1d_fn,
    "causal_conv1d_update": causal_conv1d_update,
    "chunk_gated_delta_rule": gated_delta_rule,
    "fused_recurrent_gated_delta_rule": gated_delta_rule,
}
