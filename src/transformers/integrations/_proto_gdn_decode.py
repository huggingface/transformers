# [scratch] Gated delta rule decode step for continuous batching (Qwen3-Next): one token per sequence, the recurrent
# states updated in place in the paged cache. The gather of every sequence's state, the hub's fused recurrent kernel
# (one program per 8 value columns, one warp) and the scatter back took ~53 of the ~81 ms of a 256-sequence decode step
# of Qwen3-Next-80B-A3B on a B300 (the states are 2 MiB a layer and sequence). Same math as fla's
# `fused_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel`, the gates computed in the kernel as vLLM's packed
# decode does.
import os

import torch
import triton
import triton.language as tl


_PROTO_GDN_IN_PLACE = os.environ.get("PROTO_GDN_IN_PLACE", "1") == "1"  # dev knob: 0 for the gather / scatter baseline


@triton.jit
def _gdn_decode_kernel(
    QKV, A, B, ALog, DtBias, State, ReadRows, WriteRows, Out, scale,
    stride_qkv, stride_a, stride_b, stride_slot,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
):  # fmt: skip
    """One program per (block of BV value columns, sequence and value head). The state tile [K, BV] of the sequence's
    head is read from its read row, decayed, corrected by the delta rule and written to its write row (the same row but
    for a sequence with no past, which reads zeros from the trash row)."""
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    read_row = tl.load(ReadRows + i_n).to(tl.int64)
    write_row = tl.load(WriteRows + i_n).to(tl.int64)
    tile = i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    h = tl.load(State + read_row * stride_slot + tile).to(tl.float32)

    base = QKV + i_n * stride_qkv
    q = tl.load(base + i_h * K + o_k).to(tl.float32)
    k = tl.load(base + H * K + i_h * K + o_k).to(tl.float32)
    v = tl.load(base + 2 * H * K + i_hv * V + o_v).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)

    x = tl.load(A + i_n * stride_a + i_hv).to(tl.float32) + tl.load(DtBias + i_hv).to(tl.float32)
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g = -tl.exp(tl.load(ALog + i_hv).to(tl.float32)) * softplus
    # rounded to the input dtype, as the model's `b.sigmoid()` is before fla's kernel reads it
    beta = tl.sigmoid(tl.load(B + i_n * stride_b + i_hv).to(tl.float32)).to(B.dtype.element_ty).to(tl.float32)

    h = h * tl.exp(g)
    v = (v - tl.sum(h * k[:, None], 0)) * beta
    h = h + k[:, None] * v[None, :]
    o = tl.sum(h * q[:, None], 0)
    tl.store(Out + (i_n * HV + i_hv) * V + o_v, o.to(Out.dtype.element_ty))
    tl.store(State + write_row * stride_slot + tile, h.to(State.dtype.element_ty))


# The states are a float32 view of the cache's page bytes: a mutation dynamo cannot functionalize ("input mutations on
# views with different dtypes"), so the update runs outside the compiled graph (inside the step's CUDA graph all the same)
@torch.compiler.disable
def gdn_decode_inplace(qkv, a, b, A_log, dt_bias, state, conv_view, conv_state, read_rows, write_rows, out, scale, num_k_heads):
    """`qkv` [tokens, H * K + H * K + HV * V] (after the convolution), `a` and `b` [tokens, HV] (the gate inputs),
    `state` [slots, HV, K, V] the layer's recurrent states, `read_rows` and `write_rows` [tokens] the slots of each
    token's sequence. Writes the output [tokens, HV, V] into `out`, the new recurrent states into `state` and the new
    conv states `conv_state` into `conv_view` at `write_rows`."""
    num_tokens = qkv.shape[0]
    _, num_v_heads, head_k, head_v = state.shape
    block_v = 32
    grid = (triton.cdiv(head_v, block_v), num_tokens * num_v_heads)
    _gdn_decode_kernel[grid](
        qkv, a, b, A_log, dt_bias, state, read_rows, write_rows, out, scale,
        qkv.stride(0), a.stride(0), b.stride(0), state.stride(0),
        H=num_k_heads, HV=num_v_heads, K=head_k, V=head_v, BV=block_v, num_warps=4, num_stages=1,
    )  # fmt: skip
    conv_view.index_copy_(0, write_rows, conv_state.to(conv_view.dtype))
