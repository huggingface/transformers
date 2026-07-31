# Copyright 2026 The Moonshot AI Team and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.import_utils import is_flash_linear_attention_available
from ..bamba.modeling_bamba import apply_mask_to_padding_states
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated
from ..qwen3_next.modular_qwen3_next import causal_conv1d_fn, causal_conv1d_update
from .configuration_kimi_k3 import KimiK3TextConfig


if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
else:
    FusedRMSNormGated = None
    chunk_kda, fused_recurrent_kda = None, None


class KimiK3RMSNorm(LlamaRMSNorm):
    pass


class KimiK3RMSNormGated(Qwen3NextRMSNormGated):
    pass


# TODO: replace w/ OlmoHybridShortConvolution once #47604 is merged
class KimiK3ShortConvolution(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        hidden_size: int,
        conv_state_idx: int,
        layer_idx: int,
        bias: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = config.conv_kernel_size
        self.conv_state_idx = conv_state_idx
        self.layer_idx = layer_idx
        self.activation = activation

        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            groups=hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            bias=bias,
        )

    def forward(
        self,
        input_states: torch.Tensor,
        use_precomputed_states: bool,
        seq_len: int,
        past_key_values: Cache | None = None,
        seq_idx: int | None = None,
    ) -> torch.Tensor:
        is_recurrent_decoding = use_precomputed_states and seq_len == 1
        # Convolutions (and the cached conv states) use a channel-first layout: [batch_size, hidden_size, seq_len]
        input_states = input_states.transpose(1, 2)

        # If we can modify the states in-place, do it because it's much faster
        if is_recurrent_decoding and not past_key_values.layers[self.layer_idx].record_past:  # type: ignore
            conv_state = past_key_values.layers[self.layer_idx].conv_states[self.conv_state_idx]  # type: ignore
            output_states = causal_conv1d_update(
                input_states,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            return output_states.transpose(1, 2)

        # Otherwise, apply convolution and then update the state separately
        if past_key_values is not None:
            input_states = past_key_values.update_conv_state(
                input_states, self.layer_idx, state_idx=self.conv_state_idx, conv_kernel_size=self.kernel_size
            )
        output_states = causal_conv1d_fn(
            input_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        # Drop the additional previous states
        if past_key_values is not None:
            output_states = output_states[:, :, -seq_len:]
        return output_states.transpose(1, 2)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the KDA (Kimi Delta Attention) gated delta rule, by chunking along the sequence
    dimension. KDA is a gated delta rule with per-channel decays: each key channel of the recurrent state has its own
    forget gate.
    Args:
        query: Query tensor of shape [batch_size, sequence_length, num_heads, k_head_dim]
        key: Key tensor of shape [batch_size, sequence_length, num_heads, k_head_dim]
        value: Value tensor of shape [batch_size, sequence_length, num_heads, v_head_dim]
        g: Log-decay tensor of shape [batch_size, sequence_length, num_heads, k_head_dim], one decay per key channel:
        row d of the recurrent state is multiplied by exp(g[..., d]) at each step, so entries must be <= 0.
        beta: Beta tensor of shape [batch_size, sequence_length, num_heads]
        chunk_size: Size of the chunks along the sequence dimension.
        initial_state: The recurrent state, an optional tensor of shape [batch_size, num_heads, k_head_dim, v_head_dim]
        output_final_state: Whether to output the new recurrent state along with the output.
    Returns:
        - The output tensor of shape [batch_size, sequence_length, num_heads, v_head_dim]
        - Either None or the new recurrent state tensor of shape [batch_size, num_heads, k_head_dim, v_head_dim]
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    log_decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_heads, seqlen, ...]
    query, key, value, beta, log_decay = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, log_decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))  # this adds "pad_size" padding coeffs on the right of dimension -2
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    log_decay = F.pad(log_decay, (0, 0, 0, pad_size))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
    # state influences the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    log_decay = log_decay.reshape(log_decay.shape[0], log_decay.shape[1], -1, chunk_size, log_decay.shape[-1])

    # Create a chunked-sized causal mask (with and without the diagonal)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    strictly_upper_mask = mask.triu(1)

    # Cumulative log-decay within each chunk (dim 3 is the position inside the chunk): cum_log_decay[..., t, :] is
    # the log of the total decay accumulated between the start of the chunk and position t
    cum_log_decay = log_decay.cumsum(dim=3)
    cum_decay = cum_log_decay.exp()  # cumulative in the sense of the product: decay is never summed

    # First phase: compute intra-chunk quantities, vectorized over groups of chunks that are "not too big": this strikes
    # a balance between vectorization (speed) and memory footprint. The bound matters because of the per-channel decay's
    # k_head_dim axis: materializing the full tensor for all chunks at once would cost 3 MB/token (at Kimi K3's sizes)
    pairwise_decay_numel = batch_size * num_heads * chunk_size * chunk_size * cum_log_decay.shape[-1]
    numel_bound = 2**26  # heuristic-based bound: a fp32 tensor with this many elements weighs 256 MB
    chunks_per_group = max(1, min(num_chunks, numel_bound // pairwise_decay_numel))

    vectorized_shape = (batch_size, num_heads, num_chunks, chunk_size, chunk_size)
    ut_attn = torch.empty(vectorized_shape, dtype=value.dtype, device=value.device)
    qk_attn = torch.empty(vectorized_shape, dtype=value.dtype, device=value.device)

    # Loop over groups of chunks
    for s in range(0, num_chunks, chunks_per_group):
        # Compute the pairwise decays: pairwise_decay[..., i, j] = exp(cum_log_decay_i - cum_log_decay_j) is the
        # decay accumulated between positions j and i of a chunk
        chunks = slice(s, s + chunks_per_group)
        group_cum_log_decay = cum_log_decay[:, :, chunks]
        pairwise_log_decay = group_cum_log_decay.unsqueeze(4) - group_cum_log_decay.unsqueeze(3)
        pairwise_log_decay = pairwise_log_decay.masked_fill(strictly_upper_mask.unsqueeze(-1), float("-inf"))
        pairwise_decay = pairwise_log_decay.exp()  # no overflow because positive pairwise_log_decay are masked to -inf

        # Compute auxiliary tensors: UT transform (ut_attn) and QK dot product (qk_attn) using decay-weighted dot
        # products: out[..., i, j] = sum_d a[..., i, d] * key[..., j, d] * pairwise_decay[..., i, j, d].
        # The decays cannot be factored out of the dot product, so they are folded into the (shared) keys once, and the
        # remaining reductions are batched as one matmul per row of `a`.
        decayed_keys = key[:, :, chunks].unsqueeze(-3) * pairwise_decay
        ut_attn[:, :, chunks] = -(decayed_keys @ k_beta[:, :, chunks].unsqueeze(-1)).squeeze(-1)
        ut_attn[:, :, chunks] = ut_attn[:, :, chunks].masked_fill(mask, 0)
        qk_attn[:, :, chunks] = (decayed_keys @ query[:, :, chunks].unsqueeze(-1)).squeeze(-1)

    # Apply the UT transform to the within-chunk k/v pairs. The transform computes (I + L)^-1 by forward substitution,
    # where -L is the strictly lower triangular ut_attn built above.
    for i in range(1, chunk_size):
        row = ut_attn[..., i, :i].clone()
        sub = ut_attn[..., :i, :i].clone()
        ut_attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    ut_attn = ut_attn + torch.eye(chunk_size, dtype=ut_attn.dtype, device=ut_attn.device)
    # After the UT transformation, the within-chunk k/v pairs are used to create the new_values (called "u" in the
    # DeltaNet paper and the fla kernels) and the decayed keys reading the old state (k_cumdecay). In the second
    # phase, the part of new_values that the old state already predicts is subtracted out (v_new), so that only the
    # correction is written to the recurrent state: this is the delta rule.
    new_values = ut_attn @ v_beta
    k_cumdecay = ut_attn @ (k_beta * cum_decay)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Second phase: the sequential scan over chunks. Combine the read of the previous recurrent state (attn_inter)
    # with the within-chunk attention (qk_attn), then decay + update the recurrent state
    for i in range(num_chunks):
        q_i, k_i, cum_log_decay_i = query[:, :, i], key[:, :, i], cum_log_decay[:, :, i]
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = (q_i * cum_decay[:, :, i]) @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + qk_attn[:, :, i] @ v_new
        # chunk_log_decay is the log of the total decay over the whole chunk, used to decay the recurrent state
        chunk_log_decay = cum_log_decay_i[:, :, -1]
        state_decay = chunk_log_decay.exp().unsqueeze(-1)
        key_decay = (chunk_log_decay.unsqueeze(2) - cum_log_decay_i).exp()
        last_recurrent_state = last_recurrent_state * state_decay + (k_i * key_decay).transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the original shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the KDA gated delta rule, by iterating over each token in the sequence
    dimension. Same args and return value as torch_chunk_kda, except for `chunk_size` because the sequence dim is
    not chunked.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    log_decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_heads, seqlen, ...]
    query, key, value, beta, log_decay = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, log_decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decays the key dim of the recurrent state (one decay per key channel)
        decay_t = log_decay[:, :, i].exp().unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class KimiK3DeltaAttention(nn.Module):
    # Annotations to make ty happy
    chunk_kda: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    recurrent_kda: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]

    def __init__(self, config: KimiK3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention attributes
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_attn_key_heads
        self.head_k_dim = config.linear_attn_key_head_dim
        self.num_v_heads = config.linear_attn_value_heads
        self.head_v_dim = config.linear_attn_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        # QVK modules (projections and convolutions)
        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_v_size = self.head_v_dim * self.num_v_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_v_size, bias=False)

        self.q_conv = KimiK3ShortConvolution(config, projection_k_size, conv_state_idx=0, layer_idx=layer_idx)
        self.k_conv = KimiK3ShortConvolution(config, projection_k_size, conv_state_idx=1, layer_idx=layer_idx)
        self.v_conv = KimiK3ShortConvolution(config, projection_v_size, conv_state_idx=2, layer_idx=layer_idx)

        # KDA delta rule implementations: fla kernels when available, torch fallbacks otherwise.
        self.chunk_kda = chunk_kda or torch_chunk_kda
        self.recurrent_kda = fused_recurrent_kda or torch_recurrent_kda

        self.forget_gate_down = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
        self.forget_gate_up = nn.Linear(self.head_v_dim, projection_v_size, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        A_log_init = torch.empty(self.num_v_heads, 1, dtype=torch.float32).uniform_(1, 16)  # need actual values to log
        self.A_log = torch.nn.Parameter(A_log_init.log())
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, self.head_v_dim, dtype=torch.float32))
        self.forget_gate_lower_bound = config.forget_gate_lower_bound

        # Output normalization and projection
        self.use_full_rank_output_gate = config.use_full_rank_output_gate
        if self.use_full_rank_output_gate:
            self.output_gate = nn.Linear(self.hidden_size, projection_v_size, bias=False)
        else:
            self.output_gate_down = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
            self.output_gate_up = nn.Linear(self.head_v_dim, projection_v_size, bias=False)

        norm_module = FusedRMSNormGated if FusedRMSNormGated is not None else KimiK3RMSNormGated
        self.o_norm = norm_module(self.head_v_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(projection_v_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,  # [batch, num_heads, seqlen_q, seqlen_k] or [seqlen_q, seqlen_k]
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        # Apply the 2D padding mask to the hidden states if there is one
        if attention_mask is not None:
            # Attention mask must be 2D: try switching to the padding mask if it's not
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask", attention_mask)
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"Mask must be a 0-1 matrix of shape [batch_size, seq_len] but got {attention_mask.shape = }",
                )
            apply_mask_to_padding_states(hidden_states, attention_mask)

        # Apply projections
        batch_size, seq_len = hidden_states.shape[:2]
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        # Apply convolutions and update conv_states cache. conv_states are used as left-side padding for convolution:
        # we apply convolution to groups of N tokens, so we need to keep N-1 tokens around for the next forward pass,
        # so that token T can see the token T-1, T-2, ..., T-N+1.
        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        seq_idx = kwargs.get("seq_idx")
        q_states = self.q_conv(q_states, use_precomputed_states, seq_len, past_key_values, seq_idx)
        k_states = self.k_conv(k_states, use_precomputed_states, seq_len, past_key_values, seq_idx)
        v_states = self.v_conv(v_states, use_precomputed_states, seq_len, past_key_values, seq_idx)

        # Reshape QVK states for Kimi linear attention
        key_shape = (batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value_shape = (batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        q_states = q_states.view(key_shape)
        k_states = k_states.view(key_shape)
        v_states = v_states.view(value_shape)

        # Compute the gate, ie. the log-decay of the states, called "g" in flash-linear-attention API
        gate = self.forget_gate_up(self.forget_gate_down(hidden_states))
        gate = gate.reshape(value_shape)
        log_decay_scale = self.A_log.exp()
        # If a lower bound is provided for the gate, the way to compute the log_decay is different
        if self.forget_gate_lower_bound is not None:
            gate = self.forget_gate_lower_bound * (log_decay_scale * (gate + self.dt_bias)).sigmoid()
        else:
            gate = -log_decay_scale * F.softplus(gate.float() + self.dt_bias)

        beta = self.beta_proj(hidden_states).float().sigmoid()

        # Retrieve the old recurrent state if there is one
        if use_precomputed_states:
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states[0]  # type: ignore
        else:
            recurrent_state = None

        # Apply the KDA delta rule, here in the non-chunked mode (for decoding with a cache)
        if use_precomputed_states and seq_len == 1:
            kda_fn = self.recurrent_kda
            kwargs = {}
        # Otherwise (prefill or no cache) use the "chunked" mode, which is more efficient for longer input sequences
        else:
            kda_fn = self.chunk_kda
            kwargs = {"cu_seqlens": kwargs.get("cu_seq_lens_q")}

        core_attn_out, last_recurrent_state = kda_fn(
            q_states,
            k_states,
            v_states,
            g=gate,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=past_key_values is not None,
            use_qk_l2norm_in_kernel=True,
            **kwargs,  # TODO: FLA kernel can do more and we precompute less, but it means more code divergence before
        )

        # Update cache
        if past_key_values is not None:
            past_key_values.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # Apply normalization to the attention output
        if self.use_full_rank_output_gate:
            output_gate = self.output_gate(hidden_states)
        else:
            output_gate = self.output_gate_down(self.output_gate_up(hidden_states))
        output_gate = output_gate.reshape(value_shape)
        normed_attn_out = self.o_norm(core_attn_out, output_gate)

        # Apply output projection
        normed_attn_out = normed_attn_out.reshape(batch_size, seq_len, -1)
        output = self.o_proj(normed_attn_out)
        # TODO: BUG: is there an attn mask to apply here?
        return output, None  # we add a "None" so it matches the MLA return type


class KimiK3GatedMLA(nn.Module):
    """KimiK3 Gated Multi-Head Attention module: inspired by Deepseek V2 MLA module, but diverges in two major ways:
    - there is an additional gate to control the output of the attention
    - there is no RoPE applied to the part of the keys shared between heads
    """

    is_causal: bool = True

    def __init__(self, config: KimiK3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.k_shared_head_dim = config.qk_rope_head_dim  # no RoPE is applied but is shared between heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.k_shared_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = KimiK3RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.k_shared_head_dim, bias=False)
        self.kv_a_layernorm = KimiK3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        self.output_gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.scaling = self.qk_head_dim ** (-0.5)

    def expand_kv(self, kv_nope: torch.Tensor, k_pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = kv_nope.shape
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
        kv_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = k_pe.expand(*kv_nope.shape[:-1], -1)
        key_states = torch.cat((kv_nope, k_pe), dim=-1)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = query_states.view(query_shape).transpose(1, 2)

        # Compute the latent KV states: one mixed KV, per head; the other a K state shared between heads
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_shared = torch.split(compressed_kv, [self.kv_lora_rank, self.k_shared_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        k_shared = k_shared.view(batch_size, 1, seq_length, self.k_shared_head_dim)

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            kv_nope, k_shared = past_key_values.update(kv_nope, k_shared, self.layer_idx)
        # Expand the cached latent KV states to the full key shape
        key_states, value_states = self.expand_kv(kv_nope, k_shared)

        # Regular MHA attention
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()

        # Divergence from Deepseek V2 MLA: there is an additional gate to control the output of the attention
        gate = self.output_gate_proj(hidden_states).sigmoid()
        attn_output = attn_output * gate

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def situ_and_mul(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """Applies the SituAndMul activation, as described in the Kimi K3 paper:
        out = beta * tanh(gate / beta) * sigmoid(gate) * up
    If linear_beta is provided, up is first transformed: up = linear_beta * tanh(up / linear_beta)
    """
    original_dtype = gate.dtype
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)

    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)

    out = beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * up
    return out.to(original_dtype)


class KimiK3MLP(nn.Module):
    def __init__(self, config: KimiK3TextConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(situ_and_mul(self.gate_proj(x), self.up_proj(x), self.beta, self.linear_beta))


class KimiK3TopkRouter(DeepseekV3TopkRouter):
    pass


class KimiK3Experts(DeepseekV4Experts):
    """Collection of expert weights stored as 3D tensors. Contrary to most MoEs, Kimi K3's experts' hidden dimension is
    not the same as the model's hidden dimension: it is smaller, which reduces the cost (in time and memory) of a single
    expert.
    """

    def __init__(self, config: KimiK3TextConfig):
        nn.Module.__init__(self)
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.routed_expert_hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        # Activation function (always SituAndMul) parameters
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Applies gating to the up projection of the experts. This lives on the class so the grouped_mm / batched_mm
        backends swapped in by `@use_experts_implementation` applies the same activation on top of their packed
        gate_up output instead of bypassing it."""
        gate, up = gate_up.chunk(2, dim=-1)
        return situ_and_mul(gate, up, self.beta, self.linear_beta)


class KimiK3MoE(DeepseekV3MoE):
    def __init__(self, config: KimiK3TextConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.moe_hidden_size = config.routed_expert_hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.routed_expert_down_proj = nn.Linear(self.hidden_size, self.moe_hidden_size, bias=False)
        self.routed_expert_norm = KimiK3RMSNorm(self.moe_hidden_size, self.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # KimiK3 projects the hidden states to a smaller latent space to reduce the size of the expert weights
        hidden_states = self.routed_expert_down_proj(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        # It also applies normalization inside the latent space, before exiting it
        hidden_states = self.routed_expert_norm(hidden_states)
        hidden_states = self.routed_expert_up_proj(hidden_states)
        hidden_states = hidden_states.view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class KimiK3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: KimiK3TextConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        if config.layer_types[layer_idx] == "linear_attention":
            self.self_attn = KimiK3DeltaAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = KimiK3GatedMLA(config=config, layer_idx=layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = KimiK3MoE(config)
        else:
            self.mlp = KimiK3MLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = KimiK3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiK3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_start_of_residual_block = layer_idx % config.attn_res_block_size == 0
        self.pre_attn_residual = KimiK3AttentionResidual(config)
        self.post_attn_residual = KimiK3AttentionResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        # Rather than a simple skip connection, this model applies an attention-like operation on the residuals, and
        # keeps track of the residuals across the model using a block residual
        if block_residual.shape[1] > 0:
            hidden_states = self.pre_attn_residual(hidden_states, block_residual)

        # If this layer is the start of a residual block, the current residual is used to update the block residual
        if self.is_start_of_residual_block:
            block_residual = torch.cat([block_residual, residual.view(-1, 1, self.hidden_size).float()], dim=1)
            residual = None  # residual is now part of the block residual, no need for an extra copy

        # Self attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # If the residual was not already added to the block residual, add it to the hidden states
        if not self.is_start_of_residual_block:
            hidden_states = hidden_states + residual
        # Start of the second residual connection (skips the MLP or MoE layer)
        residual = hidden_states
        hidden_states = self.post_attn_residual(hidden_states, block_residual)

        # Fully Connected (MLP or MoE)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # End of second residual connection
        hidden_states = hidden_states + residual
        return hidden_states, block_residual


class KimiK3AttentionResidual(nn.Module):
    def __init__(self, config: KimiK3TextConfig):
        super().__init__()
        self.config = config
        # Pseudo-queries have an output size of 1: they mimic a dot product along the last dim of size hidden_size
        self.pseudo_queries = nn.Linear(config.hidden_size, 1, bias=False)
        self.attn_residual_norm = KimiK3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
        """Applies an attention-like operation to the hidden_states & the residuals of previous layers. The residuals of
        some N-1 previous decoder layers are concatenated in "block_residual" this way: [num_tokens, N-1, hidden_size].
        By concatenating the current hidden_states along dim 1, we get a tensor of shape [num_tokens, N, hidden_size].
        This is both the keys and the values tensor. By matching it against a set of learnable pseudo-queries, attention
        retrieval happens along dimension 1, ie. along the previous layers axis. So for each token, the most relevant
        previous residuals are retrieved. This means the model looks back at its previous residuals, and selects the ones
        that are the most relevant to the current hidden_states: it picks which residual connections to use.
        Args:
            - hidden_states: a tensor of shape [batch_size, seq_len, hidden_size]
            - block_residual: some previous layers' residuals, concatenated along dim 1. It's a tensor of
            shape [num_tokens, N-1, hidden_size], where N-1 is the number of recorded previous layers and num_tokens is
            batch_size * seq_len.
        Returns:
            the new hidden states, a tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        initial_dtype = hidden_states.dtype

        # Flatten the batch size dimension and add the "previous layers" dimension (dim 1)
        hidden_states = hidden_states.view(-1, 1, hidden_size)
        # Concatenate the hidden states to the block residuals to get a tensor of shape [num_tokens, N, hidden_size]
        keys_or_values = torch.cat((block_residual, hidden_states.float()), dim=1)

        # Match the keys against the learned pseudo-queries and use the same tensor for the values
        normalized_keys = self.attn_residual_norm(keys_or_values)
        scores = self.pseudo_queries(normalized_keys)  # shape: [num_tokens, N, 1]
        probs = scores.transpose(1, 2).softmax(-1)  # shape: [num_tokens, 1, N]
        hidden_states = torch.matmul(probs, keys_or_values)  # shape: [num_tokens, 1, hidden_size]
        hidden_states = hidden_states.squeeze(1)  # shape: [num_tokens, hidden_size]

        return hidden_states.to(initial_dtype).view(batch_size, seq_len, hidden_size)



class KimiK3PreTrainedModel(Kimi_K25PreTrainedModel):
    _keep_in_fp32_modules_strict = ["pseudo_queries", "attn_residual_norm"]
