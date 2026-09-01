# Copyright 2026 The HuggingFace Inc. team
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
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import (
    use_kernel_func_from_hub_with_fallback,
    use_kernelized_func,
)
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ...models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)
from ...models.llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ...models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextModel,
    Qwen3NextRMSNormGated,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..bamba.modeling_bamba import apply_mask_to_padding_states


@auto_docstring(checkpoint="moonshotai/Kimi-Linear-48B-A3B-Instruct")
@strict
class KimiLinearConfig(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 8):
        Number of groups for routed experts.
    mlp_layer_types (`list[str]`, *optional*):
        List of layer types for the MLP or MoE layers. Defaults to None.
    linear_key_head_dim (`int`, *optional*):
        Dimension of each key head in linear attention layers. Defaults to 128.
    linear_num_key_heads (`int`, *optional*):
        Number of key heads for the linear attention layers. Defaults to 32.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
    """

    model_type = "kimi_linear"
    attribute_map = {
        "model_max_length": "max_position_embeddings",
        "moe_renormalize": "norm_topk_prob",
        "num_expert_group": "n_group",
        "num_local_experts": "n_routed_experts",
        "num_experts_per_token": "num_experts_per_tok",
    }

    vocab_size: int = 163840
    hidden_size: int = 2304
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    routed_scaling_factor: float = 2.446
    q_lora_rank: int | None = None
    n_group: int = 1
    mlp_layer_types: list[str] | None = None
    topk_group: int | None = 1
    norm_topk_prob: bool = True
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 163839
    bos_token_id: int | None = 163584
    eos_token_id: int | list[int] | None = 163586
    layer_types: list[str] | None = None

    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 32
    linear_conv_kernel_dim: int = 4

    rope_parameters = AttributeError()
    rope_interleave = AttributeError()
    first_k_dense_replace = AttributeError()
    num_mtp_layers = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Checkpoint stores linear attention attributes in a config sub-dict: if it's there, extract them
        linear_attn_config = kwargs.pop("linear_attn_config", {})
        self.linear_key_head_dim = linear_attn_config.get("head_dim", self.linear_key_head_dim)
        self.linear_num_key_heads = linear_attn_config.get("num_heads", self.linear_num_key_heads)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
        # Values head have the same config as key heads
        self.linear_value_head_dim = self.linear_key_head_dim
        self.linear_num_value_heads = self.linear_num_key_heads

        # For layer types, the precedence is: explicit `layer_types` > checkpoint config > default
        if self.layer_types is not None:
            pass  # nothing to do here
        elif "full_attn_layers" in linear_attn_config and "kda_layers" in linear_attn_config:
            layer_types = [None] * self.num_hidden_layers
            for layer in linear_attn_config["full_attn_layers"]:
                layer_types[layer - 1] = "full_attention"  # types are 1-indexed in the checkpoint
            for layer in linear_attn_config["kda_layers"]:
                layer_types[layer - 1] = "linear_attention"
            self.layer_types = layer_types
        else:
            self.layer_types = [
                "full_attention" if i and i % 4 == 0 else "linear_attention" for i in range(self.num_hidden_layers)
            ]

        # Same for MLP layer types, which indicate MLP or MoE
        if self.mlp_layer_types is None:
            first_k_dense_replace = kwargs.pop("first_k_dense_replace", 1)
            self.mlp_layer_types = [
                "dense" if i < first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]


class KimiLinearRMSNorm(LlamaRMSNorm):
    pass


# NOTE: The `fla` reference stays in fp32 until after the gate is applied, but the qwen norm does not. This is not an
# issue right now, but if it ever becomes one, change the parent or override `forward`.
class KimiLinearRMSNormGated(Qwen3NextRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(hidden_size, eps=eps, **kwargs)
        self.activation = "sigmoid"


class KimiLinearExperts(DeepseekV3Experts):
    pass


class KimiLinearMLP(DeepseekV3MLP):
    pass


class KimiLinearAttention(DeepseekV3Attention):
    """Multi-headed Latent Attention (MLA) from Deepseek V2 with NoPE, but the part of the keys where RoPE is applied is
    still shared."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = q_states.view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

        key_states, value_states = self.expand_kv(kv_nope, k_rot)

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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


@use_kernel_func_from_hub_with_fallback("chunk_kda", "fla")
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


@use_kernel_func_from_hub_with_fallback("fused_recurrent_kda", "fla")
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


@use_kernelized_func([torch_recurrent_kda, torch_chunk_kda, causal_conv1d_fn, causal_conv1d_update])
class KimiLinearDeltaAttention(nn.Module):  # TODO: can we try to inherit from qwen ? or something?
    # Annotations to make ty happy
    chunk_kda: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    recurrent_kda: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention attributes
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.num_v_heads = config.linear_num_value_heads
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        # QVK modules (3 projections and 1 packed convolution)
        self.projection_k_size = self.head_k_dim * self.num_k_heads
        self.projection_v_size = self.head_v_dim * self.num_v_heads
        conv_size = 2 * self.projection_k_size + self.projection_v_size

        self.q_proj = nn.Linear(self.hidden_size, self.projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.projection_v_size, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=conv_size,
            out_channels=conv_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=conv_size,
            padding=self.conv_kernel_size - 1,
        )

        # Kimi Delta Attention (KDA) modules
        self.forget_gate_down = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
        self.forget_gate_up = nn.Linear(self.head_v_dim, self.projection_v_size, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        A_log_init = torch.empty(self.num_v_heads, 1, dtype=torch.float32).uniform_(1, 16)  # need actual values to log
        self.A_log = torch.nn.Parameter(A_log_init.log())
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, self.head_v_dim, dtype=torch.float32))

        # Output normalization and projection
        self.output_gate_down = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
        self.output_gate_up = nn.Linear(self.head_v_dim, self.projection_v_size, bias=False)

        self.o_norm = KimiLinearRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_v_size, self.hidden_size, bias=False)

    @force_accelerate_hooks("conv1d")
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
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Apply projections
        batch_size, seq_len = hidden_states.shape[:2]
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        # Apply convolutions and update conv_states cache. conv_states are used as left-side padding for convolution:
        # we apply convolution to groups of N tokens, so we need to keep N-1 tokens around for the next forward pass,
        # so that token T can see the token T-1, T-2, ..., T-N+1.
        mixed_qkv = torch.cat((q_states, k_states, v_states), dim=-1).transpose(1, 2)
        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)

        if use_precomputed_states and seq_len == 1 and not past_key_values.layers[self.layer_idx].record_past:
            conv_state = past_key_values.layers[self.layer_idx].conv_states[0]
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation="silu"
            )
        else:
            if past_key_values is not None:
                mixed_qkv = past_key_values.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation="silu", **kwargs
            )
            # Drop the additional previous states
            if past_key_values is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        q_states, k_states, v_states = torch.split(
            mixed_qkv, [self.projection_k_size, self.projection_k_size, self.projection_v_size], dim=-1
        )

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
        gate = -log_decay_scale * F.softplus(gate.float() + self.dt_bias)

        beta = self.beta_proj(hidden_states).float().sigmoid()

        # Retrieve the old recurrent state if there is one
        if use_precomputed_states:
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states[0]  # type: ignore
        else:
            recurrent_state = None

        # Apply the KDA delta rule, here in the non-chunked mode (for decoding with a cache)
        if use_precomputed_states and seq_len == 1:
            kda_fn = torch_recurrent_kda
            kwargs = {}
        # Otherwise (prefill or no cache) use the "chunked" mode, which is more efficient for longer input sequences
        else:
            kda_fn = torch_chunk_kda
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
            **kwargs,  # NOTE: FLA kernel can do more and we precompute less, but it means more code divergence before
        )

        # Update cache
        if past_key_values is not None:
            past_key_values.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # Apply normalization to the attention output
        output_gate = self.output_gate_up(self.output_gate_down(hidden_states))
        output_gate = output_gate.reshape(value_shape)
        normed_attn_out = self.o_norm(core_attn_out, output_gate)

        # Apply output projection
        normed_attn_out = normed_attn_out.reshape(batch_size, seq_len, -1)
        output = self.o_proj(normed_attn_out)
        return output, None  # we add a "None" so it matches the MLA return type


class KimiLinearTopkRouter(DeepseekV3TopkRouter):
    pass


class KimiLinearMoE(DeepseekV3MoE):
    pass


class KimiLinearDecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        if config.layer_types[layer_idx] == "full_attention":
            self.self_attn = KimiLinearAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = KimiLinearDeltaAttention(config=config, layer_idx=layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = KimiLinearMoE(config)
        else:
            self.mlp = KimiLinearMLP(config)

        self.input_layernorm = KimiLinearRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiLinearRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@auto_docstring
class KimiLinearPreTrainedModel(PreTrainedModel):
    config: KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiLinearDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(KimiLinearTopkRouter, index=0),
        "hidden_states": KimiLinearDecoderLayer,
        "attentions": KimiLinearAttention,
    }
    _is_stateful = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, KimiLinearDeltaAttention):
            init.ones_(module.dt_bias)
            # Lower bound kept away from 0 so log(A) never becomes -inf
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0.01, 16).log_())
        elif isinstance(module, KimiLinearExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, KimiLinearTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)


@auto_docstring
class KimiLinearModel(Qwen3NextModel):
    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        del self.rotary_emb

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids: torch.LongTensor = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = (position_ids + past_seen_tokens).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class KimiLinearForCausalLM(DeepseekV3ForCausalLM, GenerationMixin):
    _tied_weights_keys = {}


__all__ = [
    "KimiLinearConfig",
    "KimiLinearPreTrainedModel",
    "KimiLinearModel",
    "KimiLinearForCausalLM",
]
