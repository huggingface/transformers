# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.import_utils import is_flash_linear_attention_available
from ..olmo3.configuration_olmo3 import Olmo3Config
from ..olmo3.modeling_olmo3 import (
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3Model,
    Olmo3PreTrainedModel,
    Olmo3RotaryEmbedding,
)


if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None
    ShortConvolution = None


logger = logging.get_logger(__name__)


class Olmo3_5HybridConfig(Olmo3Config):
    r"""
    This is the configuration class to store the configuration of a [`Olmo3_5HybridModel`]. It is used to instantiate
    an OLMo 3.5 Hybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [allenai/OLMo-3.5-1B-Hybrid](https://huggingface.co/allenai/OLMo-3.5-1B-Hybrid) model.

    The OLMo 3.5 Hybrid model combines standard transformer attention layers with GatedDeltaNet linear attention
    layers for improved efficiency while maintaining model quality.

    Configuration objects inherit from [`Olmo3Config`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Olmo3_5Hybrid model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Olmo3_5HybridModel`].
        hidden_size (`int`, *optional*, defaults to 3840):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 30):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 65536):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 100277):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 100257):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window for sliding window attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Can contain `"full_attention"`, `"sliding_attention"`, or
            `"linear_attention"`. Defaults to linear attention for most layers with full attention for every
            4th layer.
        linear_num_key_heads (`int`, *optional*):
            Number of key heads for the linear attention layers. Defaults to `num_attention_heads`.
        linear_num_value_heads (`int`, *optional*):
            Number of value heads for the linear attention layers. Defaults to `num_attention_heads`.
        linear_key_head_dim (`int`, *optional*):
            Dimension of each key head in linear attention layers. Defaults to `0.75 * hidden_size / linear_num_key_heads`.
        linear_value_head_dim (`int`, *optional*):
            Dimension of each value head in linear attention layers. Defaults to `2 * linear_key_head_dim`.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
        linear_use_gate (`bool`, *optional*, defaults to `True`):
            Whether to use gating in the linear attention output normalization.
        linear_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
            Whether to allow negative eigenvalues in the GatedDeltaNet recurrence. When `True`, the beta
            parameter is scaled by 2.0 to allow values in range [0, 2] instead of [0, 1].
    ```python
    >>> from transformers import Olmo3_5HybridModel, Olmo3_5HybridConfig

    >>> # Initializing an Olmo3.5 Hybrid style configuration
    >>> configuration = Olmo3_5HybridConfig()

    >>> # Initializing a model from the Olmo3.5 Hybrid style configuration
    >>> model = Olmo3_5HybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo3_5_hybrid"

    def __init__(
        self,
        vocab_size: int | None = 100352,
        hidden_size: int | None = 3840,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 30,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 65536,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 100257,
        tie_word_embeddings: bool | None = False,
        rope_parameters=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        rms_norm_eps: float | None = 1e-06,
        sliding_window: int | None = 4096,
        layer_types: list[str] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_use_gate: bool = True,
        linear_allow_neg_eigval: bool = True,
        **kwargs,
    ):
        if layer_types is None:
            # Default: linear attention for most layers, full attention every 4th layer
            layer_types = ["linear_attention"] * int(num_hidden_layers)
            for i in range(int(num_hidden_layers)):
                if i % 4 == 3:
                    layer_types[i] = "full_attention"

        if len(layer_types) != int(num_hidden_layers):
            raise ValueError(
                f"`layer_types` must have length num_hidden_layers={num_hidden_layers}, got {len(layer_types)}."
            )

        if "linear_attention" not in layer_types:
            raise ValueError("OLMo3.5 Hybrid expects at least one 'linear_attention' layer.")
        if all(t == "linear_attention" for t in layer_types):
            raise ValueError("OLMo3.5 Hybrid expects at least one attention layer (full or sliding).")

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
            layer_types=layer_types,
            **kwargs,
        )

        self.layer_types = list(layer_types)

        if linear_num_key_heads is None:
            linear_num_key_heads = int(num_attention_heads)
        if linear_num_value_heads is None:
            linear_num_value_heads = int(num_attention_heads)
        if linear_key_head_dim is None:
            linear_key_head_dim = int(0.75 * int(hidden_size) / int(linear_num_key_heads))
        if linear_value_head_dim is None:
            linear_value_head_dim = int(2 * int(linear_key_head_dim))

        self.linear_num_key_heads = int(linear_num_key_heads)
        self.linear_num_value_heads = int(linear_num_value_heads)
        self.linear_key_head_dim = int(linear_key_head_dim)
        self.linear_value_head_dim = int(linear_value_head_dim)
        self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
        self.linear_use_gate = bool(linear_use_gate)
        self.linear_allow_neg_eigval = bool(linear_allow_neg_eigval)


class Olmo3_5HybridDynamicCache:
    """
    Cache for hybrid model supporting both attention KV cache and linear attention state.

    Adapted from transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextDynamicCache
    """

    is_compileable = False

    def __init__(self, config: Olmo3_5HybridConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [i for i, t in enumerate(config.layer_types) if t != "linear_attention"]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        self.conv_states_q = [None for _ in range(config.num_hidden_layers)]
        self.conv_states_k = [None for _ in range(config.num_hidden_layers)]
        self.conv_states_v = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        batch_size = beam_idx.shape[0]
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                # Expand cache if needed (first reorder call in beam search)
                if self.key_cache[layer_idx].shape[0] < batch_size:
                    expand_ratio = batch_size // self.key_cache[layer_idx].shape[0]
                    self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(expand_ratio, dim=0)
                    self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(expand_ratio, dim=0)
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.conv_states_q[layer_idx] is not None:
                # Expand cache if needed (first reorder call in beam search)
                if self.conv_states_q[layer_idx].shape[0] < batch_size:
                    expand_ratio = batch_size // self.conv_states_q[layer_idx].shape[0]
                    self.conv_states_q[layer_idx] = self.conv_states_q[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                    self.conv_states_k[layer_idx] = self.conv_states_k[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                    self.conv_states_v[layer_idx] = self.conv_states_v[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                    self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                device = self.conv_states_q[layer_idx].device
                self.conv_states_q[layer_idx] = self.conv_states_q[layer_idx].index_select(0, beam_idx.to(device))
                self.conv_states_k[layer_idx] = self.conv_states_k[layer_idx].index_select(0, beam_idx.to(device))
                self.conv_states_v[layer_idx] = self.conv_states_v[layer_idx].index_select(0, beam_idx.to(device))
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(
                    0, beam_idx.to(device)
                )

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return (kv_length, kv_offset) for mask creation.

        For hybrid models:
        - Attention layers use the KV cache length
        - Linear attention layers don't need this (they use recurrent state)
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        return self.conv_states_q[self.last_linear_layer] is not None


class Olmo3_5HybridRMSNormGated(nn.Module):
    """RMSNorm with gating, matching FLA's FusedRMSNormGated."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate)
        return hidden_states


class Olmo3_5HybridRMSNorm(nn.Module):
    """Standard RMSNorm without gating."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


# Fallback ShortConvolution implementation when FLA is not available.
class Olmo3_5HybridShortConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D = x.shape
        W = self.kernel_size[0]

        x_conv = x.transpose(1, 2)

        # Single token update (decoding mode)
        if cache is not None and T == 1:
            cache_batch = cache.shape[0]
            if cache_batch < B:
                expand_ratio = B // cache_batch
                cache = cache.repeat_interleave(expand_ratio, dim=0)
            x_with_state = torch.cat([cache, x_conv], dim=-1)

            out = F.conv1d(
                x_with_state,
                self.weight,
                self.bias,
                padding=0,
                groups=D,
            )
            out = F.silu(out)

            new_state = x_with_state[:, :, 1:]

            return out.transpose(1, 2), new_state

        # Multi-token forward (prefill mode)
        else:
            out = super().forward(x_conv)[:, :, :T]
            out = F.silu(out)

            if output_final_state:
                if T >= W - 1:
                    new_state = x_conv[:, :, -(W - 1) :]
                else:
                    new_state = F.pad(x_conv, (W - 1 - T, 0))
            else:
                new_state = None

            return out.transpose(1, 2), new_state


def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """Compute sequence lengths from attention mask."""
    return mask.sum(dim=-1, dtype=torch.int32)


def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    """Compute cumulative sequence lengths from lengths."""
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    """Compute cumulative sequence lengths from attention mask."""
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Args:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length),
            1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def index_first_axis(input_tensor: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    Index the first axis of a tensor using the given indices.

    Args:
        input_tensor: Tensor of shape (total_tokens, ...)
        indices: 1D tensor of indices to select

    Returns:
        Tensor of shape (len(indices), ...)
    """
    return torch.index_select(input_tensor, 0, indices)


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Pad the hidden states back to the original batch/seq shape.

    Args:
        hidden_states: Tensor of shape (total_tokens, hidden_size)
        indices: The indices used for unpacking
        batch_size: Original batch size
        seq_len: Original sequence length

    Returns:
        Tensor of shape (batch_size, seq_len, hidden_size)
    """
    output = torch.zeros(
        batch_size * seq_len,
        *hidden_states.shape[1:],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output.index_copy_(0, indices, hidden_states)
    return output.view(batch_size, seq_len, *hidden_states.shape[1:])


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    g = g.clamp(min=-20, max=20)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    g = g.clamp(min=-20, max=20)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Olmo3_5HybridRotaryEmbedding(Olmo3RotaryEmbedding):
    """
    RoPE for OLMo 3.5 Hybrid that returns float32 cos/sin to match OLMo-core.

    The only difference from parent is NOT casting cos/sin back to x.dtype,
    preserving float32 precision like OLMo-core's full_precision=True.
    """

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # KEY FIX: Return float32, don't cast to x.dtype
        return cos, sin


class Olmo3_5HybridGatedDeltaNet(nn.Module):
    def __init__(self, config: Olmo3_5HybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_value_heads
        self.num_kv_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_kv_heads
        self.value_dim = self.head_v_dim * self.num_heads
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.use_gate = config.linear_use_gate
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.use_fla_conv = ShortConvolution is not None

        if self.use_fla_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation="silu",
            )
        else:
            self.q_conv1d = Olmo3_5HybridShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
            )
            self.k_conv1d = Olmo3_5HybridShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
            )
            self.v_conv1d = Olmo3_5HybridShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
            )

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Output norm - NOTE: FLA's FusedRMSNormGated uses eps=1e-5 by default
        o_norm_eps = 1e-5
        if self.use_gate:
            if FusedRMSNormGated is not None:
                self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=o_norm_eps)
            else:
                self.o_norm = Olmo3_5HybridRMSNormGated(self.head_v_dim, eps=o_norm_eps)
        else:
            self.o_norm = Olmo3_5HybridRMSNorm(self.head_v_dim, eps=o_norm_eps)

        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_flash_linear_attention_available():
            logger.warning_once(
                "FLA fast path not available. Install flash-linear-attention for better performance. "
                "See: https://github.com/fla-org/flash-linear-attention"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Olmo3_5HybridDynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        use_cache = cache_params is not None
        use_precomputed = use_cache and getattr(cache_params, "has_previous_state", False) and seq_len == 1

        indices = None
        cu_seqlens = None
        effective_batch_size = batch_size

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask_2d = (attention_mask[:, 0, -1, :] > -1e4).to(torch.int64)
                if attention_mask_2d.shape[1] > seq_len:
                    attention_mask_2d = attention_mask_2d[:, -seq_len:]
            elif attention_mask.dim() == 2:
                if attention_mask.shape[1] > seq_len:
                    attention_mask_2d = attention_mask[:, -seq_len:]
                elif attention_mask.shape[1] < seq_len:
                    pad_len = seq_len - attention_mask.shape[1]
                    attention_mask_2d = torch.nn.functional.pad(attention_mask, (pad_len, 0), value=1)
                else:
                    attention_mask_2d = attention_mask
            else:
                attention_mask_2d = None

            if attention_mask_2d is not None and attention_mask_2d.shape[1] == seq_len:
                has_padding = not torch.all(attention_mask_2d == 1)

                if has_padding and seq_len > 1:
                    indices, cu_seqlens, _ = get_unpad_data(attention_mask_2d)
                    hidden_states = index_first_axis(
                        hidden_states.reshape(-1, hidden_states.shape[-1]), indices
                    ).unsqueeze(0)
                    effective_batch_size = 1
                    seq_len = hidden_states.shape[1]

        conv_state_q = cache_params.conv_states_q[self.layer_idx] if cache_params else None
        conv_state_k = cache_params.conv_states_k[self.layer_idx] if cache_params else None
        conv_state_v = cache_params.conv_states_v[self.layer_idx] if cache_params else None
        recurrent_state = cache_params.recurrent_states[self.layer_idx] if cache_params else None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        conv_kwargs = {"output_final_state": use_cache}
        if self.use_fla_conv and cu_seqlens is not None:
            conv_kwargs["cu_seqlens"] = cu_seqlens

        q, new_conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, **conv_kwargs)
        k, new_conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, **conv_kwargs)
        v, new_conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, **conv_kwargs)

        if cache_params is not None:
            cache_params.conv_states_q[self.layer_idx] = new_conv_state_q
            cache_params.conv_states_k[self.layer_idx] = new_conv_state_k
            cache_params.conv_states_v[self.layer_idx] = new_conv_state_v

        # === FIX: Use effective_batch_size for reshaping ===
        q = q.view(effective_batch_size, seq_len, self.num_kv_heads, self.head_k_dim)
        k = k.view(effective_batch_size, seq_len, self.num_kv_heads, self.head_k_dim)
        v = v.view(effective_batch_size, seq_len, self.num_heads, self.head_v_dim)

        if self.num_heads > self.num_kv_heads:
            expand_ratio = self.num_heads // self.num_kv_heads
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, expand_ratio, -1)
                .reshape(effective_batch_size, seq_len, self.num_heads, self.head_k_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, expand_ratio, -1)
                .reshape(effective_batch_size, seq_len, self.num_heads, self.head_k_dim)
            )

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        use_recurrent_mode = use_precomputed or (seq_len <= 64 and not self.training)

        delta_kwargs = {
            "initial_state": recurrent_state,
            "output_final_state": use_cache,
            "use_qk_l2norm_in_kernel": True,
        }

        if cu_seqlens is not None and is_flash_linear_attention_available():
            delta_kwargs["cu_seqlens"] = cu_seqlens

        if use_recurrent_mode:
            output, new_recurrent_state = self.recurrent_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                **delta_kwargs,
            )
        else:
            output, new_recurrent_state = self.chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                **delta_kwargs,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = new_recurrent_state

        if self.use_gate:
            gate = self.g_proj(hidden_states)
            gate = gate.view(effective_batch_size, seq_len, self.num_heads, self.head_v_dim)
            if FusedRMSNormGated is not None:
                output = self.o_norm(output, gate)
            else:
                output = output.reshape(-1, self.head_v_dim)
                gate = gate.reshape(-1, self.head_v_dim)
                output = self.o_norm(output, gate)
                output = output.view(effective_batch_size, seq_len, self.num_heads, self.head_v_dim)
        else:
            output = output.reshape(-1, self.head_v_dim)
            output = self.o_norm(output)
            output = output.view(effective_batch_size, seq_len, self.num_heads, self.head_v_dim)

        output = output.reshape(effective_batch_size, seq_len, self.value_dim)
        output = self.o_proj(output)

        if indices is not None:
            output = pad_input(output.squeeze(0), indices, batch_size, attention_mask_2d.shape[-1])

        return output


class Olmo3_5HybridDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: Olmo3_5HybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Olmo3_5HybridGatedDeltaNet(config, layer_idx=layer_idx)
            # For linear attention, we need a PRE-norm (fla_norm)
            # The post_attention_layernorm from parent becomes the fla_norm
            # We rename it conceptually but keep the same weight
            del self.self_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            # OLMo-core FLABlock: h = x + fla(fla_norm(x))
            # post_attention_layernorm is used as fla_norm (pre-norm)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)  # Norm BEFORE FLA
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states

            # MLP: h = h + mlp(mlp_norm(h))
            residual = hidden_states
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # Norm BEFORE MLP
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Standard attention layers: OLMo-core ReorderedNormTransformerBlock
            # h = x + post_attn_norm(attn(x))
            residual = hidden_states
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = self.post_attention_layernorm(hidden_states)  # Norm AFTER attention
            hidden_states = residual + hidden_states

            # MLP: h = h + post_ff_norm(mlp(h))
            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # Norm AFTER MLP
            hidden_states = residual + hidden_states

        return hidden_states


class Olmo3_5HybridPreTrainedModel(Olmo3PreTrainedModel):
    _is_stateful = True


class Olmo3_5HybridModel(Olmo3Model):
    def __init__(self, config: Olmo3_5HybridConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Olmo3_5HybridDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None or not isinstance(past_key_values, Olmo3_5HybridDynamicCache):
                past_key_values = Olmo3_5HybridDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        causal_mask = create_causal_mask(**mask_kwargs)
        sliding_mask = None
        if any(t == "sliding_attention" for t in self.config.layer_types):
            sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)

        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            if decoder_layer.layer_type == "linear_attention":
                layer_mask = linear_attn_mask
            elif decoder_layer.layer_type == "full_attention":
                layer_mask = causal_mask
            elif decoder_layer.layer_type == "sliding_attention":
                if sliding_mask is None:
                    sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)
                layer_mask = sliding_mask
            else:
                raise ValueError(f"Unknown layer type {decoder_layer.layer_type!r}.")

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask: torch.Tensor | None, cache_position: torch.Tensor):
        linear_attn_mask = attention_mask
        if cache_position.numel() > 0 and (
            cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1))
        ):
            linear_attn_mask = None
        return linear_attn_mask


class Olmo3_5HybridForCausalLM(Olmo3ForCausalLM, GenerationMixin):
    pass


__all__ = [
    "Olmo3_5HybridConfig",
    "Olmo3_5HybridForCausalLM",
    "Olmo3_5HybridModel",
    "Olmo3_5HybridPreTrainedModel",
]
