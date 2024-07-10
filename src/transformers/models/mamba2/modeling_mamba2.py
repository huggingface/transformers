# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
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
"""PyTorch MAMBA2 model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import DynamicCache
from ...modeling_utils import PreTrainedModel
from ...utils.import_utils import (
    get_torch_version,
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_mamba_ssm_greater_or_equal_2_0_4,
    logging,
)
from .configuration_mamba2 import Mamba2Config


logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

if is_mamba_ssm_greater_or_equal_2_0_4():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
else:
    selective_state_update, mamba_chunk_scan_combined = None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, mamba_chunk_scan_combined, causal_conv1d_fn, causal_conv1d_update)
)


# Adapted from transformers.models.jamba.modeling_jamba.HybridMamba2AttentionDynamicCache
class HybridMamba2AttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba2 cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache`, `value_cache`, and 'conv_states' for attention cache and
    `conv_states` and `ssm_states` for mamba2 cache. Each of these lists has `num_layers` tensors.

    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` has a shape of `(batch_size, attn_head_dim * (attn_num_heads + 2 * attn_num_heads_kv), d_conv)`
    and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).

    For mamba2 layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner + 2 * d_state, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, num_heads, head_dim, d_state)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.has_previous_state = False

        in_channels = config.intermediate_size + 2 * config.state_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel
        mamba2_num_heads = config.mamba2_num_heads
        mamba2_head_dim = config.mamba2_head_dim
        attention_head_dim = config.attention_head_dim
        attention_num_heads = config.attention_num_heads
        attention_num_heads_kv = config.attention_num_key_value_heads
        attention_qkv_dim = attention_head_dim * (attention_num_heads + 2 * attention_num_heads_kv)

        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if i not in config.attention_layers_idx:
                self.conv_states += [
                    torch.zeros(batch_size, in_channels, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, mamba2_num_heads, mamba2_head_dim, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                self.conv_states += [
                    torch.zeros(batch_size, attention_qkv_dim, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")


class Mamba2MLP(nn.Module):
    def __init__(self, config: Mamba2Config):
        self.hidden_size = config.hidden_size
        self.original_intermediate_size = config.mlp_intermediate_size
        self.intermediate_padding_multiple = config.mlp_shape_padding_size

        self.intermediate_size = (
            (self.original_intermediate_size + self.intermediate_padding_multiple - 1)
            // self.intermediate_padding_multiple
            * self.intermediate_padding_multiple
        )

        self.fc1 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.use_mlp_bias)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_mlp_bias)

    def forward(self, x):
        y = self.fc1(x)
        y, z = y.chunk(2, dim=-1)
        y = y * self.act(z)
        y = self.fc2(y)
        return y

# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXRotaryEmbedding
class Mamba2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


# TODO @gante bring compatibility back, is that still valid?
# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLinearScalingRotaryEmbedding
class Mamba2LinearScalingRotaryEmbedding(Mamba2RotaryEmbedding):
    """Mamba2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXDynamicNTKScalingRotaryEmbedding
class Mamba2DynamicNTKScalingRotaryEmbedding(Mamba2RotaryEmbedding):
    """Mamba2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # TODO @gante no longer copied from, is that still valid?
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Adapted from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention
class Mamba2Attention(nn.Module):
    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.conv_kernel_size = config.conv_kernel
        self.head_dim = config.attention_head_dim
        self.num_heads = config.attention_num_heads
        self.num_heads_kv = config.attention_num_key_value_heads
        self.num_groups_kv = self.num_heads // self.num_heads_kv
        self.qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        self.out_dim = self.head_dim * self.num_heads

        # only used for causal mask creation
        self._init_bias(config.max_position_embeddings)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.rotary_emb_dim = config.rope_emb_dim
        self._init_rope()

        self.in_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=config.use_attention_qkv_bias)
        self.conv1d = nn.Conv1d(
            self.qkv_dim, self.qkv_dim, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size-1, groups=self.qkv_dim
        )
        self.out_proj = nn.Linear(self.out_dim, self.hidden_size, bias=config.use_attention_out_bias)

        self.is_causal = True
        self.layer_idx = layer_idx

    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Mamba2RotaryEmbedding(
                self.rotary_emb_dim, self.config.max_position_embeddings, base=self.config.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Mamba2LinearScalingRotaryEmbedding(
                    self.rotary_emb_dim,
                    self.config.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Mamba2DynamicNTKScalingRotaryEmbedding(
                    self.rotary_emb_dim,
                    self.config.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor,
            cache: Optional[HybridMamba2AttentionDynamicCache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        # Apply attention-conv1d-specific projections and rope
        query, key, value, cache = self._attn_conv1d_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache, use_cache=use_cache
        )

        # Repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, cache

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _conv1d(self, qkv, seq_len, cache, cached_start, cached_forward):
        # Init cache with first "real" values
        if cached_start:
            qkv_t = qkv.transpos(1, 2)
            cache.conv_states[self.layer_idx].copy_(
                nn.functional.pad(qkv_t, (self.conv_kernel_size - qkv_t.shape[-1], 0))
            )

        if is_fast_path_available:
            if cached_forward:
                qkv = causal_conv1d_update(
                    qkv.squeeze(1),
                    cache.conv_states[self.layer_idx],
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                ).unsqueeze(1)
            else:
                qkv = causal_conv1d_fn(
                    qkv.transpose(1, 2),
                    self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                ).transpose(1, 2)
        else:
            if cached_forward:
                cache.conv_states[self.layer_idx].copy_(
                    torch.roll(cache.conv_states[self.layer_idx], shifts=-1, dims=-1)
                )
                cache.conv_states[self.layer_idx][:, :, -1] = qkv.squeeze(1)
                qkv = torch.sum(
                    cache.conv_states[self.layer_idx] * self.conv1d.weight.squeeze(1), dim=-1
                )
                if self.conv1d.bias is not None:
                    qkv = qkv + self.conv1d.bias
                qkv = qkv.unsqueeze(1)
            else:
                qkv = self.conv1d(qkv.transpose(1, 2))[..., :seq_len].transpose(1, 2).contiguous()

        return qkv

    def _attn_conv1d_projections_and_rope(
            self,
            hidden_states: torch.FloatTensor,
            position_ids: torch.LongTensor,
            cache: Optional[HybridMamba2AttentionDynamicCache] = None,
            use_cache: Optional[bool] = False,
    ):
        # Managing cache state
        has_layer_past = cache is not None
        if has_layer_past:
            cached_start = not cache.has_previous_state
            cached_forward = not cached_start
        else:
            cached_start = False
            cached_forward = False

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (num_heads(_q) * head_dim + 2 * num_heads_kv * head_dim)]
        qkv = self.in_proj(hidden_states)

        # Apply Conv1d, caching is applied in-place
        qkv = self._conv1d(qkv, seq_len=qkv.shape[1], cache=cache, cached_start=cached_start, cached_forward=cached_forward)

        # Get the respective matrices from the parallel projection back
        q, k, v = qkv.split([self.num_heads * self.head_dim, self.num_heads_kv * self.head_dim, self.num_heads_kv * self.head_dim], dim=-1)

        # Split combined hidden dims back into respective attention heads
        # [batch, seq_len, hidden_size] --> [batch, num_heads, seq_len, head_dim]
        query = self._split_heads(q, num_attention_heads=self.num_heads, attn_head_size=self.head_dim)
        key = self._split_heads(k, num_attention_heads=self.num_heads_kv, attn_head_size=self.head_dim)
        value = self._split_heads(v, num_attention_heads=self.num_heads_kv, attn_head_size=self.head_dim)

        # Compute rotary embeddings on rotary_emb_dim
        query_rot = query[..., : self.rotary_emb_dim]
        query_pass = query[..., self.rotary_emb_dim :]
        key_rot = key[..., : self.rotary_emb_dim]
        key_pass = key[..., self.rotary_emb_dim :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += cache.key_cache[self.layer_idx].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache KV values
        if has_layer_past:
            key, value = cache.update(key, value, self.layer_idx)

        return query, key, value, cache

    def _attn(self, query, key, value, attention_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
            )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class Mamba2FlashAttention2(Mamba2Attention):
    """
    Mamba2 flash attention module. This module inherits from `Mamba2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor,
            cache: Optional[HybridMamba2AttentionDynamicCache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        # Apply attention-conv1d-specific projections and rope
        query, key, value, cache = self._attn_conv1d_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache, use_cache=use_cache
        )

        # Repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        query_length = query.shape[-2]

        # Mamba2 casts query and key in fp32 to apply rotary embedding in full precision
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)

        # Permute to get the expected shape for Flash Attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 / bfloat16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        input_dtype = query.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.in_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        # Compute attention
        attn_weights = self._flash_attention_forward(
            query, key, value, attention_mask, query_length, dropout=0.0, softmax_scale=None
        )

        # Reshape outputs
        attn_output = attn_weights.reshape(
            attn_weights.shape[0], attn_weights.shape[1], self.num_heads * self.head_dim
        )
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, cache

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input with num_heads->num_attention_heads
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Mamba2SdpaAttention(Mamba2Attention):
    """
    Mamba2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Mamba2Attention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, config):
        super().__init__(config)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor,
            cache: Optional[HybridMamba2AttentionDynamicCache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        if output_attentions:
            logger.warning_once(
                "`Mamba2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual "
                "implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                cache=cache,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        # Apply attention-conv1d-specific projections and rope
        query, key, value, cache = self._attn_conv1d_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache, use_cache=use_cache
        )

        # Repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        # Mamba2 casts query and key in fp32 to apply rotary embedding in full precision
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, cache


MAMBA2_ATTENTION_CLASSES = {
    "eager": Mamba2Attention,
    "flash_attention_2": Mamba2FlashAttention2,
    "sdpa": Mamba2SdpaAttention,
}


class Mamba2Mixer(nn.Module):
    """
    Using the found relation to the attention mechanism under certain conditions (State-Space-Duality SSD),
    we use the Multi-input SSM which can be seen as a counterpart to the Multi-value Attention with analogues:
    - X ~= V
    - B ~= Q
    - C ~= K
    - A (1-SS(a)) ~= Attention Mask

    For an overview, see the mamba2 paper, section 6, figure 4.
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.chunk_size = config.chunk_size
        self.dt_min = config.time_step_limit[0]
        self.dt_max = config.time_step_limit[1]
        self.layer_idx = layer_idx
        self.use_bias = config.use_bias
        self.use_conv_bias = config.use_conv_bias
        self.use_triton_kernels = config.use_triton_kernels

        # Parallel projection of the input hidden states
        self.in_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=2 * (self.intermediate_size + self.ssm_state_size) + config.num_heads,
            bias=config.use_bias,
        )

        conv1d_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=conv1d_dim,
            out_channels=conv1d_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=conv1d_dim,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # We only use a bias as parameter
        self.dt_bias = nn.Parameter(torch.rand(size=(config.num_heads,)))

        # Scalar initialization of A, i.e. 1-Semi-Separable Matrix of A (== 1-SS(a))
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*config.A_initializer_range)
        self.A_log = nn.Parameter(torch.log(A))

        # As D is a skip connection with A, it is also a scalar of the same shape as A
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # Residual normalization introduced for instability, see section 7 of the paper
        self.norm = Mamba2RMSNorm(self.intermediate_size, eps=1e-5, normalize=True)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def _conv1d(self, xBC, seq_len, use_triton_kernels, cache, cached_start, cached_forward):
        # Init cache with first "real" values
        if cached_start:
            xBC_t = rearrange(xBC, "b l d -> b d l")
            cache.conv_states[self.layer_idx].copy_(
                nn.functional.pad(xBC_t, (self.conv_kernel_size - xBC_t.shape[-1], 0))
            )

        if is_fast_path_available and use_triton_kernels:
            if cached_forward:
                xBC = causal_conv1d_update(
                    xBC,
                    cache.conv_states[self.layer_idx],
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
        else:
            if cached_forward:
                cache.conv_states[self.layer_idx].copy_(
                    torch.roll(cache.conv_states[self.layer_idx], shifts=-1, dims=-1)
                )
                cache.conv_states[self.layer_idx][:, :, -1] = xBC
                xBC = torch.sum(
                    cache.conv_states[self.layer_idx] * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
                )
                if self.conv1d.bias is not None:
                    xBC = xBC + self.conv1d.bias
                xBC = self.act(xBC)
            else:
                xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        return xBC

    def _ssd_naive(self, x, dt, A, B, C, chunk_size, dt_min, dt_max, initial_states=None, return_final_states=False):
        """
        Arguments:
            x:  (batch_size, seq_len, num_heads, head_dim)
            dt: (batch_size, seq_len, num_heads)
            A:  (num_heads)
            B:  (batch_size, seq_len, num_heads, ssm_state_size)
            C:  (batch_size, seq_len, num_heads, ssm_state_size)
        Return:
            y:  (batch_size, seq_len, num_heads, head_dim)
        """

        def pad_by_size(x, pad_size):
            """
            Padding x tensor with `pad_size` on the seq_len dim (dim=1)

            Assumes that we only have tensors of either size 4 or 3
            """
            assert 2 < len(x.shape) < 5

            pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(x.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

            return nn.functional.pad(x, pad_shape, mode="constant", value=0)

        def segsum(x):
            """
            More stable segment sum calculation
            """
            T = x.size(-1)
            x = repeat(x, "... d -> ... d e", e=T)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
            x = x.masked_fill(~mask, 0)
            x_segsum = torch.cumsum(x, dim=-2)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
            x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
            return x_segsum

        # Since it is parallelized by chunks they have to be of the same size which we ensure by padding
        seq_len = x.shape[1]
        pad_size = chunk_size - (seq_len % chunk_size)

        # dt softplus and clamping
        dt = nn.functional.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, dt_min, dt_max)

        D_residual = rearrange(self.D, "h -> 1 1 h 1") * pad_by_size(x, pad_size)

        # Discretize x and A
        x = x * dt.unsqueeze(-1)
        A = A * dt

        # Rearrange into blocks/chunks
        x, A, B, C = [
            rearrange(pad_by_size(t, pad_size), "b (c l) ... -> b c l ...", l=chunk_size) for t in (x, A, B, C)
        ]

        A = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(segsum(A))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(segsum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        # Add D residual to final output
        y = y + D_residual

        # Cutting off padded chunks
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        if not return_final_states:
            return y
        else:
            return y, final_state

    def _ssd(
        self, x, B, C, dt, initial_state, return_final_state, use_triton_kernels, cache, cached_start, cached_forward
    ):
        # Discretize 1-SS(a)
        A = -torch.exp(self.A_log.float())  # .float() to avoid infs/nans

        last_state = None
        if not cached_forward:
            if use_triton_kernels:
                y = mamba_chunk_scan_combined(
                    x=rearrange(x, pattern="b l (h p) -> b l h p", p=self.head_dim),
                    dt=dt,
                    A=A,
                    B=rearrange(B, pattern="b l n -> b l 1 n"),
                    C=rearrange(C, pattern="b l n -> b l 1 n"),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    initial_states=initial_state,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=None,
                    dt_limit=(self.dt_min, self.dt_max),
                    return_final_states=cached_start or return_final_state,
                )
            else:
                initial_state = rearrange(initial_state, "b n h p -> b 1 n h p") if initial_state is not None else None
                y = self._ssd_naive(
                    x=rearrange(x, pattern="b l (h p) -> b l h p", p=self.head_dim),
                    dt=dt,
                    A=A,
                    B=rearrange(B, pattern="b l n -> b l 1 n"),
                    C=rearrange(C, pattern="b l n -> b l 1 n"),
                    chunk_size=self.chunk_size,
                    initial_states=initial_state,
                    dt_min=self.dt_min,
                    dt_max=self.dt_max,
                    return_final_states=cached_start or return_final_state,
                )
            if cached_start or return_final_state:
                y, last_state = y
                if cached_start:
                    cache.ssm_states[self.layer_idx].copy_(last_state)

            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            if use_triton_kernels:
                # Preparing values for single step
                A = repeat(A, "h -> h p n", p=self.head_dim, n=self.ssm_state_size).to(dtype=torch.float32)
                dt = repeat(dt, "b 1 h -> b h p", p=self.head_dim)
                dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
                D = repeat(self.D, "h -> h p", p=self.head_dim)
                x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.head_dim)

                # Triton kernel for updating states in-place and returning the hidden state
                y = selective_state_update(
                    state=cache.ssm_states[self.layer_idx],
                    x=x_reshaped,
                    dt=dt,
                    A=A,
                    B=B,
                    C=C,
                    D=D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                )
            else:
                # Get time step with softplus and bias
                dt = nn.functional.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
                dt = rearrange(dt, "b 1 h -> b h")

                # Discretize A
                dA = torch.exp(dt * A)

                # Discretize B and x
                x = rearrange(x, "b (h p) -> b h p", p=self.head_dim)
                dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)

                # State calculation
                cache.ssm_states[self.layer_idx].copy_(
                    cache.ssm_states[self.layer_idx] * rearrange(dA, "b h -> b h 1 1") + dBx
                )

                # Subsequent output
                y = torch.einsum("bhpn,bn->bhp", cache.ssm_states[self.layer_idx], C)

                # D skip connection
                y = y + rearrange(self.D, "h -> h 1") * x

            # Reshaping to have seq_len == 1
            y = rearrange(y, "b h p -> b 1 (h p)")

            # Optional output of last state
            if return_final_state:
                last_state = cache.ssm_states[self.layer_idx].clone()

        return y, last_state

    def _forward(
        self,
        hidden_states,
        use_triton_kernels,
        initial_state=None,
        return_final_state=False,
        cache: Optional[Mamba2Cache] = None,
    ):
        # Managing cache state
        if cache is not None:
            cached_start = cache.seq_offset == 0
            cached_forward = not cached_start
        else:
            cached_start = False
            cached_forward = False

        # Supporting cached values as well as passing initial states but not both at the same time
        if initial_state is not None and cached_forward:
            raise ValueError("Subsequent caching and passing initial states is not possible at the same time!")

        # 1. Parallel projection for the input
        zxbcdt = self.in_proj(hidden_states)

        # 2-5. Training combined into one triton kernel
        if self.training and cache is None and is_fast_path_available and use_triton_kernels:
            y = mamba_split_conv1d_scan_combined(
                zxbcdt=zxbcdt,
                conv1d_weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                conv1d_bias=self.conv1d.bias,
                dt_bias=self.dt_bias,
                A=-torch.exp(self.A_log),
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.head_dim,
                ngroups=1,
                norm_before_gate=False,  # not the same as our variant's normalization var
                dt_limit=(self.dt_min, self.dt_max),
                initial_states=initial_state,
                return_final_states=return_final_state,
            )
            last_state = None
            if return_final_state:
                y, last_state = y
            return y, last_state

        # Reconstructing the necessary vars
        d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.intermediate_size, self.intermediate_size + 2 * self.ssm_state_size, self.num_heads],
            dim=-1,
        )

        # 2. Causal convolution for partial set of variables ("input", B, C)
        xBC = self._conv1d(
            xBC=xBC,
            seq_len=hidden_states.shape[1],
            use_triton_kernels=use_triton_kernels,
            cache=cache,
            cached_start=cached_start,
            cached_forward=cached_forward,
        )

        # Reconstruct causal convolution vars
        x, B, C = torch.split(xBC, [self.intermediate_size, self.ssm_state_size, self.ssm_state_size], dim=-1)

        # 3. State Space Duality (SSD)
        y, last_state = self._ssd(
            x=x,
            B=B,
            C=C,
            dt=dt,
            initial_state=initial_state,
            return_final_state=return_final_state,
            use_triton_kernels=use_triton_kernels,
            cache=cache,
            cached_start=cached_start,
            cached_forward=cached_forward,
        )

        # 4. Gate normalization introduced for instability, see section 7 of the paper
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = torch.cat([self.act(z0) * x0, y], dim=-1)

        # 5. Out projecting
        y = self.out_proj(y)

        return y, last_state

    def forward(
        self, hidden_states, initial_state=None, return_final_state=False, cache: Optional[Mamba2Cache] = None
    ):
        use_triton_kernels = "cuda" in self.in_proj.weight.device.type and self.use_triton_kernels

        # AMD might be available later on with https://github.com/state-spaces/mamba/pull/359
        if use_triton_kernels:
            if not is_fast_path_available:
                logger.warning_once(
                    "Faster path is not available because `(causal_conv1d_fn, causal_conv1d_update)` is None. "
                    "Falling back to slower implementation. To install follow https://github.com/Dao-AILab/causal-conv1d"
                )
        else:
            logger.warning_once(
                "Fast path is not available because the GPU is not properly utilized. "
                "Falling back to naive implementation."
            )
        return self._forward(hidden_states, use_triton_kernels, initial_state, return_final_state, cache)


class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, normalize=False):
        """
        Mamba2RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm but with optional residual normalizing
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.normalize = normalize

    def forward(self, hidden_states, residual=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # residual normalization introduced for instability, see section 7 of the paper
        if residual is not None and self.normalize:
            hidden_states = hidden_states * nn.functional.silu(residual.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * self.weight

        return hidden_states.to(input_dtype)


class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(
        self, hidden_states, initial_state=None, return_final_state=False, cache: Optional[Mamba2Cache] = None
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states, last_state = self.mixer(
            hidden_states, initial_state=initial_state, return_final_state=return_final_state, cache=cache
        )
        hidden_states = residual + hidden_states
        return hidden_states, last_state


@dataclass
class Mamba2Output(ModelOutput):
    """
    Class for the MAMBA2 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the last state returned by the SSD call of the State Space Machine, and the Causal Convolutional states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        last_ssm_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_last_ssm_states=True` is passed or when `config.output_last_ssm_states=True`):
            Tuple of `torch.FloatTensor` (one for the last state of the ssd block each) of shape `(batch_size, num_heads, head_dim, ssm_state_size)`.

            Last SSM-states of the model at the final state of an SSD block.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_ssm_states: Optional[Tuple[torch.FloatTensor]] = None


class Mamba2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Mamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True
            module.dt_bias._no_weight_decay = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.emb_initializer_range)
        elif isinstance(module, nn.Conv1d):
            if self.config.conv_initializer_range is not None:
                nn.init.uniform_(
                    module.weight, -self.config.conv_initializer_range, self.config.conv_initializer_range
                )

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


class Mamba2Model(Mamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        use_cache: Optional[bool] = None,
        initial_states: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_last_ssm_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, Mamba2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_last_ssm_states = (
            output_last_ssm_states if output_last_ssm_states is not None else self.config.output_last_ssm_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = Mamba2Cache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        initial_states = [None] * self.config.num_hidden_layers if initial_states is None else initial_states
        if len(initial_states) != self.config.num_hidden_layers:
            raise ValueError(
                "Initial states have been passed but not for all layers making it ambiguous. "
                "To ensure correctness, fill layers without an initial state with None."
            )
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_last_ssm_states = () if output_last_ssm_states else None
        for mixer_block, initial_state in zip(self.layers, initial_states):
            if self.gradient_checkpointing and self.training:
                out = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, initial_state, output_last_ssm_states, cache_params
                )
            else:
                out = mixer_block(
                    hidden_states,
                    initial_state=initial_state,
                    return_final_state=output_last_ssm_states,
                    cache=cache_params,
                )

            hidden_states = out[0]
            last_state = out[1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_last_ssm_states:
                all_last_ssm_states = all_last_ssm_states + (last_state,)

        if use_cache:
            cache_params.seq_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            last_ssm_states=all_last_ssm_states,
        )


@dataclass
class Mamba2CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the last state returned by the SSD call of the State Space Machine, and the Causal Convolutional states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        last_ssm_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_last_ssm_states=True` is passed or when `config.output_last_ssm_states=True`):
            Tuple of `torch.FloatTensor` (one for the last state of the ssd block each) of shape `(batch_size, num_heads, head_dim, ssm_state_size)`.

            Last SSM-states of the model at the final state of an SSD block.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_ssm_states: Optional[Tuple[torch.FloatTensor]] = None


class Mamba2ForCausalLM(Mamba2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "backbone.embeddings.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tie_weights()

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        # probably overwritten by `_tied_weights_keys` but just to be sure
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[Mamba2Cache] = None,
        **kwargs,
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        initial_states: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_last_ssm_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, Mamba2CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            initial_states=initial_states,
            output_hidden_states=output_hidden_states,
            output_last_ssm_states=output_last_ssm_states,
            return_dict=return_dict,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
            last_ssm_states=mamba_outputs.last_ssm_states,
        )
