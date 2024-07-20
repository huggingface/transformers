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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import (
    get_torch_version,
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_mamba_ssm_greater_or_equal_2_0_4,
)
from .configuration_mamba2 import Mamba2Config


logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

if is_mamba_ssm_greater_or_equal_2_0_4():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)


_CONFIG_FOR_DOC = "MambaConfig"


# Adapted from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache with Mamba->Mamba2
class HybridMamba2AttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba2 cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache`, `value_cache`, and 'conv_states' for attention cache and
    `conv_states` and `ssm_states` for mamba2 cache. Each of these lists has `num_layers` tensors.

    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_key_value_heads, seq_len, attention_head_dim)`,
    while `conv_states` has a shape of `(batch_size, attention_head_dim * (num_attention_heads + 2 * num_key_value_heads), attention_conv_kernel)`
    or `(batch_size, 0)` (empty tensors) and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).

    For mamba2 layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, intermediate_size + 2 * state_size, mamba2_conv_kernel)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, mamba2_num_heads, mamba2_head_dim, state_size)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.has_previous_state = False

        in_channels = config.intermediate_size + 2 * config.state_size
        ssm_state_size = config.state_size
        mamba2_conv_kernel_size = config.mamba2_conv_kernel
        attention_conv_kernel_size = config.attention_conv_kernel
        mamba2_num_heads = config.mamba2_num_heads
        mamba2_head_dim = config.mamba2_head_dim
        attention_head_dim = config.attention_head_dim
        attention_num_heads = config.num_attention_heads
        attention_num_heads_kv = config.num_key_value_heads
        attention_qkv_dim = attention_head_dim * (attention_num_heads + 2 * attention_num_heads_kv)

        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if i not in config.attention_layers_idx:
                self.conv_states += [
                    torch.zeros(batch_size, in_channels, mamba2_conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(
                        batch_size, mamba2_num_heads, mamba2_head_dim, ssm_state_size, device=device, dtype=dtype
                    )
                ]
            else:
                # Conv1d is optional for the attention layer
                if attention_conv_kernel_size > 0:
                    self.conv_states += [
                        torch.zeros(
                            batch_size, attention_qkv_dim, attention_conv_kernel_size, device=device, dtype=dtype
                        )
                    ]
                else:
                    self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.update
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

    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.reorder_cache
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

    # Adapted from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.get_seq_length
    # Fixes issues when accessing on empty cache and allow mamba2 pure architectures
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Mamba2 layers don't need the seq_len either way
        if len(self.transformer_layers) == 0:
            return 0

        # Take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0

        # We also allow seq_len checks on empty tensors
        size_idx = -2 if len(self.key_cache[layer_idx].shape) > 2 else -1

        return self.key_cache[layer_idx].shape[size_idx]

    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.to_legacy_cache with Mamba->Mamba2
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMamba2AttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.from_legacy_cache with Mamba->Mamba2
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMamba2AttentionDynamicCache does not have a legacy cache equivalent.")


class Mamba2MLP(nn.Module):
    def __init__(self, config: Mamba2Config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.original_intermediate_size = config.mlp_intermediate_size
        self.mlp_padding_size = config.mlp_padding_size

        self.intermediate_size = (
            (self.original_intermediate_size + self.mlp_padding_size - 1)
            // self.mlp_padding_size
            * self.mlp_padding_size
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


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mamba2
class Mamba2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->Mamba2
class Mamba2LinearScalingRotaryEmbedding(Mamba2RotaryEmbedding):
    """Mamba2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->Mamba2
class Mamba2DynamicNTKScalingRotaryEmbedding(Mamba2RotaryEmbedding):
    """Mamba2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
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


# Adapted from transformers.models.llama.modeling_llama.LlamaAttention with Llama->Mamba2
class Mamba2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Possible switch to MQA when num_heads_kv < num_heads_q.
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.conv_kernel_size = config.attention_conv_kernel
        self.head_dim = config.attention_head_dim
        self.num_heads = config.num_attention_heads
        self.num_heads_kv = config.num_key_value_heads
        self.num_groups_kv = self.num_heads // self.num_heads_kv
        # See https://github.com/state-spaces/mamba/issues/457#issuecomment-2221116217
        # hidden_size % num_heads == 0 is not necessary due to this custom head projection dim
        self.qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        self.out_dim = self.head_dim * self.num_heads

        # Optional RoPE
        self.rotary_emb_dim = config.rope_emb_dim
        self.rope_theta = config.rope_theta
        self._init_rope()

        self.in_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=config.use_attention_qkv_bias)
        # Optional conv1d
        self._init_conv1d()
        self.out_proj = nn.Linear(self.out_dim, self.hidden_size, bias=config.use_attention_out_bias)

        self.is_causal = True
        self.layer_idx = layer_idx

        # We throw a similar fast path warning, in case no mamba2 block is used
        if config.num_hidden_layers == len(config.attention_layers_idx):
            if not is_causal_conv1d_available():
                logger.warning_once(
                    "Convolution implementation in Mamba2 attention is falling back to naive implementation because `(causal_conv1d_fn, causal_conv1d_update)`"
                    "is None. To install follow https://github.com/Dao-AILab/causal-conv1d."
                )

    # Adapted from transformers.models.llama.modeling_llama.LlamaAttention._init_rope
    # Rope is optional and can be ignored if rope_emb_dim <= 0
    def _init_rope(self):
        # RoPE is optional
        if self.rotary_emb_dim < 1:
            return

        if self.config.rope_scaling is None:
            self.rotary_emb = Mamba2RotaryEmbedding(
                self.rotary_emb_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Mamba2LinearScalingRotaryEmbedding(
                    self.rotary_emb_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Mamba2DynamicNTKScalingRotaryEmbedding(
                    self.rotary_emb_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _init_conv1d(self):
        # Conv1d is optional
        if self.conv_kernel_size < 1:
            return

        self.conv1d = nn.Conv1d(
            self.qkv_dim,
            self.qkv_dim,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=self.qkv_dim,
        )

    # Adapted from transformers.models.llama.modeling_llama.LlamaAttention.forward
    # Custom projections involving optional causal-conv-1d and optional (partial) RoPE
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        cache: Optional[HybridMamba2AttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        bsz, q_len, _ = hidden_states.shape

        # Apply attention-conv1d-specific projections and rope
        query, key, value = self._attn_conv1d_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache, use_cache=use_cache
        )

        # Repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Final projection
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _conv1d(self, qkv, seq_len, cache, cached_start, cached_forward):
        # Init cache with first "real" values
        if cached_start:
            qkv_t = qkv.transpose(1, 2)
            cache.conv_states[self.layer_idx].copy_(
                nn.functional.pad(qkv_t, (self.conv_kernel_size - qkv_t.shape[-1], 0))
            )

        if is_causal_conv1d_available():
            if cached_forward:
                qkv = causal_conv1d_update(
                    x=qkv.squeeze(1),
                    conv_state=cache.conv_states[self.layer_idx],
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                ).unsqueeze(1)
            else:
                qkv = causal_conv1d_fn(
                    x=qkv.transpose(1, 2),
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                ).transpose(1, 2)
        else:
            if cached_forward:
                cache.conv_states[self.layer_idx].copy_(
                    torch.roll(cache.conv_states[self.layer_idx], shifts=-1, dims=-1)
                )
                cache.conv_states[self.layer_idx][:, :, -1] = qkv.squeeze(1)
                qkv = torch.sum(cache.conv_states[self.layer_idx] * self.conv1d.weight.squeeze(1), dim=-1)
                if self.conv1d.bias is not None:
                    qkv = qkv + self.conv1d.bias
                qkv = qkv.unsqueeze(1)
            else:
                qkv = self.conv1d(qkv.transpose(1, 2))[..., :seq_len].transpose(1, 2).contiguous()

        return qkv

    # Moved to a separate function since it's optional
    # Mixture of transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention._attn_projections_and_rope and
    # transformers.models.llama.modeling_llama.LlamaAttention.forward RoPE parts
    # GPTNeoX for the partial (on dim) RoPE application, Llama for the general RoPE embeddings
    def _apply_rope(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        position_ids: torch.LongTensor,
    ):
        # Compute rotary embeddings on rotary_emb_dim
        query_rot = query[..., : self.rotary_emb_dim]
        query_pass = query[..., self.rotary_emb_dim :]
        key_rot = key[..., : self.rotary_emb_dim]
        key_pass = key[..., self.rotary_emb_dim :]

        # Compute RoPE and stitch it back together
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        return query, key

    def _attn_conv1d_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        cache: Optional[HybridMamba2AttentionDynamicCache] = None,
        use_cache: Optional[bool] = False,
    ):
        bsz, q_len, _ = hidden_states.shape

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
        #   --> [batch, seq_len, (head_dim * (num_heads(_q) + 2 * num_heads_kv)]
        qkv = self.in_proj(hidden_states)

        # (Optional) Apply Conv1d, caching is applied in-place
        if self.conv_kernel_size > 0:
            qkv = self._conv1d(
                qkv, seq_len=qkv.shape[1], cache=cache, cached_start=cached_start, cached_forward=cached_forward
            )

        # Get the respective matrices from the parallel projection back
        q, k, v = qkv.split(
            [self.num_heads * self.head_dim, self.num_heads_kv * self.head_dim, self.num_heads_kv * self.head_dim],
            dim=-1,
        )

        # Split combined hidden dims back into respective attention heads
        # [batch, seq_len, hidden_size] --> [batch, seq_len, num_heads, head_dim]
        query = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = k.reshape(bsz, q_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        value = v.reshape(bsz, q_len, self.num_heads_kv, self.head_dim).transpose(1, 2)

        # (Optional) RoPE
        if self.rotary_emb_dim > 0:
            # TODO do we need to cache sin and cos for RoPE, llama doesn't seem to cache it (except when using sink cache)?
            query, key = self._apply_rope(query, key, value, position_ids)

        # Cache KV values
        if has_layer_past:
            key, value = cache.update(key, value, self.layer_idx)

        return query, key, value


# Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->Mamba2
class Mamba2FlashAttention2(Mamba2Attention):
    """
    Mamba2 flash attention module. This module inherits from `Mamba2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    # Custom projections involving optional causal-conv-1d and optional (partial) RoPE
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        cache: Optional[HybridMamba2AttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        bsz, q_len, _ = hidden_states.shape

        # Apply attention-conv1d-specific projections and rope
        query, key, value = self._attn_conv1d_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache, use_cache=use_cache
        )

        # Repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

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
        attn_weights = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            q_len,
            dropout=0.0,
            softmax_scale=None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        # Reshape outputs
        attn_output = attn_weights.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# Adapted from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Mamba2
class Mamba2SdpaAttention(Mamba2Attention):
    """
    Mamba2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Mamba2Attention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    # Adapted from transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    # Custom projections involving optional causal-conv-1d and optional (partial) RoPE
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        cache: Optional[HybridMamba2AttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
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
        query, key, value = self._attn_conv1d_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache, use_cache=use_cache
        )

        # Repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

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
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


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
        self.conv_kernel_size = config.mamba2_conv_kernel
        self.intermediate_size = config.intermediate_size
        self.head_dim = config.mamba2_head_dim
        self.num_heads = config.mamba2_num_heads
        self.chunk_size = config.chunk_size
        self.dt_min = config.time_step_limit[0]
        self.dt_max = config.time_step_limit[1]
        self.layer_idx = layer_idx
        self.use_bias = config.use_mamba2_bias
        self.use_conv_bias = config.use_conv_bias

        # Parallel projection of the input hidden states
        self.in_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=2 * (self.intermediate_size + self.ssm_state_size) + self.num_heads,
            bias=self.use_bias,
        )

        conv1d_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=conv1d_dim,
            out_channels=conv1d_dim,
            bias=config.use_conv_bias,
            kernel_size=config.mamba2_conv_kernel,
            groups=conv1d_dim,
            padding=config.mamba2_conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # We only use a bias as parameter
        self.dt_bias = nn.Parameter(torch.rand(size=(self.num_heads,)))

        # Scalar initialization of A, i.e. 1-Semi-Separable Matrix of A (== 1-SS(a))
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*config.A_initializer_range)
        self.A_log = nn.Parameter(torch.log(A))

        # As D is a skip connection with A, it is also a scalar of the same shape as A
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # Residual normalization introduced for instability, see section 7 of the paper
        self.norm = Mamba2RMSNorm(self.intermediate_size, eps=1e-5)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of "
                "`(selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def triton_kernels_forward(self, hidden_states, cache):
        # Managing cache state
        if cache is not None:
            cached_start = not cache.has_previous_state
            cached_forward = not cached_start
        else:
            cached_start = False
            cached_forward = False

        # 1. Parallel projection for the input
        zxbcdt = self.in_proj(hidden_states)

        # 2-5. Training combined into one triton kernel
        if self.training and cache is None:
            y = mamba_split_conv1d_scan_combined(
                zxbcdt=zxbcdt,
                conv1d_weight=self.conv1d.weight.squeeze(1),
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
                norm_before_gate=False,
                dt_limit=(self.dt_min, self.dt_max),
                initial_states=None,
                return_final_states=False,
            )
            return y

        # Reconstructing the necessary vars
        d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.intermediate_size, self.intermediate_size + 2 * self.ssm_state_size, self.num_heads],
            dim=-1,
        )

        # 2. Causal convolution for partial set of variables ("input" (x), B, C)
        # Init cache with first "real" values
        if cached_start:
            xBC_t = xBC.transpose(1, 2)
            cache.conv_states[self.layer_idx].copy_(F.pad(xBC_t, (self.conv_kernel_size - xBC_t.shape[-1], 0)))

        if cached_forward:
            xBC = causal_conv1d_update(
                x=xBC.squeeze(1),
                conv_state=cache.conv_states[self.layer_idx],
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

        # Reconstruct causal convolution vars
        x, B, C = torch.split(xBC, [self.intermediate_size, self.ssm_state_size, self.ssm_state_size], dim=-1)

        # 3. State Space Duality (SSD)
        # Discretize 1-SS(a)
        A = -torch.exp(self.A_log.float())  # .float() to avoid infs/nans

        if not cached_forward:
            y = mamba_chunk_scan_combined(
                x=x.reshape(x.shape[0], x.shape[1], -1, self.head_dim),
                dt=dt,
                A=A,
                B=B.unsqueeze(-2),
                C=C.unsqueeze(-2),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                initial_states=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=None,
                dt_limit=(self.dt_min, self.dt_max),
                return_final_states=cached_start,
            )

            if cached_start:
                y, last_state = y
                if cached_start:
                    cache.ssm_states[self.layer_idx].copy_(last_state)

            # [bsz, seq_len, num_heads, head_dim] -> [bsz, seq_len, intermediate size]
            y = y.reshape(y.shape[0], y.shape[1], -1)
        else:
            # Preparing values for single step
            # [num_heads] -> [num_heads, head_dim, state_size]
            A = (
                A.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(A.shape[0], self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            # [bsz, 1, num_heads] -> [bsz, num_heads, head_dim]
            dt = dt.transpose(1, 2).expand(dt.shape[0], dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias.unsqueeze(-1).expand(self.dt_bias.shape[0], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            D = self.D.unsqueeze(-1).expand(self.D.shape[0], self.head_dim)
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            x_reshaped = x.reshape(x.shape[0], -1, self.head_dim)

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

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(y.shape[0], -1).unsqueeze(1)

        # 4. Gate normalization introduced for instability, see section 7 of the paper
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = torch.cat([self.act(z0) * x0, y], dim=-1)

        # 5. Out projecting
        y = self.out_proj(y)

        return y

    @classmethod
    def _ssd_naive(
        cls, x, dt, A, B, C, D, chunk_size, dt_bias, dt_min, dt_max, initial_states=None, return_final_states=False
    ):
        """
        Arguments:
            x:       (batch_size, seq_len, num_heads, head_dim)
            dt:      (batch_size, seq_len, num_heads)
            A:       (num_heads)
            B:       (batch_size, seq_len, num_heads, ssm_state_size)
            C:       (batch_size, seq_len, num_heads, ssm_state_size)
            D:       (num_heads)
            dt_bias: (num_heads)
        Return:
            y:       (batch_size, seq_len, num_heads, head_dim)
        """

        def pad_by_size(x, pad_size):
            """
            Padding x tensor with `pad_size` on the seq_len dim (dim=1)

            Assumes that we only have tensors of either size 4 or 3
            """
            assert 2 < len(x.shape) < 5

            pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(x.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

            return F.pad(x, pad_shape, mode="constant", value=0)

        def reshape_into_chunks(x, pad_size, chunk_size):
            """
            Padding x tensor with `pad_size` on the seq_len dim (dim=1) and
            simultaneously splitting it into chunk sequences.

            Assumes that we only have tensors of either size 4 or 3
            """
            # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
            x = pad_by_size(x, pad_size)

            if len(x.shape) == 3:
                # b (l c) h -> b l c h with c=chunk_size
                # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
                return x.reshape(x.shape[0], -1, chunk_size, x.shape[2])
            else:
                # b (l c) h p -> b l c h p with c=chunk_size
                # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
                return x.reshape(x.shape[0], -1, chunk_size, x.shape[2], x.shape[3])

        def segsum(x):
            """
            More stable segment sum calculation
            """
            T = x.size(-1)
            # [..., chunk_size] -> [..., chunk_size, chunk_size]
            x = x.unsqueeze(-1).expand(*x.size(), T)
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
        dt = F.softplus(dt + dt_bias)
        dt = torch.clamp(dt, dt_min, dt_max)

        D_residual = D.unsqueeze(-1) * pad_by_size(x, pad_size)

        # Discretize x and A
        x = x * dt.unsqueeze(-1)
        A = A.to(x.dtype) * dt

        # Rearrange into blocks/chunks
        x, A, B, C = [reshape_into_chunks(t, pad_size, chunk_size) for t in (x, A, B, C)]

        # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
        A = A.permute(0, 3, 1, 2)
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
        y = Y_diag + Y_off
        # [bsz, -1, chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
        y = y.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])

        # Add D residual to final output
        y = y + D_residual

        # Cutting off padded chunks
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        if not return_final_states:
            return y
        else:
            return y, final_state

    def slow_forward(self, hidden_states, cache):
        seq_len = hidden_states.shape[1]

        # Managing cache state
        if cache is not None:
            cached_start = not cache.has_previous_state
            cached_forward = not cached_start
        else:
            cached_start = False
            cached_forward = False

        # 1. Parallel projection for the input
        zxbcdt = self.in_proj(hidden_states)

        # Reconstructing the necessary vars
        d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.intermediate_size, self.intermediate_size + 2 * self.ssm_state_size, self.num_heads],
            dim=-1,
        )

        # 2. Causal convolution for partial set of variables ("input" (x), B, C)
        # Init cache with first "real" values
        if cached_start:
            xBC_t = xBC.transpose(1, 2)
            cache.conv_states[self.layer_idx].copy_(F.pad(xBC_t, (self.conv_kernel_size - xBC_t.shape[-1], 0)))

        if cached_forward:
            cache.conv_states[self.layer_idx].copy_(torch.roll(cache.conv_states[self.layer_idx], shifts=-1, dims=-1))
            cache.conv_states[self.layer_idx][:, :, -1] = xBC.squeeze(1)
            xBC = torch.sum(cache.conv_states[self.layer_idx] * self.conv1d.weight.squeeze(1), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC)
        else:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        # Reconstruct causal convolution vars
        x, B, C = torch.split(xBC, [self.intermediate_size, self.ssm_state_size, self.ssm_state_size], dim=-1)

        # 3. State Space Duality (SSD)
        # Discretize 1-SS(a)
        A = -torch.exp(self.A_log.float())  # .float() to avoid infs/nans

        if not cached_forward:
            y = self._ssd_naive(
                # [bsz, seq_len, intermediate_size] -> [bsz, seq_len, num_heads, head_dim]
                x=x.reshape(x.shape[0], x.shape[1], -1, self.head_dim),
                dt=dt,
                A=A,
                # [bsz, seq_len, state_size] -> [bsz, seq_len, num_groups=1, state_size]
                B=B.unsqueeze(-2),
                # [bsz, seq_len, state_size] -> [bsz, seq_len, num_groups=1, state_size]
                C=C.unsqueeze(-2),
                chunk_size=self.chunk_size,
                D=self.D,
                initial_states=None,
                dt_bias=self.dt_bias,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                return_final_states=cached_start,
            )

            if cached_start:
                y, last_state = y
                if cached_start:
                    cache.ssm_states[self.layer_idx].copy_(last_state)

            # [bsz, seq_len, num_heads, head_dim] -> [bsz, seq_len, intermediate_size]
            y = y.reshape(y.shape[0], y.shape[1], -1)
        else:
            # Get time step with softplus and bias
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            dt = dt.squeeze(1)

            # Discretize A
            dA = torch.exp(dt * A)

            # Discretize B and x
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            x = x.reshape(x.shape[0], -1, self.head_dim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)

            # State calculation
            cache.ssm_states[self.layer_idx].copy_(
                cache.ssm_states[self.layer_idx] * dA.unsqueeze(-1).unsqueeze(-1) + dBx
            )

            # Subsequent output
            y = torch.einsum("bhpn,bn->bhp", cache.ssm_states[self.layer_idx].to(C.dtype), C)

            # D skip connection
            y = y + self.D.unsqueeze(-1) * x

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(y.shape[0], -1).unsqueeze(1)

        # 4. Gate normalization introduced for instability, see section 7 of the paper
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = torch.cat([self.act(z0) * x0, y], dim=-1)

        # 5. Out projecting
        y = self.out_proj(y)

        return y

    def forward(self, hidden_states, cache: Optional[HybridMamba2AttentionDynamicCache] = None):
        # TODO: check version for AMD support?
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.triton_kernels_forward(hidden_states, cache)
        return self.slow_forward(hidden_states, cache)


# Adapted from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mamba2
# An optional residual normalization has been integrated
class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Mamba2RMSNorm is equivalent to LlamaRMSNorm but with optional residual normalizing
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Residual normalization introduced for instability, see section 7 of the paper
        if residual is not None:
            hidden_states = hidden_states * F.silu(residual.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


# Adapted from transformers.models.mamba.modeling_mamba.MambaBlock
# Allows attention instead of mamba2 and an optional MLP layer afterward
class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_layer = layer_idx in config.attention_layers_idx
        self.mlp_layer = config.mlp_intermediate_size > 0
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Mixer is either attention layer or mamba2 layer
        if self.attention_layer:
            self.mixer = MAMBA2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        else:
            self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

        # Following mlp layer is optional
        if self.mlp_layer:
            self.norm2 = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.mlp = Mamba2MLP(config, layer_idx=layer_idx)
        else:
            self.norm2 = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        cache: Optional[HybridMamba2AttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        dtype = hidden_states.dtype

        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # Mamba2 path
        if not self.attention_layer:
            hidden_states = self.mixer(hidden_states, cache=cache)
            attn_weights = None
        # Attention path
        else:
            hidden_states, attn_weights = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        hidden_states = (residual + hidden_states).to(dtype)

        if self.mlp_layer:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            hidden_states = self.mlp(hidden_states)
            hidden_states = (hidden_states + residual).to(dtype)

        return hidden_states, attn_weights


class Mamba2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True  # Note: only supports HybridMamba2AttentionDynamicCache
    _is_stateful = True

    # Adapted from transformers.models.mamba.modeling_mamba.MambaPreTrainedModel._init_weights
    # Only using dt bias and rescale_prenorm_residual is expanded when using the additional MLP layer
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Mamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.mamba2_num_heads)
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

                    # mlp layer is considered as an additional overhead
                    n_residuals = 2 if self.config.mlp_intermediate_size > 0 else 1
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals * self.config.num_hidden_layers)


MAMBA2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Mamba2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MAMBA2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`HybridMamba2AttentionDynamicCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A HybridMamba2AttentionDynamicCache object containing pre-computed hidden-states (keys, values, and, if used, the convolution in the
            self-attention blocks and convolution and ssm states in the mamba2 blocks) that can be used (see `past_key_values` input)
            to speed up sequential decoding.
            Key and value cache tensors have shape `(batch_size, num_key_value_heads, seq_len, attention_head_dim)`.
            Convolution and ssm states tensors have shape `(batch_size, intermediate_size + 2 * state_size, mamba2_conv_kernel)` if used in the mamba2 block
            else it has shape `(batch_size, attention_head_dim * (num_attention_heads + 2 * num_key_value_heads), attention_conv_kernel)`
            and `(batch_size, mamba2_num_heads, mamba2_head_dim, state_size)` respectively.
            See the `HybridMamba2AttentionDynamicCache` class for more details.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare MAMBA2 Model outputting raw hidden-states without any specific head on top.",
    MAMBA2_START_DOCSTRING,
)
class Mamba2Model(Mamba2PreTrainedModel):
    # Adapted from transformers.models.mamba.modeling_mamba.MambaModel.__init__ with Mamba->Mamba2
    # Additional information about possible attention layers
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self._attn_implementation = config._attn_implementation
        self._uses_attention_layers = len(config.attention_layers_idx) > 0

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    # Copied from transformers.models.mamba.modeling_mamba.MambaModel.load_hook
    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # Adapted from transformers.models.jamba.modeling_jamba.JambaModel.forward
    # No MoE logic, inits cache itself like Mamba does, and handles position_ids like Llama
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMamba2AttentionDynamicCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # We allow empty caches on initial forward
        if past_key_values is None and use_cache:
            past_key_values = HybridMamba2AttentionDynamicCache(
                config=self.config,
                batch_size=inputs_embeds.shape[0],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        # LLama based positions
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for mixer_block in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                out = self._gradient_checkpointing_func(
                    mixer_block.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                out = mixer_block(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = out[0]

            if output_attentions:
                if out[1] is not None:
                    # Append attentions only of attention layers. Mamba2 layers return `None` as the attention weights
                    all_self_attns += (out[1],)

        hidden_states = self.norm_f(hidden_states)

        # Add hidden states from the last block
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Adapted from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    # Custom hybrid cache instead
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridMamba2AttentionDynamicCache,
        output_attentions: bool,
    ):
        if not self._uses_attention_layers:
            return None

        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # TODO: check if this is compatible with this custom cache format
        if self._attn_implementation == "sdpa" and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


@add_start_docstrings(
    """
    The MAMBA2 Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    MAMBA2_START_DOCSTRING,
)
class Mamba2ForCausalLM(Mamba2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    # Adapted from transformers.models.jamba.modeling_jamba.JambaForCausalLM.prepare_inputs_for_generation
    # We omit some args Mamba2 doesn't use such as output_router_logits and num_logits_to_keep; additional optional reinit of the cache
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if not empty_past_kv:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # Initialize cache, if necessary
        if empty_past_kv:
            past_key_values = HybridMamba2AttentionDynamicCache(
                config=self.config,
                batch_size=input_ids.shape[0],
                device=self.device,
                dtype=self.dtype,
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(MAMBA2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMamba2AttentionDynamicCache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.bart.modeling_bart.BartClassificationHead with Bart->Mamba2, torch.tanh->F.silu
class Mamba2ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    Mamba2 Model backbone with a sequence classification/regression head on top
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.

    [`Mamba2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token.
    If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row.
    If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MAMBA2_START_DOCSTRING,
)
class Mamba2ForSequenceClassification(Mamba2PreTrainedModel):
    # Copied from transformers.models.bart.modeling_bart.BartForSequenceClassification.__init__ with Bart->Mamba2,d_model->hidden_size,model->backbone
    def __init__(self, config: Mamba2Config, **kwargs):
        super().__init__(config, **kwargs)
        self.backbone = Mamba2Model(config)
        self.classification_head = Mamba2ClassificationHead(
            config.hidden_size,
            config.hidden_size,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def set_input_embeddings(self, value):
        self.backbone.embeddings = value

    @add_start_docstrings_to_model_forward(MAMBA2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralForSequenceClassification.forward with self.num_labels->self.config.num_labels,self.score->self.classification_head,self.model->self.backbone
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.classification_head(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
