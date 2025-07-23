# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Union

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from ...utils.generic import check_model_inputs, OutputRecorder

from .configuration_blt import (
    BLTConfig,
    BLTGlobalTransformerConfig,
    BLTLocalDecoderConfig,
    BLTLocalEncoderConfig,
    BLTPatcherConfig,
)


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


class BLTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Ignore copy
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, **kwargs: Unpack[TransformersKwargs]):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class BLTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BLTRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, **kwargs: Unpack[TransformersKwargs]):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BLTRotaryEmbedding(nn.Module):
    def __init__(self, config: BLTConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_type = (
            config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            if config.rope_scaling is not None
            else "default"
        )
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, **kwargs: Unpack[TransformersKwargs]):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Modified from transformers.models.llama.modeling_llama.LlamaDecoderLayer
class BLTTransformerLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BLTSelfAttention(config=config, layer_idx=layer_idx)
        self.mlp = BLTMLP(config)
        self.input_layernorm = BLTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BLTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # TODO: not exactly equivalent to other transformers implementations,, need feedback
    # Extract first head_dim//2 elements which correspond to the unique frequencies
    # This matches the original BLT approach which uses head_dim//2 frequency pairs
    head_dim = q.shape[-1]
    cos_freqs = cos[..., : head_dim // 2]  # [B, S, D/2]
    sin_freqs = sin[..., : head_dim // 2]  # [B, S, D/2]

    # Expand cos/sin to match query/key tensor format [B, H, S, D/2]
    cos_freqs = cos_freqs.unsqueeze(1)  # [B, 1, S, D/2] -> [B, H, S, D/2]
    sin_freqs = sin_freqs.unsqueeze(1)  # [B, 1, S, D/2] -> [B, H, S, D/2]

    # Split q and k into pairs for rotation: (d0, d1), (d2, d3), ...
    q_pairs = q.view(*q.shape[:-1], head_dim // 2, 2)  # [B, H, S, D/2, 2]
    k_pairs = k.view(*k.shape[:-1], head_dim // 2, 2)  # [B, H, S, D/2, 2]

    # Extract real and i parts
    q_real, q_imag = q_pairs[..., 0], q_pairs[..., 1]  # [B, H, S, D/2]
    k_real, k_imag = k_pairs[..., 0], k_pairs[..., 1]  # [B, H, S, D/2]

    # Apply rotation: [real', imag'] = [cos*real - sin*imag, sin*real + cos*imag]
    q_real_rot = cos_freqs * q_real - sin_freqs * q_imag
    q_imag_rot = sin_freqs * q_real + cos_freqs * q_imag
    k_real_rot = cos_freqs * k_real - sin_freqs * k_imag
    k_imag_rot = sin_freqs * k_real + cos_freqs * k_imag

    # Recombine pairs and reshape back to original format
    q_rot = torch.stack([q_real_rot, q_imag_rot], dim=-1).view(*q.shape)  # [B, H, S, D]
    k_rot = torch.stack([k_real_rot, k_imag_rot], dim=-1).view(*k.shape)  # [B, H, S, D]

    return q_rot.type_as(q), k_rot.type_as(k)


class BLTSelfAttention(nn.Module):
    """BLT variant of MllamaTextSelfAttention. Inherits all logic directly."""

    def __init__(self, config: BLTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.is_causal = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Ensure hidden_states is always 3D (batch, seq_len, hidden_dim)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attn_impl = getattr(self.config, "_attn_implementation", None) or "eager"
        attention_interface: Callable = eager_attention_forward
        if attn_impl != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Decoder-specific layer class for automatic hidden states collection  
class BLTDecoderLayer(BLTTransformerLayer):
    """
    BLT decoder layer - identical to BLTTransformerLayer but with a different class name
    for selective automatic collection of hidden states from decoder layers only.
    """
    pass


# BLT-SPECIFIC COMPONENTS (no Mllama equivalent)


class BLTLocalEncoder(nn.Module):
    def __init__(self, config: BLTLocalEncoderConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.config = config
        self.layers = nn.ModuleList(
            [BLTTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = BLTRotaryEmbedding(config=config)
        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * config.cross_attn_k,
            bias=False,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cross_attn_layers = nn.ModuleList()
        layers_to_add = config.num_hidden_layers if config.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BLTCrossAttention(config=config, layer_idx=layer_idx, hidden_size=config.hidden_size)
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        batch_size, _, _ = input_embeds.shape
        hidden_states = F.dropout(input_embeds, p=self.config.dropout, training=self.training)
        if position_ids is None:
            position_ids = (
                torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            if idx == len(self.layers) - 1 or self.config.cross_attn_all_layers:
                patch_embeds = self.patch_reduce(hidden_states, num_patches, "amax", patch_ids)
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size
                )
                layer_idx = idx if self.config.cross_attn_all_layers else 0
                # Remove cross_attention_states from kwargs if present to avoid multiple values error
                kwargs.pop("cross_attention_states", None)
                cross_attention_output, _ = self.cross_attn_layers[layer_idx](
                    hidden_states=patch_embeds,
                    cross_attention_states=hidden_states,
                    attention_mask=cross_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    **kwargs,
                )
                patch_embeds = patch_embeds + cross_attention_output
        encoder_cross_states = patch_embeds
        return hidden_states, encoder_cross_states

    def patch_reduce(self, hidden_states, max_num_patches, reduction, patch_ids):
        """
        Reduce variable length patches to single embedding per patch
        Note: this works with variable number of patches for different sequences in the batch
        It handles variable length patches by assuming that patch_lengths will be 0 for any
        extra patches on the *right*. Since there can be a variable number of patches
        this function also return the number of patches for each sequence in the batch.
        Any embeddings on the right that are not allocated to a patch
        (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
        will be sent to a dummy patch, which is trimmed before returning.
        """
        batch_size, _, embedding_dim = hidden_states.shape

        patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])

        reduced_embeddings = torch.zeros(
            (batch_size, max_num_patches, embedding_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        reduced_embeddings = reduced_embeddings.scatter_reduce(
            src=hidden_states,
            dim=1,
            index=patch_ids,
            reduce=reduction,
            include_self=False,
        )
        reduced_embeddings = reduced_embeddings[:, :max_num_patches, :]

        return reduced_embeddings


class BLTLocalDecoder(nn.Module):
    def __init__(self, config: BLTLocalDecoderConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.config = config
        self.cross_attn_decoder = True
        self.layers = nn.ModuleList(
            [BLTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = BLTRotaryEmbedding(config=config)
        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size_global,
            out_features=config.hidden_size * config.cross_attn_k,
            bias=False,
        )
        self.norm = BLTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_layers = nn.ModuleList()
        layers_to_add = config.num_hidden_layers if config.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BLTCrossAttention(config=config, layer_idx=layer_idx, hidden_size=config.hidden_size)
            )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, _, _ = embeds.shape
        hidden_states = embeds
        patch_embeds = self.patch_embedding_projection(patch_embeds)
        patch_embeds = patch_embeds.reshape(
            batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size
        )
        if patch_embeds is not None and not self.cross_attn_decoder:
            hidden_states = hidden_states + patch_embeds
        if position_ids is None:
            position_ids = (
                torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0).expand(batch_size, -1)
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if i == 0 or self.config.cross_attn_all_layers:
                # Remove cross_attention_states from kwargs if present to avoid multiple values error
                kwargs.pop("cross_attention_states", None)
                cross_attention_output, _ = self.cross_attn_layers[i](
                    hidden_states=hidden_states,
                    cross_attention_states=patch_embeds,
                    attention_mask=cross_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    **kwargs,
                )
                hidden_states = hidden_states + cross_attention_output
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        logits = self.norm(hidden_states)
        return logits


class BLTCrossAttention(nn.Module):
    """Cross-attention module for BLT, following transformers style"""

    def __init__(self, config: BLTConfig, layer_idx: int, hidden_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # needs to stay hidden_size, NOT head_dim
        self.q_norm = BLTRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = BLTRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(hidden_states)
        query_states = self.q_proj(query_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if cross_attention_states is not None:
            cross_attention_states = self.k_norm(cross_attention_states)
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        elif cache_position is not None and cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )
        attn_impl = getattr(self.config, "_attn_implementation", None) or "eager"
        attention_interface: Callable = eager_attention_forward
        if attn_impl != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = attn_output + hidden_states
        return attn_output, attn_weights


class BLTGlobalTransformer(nn.Module):
    def __init__(self, config: BLTGlobalTransformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(BLTTransformerLayer(config, layer_idx))
        self.rotary_emb = BLTRotaryEmbedding(config=config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = input_embeds.shape
        hidden_states = input_embeds
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        if position_ids is None:
            position_ids = (
                torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return hidden_states


def rolling_polynomial_hash(token_tensor, hash_func_nb: int = 0):
    primes = [
        1000000007,
        5915587277,
        1500450271,
        3267000013,
        5754853343,
        4093082899,
        9576890767,
        3628273133,
        2860486313,
        5463458053,
        3367900313,
    ]
    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64, device=token_tensor.device)
    powers = torch.arange(token_tensor.shape[-1], device=token_tensor.device)
    prime_powers = prime**powers
    return torch.sum(token_tensor * prime_powers, dim=-1)


def byte_group_hash_function(
    token_ids: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
):
    """Hash token groups and map to range [0, max_hash]."""
    with torch.no_grad():
        batch_size, seq_len = token_ids.shape
        # Add padding for sliding window
        padding = torch.zeros(batch_size, group_size - 1, dtype=torch.int64, device=token_ids.device)
        padded_tokens = torch.cat([padding, token_ids], dim=1)

        # Create sliding windows and compute hashes
        windows = padded_tokens.unfold(1, group_size, 1)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values = hashes % max_hash

    return hash_values


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """Compute token embeddings enhanced with hash-based embeddings."""
    embeddings = local_encoder.embed_tokens(local_encoder_tokens)
    embedding_idx = 0
    for func_nb in range(encoder_hash_byte_group_nb_functions):
        for group_size in encoder_hash_byte_group_size:
            hash_ids = byte_group_hash_function(
                local_encoder_tokens, group_size, func_nb, encoder_hash_byte_group_vocab
            )
            embeddings += encoder_hash_tok_embedding[embedding_idx](hash_ids)
            embedding_idx += 1

    return embeddings


def _prepare_patch_cross_attention_mask(
    patch_ids: torch.Tensor,
    num_patches: int,
    sequence_length: int,
    patches_as_queries: bool = False,
    cross_attn_k: int = 1,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare cross-attention mask for patch-based attention, following mllama's robust approach.

    This function creates masks that control which patches can attend to which other patches,
    with support for query/key role swapping and cross-attention multipliers.

    Args:
        patch_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        sequence_length (int): Length of the sequence.
        patches_as_queries (bool): If True, patches are used as queries, otherwise as keys.
        cross_attn_k (int): Cross-attention multiplier for repeating patches.
        dtype (torch.dtype): Data type for the output mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - cross_attention_mask: 4D tensor [batch_size, 1, q_len, kv_len]
            - full_text_row_masked_out_mask: 4D tensor indicating fully masked rows
    """
    batch_size, seq_len = patch_ids.shape
    device = patch_ids.device

    # Determine query and key lengths based on configuration
    if patches_as_queries:
        q_len = num_patches * cross_attn_k
        kv_len = sequence_length
        # Create patch-to-sequence mapping
        q_patch_ids = (
            torch.arange(num_patches, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, num_patches, seq_len)
        )
        kv_patch_ids = patch_ids.unsqueeze(1).expand(batch_size, num_patches, seq_len)
    else:
        q_len = sequence_length
        kv_len = num_patches * cross_attn_k
        # Create sequence-to-patch mapping
        q_patch_ids = patch_ids.unsqueeze(-1).expand(batch_size, seq_len, num_patches)
        kv_patch_ids = (
            torch.arange(num_patches, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, num_patches)
        )

    # Create base attention mask - boolean mask where True means "should attend"
    # Exact patch matching
    cross_attention_mask = q_patch_ids == kv_patch_ids

    # Handle cross_attn_k multiplier by repeating along appropriate dimension
    repeat_dim = 1 if patches_as_queries else -1
    cross_attention_mask = cross_attention_mask.repeat_interleave(cross_attn_k, dim=repeat_dim)

    # Validate dimensions
    expected_shape = (batch_size, q_len, kv_len)
    if cross_attention_mask.shape != expected_shape:
        raise ValueError(
            f"Cross attention mask shape {cross_attention_mask.shape} doesn't match expected {expected_shape}"
        )

    # Reshape so it can be used by attn module - add head dimension
    cross_attention_mask = cross_attention_mask.unsqueeze(1)  # [batch_size, 1, q_len, kv_len]

    # Invert the mask (following mllama pattern exactly)
    # True -> 0.0 (attend), False -> 1.0 (will become -inf)
    inverted_cross_attn_mask = 1.0 - cross_attention_mask.to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # Apply full-row bias (following mllama pattern exactly)
    # Return 4D tensor of shape [B, H, S1, 1] where value is 0 if a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


def process_patch_lengths(patch_lengths: torch.Tensor, max_patch_length: Optional[int]) -> torch.Tensor:
    """
    Splits patch lengths into smaller segments if they exceed `max_patch_length`.
    Pads the result to uniform length across the batch.

    Args:
        patch_lengths (torch.Tensor): [batch_size, num_patches] tensor of patch lengths.
        max_patch_length (int, optional): Maximum allowed length per patch.

    Returns:
        torch.Tensor: [batch_size, max_len] tensor of split and padded patch lengths.
    """
    if max_patch_length is None:
        return patch_lengths

    batch_size = patch_lengths.size(0)
    processed = []

    for seq in patch_lengths:
        splits = []
        for length in seq[seq > 0]:
            length = length.item()
            full_chunks, remainder = divmod(length, max_patch_length)
            splits.extend([max_patch_length] * full_chunks)
            if remainder:
                splits.append(remainder)
        processed.append(splits)

    # Find max length to pad to
    max_len = max(len(splits) for splits in processed)
    padded = torch.zeros((batch_size, max_len), dtype=patch_lengths.dtype, device=patch_lengths.device)

    for i, splits in enumerate(processed):
        if splits:
            padded[i, : len(splits)] = torch.tensor(splits, dtype=patch_lengths.dtype, device=patch_lengths.device)

    # Trim zero columns
    if (padded != 0).any(dim=0).sum() < padded.shape[1]:
        last_nonzero = (padded != 0).any(dim=0).nonzero().max().item() + 1
        padded = padded[:, :last_nonzero]

    return padded



@auto_docstring
class BLTPreTrainedModel(PreTrainedModel):
    """BLT PreTrainedModel inheriting from Mllama but with BLT-specific init."""

    config: BLTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BLTTransformerLayer", "BLTLocalEncoder", "BLTLocalDecoder", "BLTGlobalTransformer"]

    _supports_static_cache = False  # static cache cannot have different shapes for each layer
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    config_class = BLTConfig
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_cache_class = False

    _can_record_outputs = {
        "hidden_states": BLTDecoderLayer,
        "attentions": OutputRecorder(BLTSelfAttention, index=1, layer_name="local_decoder"),
        "encoder_attentions": OutputRecorder(BLTSelfAttention, index=1, layer_name="local_encoder"),
        "global_attentions": OutputRecorder(BLTSelfAttention, index=1, layer_name="global_transformer"),
    }

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, BLTRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# Top-level model classes
class BLTModel(BLTPreTrainedModel):
    def __init__(self, config: BLTConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.config = config
        self.local_encoder = BLTLocalEncoder(config.encoder_config)
        self.global_transformer = BLTGlobalTransformer(config.global_config)
        self.local_decoder = BLTLocalDecoder(config.decoder_config)
        num_embeddings = config.encoder_hash_byte_group_nb_functions * len(config.encoder_hash_byte_group_size)
        embeddings = [
            nn.Embedding(config.encoder_hash_byte_group_vocab, config.encoder_config.hidden_size)
            for _ in range(num_embeddings)
        ]
        self.encoder_hash_tok_embedding = nn.ModuleList(embeddings)
        if self.config.patch_in_forward:
            self.patcher = BLTPatcher(config.patcher_config)
            self.patcher.eval()
            for param in self.patcher.parameters():
                param.requires_grad = False
        else:
            self.patcher = None
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape
        else:
            batch_size, sequence_length, _ = inputs_embeds.shape
        if patch_lengths is None:
            if self.config.patching_mode == "entropy" and self.patcher is not None:
                if input_ids is None:
                    raise ValueError("input_ids is required for entropy-based patching")
                _, patch_lengths, _ = self.patcher(
                    input_ids,
                    patch_size=self.config.patch_size,
                    threshold=self.config.patching_threshold,
                    max_patch_length=self.config.max_patch_length,
                    patching_batch_size=self.config.patching_batch_size,
                    device=input_ids.device,
                )
            else:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                dtype = input_ids.dtype if input_ids is not None else inputs_embeds.dtype
                patch_lengths = process_patch_lengths(
                    torch.ones((batch_size, sequence_length + 1), dtype=dtype, device=device),
                    self.config.max_patch_length,
                )
        patch_ids = self._patch_ids_from_lengths(patch_lengths, sequence_length)
        if inputs_embeds is not None:
            encoder_embeds = inputs_embeds
        else:
            encoder_embeds = compute_hash_embeddings(
                input_ids,
                self.local_encoder,
                self.encoder_hash_tok_embedding,
                self.config.encoder_hash_byte_group_nb_functions,
                self.config.encoder_hash_byte_group_size,
                self.config.encoder_hash_byte_group_vocab,
            )
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + encoder_embeds.shape[1], device=encoder_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, encoder_embeds, cache_position, past_key_values
        )
        cross_attn_mask_enc, full_text_row_masked_out_mask_enc = _prepare_patch_cross_attention_mask(
            patch_ids, patch_lengths.shape[1], sequence_length, True, self.config.cross_attn_k, encoder_embeds.dtype
        )
        # Remove full_text_row_masked_out_mask from kwargs if present to avoid multiple values error
        kwargs.pop("full_text_row_masked_out_mask", None)
        encoder_hidden_states, encoder_cross_states = self.local_encoder(
            input_ids=input_ids,
            input_embeds=encoder_embeds,
            patch_embeds=None,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=None,
            cross_mask=cross_attn_mask_enc,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
            **kwargs,
        )
        global_hidden_states = encoder_cross_states.view(batch_size, patch_lengths.shape[1], -1)
        global_cache_position = torch.arange(
            0, global_hidden_states.shape[1], device=global_hidden_states.device
        )
        global_position_ids = global_cache_position.unsqueeze(0)
        global_causal_mask = self._update_causal_mask(
            None, global_hidden_states, global_cache_position, None
        )
        global_hidden_states = self.global_transformer(
            input_embeds=global_hidden_states,
            attention_mask=global_causal_mask,
            position_ids=global_position_ids,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        )
        decoder_patch_ids = self._patch_ids_from_lengths(patch_lengths[:, 1:], sequence_length)
        cross_attn_mask_dec, full_text_row_masked_out_mask_dec = _prepare_patch_cross_attention_mask(
            decoder_patch_ids,
            patch_lengths.shape[1],
            sequence_length,
            False,
            self.config.cross_attn_k,
            encoder_embeds.dtype,
        )
        output = self.local_decoder(
            input_ids=input_ids,
            embeds=encoder_hidden_states,
            patch_embeds=global_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            mask=None,
            cross_mask=cross_attn_mask_dec,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask_dec,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values if use_cache else None,
        )

    def get_input_embeddings(self):
        return self.local_encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.local_encoder.embed_tokens = value

    def _patch_ids_from_lengths(self, patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = patch_lengths.shape[0]
        patch_starts = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
                patch_lengths.cumsum(dim=-1)[:, :-1],
            ],
            dim=-1,
        )
        token_positions = torch.arange(seq_len, device=patch_lengths.device)
        return (patch_starts.unsqueeze(1) <= token_positions.unsqueeze(0).unsqueeze(-1)).sum(dim=-1) - 1


class BLTPatcher(BLTPreTrainedModel):
    def __init__(self, config: BLTPatcherConfig):
        super().__init__(config)
        self.rotary_emb = BLTRotaryEmbedding(config=self.config)
        self.layers = nn.ModuleList()
        for layer_idx in range(self.config.num_hidden_layers):
            self.layers.append(BLTTransformerLayer(self.config, layer_idx))
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.norm = BLTRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_values: torch.Tensor,
        patch_size: Optional[int] = None,
        threshold: Optional[float] = None,
        max_patch_length: Optional[int] = None,
        patching_batch_size: int = 1,
        device: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        entropies = []
        predictions = []
        max_length = self.config.max_position_embeddings
        batch_numel = max_length * patching_batch_size
        splits = torch.split(token_values.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(pad_size, dtype=split.dtype, device=split.device, requires_grad=False)
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            batch_size, sequence_length = split.shape
            input_embeds = self.embed_tokens(split)
            hidden_states = input_embeds
            batch_size, _, _ = input_embeds.shape
            position_ids = torch.arange(split.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            cache_position = torch.arange(sequence_length, device=input_embeds.device)
            causal_mask = self._update_causal_mask(
                None,  # attention_mask
                input_embeds,
                cache_position,
                None,  # past_key_values
            )

            for i, layer in enumerate(self.layers):
                layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask)
                hidden_states = layer_outputs[0]

            logits = self.lm_head(self.norm(hidden_states))
            logits = logits.reshape(-1, logits.shape[-1])[: split.numel() - pad_size, :]
            predictions.append(logits)
            prediction_entropies = torch.distributions.Categorical(logits=logits).entropy()
            entropies.append(prediction_entropies)
        concat_entropies = torch.cat(entropies, dim=0).reshape(token_values.shape)
        concat_predictions = torch.cat(predictions, dim=0).reshape(token_values.shape[0], -1)
        batch_size, sequence_length = token_values.shape
        if patch_size is not None:
            patch_lengths = self.patch_lengths_from_entropies(
                entropies=concat_entropies,
                sequence_length=sequence_length,
                patch_size=patch_size,
                threshold=threshold,
            )
        else:
            patch_lengths = torch.ones(
                (batch_size, sequence_length), dtype=token_values.dtype, device=token_values.device
            )
        patch_lengths = process_patch_lengths(patch_lengths, max_patch_length)
        return concat_entropies, patch_lengths, concat_predictions

    @staticmethod
    def patch_lengths_from_entropies(
        entropies,
        sequence_length,
        patch_size=None,
        threshold=None,
    ):
        """
        Computes patch lengths from token entropies.

        Depending on whether a threshold is provided, the function uses either:
        - Top-k selection based on entropy (when `threshold` is None), or
        - Thresholding the entropy values (when `threshold` is set).
        """

        batch_size = entropies.shape[0]

        # Always include token 0 and 1 as starting tokens
        init_tokens = (
            torch.tensor([0, 1], dtype=torch.long, device=entropies.device).unsqueeze(0).repeat(batch_size, 1)
        )
        offset = init_tokens.shape[1]

        # Ignore first token entropy (BOS)
        entropies = entropies[:, 1:]

        if threshold is None:
            # Use top-k entropy values to define patch start points
            num_patches = sequence_length // patch_size
            topk_indices = entropies.topk(num_patches - 2, dim=1).indices
            patch_starts = topk_indices.sort(dim=1).values
        else:
            # Threshold the entropy values to define patch start points
            patch_mask = entropies > threshold

            seq_len = patch_mask.shape[1]

            # Create patch IDs (token indices), and add a sentinel to ensure alignment
            token_indices = torch.arange(seq_len, device=entropies.device).unsqueeze(0).expand(batch_size, -1)
            sentinel = torch.full_like(token_indices, seq_len)
            padded_indices = torch.cat([token_indices, sentinel], dim=1)

            # Pad mask with inverse to align sentinel correctly
            padded_mask = torch.cat([patch_mask, ~patch_mask], dim=1)

            # Select indices where mask is True
            patch_starts = padded_indices[padded_mask].reshape(batch_size, seq_len)
            max_valid_patches = patch_mask.sum(dim=1).max()
            patch_starts = patch_starts[:, :max_valid_patches]

        # Offset patch starts to account for the two initial tokens
        patch_start_ids = torch.cat((init_tokens, patch_starts + offset), dim=1)

        # Compute patch end positions by shifting start positions
        last_token = torch.full_like(patch_start_ids[:, :1], sequence_length - 1)
        patch_ends = torch.cat((patch_start_ids[:, 1:] - 1, last_token), dim=1)

        patch_lengths = patch_ends - patch_start_ids + 1

        return patch_lengths


@auto_docstring(
    custom_intro="""
    The BLT Text Model with a language modeling head on top.
    """
)
class BLTForCausalLM(BLTPreTrainedModel, GenerationMixin):
    config: BLTConfig
    _supports_static_cache = True  # only the LLM without cross attn can do compile
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["BLTTransformerLayer", "BLTLocalEncoder", "BLTLocalDecoder", "BLTGlobalTransformer"]

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = config.vocab_size
        self.model = BLTModel(config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.local_encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.local_encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        cross_attention_states (`torch.FloatTensor`, *optional*):
            Output of the vision model, used for cross-attention. This tensor contains the processed image features that
            the language model will attend to.
        cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
            Cross-attention mask to control the interaction between text tokens and image tiles.
            This 4D tensor defines which image tiles each text token should attend to.

            For each text token (in seq_length):
            - 1 indicates the token **should attend** to the corresponding image tile
            - 0 indicates the token **should not attend** to the corresponding image tile
        full_text_row_masked_out_mask (`tuple[torch.Tensor, torch.Tensor]`, *optional*):
            A tuple containing two tensors that mask out rows in the cross-attention mechanism:
            - The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
              A value of 0 indicates that the corresponding text token's entire row in the cross-attention
              matrix should be masked out (all image tokens ignored).
            - The second tensor has the same shape and is used internally to apply the masking during
              the forward pass of cross-attention layers.
            This mask is derived from the cross_attention_mask and is used to handle cases where a text token
            should not attend to any image token.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BLTForCausalLM

        >>> model = BLTForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
        >>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

        >>> prompt = "If I had to write a haiku, it would be:"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
        >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(result)
        If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
        I love the idea of snowflakes gently falling, each one
        ```
        """
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
       # Add BLT-specific attention outputs
        if hasattr(outputs, 'encoder_attentions'):
            output.encoder_attentions = outputs.encoder_attentions
        if hasattr(outputs, 'global_attentions'):
            output.global_attentions = outputs.global_attentions
            
        return output


__all__ = [
    "BLTPreTrainedModel",
    "BLTModel",
    "BLTPatcher",
    "BLTForCausalLM",
]