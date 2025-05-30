# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mllama model."""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast, \
    BaseModelOutputWithPastAndCrossAttentions
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from .configuration_blt import BLTConfig, MllamaTextConfig, MllamaVisionConfig

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)


def _prepare_cross_attention_mask(
        cross_attention_mask: torch.Tensor,
        num_vision_tokens: int,
        dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


def _prepare_aspect_ratio_attention_mask(
        aspect_ratio_mask: torch.Tensor,
        num_patches: int,
        target_length: int,
        dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size ** -0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MllamaVision
class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


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


# Copied from transformers.models.llama.modeling_llama.eager_attention_forward
def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
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


class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, q_seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionAttention(config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)

        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state, attn_weights = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = self.gate_attn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        if self.is_gated:
            hidden_state = self.gate_ffn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MllamaVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MllamaEncoderLayer`].

    Args:
        config: BLTConfig
    """

    def __init__(self, config: MllamaVisionConfig, num_layers=32, is_gated=False):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


#Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->MllamaText
class MllamaTextRMSNorm1(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MllamaTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * normed

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return self.weight * (x / (rms + self.eps))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class BLTCrossAttention(nn.Module):
    def __init__(
        self,
        config: Optional[MllamaTextConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Cross attention specific norms
        self.cross_attn_norm_q = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_norm_kv = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        _, kv_len, _ = cross_attention_states.shape

        # Normalize inputs
        query_states = self.cross_attn_norm_q(hidden_states)
        key_value_states = self.cross_attn_norm_kv(cross_attention_states)

        # Project queries, keys and values
        query_states = self.q_proj(query_states)
        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim)

        # Repeat KV heads if needed
        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.transpose(1, 2)  # (bsz, num_heads, kv_len, head_dim)
        value_states = value_states.transpose(1, 2)  # (bsz, num_heads, kv_len, head_dim)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Add residual connection
        attn_output = hidden_states + attn_output

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# class BLTCrossAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(
#             self,
#             config: Optional[MllamaTextConfig] = None,
#             layer_idx: Optional[int] = None,
#     ):
#         super().__init__()
#         self.config = config
#         self.num_heads = self.config.num_attention_heads
#         self.num_key_value_heads = self.config.num_key_value_heads
#         self.dropout = config.dropout
#         self.hidden_size = config.hidden_size
#         self.head_dim = config.hidden_size // self.num_heads
#         self.layer_idx = layer_idx
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.scaling = self.head_dim ** -0.5

#         # Apply normalization before projection
#         self.q_norm = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
#         self.k_norm = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
#         self.v_norm = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
#         self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             cross_attention_states: Optional[torch.Tensor] = None,
#             past_key_value: Optional[Cache] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             output_attentions: bool = False,
#             use_cache: Optional[bool] = None,
#             cache_position: Optional[torch.LongTensor] = None,
#             **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""
#         bsz, q_len, _ = hidden_states.size()

#         # TODO: equivalent to BLT in CrossAttention.forward --> self.cross_attn_norm_q(x) --> RMSNorm, the mormalize before projection
#         query_states = self.q_proj(query_states)
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         query_states = self.q_norm(query_states)

#         if cross_attention_states is not None:
#             # TODO: check blt which Normalize before projection for key and value
#             # key_states = self.k_norm(cross_attention_states)
#             # value_states = self.v_norm(cross_attention_states)
#             key_states = self.k_proj(cross_attention_states)
#             value_states = self.v_proj(cross_attention_states)
#             key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#             value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#             key_states = repeat_kv(key_states, self.num_key_value_groups)
#             value_states = repeat_kv(value_states, self.num_key_value_groups)

#             #TODO: blt would then skip the norm here:
#             key_states = self.k_norm(key_states)
#             if past_key_value is not None:
#                 # if we have a new image + new tokens, we only computed key_states on that new image
#                 # we still update the cross key states, past_image, new_image. And use it!
#                 key_states, value_states = past_key_value.update(
#                     key_states, value_states, self.layer_idx, {"cache_position": cache_position}
#                 )
#         elif cache_position[0] != 0:
#             key_states, value_states = (
#                 past_key_value.key_cache[self.layer_idx],
#                 past_key_value.value_cache[self.layer_idx],
#             )
#         else:
#             raise ValueError(
#                 "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
#             )

#         attention_interface: Callable = eager_attention_forward

#         if self.config._attn_implementation != "eager":
#             if self.config._attn_implementation == "sdpa" and output_attentions:
#                 logger.warning_once(
#                     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
#                     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#                 )
#             else:
#                 attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.dropout,
#             scaling=self.scaling,
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
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


class BLTSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: torch.Tensor,
            output_attentions: bool = False,
            use_cache: bool = False,
            past_key_value=None,
            cache_position=None,
            **kwargs,
    ):
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
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BLTFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        hidden_dim = 4 * self.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if getattr(config, "ffn_dim_multiplier", None) is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        self.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BLTLocalDecoderLayer(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model

        # Add patch projection layer
        self.patch_embedding_projection = nn.Linear(
            config.patch_embedding_dim,  # Need to add this to config
            config.hidden_size,
            bias=False
        )

        # BLT flags
        self.cross_attn_decoder = getattr(config, 'cross_attn_decoder', False)
        self.cross_attn_all_layers_decoder = getattr(config, 'cross_attn_all_layers_decoder', False)

        self.self_attn = BLTSelfAttention(config=config, layer_idx=layer_idx)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = BLTCrossAttention(config, layer_idx=layer_idx)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        self.ffn_norm = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.ffn = BLTFeedForward(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            # No pre layer norm
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            # No dropout yet
            # TODO: = BLT does x + output INSIDE CrossAttention, we return the o_proj only in cross_attn, so we need to add the (+ x) here:
            # in BLT: hidden_states == output, residual == x (hidden_states that were passed in)
            h_cross = residual + hidden_states

            # TODO: = BLT THEN again does h = h + cross_attn - CHECK THIS - adding residual twice?
            hidden_states = residual + h_cross

        # Self Attention #TODO: below is all in TransformerBlock, h=layer(
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=self.self_attn_layer_norm(hidden_states),
            # TODO: = BLT, attn_out = self.attention(self.attention_norm(x), in TransformerBlock.forward,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        hidden_states_norm = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.ffn(hidden_states_norm)
        # end TransformerBlock , h=layer)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def _create_patch_projection(self, args):
        dimension_mismatch = (getattr(args, "dim_patch_emb") and args.dim_patch_emb != self.dim)

        # Check cross attention conditions
        cross_attn_conditions = (args.cross_attn_encoder and args.cross_attn_init_by_pooling) or (
                    args.cross_attn_decoder and args.cross_attn_init_by_pooling)

        if not (dimension_mismatch or cross_attn_conditions):
            return None

        output_dim = args.dim_token_emb * (self.cross_attn_k or 1)

        patch_embeds = nn.Linear(
            in_features=args.dim_patch_emb,
            out_features=output_dim,
            bias=False,
        )

        return patch_embeds.reshape(
            bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
        )


class BLTLocalEncoderLayer(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_patches = config.num_patches  # TODO: length of pathces from batch_to_tensors_and_gpu?
        self.hidden_size = config.hidden_size
        self.cross_attn_k = getattr(config, 'cross_attn_k', 1)


        # Add patch projection layer
        # BLT flags
        self.cross_attn_decoder = getattr(config, 'cross_attn_decoder', False)
        self.cross_attn_all_layers_decoder = getattr(config, 'cross_attn_all_layers_decoder', False)

        self.self_attn = BLTSelfAttention(config=config, layer_idx=layer_idx)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = BLTCrossAttention(config, layer_idx=layer_idx)

        self.ffn_norm = MllamaTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
       # self.attention_norm = nn.RMSNorm(1024, 1e-05).cuda()
        self.ffn = BLTFeedForward(config)


    @staticmethod
    def patch_reduce(h, max_num_patches, reduction, patch_ids):
        """
        TODO: copied from BLT
        Reduce variable length patches to single embedding per patch
        Note: this works with variable number of patches for different sequences in the batch
        It handles variable length patches by assuming that patch_lengths will be 0 for any
        extra patches on the *right*. Since there can be a variable number of patches
        this function also return the number of patches for each sequence in the batch.
        Any embeddings on the right that are not allocated to a patch
        (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
        will be sent to a dummy patch, which is trimmed before returning.
        """
        bs, seq_len, emb_dim = h.shape

        patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

        reduced_embs = torch.zeros(
            (bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device
        )
        reduced_embs = reduced_embs.scatter_reduce(
            src=h,
            dim=1,
            index=patch_ids,
            reduce="amax",
            include_self=False,
        )
        reduced_embs = reduced_embs[:, :max_num_patches, :]

        return reduced_embs


    def forward(
            self,
            hidden_states: torch.Tensor,
            patch_ids,  # TODO: BLT check this ITA
            patch_embedding_projection, 
            position_embeddings: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # Self Attention #TODO: below is all in TransformerBlock, h=layer(
        residual = hidden_states
        norm_hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=norm_hidden_states,
            # TODO: = BLT, attn_out = self.attention(self.attention_norm(x), in TransformerBlock.forward,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )

        hidden_states = residual + hidden_states
        hidden_states_norm = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.ffn(hidden_states_norm)
        # end TransformerBlock , h=layer)

        # Cross-Attention "apply_cross_attention" Block
        max_num_patches = self.num_patches

        # BLT pooling_downsample with "max"
        patch_embeds = self.patch_reduce(hidden_states, max_num_patches, "max", patch_ids)
        if patch_embedding_projection is not None:
            patch_embeds = patch_embedding_projection(patch_embeds)  
            bs = hidden_states.shape[0] 
            dim = hidden_states.shape[2]
            patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, dim
                )    
            
        # End Cross-Attention "apply_cross_attention" Block

        # Cross-Attention Block
        cross_attn_weights = None
        if patch_embeds is not None:
            residual = hidden_states
     #       hidden_states = hidden_states + patch_embeds

            # No pre layer norm
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                cross_attention_states=patch_embeds,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            # No dropout yet
            # TODO: = BLT does x + output INSIDE CrossAttention, we return the o_proj only in cross_attn, so we need to add the (+ x) here:
            # in BLT: hidden_states == output, residual == x (hidden_states that were passed in)
            h_cross = residual + hidden_states

            # TODO: = BLT THEN again does h = h + cross_attn - CHECK THIS - adding residual twice?
            hidden_states = residual + h_cross

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class MllamaRotaryEmbedding(nn.Module):
    def __init__(self, config: MllamaTextConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling["rope_type"]
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


BLTConfig
class MllamaPreTrainedModel(PreTrainedModel):
    config_class = BLTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "MllamaVisionEncoderLayer",
        "MllamaCrossAttentionDecoderLayer",
        "MllamaSelfAttentionDecoderLayer",
    ]
    _supports_cache_class = True
    _supports_static_cache = False  # static cache cannot have different shapes for each layer
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_quantized_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

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
        elif isinstance(module, MllamaTextRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, MllamaVisionModel):
            nn.init.normal_(module.class_embedding.data, std=std)
        elif isinstance(module, MllamaPrecomputedPositionEmbedding):
            nn.init.normal_(module.embedding.data, std=std)
            nn.init.zeros_(module.gate.data)
        elif isinstance(module, MllamaVisionEncoderLayer) and module.is_gated:
            nn.init.normal_(module.gate_attn.data, std=std)
            nn.init.normal_(module.gate_ffn.data, std=std)
        elif isinstance(module, BLTLocalDecoderLayer):
            module.cross_attn_attn_gate.data.zero_()
            module.cross_attn_mlp_gate.data.zero_()
        elif isinstance(module, MllamaPrecomputedAspectRatioEmbedding):
            if module.is_gated:
                module.gate.data.zero_()

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
            self,
            attention_mask: Union[torch.Tensor, "BlockMask"],
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool = False,
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
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
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
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
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


@auto_docstring(
    custom_intro="""
    The Mllama Vision Model which consists of two vision encoders.
    """
)
class MllamaVisionModel(MllamaPreTrainedModel):
    config_class = MllamaVisionConfig
    base_model_prefix = "vision_model"

    def __init__(self, config: MllamaVisionConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size ** -0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

        self.post_init()

    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        return self.patch_embedding

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    @auto_docstring
    def forward(
            self,
            pixel_values: torch.Tensor,
            aspect_ratio_ids: torch.Tensor,
            aspect_ratio_mask: torch.Tensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        r"""
        aspect_ratio_ids (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*):
            Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
            These ids correspond to indices in the model's list of supported aspect ratios, offset by 1.

            For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:
            - An image with aspect ratio [1, 1] would have ID 1
            - An image with aspect ratio [1, 2] would have ID 2
            - An image with aspect ratio [2, 1] would have ID 3

            The id 0 is reserved for padding (i.e., no image).

            If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
        aspect_ratio_mask (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*):
            Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:

            - 1 for tiles that are **not masked**,
            - 0 for tiles that are **masked**.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaVisionModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaVisionModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 1, 4, 1025, 7680])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # Patch embedding
        target_dtype = self.patch_embedding.weight.dtype
        target_device = self.patch_embedding.weight.device
        patch_embeds = self.patch_embedding(pixel_values.to(target_device, target_dtype))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_state = output[0]

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]

        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = [output[1][i] for i in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            # global transformer in contrast to `self.transformer` doesn't always return hidden states so we might go index out-of-range
            global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


@auto_docstring(
    custom_intro="""
    The BLT Local Decoder Model which consists of transformer with self and cross attention layers.
    """
)
class BLTLocalDecoder(MllamaPreTrainedModel):
    config_class = MllamaTextConfig
    base_model_prefix = "language_model.model"

    def __init__(self, config: MllamaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size + 8, config.hidden_size, self.padding_idx)

        # Add patch projection layer
        self.patch_embedding_projection = _create_patch_projection()

        # BLT flags
        self.cross_attn_decoder = getattr(config, 'cross_attn_decoder', False)
        self.cross_attn_all_layers_decoder = getattr(config, 'cross_attn_all_layers_decoder', False)

        # Create all layers as self-attention layers
        self.layers = nn.ModuleList([
            BLTLocalDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MllamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cross_attention_states: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            patch_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        full_text_row_masked_out_mask (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
            A tuple containing two tensors that mask out rows in the cross-attention mechanism:
            - The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
              A value of 0 indicates that the corresponding text token's entire row in the cross-attention
              matrix should be masked out (all image tokens ignored).
            - The second tensor has the same shape and is used internally to apply the masking during
              the forward pass of cross-attention layers.
            This mask is derived from the cross_attention_mask and is used to handle cases where a text token
            should not attend to any image token.

        Example:

        ```python
        >>> from transformers import AutoProcessor, MllamaTextModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaTextModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> text = "<|image|>If I had to write a haiku for this one"
        >>> inputs = processor(text=text, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 13, 4096])
        ```
        patch_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_embedding_dim)`, *optional*):
            Patch embeddings to be projected and combined with token embeddings. These are typically the output of a vision encoder
            or patch embedding layer. If provided, they will be projected to the model's hidden size and added to the token embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        if inputs_embeds is None:  # TODO: blt asserts defined so might rm
            inputs_embeds = self.embed_tokens(input_ids)

        # TODO: check this: Project and combine patch embeddings
        if patch_embeds is not None:
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            hidden_states = inputs_embeds + patch_embeds
        else:
            hidden_states = inputs_embeds

        # TODO: eq to h=F.dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # TODO: BLT =  h_cross = self.cross_attn_layers[i](
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    cross_attention_states,
                    cross_attention_mask,
                    causal_mask,
                    full_text_row_masked_out_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    cross_attention_states=cross_attention_states,
                    cross_attention_mask=cross_attention_mask,
                    attention_mask=causal_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # TODO: BLT = LocalDecoder norms to h_preds
        hidden_states = self.norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)  # applied twice?
        # TODO:  h_preds = self.output(h_preds) # in causal lm (head)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring(
    custom_intro="""
    The Mllama Text Model with a language modeling head on top.
    """
)
class MllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    config_class = MllamaTextConfig
    _supports_static_cache = True  # only the LLM without cross attn can do compile
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = MllamaTextModel._from_config(self.text_config)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cross_attention_states: Optional[torch.LongTensor] = None,
            cross_attention_mask: Optional[torch.LongTensor] = None,
            full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        full_text_row_masked_out_mask (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
            A tuple containing two tensors that mask out rows in the cross-attention mechanism:
            - The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
              A value of 0 indicates that the corresponding text token's entire row in the cross-attention
              matrix should be masked out (all image tokens ignored).
            - The second tensor has the same shape and is used internally to apply the masking during
              the forward pass of cross-attention layers.
            This mask is derived from the cross_attention_mask and is used to handle cases where a text token
            should not attend to any image token.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MllamaForCausalLM

        >>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Mllama model which consists of a vision encoder and a language model without language modeling head.
    """
)
class MllamaModel(MllamaPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}
    _supports_quantized_cache = False  # quant cache not supported in encoder-decoder setting

    def __init__(self, config: BLTConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.vision_model = MllamaVisionModel._from_config(config.vision_config)
        self.language_model = MllamaTextModel._from_config(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            aspect_ratio_mask: Optional[torch.Tensor] = None,
            aspect_ratio_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            cross_attention_states: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        aspect_ratio_mask (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*):
            Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:

            - 1 for tiles that are **not masked**,
            - 0 for tiles that are **masked**.
        aspect_ratio_ids (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*):
            Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
            These ids correspond to indices in the model's list of supported aspect ratios, offset by 1.

            For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:
            - An image with aspect ratio [1, 1] would have ID 1
            - An image with aspect ratio [1, 2] would have ID 2
            - An image with aspect ratio [2, 1] would have ID 3

            The id 0 is reserved for padding (i.e., no image).

            If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
        cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
            Cross-attention mask to control the interaction between text tokens and image tiles.
            This 4D tensor defines which image tiles each text token should attend to.

            For each text token (in seq_length):
            - 1 indicates the token **should attend** to the corresponding image tile
            - 0 indicates the token **should not attend** to the corresponding image tile
        cross_attention_states (`torch.FloatTensor`, *optional*):
            Output of the vision model, used for cross-attention. This tensor contains the processed image features that
            the language model will attend to.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Mllama model which consists of a vision encoder and a language model.
    """,
)
class MllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_model": "model.vision_model",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _supports_quantized_cache = False  # quant cache not supported in encoder-decoder setting
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BLTConfig):
        super().__init__(config)
        self.model = MllamaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_model(self):
        return self.model.vision_model

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            aspect_ratio_mask: Optional[torch.Tensor] = None,
            aspect_ratio_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            cross_attention_states: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        aspect_ratio_mask (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*):
            Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:

            - 1 for tiles that are **not masked**,
            - 0 for tiles that are **masked**.
        aspect_ratio_ids (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*):
            Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
            These ids correspond to indices in the model's list of supported aspect ratios, offset by 1.

            For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:
            - An image with aspect ratio [1, 1] would have ID 1
            - An image with aspect ratio [1, 2] would have ID 2
            - An image with aspect ratio [2, 1] would have ID 3

            The id 0 is reserved for padding (i.e., no image).

            If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
        cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
            Cross-attention mask to control the interaction between text tokens and image tiles.
            This 4D tensor defines which image tiles each text token should attend to.

            For each text token (in seq_length):
            - 1 indicates the token **should attend** to the corresponding image tile
            - 0 indicates the token **should not attend** to the corresponding image tile
        cross_attention_states (`torch.FloatTensor`, *optional*):
            Output of the vision model, used for cross-attention. This tensor contains the processed image features that
            the language model will attend to.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> prompt = "<|image|>If I had to write a haiku for this one"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> output = model.generate(**inputs, max_new_tokens=15)

        >>> prompt_len = inputs.input_ids.shape[-1]
        >>> generated_ids = output[:, prompt_len:]
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(generated_text)
        [', it would be:.\\nA stop sign in Chinatown.\\n']
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            cross_attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            position_ids=None,
            pixel_values=None,
            aspect_ratio_ids=None,
            aspect_ratio_mask=None,
            cross_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            logits_to_keep=None,
            **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["aspect_ratio_ids"] = None
            model_inputs["aspect_ratio_mask"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs


class BLTLocalEncoder(MllamaPreTrainedModel):
    config_class = MllamaTextConfig
    base_model_prefix = "language_model.model"

    def __init__(self, config: MllamaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.dropout = config.dropout
        self.embed_tokens = nn.Embedding(config.vocab_size + 8, config.hidden_size, self.padding_idx)

        # BLT flags
        self.cross_attn_encoder = getattr(config, 'cross_attn_encoder', False)
        self.cross_attn_all_layers_encoder = getattr(config, 'cross_attn_all_layers_encoder', False)

        # Create all layers as self-attention layers
        self.layers = nn.ModuleList([
            BLTLocalEncoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

         # Add patch embedding projection
        self.patch_embedding_projection = self._create_patch_projection(config)

      #  self.norm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MllamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _create_patch_projection(self, config):
        # Always create the projection if cross attention is enabled
        if config.cross_attn_encoder:
            output_dim = config.dim_token_emb * (getattr(config, 'cross_attn_k', 1) or 1)
            return nn.Linear(
                in_features=config.dim_patch_emb,
                out_features=output_dim,
                bias=False,
            )
        return None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            patch_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cross_attention_states: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            patch_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutput]:
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
        full_text_row_masked_out_mask (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
            A tuple containing two tensors that mask out rows in the cross-attention mechanism:
            - The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
              A value of 0 indicates that the corresponding text token's entire row in the cross-attention
              matrix should be masked out (all image tokens ignored).
            - The second tensor has the same shape and is used internally to apply the masking during
              the forward pass of cross-attention layers.
            This mask is derived from the cross_attention_mask and is used to handle cases where a text token
            should not attend to any image token.

        patch_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_embedding_dim)`, *optional*):
            Patch embeddings to be projected and combined with token embeddings. These are typically the output of a vision encoder
            or patch embedding layer. If provided, they will be projected to the model's hidden size and added to the token embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # encoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and cross_attention_states is not None) else None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # TODO: = BLT  h = self.apply_embedding(tokens, embeds)

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)  # = BLT h = F.dropout(h, p=self.dropout, training=self.training)

        # create position embeddings to be shared across the encoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # = BLT self.rope

        # encoder layers
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    cross_attention_states,
                    cross_attention_mask,
                    attention_mask,
                    full_text_row_masked_out_mask,
                    position_ids,
                    None,  # No past key values for encoder
                    output_attentions,
                    False,  # No cache for encoder
                    None,  # No cache position for encoder
                    position_embeddings,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    patch_ids=patch_ids,
                    encoder_hidden_states=cross_attention_states,
                    patch_embedding_projection=self.patch_embedding_projection,
                    cross_attention_mask=cross_attention_mask,
                    attention_mask=attention_mask,
               #     full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    past_key_value=None,  # No past key values for encoder
                    output_attentions=output_attentions,
                    use_cache=False,  # No cache for encoder
                    cache_position=None,  # No cache position for encoder
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

      #  hidden_states = self.norm(hidden_states)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if cross_attention_states is not None:
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last encoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


__all__ = [
    "MllamaForConditionalGeneration",
    "MllamaForCausalLM",
    "MllamaTextModel",
    "MllamaVisionModel",
    "MllamaPreTrainedModel",
    "MllamaModel",
    "BLTLocalEncoder",
    "BLTLocalDecoder",
]
