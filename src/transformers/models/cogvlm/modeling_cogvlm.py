# coding=utf-8
# Copyright 2024 THUDM and The HuggingFace Team. All rights reserved.
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
""" PyTorch CogVLM model."""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.logging import get_logger

from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_cogvlm import CogvlmConfig


if TYPE_CHECKING:
    from transformers.utils import ModelOutput

logger = get_logger(__name__)

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

_CONFIG_FOR_DOC = "CogvlmConfig"


COGVLM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CogvlmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

COGVLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class CogvlmPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.proj = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        embeddings = self.proj(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_token, embeddings), dim=1)
        embeddings += self.position_embedding.weight.unsqueeze(0)
        return embeddings


class CogvlmVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, sequence_length, _ = hidden_state.shape
        qkv = self.query_key_value(hidden_state)

        # reshape to (3, batch_size, sequence_length, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # import xformers.ops as xops

        # out = xops.memory_efficient_attention(
        #     queries,
        #     keys,
        #     values,
        #     scale=self.scale,
        # )

        # output = self.dense(out.view(batch_size, sequence_length, -1))
        # output = self.output_dropout(output)
        # return output

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = queries * self.scale
        keys = keys
        attention_scores = queries @ keys.transpose(-2, -1)

        # PyTorch already accumulates softmax on fp32 (Reference: https://github.com/pytorch/pytorch/pull/103167)
        attention_probs = attention_scores.softmax(-1)
        attention_output = attention_probs @ values
        attention_output = attention_output.transpose(1, 2).contiguous()

        output = self.dense(attention_output.view(batch_size, sequence_length, -1))
        output = self.output_dropout(output)
        return output


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->CogvlmVision
class CogvlmVisionMLP(nn.Module):
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


class CogvlmVisionTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = CogvlmVisionAttention(config)
        self.mlp = CogvlmVisionMLP(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states

        attention_output = self.attention(attention_input)

        attention_output = self.input_layernorm(attention_output)
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class CogvlmVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CogvlmVisionTransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for idx, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class CogvlmVisionGLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_state):
        hidden_state = self.linear_proj(hidden_state)
        hidden_state = self.act1(self.norm1(hidden_state))
        hidden_state = self.act2(self.gate_proj(hidden_state)) * self.dense_h_to_4h(hidden_state)
        hidden_state = self.dense_4h_to_h(hidden_state)
        return hidden_state


class CogvlmVisionModel(nn.Module):
    def __init__(self, config: CogvlmConfig):
        super().__init__()

        self.patch_embedding = CogvlmPatchEmbedding(config.vision_config)
        self.transformer = CogvlmVisionTransformer(config.vision_config)
        self.linear_proj = CogvlmVisionGLU(config, in_features=config.vision_config.hidden_size)
        # parameters for beginning of image (boi) and end of image (eoi)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.patch_embedding(pixel_values)
        hidden_state = self.transformer(hidden_state)
        hidden_state = hidden_state[:, 1:]
        hidden_state = self.linear_proj(hidden_state)
        beginning_of_image_features = self.boi.expand(hidden_state.shape[0], -1, -1)
        end_of_image_features = self.eoi.expand(hidden_state.shape[0], -1, -1)
        hidden_state = torch.cat((beginning_of_image_features, hidden_state, end_of_image_features), dim=1)
        return hidden_state


# Copied from transformers.models.mistral.modeling_mistral.MistralRMSNorm with Mistral->Cogvlm
class CogvlmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        CogvlmRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    # Ignore copy
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Cogvlm
class CogvlmMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def get_expert_mask(token_type_ids: torch.LongTensor) -> (torch.BoolTensor, torch.BoolTensor):
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (
        token_type_ids[:, 1:] == VISION_TOKEN_TYPE
    )
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class CogvlmVisionExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = CogvlmMLP(config)
        self.vision_mlp = CogvlmMLP(config)

    def forward(self, hidden_states: torch.FloatTensor, token_type_ids: torch.LongTensor):
        output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


class CogvlmRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device) / self.dim))

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[:, None, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, :].to(dtype), persistent=False)

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


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


class CogvlmVisionExpertAttention(nn.Module):
    def __init__(self, config: CogvlmConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx

        self.rotary_emb = CogvlmRotaryEmbedding(self.head_dim)
        self.vision_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.vision_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.language_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.language_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Reference: https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118 and SDPA C++ implementation.
        self.sqrt_scale = math.sqrt(1 / math.sqrt(self.head_dim))

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [batch_size, sequence_length, num_head*head_dim] into a 4D tensor with size
        [batch_size, num_heads, seq_length, head_dim]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, q_len, hidden_size = hidden_states.size()
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        mixed_raw_layer = torch.empty(
            (batch_size, q_len, hidden_size * 3), dtype=hidden_states.dtype, device=hidden_states.device
        )
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])

        query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        query_states = self._transpose_for_scores(query_states)
        key_states = self._transpose_for_scores(key_states)
        value_states = self._transpose_for_scores(value_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        cos, sin = (
            nn.functional.embedding(position_ids, cos.squeeze(1)),
            nn.functional.embedding(position_ids, sin.squeeze(1)),
        )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=1
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(query_states * self.sqrt_scale, (key_states * self.sqrt_scale).transpose(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # PyTorch already accumulates softmax on fp32 (Reference: https://github.com/pytorch/pytorch/pull/103167)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        context_layer = torch.matmul(attn_weights, value_states)

        if context_layer.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.size()}"
            )

        context_layer = context_layer.transpose(1, 2).contiguous()

        context_layer = context_layer.reshape(batch_size, q_len, self.hidden_size)

        attn_output = torch.empty(context_layer.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])

        return (
            (attn_output, attn_weights, past_key_value) if output_attentions else (attn_output, None, past_key_value)
        )


class CogvlmVisionExpertSdpaAttention(CogvlmVisionExpertAttention):
    """
    CogVLM attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `CogvlmVisionExpertAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from CogvlmVisionExpertAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "CogVLM is using CogvlmVisionExpertSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

        batch_size, q_len, hidden_size = hidden_states.size()
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        mixed_raw_layer = torch.empty(
            (batch_size, q_len, hidden_size * 3), dtype=hidden_states.dtype, device=hidden_states.device
        )
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])

        query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        query_states = self._transpose_for_scores(query_states)
        key_states = self._transpose_for_scores(key_states)
        value_states = self._transpose_for_scores(value_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        cos, sin = (
            nn.functional.embedding(position_ids, cos.squeeze(1)),
            nn.functional.embedding(position_ids, sin.squeeze(1)),
        )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=1
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_mask_bool = attention_mask == 0
        is_full = (attention_mask_bool > 0).all()

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=not is_full,
        )

        if context_layer.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.size()}"
            )

        context_layer = context_layer.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.hidden_size)

        attn_output = torch.empty(context_layer.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])

        return (attn_output, None, past_key_value)


COGVLM_ATTENTION_CLASSES = {
    "eager": CogvlmVisionExpertAttention,
    "sdpa": CogvlmVisionExpertSdpaAttention,
}


class CogvlmDecoderLayer(nn.Module):
    def __init__(self, config: CogvlmConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = COGVLM_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = CogvlmVisionExpertMLP(config)
        self.input_layernorm = CogvlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CogvlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CogvlmPreTrainedModel(PreTrainedModel):
    config_class = CogvlmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _no_split_modules = ["CogvlmDecoderLayer", "CogvlmVisionTransformerLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

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


def build_position_ids(
    token_type_ids: torch.BoolTensor, attention_mask: Optional[torch.BoolTensor] = None
) -> torch.LongTensor:
    """
    Create position_ids based on provided token_type_ids and attention_mask.
    """

    if attention_mask is not None:
        tmp = token_type_ids.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = token_type_ids.clone()

    # image beginning-of-image (boi), end-of-image (eoi) token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(token_type_ids, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= tmp[:, 0] == VISION_TOKEN_TYPE
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= tmp[:, -1] == VISION_TOKEN_TYPE
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE

    # final position ids
    position_ids = torch.zeros_like(token_type_ids, dtype=torch.long)
    position_ids[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (
        (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    )
    position_ids = position_ids.cumsum(dim=-1)
    return position_ids


@add_start_docstrings(
    """
    CogVLM model without any head on top, just outputting raw hidden states.
    """,
    COGVLM_START_DOCSTRING,
)
class CogvlmModel(CogvlmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.vision = CogvlmVisionModel(config)
        self.num_vision_tokens = (
            self.config.vision_config.image_size // self.config.vision_config.patch_size
        ) ** 2 + 2

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CogvlmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CogvlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        vision_input_ids = torch.tensor(
            [self.config.bos_token_id] + [self.config.pad_token_id] * self.num_vision_tokens,
        )
        self.register_buffer("vision_input_ids", vision_input_ids, persistent=False)
        vision_token_type_ids = torch.tensor([LANGUAGE_TOKEN_TYPE] + [VISION_TOKEN_TYPE] * self.num_vision_tokens)
        self.register_buffer("vision_token_type_ids", vision_token_type_ids, persistent=False)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def encode_images(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        images_features = self.vision(pixel_values)
        return images_features

    @add_start_docstrings_to_model_forward(COGVLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: List[List[torch.Tensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored,
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import CogvlmProcessor, CogvlmModel
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> processor = CogvlmProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        >>> model = CogvlmModel.from_pretrained("THUDM/cogvlm-chat-hf")

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query = "Describe this image"

        >>> prompt = f"Question: {query} Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None:
            # generate mode with past_key_values. the image features are already mapped
            # update attention_mask
            if pixel_values is not None:
                batch_size = pixel_values.shape[0]
                vision_mask = torch.full(
                    size=(batch_size, self.num_vision_tokens), fill_value=1, device=attention_mask.device
                )
                attention_mask = torch.cat(
                    [attention_mask[:, :-1], vision_mask, attention_mask[:, -1].repeat(batch_size, 1)], dim=1
                )
        else:
            # not allow for inputs_embeds, because we want to process image feature
            # assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"

            if pixel_values is not None and input_ids is not None:
                # multi-modality
                if len(input_ids) != len(pixel_values):
                    raise ValueError("Make sure to pass as many texts as images")

                # prepend the input_ids and token_type_ids with image tokens
                batch_size = input_ids.shape[0]

                vision_input_ids = self.vision_input_ids.repeat(batch_size, 1)
                vision_input_ids = vision_input_ids.to(input_ids.device)

                vision_token_type_ids = self.vision_token_type_ids.repeat(batch_size, 1)
                vision_token_type_ids = vision_token_type_ids.to(token_type_ids.device)

                input_ids = torch.cat([vision_input_ids, input_ids[:, 1:]], dim=1)
                token_type_ids = torch.cat([vision_token_type_ids, token_type_ids[:, 1:]], dim=1)
                attention_mask = torch.ones_like(input_ids)

                inputs_embeds = self.embed_tokens(input_ids)

                images_features = self.encode_images(pixel_values)
                images_features = images_features.reshape(-1, images_features.shape[-1])
                images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)

            else:
                # TODO verify single-modality
                if token_type_ids is None:
                    token_type_ids = (
                        torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device) * LANGUAGE_TOKEN_TYPE
                    )
                if (token_type_ids == VISION_TOKEN_TYPE).any():
                    raise ValueError("Token type ids should not contain the VISION_TOKEN_TYPE")
                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)

            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None

        # next: forward pass, which largely is copy from llama and adapted to add `token_type_ids`

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


@add_start_docstrings(
    """
    CogVLM model with a language modeling head on top (a linear layer on top of the hidden states).
    """,
    COGVLM_START_DOCSTRING,
)
class CogvlmForCausalLM(CogvlmPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.model = CogvlmModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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

    @add_start_docstrings_to_model_forward(COGVLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored,
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import CogvlmProcessor, CogvlmForCausalLM
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> processor = CogvlmProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        >>> model = CogvlmForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf")

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query = "Describe this image"

        >>> prompt = f"Question: {query} Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")
        >>> outputs = model.generate(**inputs)

        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)  # type: ignore

    def prepare_inputs_for_generation(
        self,
        input_ids,
        token_type_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        if past_key_values is not None:
            if position_ids is None:
                # the reason we add + 2 + 1 here is because we have 2 additional vision tokens,
                # and we need to add 1 to take into account the one extra token that is going to
                # be sent through the model
                position_ids = build_position_ids(token_type_ids, attention_mask) + 2 + 1
            position_ids = position_ids[:, -1:]
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "pixel_values": pixel_values,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: "ModelOutput",
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        model_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = (
                torch.ones(size=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype, device=token_type_ids.device)
                * LANGUAGE_TOKEN_TYPE
            )
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token_type_ids], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
