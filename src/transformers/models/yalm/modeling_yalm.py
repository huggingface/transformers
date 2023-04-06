# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch YaLM model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_yalm import YalmConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "YalmConfig"


class YalmScaleMaskSoftmax(nn.Module):
    def __init__(self, mask_func, scale):
        super().__init__()
        self.mask_func = mask_func
        self.scale = scale

    def forward(self, input, mask):
        # [b, np, s, s]
        assert input.dim() == 4

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask)
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        return probs


class YalmRotaryPositionEncoding(nn.Module):
    def __init__(self, max_seq_length: int, hidden_size_per_attention_head: int):
        super().__init__()
        cos_cached, sin_cached = YalmRotaryPositionEncoding.get_cache_multipliers(
            max_seq_length, hidden_size_per_attention_head
        )
        self.register_buffer("cos_cached", cos_cached.unsqueeze(1).unsqueeze(2), persistent=False)
        self.register_buffer("sin_cached", sin_cached.unsqueeze(1).unsqueeze(2), persistent=False)

    def forward(self, hidden_state, context_position):
        seq_length = hidden_state.shape[0]
        cache_slice = slice(context_position, context_position + seq_length)
        return self.apply_rotary_position_encoding(
            hidden_state, self.cos_cached[cache_slice], self.sin_cached[cache_slice]
        )

    @staticmethod
    def get_cache_multipliers(max_seq_length, hidden_size):
        inv_freqs = 1e-4 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size)
        positions = torch.arange(max_seq_length, dtype=torch.float)
        angles = positions.unsqueeze(-1) * inv_freqs

        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def apply_rotary_position_encoding(hidden_state, cos_cached, sin_cached):
        sq, b, np, hn = hidden_state.shape
        half_hn = hn // 2
        left, right = hidden_state[..., :half_hn], hidden_state[..., half_hn:]
        encoded_left = cos_cached * left - sin_cached * right
        encoded_right = sin_cached * left + cos_cached * right
        return torch.cat((encoded_left, encoded_right), dim=3)


class YalmSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, config: YalmConfig, layer_idx: int):
        super().__init__()

        self.attention_mask_func = None
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = max(1, layer_idx)
        self.pos_encoding_type = "rotary"

        # Per attention head and per partition values.
        self.hidden_size_per_partition = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        if (self.hidden_size_per_attention_head * self.num_attention_heads) != self.hidden_size_per_partition:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.num_attention_heads_per_partition = config.num_attention_heads

        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.scale_attn_by_inverse_layer_idx:
            coeff = self.layer_idx + 1
            self.norm_factor *= coeff

        self.scale_mask_softmax = YalmScaleMaskSoftmax(self.attention_mask_func, coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

        # Output.
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.rotary_position_encoding = YalmRotaryPositionEncoding(
            config.max_position_embeddings,
            self.hidden_size_per_attention_head,
        )

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size()
        if num_splits_first:
            """[s, b, num_splits * np * hn]
            -->(view) [s, b, num_splits, np, hn]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn]"""

            intermediate_shape = input_shape[:-1] + (
                num_splits,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits]
            -->(view) [s, b, np, hn, num_splits]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn]"""

            intermediate_shape = input_shape[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
                num_splits,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)

        return mixed_layer

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor, int]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, self.hidden_size_per_partition, dim=-1)

        context_position = 0 if layer_past is None else layer_past[2]
        query_layer = self.rotary_position_encoding(query_layer, context_position)
        key_layer = self.rotary_position_encoding(key_layer, context_position)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value, sq_length = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
            sq_length += 1
        else:
            sq_length = key_layer.size()[0]

        present = (key_layer, value_layer, sq_length) if use_cache else None

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if layer_past is not None:
            attention_mask = attention_mask[..., attention_scores.size(3) - 1, : attention_scores.size(3)].unsqueeze(2)
        else:
            attention_mask = attention_mask[..., : attention_scores.size(3), : attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)
        output = (output, present)
        if output_attentions:
            outputs += (attention_probs,)

        return output


class YalmMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, config: YalmConfig):
        super().__init__()

        self.dense_ffn_hidden = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
        )

        self.activation_type = config.activation_type
        self.is_gated = config.activation_type in ["geglu"]

        self.activation_func = torch.nn.functional.gelu

        if self.is_gated:
            self.dense_ffn_gate = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
            )

        self.dense_ffn_output = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
        )

    def forward(self, hidden_states):
        intermediate_parallel, bias_parallel = self.dense_ffn_hidden(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        if self.is_gated:
            gate = self.dense_ffn_gate(hidden_states)[0]
            intermediate_gated = intermediate_parallel * gate
        else:
            intermediate_gated = intermediate_parallel

        output, output_bias = self.dense_ffn_output(intermediate_gated)
        return output, output_bias


class YalmTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, config: YalmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        if self.layer_idx > 0:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Self attention.
        self.attention = YalmSelfAttention(layer_idx)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the input data.
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # MLP
        self.mlp = YalmMLP()

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, int]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        if self.layer_idx > 0:
            attention_input = self.input_layernorm(hidden_states)
        else:
            attention_input = hidden_states

        # Self attention.
        attention_layer_outputs = self.attention(
            attention_input,
            attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attention_output = attention_layer_outputs[0]  # output_attn: attention_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = attention_input
        else:
            residual = hidden_states

        layernorm_input = (
            torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training) + residual
        )

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)
        residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training) + residual

        if use_cache:
            outputs = (output,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (output,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class YalmTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config: YalmConfig):
        super().__init__()

        # Number of layers:
        self.num_layers = config.num_layers

        self.layers = torch.nn.ModuleList([YalmTransformerLayer(layer_idx=i) for i in range(self.num_layers)])

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for layer_past
                        return module(*inputs, use_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # reverting data format change [s b h] --> [b s h]
        output = hidden_states.transpose(0, 1).contiguous()

        return output, presents, all_hidden_states, all_attentions


class YalmProjector(nn.Module):
    def __init__(self, config: YalmConfig):
        super().__init__()

        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        if not self.apply_residual_connection_post_layernorm:
            self.input_layernorm = nn.LayerNorm(config.embedding_size, eps=config.layernorm_epsilon)

        if config.embedding_size != config.hidden_size:
            self.register_buffer(
                "projector",
                torch.eye(config.embedding_size, config.hidden_size).to(config.params_dtype),
                persistent=False,
            )

    def forward(self, data):
        if self.apply_residual_connection_post_layernorm:
            hidden_states = data
        else:
            hidden_states = self.input_layernorm(data)

        if self.embedding_size != self.hidden_size:
            hidden_states = hidden_states @ self.projector

        return hidden_states


YALM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`YalmConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Yalm Model outputting raw hidden-states without any specific head on top.",
    YALM_START_DOCSTRING,
)
class YalmPreTrainedModel(PreTrainedModel):
    config_class = YalmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, YalmModel):
            module.gradient_checkpointing = value


YALM_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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
"""


@add_start_docstrings(
    "The bare YaLM Model outputting raw hidden-states without any specific head on top.",
    YALM_START_DOCSTRING,
)
class YalmModel(YalmPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`YalmDecoderLayer`]

    Args:
        config: YalmConfig
    """

    def __init__(self, config: YalmConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.padded_vocab_size = config.padded_vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size, self.padding_idx)
        self.projector = YalmProjector(config)
        self.transformer = YalmTransformer(config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(YALM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.projector(inputs_embeds)

        hidden_states, presents, all_hidden_states, all_attentions = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    """
    YaLM Model with a `language modeling` head on top (linear layer with weights tied to the input
    embeddings).
    """,
    YALM_START_DOCSTRING,
)
class YalmCausalLM(YalmPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: YalmConfig):
        super().__init__(config)

        self.yalm = YalmModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False)
        self.out_bias = torch.nn.Parameter(
            torch.zeros(
                config.padded_vocab_size,
            )
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    @add_start_docstrings_to_model_forward(YALM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, YalmForCausalLM, YalmConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("TODO")
        >>> config = YalmConfig.from_pretrained("TODO")
        >>> config.is_decoder = True
        >>> model = YalmForCausalLM.from_pretrained("TODO", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.yalm(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states) + self.out_bias

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
