# coding=utf-8
# Copyright 2023 Amazon and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Informer model."""

from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from ...cache_utils import EncoderDecoderCache
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
    auto_docstring,
    is_torch_flex_attn_available,
)
from ..bart.modeling_bart import BartAttention
from ..time_series_transformer.modeling_time_series_transformer import (
    TimeSeriesFeatureEmbedder,
    TimeSeriesMeanScaler,
    TimeSeriesNOPScaler,
    TimeSeriesSinusoidalPositionalEmbedding,
    TimeSeriesStdScaler,
    TimeSeriesTransformerDecoder,
    TimeSeriesTransformerDecoderLayer,
    TimeSeriesTransformerEncoder,
    TimeSeriesTransformerEncoderLayer,
    TimeSeriesTransformerForPrediction,
    TimeSeriesTransformerModel,
    TimeSeriesValueEmbedding,
)
from .configuration_informer import InformerConfig


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import make_flex_block_causal_mask


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


class InformerFeatureEmbedder(TimeSeriesFeatureEmbedder):
    pass


class InformerStdScaler(TimeSeriesStdScaler):
    pass


class InformerMeanScaler(TimeSeriesMeanScaler):
    pass


class InformerNOPScaler(TimeSeriesNOPScaler):
    pass


class InformerSinusoidalPositionalEmbedding(TimeSeriesSinusoidalPositionalEmbedding):
    pass


class InformerValueEmbedding(TimeSeriesValueEmbedding):
    pass


@auto_docstring
class InformerPreTrainedModel(PreTrainedModel):
    config: InformerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, InformerSinusoidalPositionalEmbedding):
            module._init_weight()

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask

    # Copied from transformers.models.musicgen.modeling_musicgen.MusicgenDecoder._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.config._attn_implementation == "sdpa":
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        elif self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            # Other attention flavors support in-built causal (when `mask is None`)
            # while we need to create our specific block mask regardless
            elif attention_mask is None:
                attention_mask = make_flex_block_causal_mask(
                    torch.ones(
                        size=(input_shape),
                        device=inputs_embeds.device,
                    )
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        return attention_mask

    # Copied from transformers.models.musicgen.modeling_musicgen.MusicgenDecoder._update_cross_attn_mask
    def _update_cross_attn_mask(
        self,
        encoder_hidden_states: Union[torch.Tensor, None],
        encoder_attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
    ):
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(encoder_attention_mask, torch.Tensor):
                    encoder_attention_mask = make_flex_block_causal_mask(
                        encoder_attention_mask,
                        query_length=input_shape[-1],
                        is_causal=False,
                    )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        return encoder_attention_mask


class InformerAttention(BartAttention):
    pass


class InformerProbSparseAttention(nn.Module):
    """Probabilistic Attention mechanism to select the "active"
    queries rather than the "lazy" queries and provides a sparse Transformer thus mitigating the quadratic compute and
    memory requirements of vanilla attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        sampling_factor: int = 5,
        bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.factor = sampling_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.layers[self.layer_idx].keys
            value_states = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(*kv_input_shape).transpose(1, 2)
            value_states = value_states.view(*kv_input_shape).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        key_states_time_length = key_states.size(1)  # L_K
        log_key_states_time_length = np.ceil(np.log1p(key_states_time_length)).astype("int").item()  # log_L_K

        query_states_time_length = query_states.size(1)  # L_Q
        log_query_states_time_length = np.ceil(np.log1p(query_states_time_length)).astype("int").item()  # log_L_Q

        u_part = min(self.factor * query_states_time_length * log_key_states_time_length, key_states_time_length)
        u = min(self.factor * log_query_states_time_length, query_states_time_length)

        if key_states_time_length > 0:
            index_sample = torch.randint(0, key_states_time_length, (u_part,))
            k_sample = key_states[:, index_sample, :]
        else:
            k_sample = key_states

        queries_keys_sample = torch.bmm(query_states, k_sample.transpose(1, 2))  # Q_K_sampled

        # find the Top_k query with sparsity measurement
        if u > 0:
            sparsity_measurement = queries_keys_sample.max(dim=-1)[0] - torch.div(
                queries_keys_sample.sum(dim=-1), key_states_time_length
            )  # M
            top_u_sparsity_measurement = sparsity_measurement.topk(u, sorted=False)[1]  # M_top

            # calculate q_reduce: query_states[:, top_u_sparsity_measurement]
            dim_for_slice = torch.arange(query_states.size(0)).unsqueeze(-1)
            q_reduce = query_states[dim_for_slice, top_u_sparsity_measurement]
        else:
            q_reduce = query_states
            top_u_sparsity_measurement = None

        # Use q_reduce to calculate attention weights
        attn_weights = torch.bmm(q_reduce, key_states.transpose(1, 2))

        src_len = key_states.size(1)
        if attn_weights.size() != (bsz * self.num_heads, u, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, u, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            prob_mask = attention_mask.expand(bsz, self.num_heads, tgt_len, src_len).reshape(
                bsz * self.num_heads, tgt_len, src_len
            )

            if top_u_sparsity_measurement is not None:
                dim_for_slice = torch.arange(prob_mask.size(0)).unsqueeze(-1)
                prob_mask = prob_mask[dim_for_slice, top_u_sparsity_measurement, :]

            attn_weights = attn_weights.view(bsz, self.num_heads, u, src_len) + prob_mask.view(
                bsz, self.num_heads, u, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, u, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, u, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, u, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, u, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, u, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        # calculate context for updating the attn_output, based on:
        # https://github.com/zhouhaoyi/Informer2020/blob/ac59c7447135473fb2aafeafe94395f884d5c7a5/models/attn.py#L74
        if self.is_decoder:
            # cast to float32 before operation to avoid overflow
            context = value_states.cumsum(dim=-2, dtype=torch.float32).to(value_states.dtype)
        else:
            v_mean_dim_time = value_states.mean(dim=-2)
            context = (
                v_mean_dim_time.unsqueeze(dim=1)
                .expand(bsz * self.num_heads, query_states_time_length, v_mean_dim_time.size(-1))
                .clone()
            )

        if top_u_sparsity_measurement is not None:
            # update context: copy the attention output to the context at top_u_sparsity_measurement index
            dim_for_slice = torch.arange(context.size(0)).unsqueeze(-1)
            context[dim_for_slice, top_u_sparsity_measurement, :] = attn_output
            attn_output = context

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# source: https://github.com/zhouhaoyi/Informer2020/blob/main/models/encoder.py
class InformerConvLayer(GradientCheckpointingLayer):
    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class InformerEncoderLayer(TimeSeriesTransformerEncoderLayer):
    def __init__(self, config: InformerConfig):
        super().__init__(config)

        del self.self_attn

        if config.attention_type == "prob":
            self.self_attn = InformerProbSparseAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                sampling_factor=config.sampling_factor,
            )
        else:
            self.self_attn = InformerAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )


class InformerDecoderLayer(TimeSeriesTransformerDecoderLayer):
    def __init__(self, config: InformerConfig, layer_idx: Optional[int] = None):
        super().__init__(config)

        del self.self_attn

        if config.attention_type == "prob":
            self.self_attn = InformerProbSparseAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                sampling_factor=config.sampling_factor,
                is_decoder=True,
                layer_idx=layer_idx,
            )
        else:
            self.self_attn = InformerAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                config=config,
                layer_idx=layer_idx,
            )


class InformerEncoder(TimeSeriesTransformerEncoder):
    def __init__(self, config: InformerConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.gradient_checkpointing = False
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = InformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList([InformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        if config.distil:
            self.conv_layers = nn.ModuleList(
                [InformerConvLayer(config.d_model) for _ in range(config.encoder_layers - 1)]
            )
            self.conv_layers.append(None)
        else:
            self.conv_layers = [None] * config.encoder_layers

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        Args:
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
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

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())

        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, (encoder_layer, conv_layer) in enumerate(zip(self.layers, self.conv_layers)):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                if conv_layer is not None:
                    output = conv_layer(layer_outputs[0])
                    layer_outputs = (output,) + layer_outputs[1:]

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class InformerDecoder(TimeSeriesTransformerDecoder):
    def __init__(self, config: InformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = InformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList([InformerDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class InformerModel(TimeSeriesTransformerModel, nn.Module):
    def __init__(self, config: InformerConfig):
        nn.Module().__init__(config)

        if config.scaling == "mean" or config.scaling is True:
            self.scaler = InformerMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = InformerStdScaler(config)
        else:
            self.scaler = InformerNOPScaler(config)

        if config.num_static_categorical_features > 0:
            self.embedder = InformerFeatureEmbedder(
                cardinalities=config.cardinality,
                embedding_dims=config.embedding_dimension,
            )

        # transformer encoder-decoder and mask initializer
        self.encoder = InformerEncoder(config)
        self.decoder = InformerDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **super_kwargs):
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):
            Past values of the time series, that serve as context in order to predict the future. The sequence size of
            this tensor must be larger than the `context_length` of the model, since the model will use the larger size
            to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
            context".

            The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
            `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
            look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
            the past.

            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

            Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
            Required time features, which the model internally will add to `past_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step. Holiday features are also a good example of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
            `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
            Optional static categorical features for which the model will learn an embedding, which it will add to the
            values of the time series.

            Static categorical features are features which have the same value for all time steps (static over time).

            A typical example of a static categorical feature is a time series ID.
        static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
            Optional static real features which the model will add to the values of the time series.

            Static real features are features which have the same value for all time steps (static over time).

            A typical example of a static real feature is promotion information.
        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`.

            The sequence length here is equal to `prediction_length`.

            See the demo notebook and code snippets for details.

            Optionally, during training any missing values need to be replaced with zeros and indicated via the
            `future_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):
            Required time features for the prediction window, which the model internally will add to `future_values`.
            These could be things like "month of year", "day of the month", etc. encoded as vectors (for instance as
            Fourier features). These could also be so-called "age" features, which basically help the model know "at
            which point in life" a time-series is. Age features have small values for distant past time steps and
            increase monotonically the more we approach the current time step. Holiday features are also a good example
            of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import InformerModel

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        super().forward(**super_kwargs)


class InformerForPrediction(TimeSeriesTransformerForPrediction, nn.Module):
    def __init__(self, config: InformerConfig):
        nn.Module().__init__(config)

        self.model = InformerModel(config)
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.d_model)
        self.target_shape = self.distribution_output.event_shape

        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # Initialize weights of distribution_output and apply final processing
        self.post_init()

    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):
            Past values of the time series, that serve as context in order to predict the future. The sequence size of
            this tensor must be larger than the `context_length` of the model, since the model will use the larger size
            to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
            context".

            The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
            `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
            look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
            the past.

            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

            Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
            Required time features, which the model internally will add to `past_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step. Holiday features are also a good example of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
            `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
            Optional static categorical features for which the model will learn an embedding, which it will add to the
            values of the time series.

            Static categorical features are features which have the same value for all time steps (static over time).

            A typical example of a static categorical feature is a time series ID.
        static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
            Optional static real features which the model will add to the values of the time series.

            Static real features are features which have the same value for all time steps (static over time).

            A typical example of a static real feature is promotion information.
        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`.

            The sequence length here is equal to `prediction_length`.

            See the demo notebook and code snippets for details.

            Optionally, during training any missing values need to be replaced with zeros and indicated via the
            `future_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):
            Required time features for the prediction window, which the model internally will add to `future_values`.
            These could be things like "month of year", "day of the month", etc. encoded as vectors (for instance as
            Fourier features). These could also be so-called "age" features, which basically help the model know "at
            which point in life" a time-series is. Age features have small values for distant past time steps and
            increase monotonically the more we approach the current time step. Holiday features are also a good example
            of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        future_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            This mask is used to filter out missing values for the final loss calculation.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import InformerForPrediction

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = InformerForPrediction.from_pretrained(
        ...     "huggingface/informer-tourism-monthly"
        ... )

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # during inference, one only provides past values
        >>> # as well as possible additional features
        >>> # the model autoregressively generates future values
        >>> outputs = model.generate(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> mean_prediction = outputs.sequences.mean(dim=1)
        ```"""
        super().forward(**super_kwargs)


__all__ = ["InformerForPrediction", "InformerModel", "InformerPreTrainedModel"]
