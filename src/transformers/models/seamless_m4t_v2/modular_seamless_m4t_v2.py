# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch SeamlessM4Tv2 model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from ..bart.modeling_bart import BartAttention
from ..m2m_100.modeling_m2m_100 import (
    M2M100ScaledWordEmbedding,
    M2M100SinusoidalPositionalEmbedding,
)
from ..nllb_moe.modeling_nllb_moe import NllbMoeDenseActDense
from ..seamless_m4t.modeling_seamless_m4t import (
    SeamlessM4TCodeHifiGan,
    SeamlessM4TConformerAdapter,
    SeamlessM4TConformerAdapterLayer,
    SeamlessM4TConformerFeatureProjection,
    SeamlessM4TConformerFeedForward,
    SeamlessM4TDecoder,
    SeamlessM4TDecoderLayer,
    SeamlessM4TEncoder,
    SeamlessM4TEncoderLayer,
    SeamlessM4TForSpeechToSpeech,
    SeamlessM4TForSpeechToText,
    SeamlessM4TForTextToSpeech,
    SeamlessM4TForTextToText,
    SeamlessM4TGenerationOutput,
    SeamlessM4THifiGan,
    SeamlessM4TModel,
    SeamlessM4TPreTrainedModel,
    SeamlessM4TSpeechEncoder,
    SeamlessM4TTextToUnitForConditionalGeneration,
    SeamlessM4TTextToUnitModel,
    format_speech_generation_kwargs,
)
from ..speecht5.modeling_speecht5 import HifiGanResidualBlock
from ..wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoderLayer,
)
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config


logger = logging.get_logger(__name__)


class SeamlessM4Tv2GenerationOutput(SeamlessM4TGenerationOutput):
    pass


@dataclass
class SeamlessM4Tv2TextToUnitOutput(ModelOutput):
    """
        Class defining the outputs from [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] and
        [`SeamlessM4Tv2TextToUnitModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    padding_mask: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


@dataclass
class SeamlessM4Tv2TextToUnitDecoderOutput(ModelOutput):
    """
    Class defining the outputs from [`SeamlessM4Tv2TextToUnitDecoder`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    padding_mask: Optional[torch.Tensor] = None

def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask

############ SPEECH ENCODER related code ################


class SeamlessM4Tv2ConformerFeatureProjection(SeamlessM4TConformerFeatureProjection):
    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states.to(self.layer_norm.weight.dtype))
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SeamlessM4Tv2ConformerFeedForward(SeamlessM4TConformerFeedForward):
    pass


class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block. Uses a causal depthwise convolution similar to that
    described in Section 2.1 of `https://doi.org/10.48550/arxiv.1609.03499"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError(
                "`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding"
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = torch.nn.functional.pad(
            hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0)
        )

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SeamlessM4Tv2ConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4Tv2ConformerSelfAttention object.
    Can be enhanced with relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        super().__init__()

        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        self.num_heads = config.speech_encoder_attention_heads
        self.position_embeddings_type = (
            config.position_embeddings_type if use_position_embeddings else None
        )

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = (
                self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            )
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = torch.arange(
                query_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(
                distance, -self.left_max_position_embeddings, self.right_max_position_embeddings
            )

            positional_embedding = self.distance_embedding(
                distance + self.left_max_position_embeddings
            )
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = torch.einsum(
                "bhld,lrd->bhlr", query, positional_embedding
            )
            attn_weights = attn_weights + (
                relative_position_attn_weights / math.sqrt(self.head_size)
            )

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = torch.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_size
        )
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class SeamlessM4Tv2ConformerEncoderLayer(Wav2Vec2ConformerEncoderLayer):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""
    def __init__(self, config):
        super().__init__(config)
        dropout = config.speech_encoder_dropout
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)


    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


class SeamlessM4Tv2ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList(
            [
                SeamlessM4Tv2ConformerEncoderLayer(config)
                for _ in range(config.speech_encoder_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def _apply_chunk_attention(self, attention_mask, hidden_states):
        """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
        sequence_len = hidden_states.shape[1]

        chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
        chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()

        start_indices = torch.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(
                min=0
            )
            start_indices = start_indices * self.config.speech_encoder_chunk_size
            start_indices = start_indices
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)

        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(
            max=sequence_len
        )

        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)

        indices = (
            torch.arange(sequence_len, device=hidden_states.device)
            .unsqueeze(0)
            .expand(sequence_len, -1)
        )

        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)

        attention_mask = (
            chunk_mask if attention_mask is None else (attention_mask.bool() | chunk_mask)
        )
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        if self.config.speech_encoder_chunk_size is not None:
            attention_mask = self._apply_chunk_attention(attention_mask, hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = (
                True
                if self.training and (dropout_probability < self.config.speech_encoder_layerdrop)
                else False
            )
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                        conv_attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SeamlessM4Tv2ConformerAdapterLayer(SeamlessM4TConformerAdapterLayer):
    pass


class SeamlessM4Tv2ConformerAdapter(SeamlessM4TConformerAdapter):
    pass


############ TEXT / UNITS related code ################


class SeamlessM4Tv2ScaledWordEmbedding(M2M100ScaledWordEmbedding):
    pass


class SeamlessM4Tv2SinusoidalPositionalEmbedding(M2M100SinusoidalPositionalEmbedding):
    pass


class SeamlessM4Tv2Attention(BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_length = hidden_states.shape[:2]

        # use encoder_hidden_states if cross attention
        current_states = (
            encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        )
        # checking that the `sequence_length` of the `past_key_value` is the same as the he provided
        # `encoder_hidden_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value
            and past_key_value[0].shape[2] == current_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k_proj(current_states))
            value_states = self._shape(self.v_proj(current_states))
            if past_key_value is not None and not is_cross_attention:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        query_states = self._shape(self.q_proj(hidden_states) * self.scaling)
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).type_as(
            attention_scores
        )
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        #  attn_output = torch.bmm(attn_probs, value_states) ?
        context_states = torch.matmul(attn_weights, value_states)
        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim) ?
        context_states = (
            context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        )
        attn_output = self.out_proj(context_states)

        if output_attentions:
            return attn_output, attn_weights, past_key_value
        else:
            return attn_output, None, past_key_value


class SeamlessM4Tv2FeedForwardNetwork(NllbMoeDenseActDense):
    def __init__(self, config: SeamlessM4Tv2Config, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)


class SeamlessM4Tv2EncoderLayer(SeamlessM4TEncoderLayer):
    pass


class SeamlessM4Tv2DecoderLayer(SeamlessM4TDecoderLayer):
    pass


class SeamlessM4Tv2TextToUnitDecoderLayer(nn.Module):
    def __init__(
        self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None
    ):
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = (
            config.decoder_attention_heads
            if decoder_attention_heads is None
            else decoder_attention_heads
        )
        self.dropout = config.dropout
        self.embed_dim = config.hidden_size

        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.conv1 = nn.Conv1d(
            self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding="same"
        )
        self.activation_fn = ACT2FN[config.activation_function]
        self.conv2 = nn.Conv1d(
            self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding="same"
        )

        self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
        self.conv_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*
                or 0 for *masked*
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Conv
        residual = hidden_states

        # Apply padding mask to avoid leaking padded positions in the convolution layer
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.transpose(1, 2)).transpose(1, 2)

        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv2(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = self.conv_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.conv_layer_norm(hidden_states)

        outputs = (hidden_states, present_key_value)

        if output_attentions:
            outputs += self_attn_weights

        return outputs


############ SUB-MODELS related code ################


@auto_docstring
class SeamlessM4Tv2PreTrainedModel(SeamlessM4TPreTrainedModel):
    config_class = SeamlessM4Tv2Config
    base_model_prefix = "seamless_m4t_v2"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "SeamlessM4Tv2EncoderLayer",
        "SeamlessM4Tv2DecoderLayer",
        "SeamlessM4Tv2ConformerEncoderLayer",
        "SeamlessM4Tv2TextToUnitDecoderLayer",
    ]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SeamlessM4Tv2ConformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, SeamlessM4Tv2ConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def compute_last_hidden_states_per_sample(self):
        raise AttributeError("Not needed for SeamlessM4Tv2")


    def _indices_to_subwords(self, input_ids):
        """
        Returns the corresponding text string for each input id.
        """
        if not hasattr(self.generation_config, "id_to_text"):
            raise ValueError(
                """This model generation config doesn't have a `id_to_text` key which maps
                token ids to subwords. Make sure to load the right generation config."""
            )
        batch_size, sequence_len = input_ids.shape

        subwords_batch = []
        for batch_id in range(batch_size):
            subwords = []
            for i in range(sequence_len):
                subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
                subwords.append(str(subword))
            subwords_batch.append(subwords)
        return subwords_batch

    def _count_character_length_in_subword(
        self,
        input_ids,
        subwords_batch,
        merge_space_with_prev_subword=False,
        pad_token_id=0,
        unk_token_id=1,
        space="▁",
    ):
        """
        Counts the number of characters per text string associated with the input token id.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            merge_space_with_prev_subword (`bool`, *optional*, defaults to `False`):
                Indicates if the space character is merged with the previous subword. If `False`, it will be merged
                with the next subword.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
            space (`str`, *optional*, defaults to `"▁"`):
                The space character.
        """
        batch_size, _ = input_ids.shape

        char_count_per_id = input_ids.new_zeros(input_ids.size())

        subword_lens = input_ids.ne(pad_token_id).sum(1)

        for batch_id in range(batch_size):
            # We slice out the tensor till the padding index.
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]

            is_next_start_with_space = [
                len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space
                if i < len(subwords) - 1
                else False
                for i in range(len(subwords))
            ]
            is_punc = [
                len(subwords[i]) == 1
                and not subwords[i].isalpha()
                and not subwords[i].isnumeric()
                and subwords[i] != space
                for i in range(len(subwords))
            ]
            for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
                if subword_idx == pad_token_id:
                    break

                if subword_idx == unk_token_id:
                    # We set char_len to 1 for an unk token.
                    char_len = 1

                    if merge_space_with_prev_subword and is_next_start_with_space[i]:
                        char_len += 1
                else:
                    # By default, spaces are merged with the next subword.
                    # char_len includes the space.
                    char_len = len(subword)

                    if merge_space_with_prev_subword:
                        # Add the space for the next subword.
                        if is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the current subword.
                        if i > 0 and is_next_start_with_space[i - 1]:
                            char_len -= 1
                    else:
                        # Merge space with punctuation mark by default.
                        if is_punc[i] and is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the subword succeeding the punctuation mark.
                        elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
                            char_len -= 1

                char_count_per_id[batch_id, i] = char_len

        return char_count_per_id

    def _get_char_input_ids(
        self, input_ids, subwords_batch, char_count_per_id, pad_token_id=0, unk_token_id=1
    ):
        """
        Returns the corresponding character input id for each character of `subwords_batch`.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
        Returns:
            `torch.Tensor`: Tensor of shape `(batch_size, char_sequence_length)` containing the id of each character.
        """
        if not hasattr(self.generation_config, "char_to_id"):
            raise ValueError(
                """This model generation config doesn't have a `char_to_id` key which maps
                characters to character ids. Make sure to load the right generation config."""
            )

        batch_size = input_ids.shape[0]
        max_len = int(char_count_per_id.sum(1).max().item())

        char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)

        subword_lens = input_ids.ne(pad_token_id).sum(1)

        for batch_id in range(batch_size):
            total = 0
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == unk_token_id:
                    char_ids = [unk_token_id]
                else:
                    # Get char token indices corresponding to the subwords.
                    char_ids = [
                        self.generation_config.char_to_id.get(ch, unk_token_id)
                        for ch in list(subword)
                    ]
                char_seq_len = len(char_ids)
                char_seqs[batch_id, total : total + char_seq_len] = torch.tensor(char_ids).to(
                    char_seqs
                )
                total += char_seq_len
        return char_seqs

    def _hard_upsample(self, hidden_states, durations):
        """
        Repeats the time dimension of each sample in the batch based on the corresponding duration.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, *)`, *optional*):
                The sequence to repeat, where `*` is any number of sequence-specific dimensions including none.
            durations (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates how many times to repeat time segments.
        """
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
        else:
            # if batched sample, need to interleave per sample, and pad -> loss of parallelism
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning_once(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=0)
                for (hidden_state, duration) in zip(hidden_states, durations)
            ]

            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)

        return hidden_states


@auto_docstring(
    custom_intro="""
    Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.
    Each layer is a [`SeamlessM4Tv2ConformerEncoderLayer`].
    """
)
class SeamlessM4Tv2SpeechEncoder(SeamlessM4TSpeechEncoder):
    pass


@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4Tv2EncoderLayer`].
    """
)
class SeamlessM4Tv2Encoder(SeamlessM4TEncoder):
    pass


@auto_docstring(
    custom_intro="""
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4Tv2DecoderLayer`].
    """
)
class SeamlessM4Tv2Decoder(SeamlessM4TDecoder):
    pass


@auto_docstring(
    custom_intro="""
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4Tv2DecoderLayer`].
    """
)
class SeamlessM4Tv2TextToUnitDecoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        r"""
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # if embed_tokens defined, use its shape instead
            self.embed_tokens = nn.Embedding(
                embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx
            )
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        self.embed_char = nn.Embedding(config.char_vocab_size, config.hidden_size)
        self.embed_char_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.pos_emb_alpha_char = nn.Parameter(torch.ones(1))
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.duration_predictor = SeamlessM4Tv2VariancePredictor(
            config.variance_predictor_embed_dim,
            config.variance_predictor_hidden_dim,
            config.variance_predictor_kernel_size,
            config.variance_pred_dropout,
        )

        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4Tv2TextToUnitDecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        char_input_ids: Optional[torch.LongTensor] = None,
        char_count_per_id: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SeamlessM4Tv2TextToUnitDecoderOutput]:
        r"""
        Args:
            char_input_ids (`torch.LongTensor` of shape `(batch_size, char_sequence_length)`):
                Character indices. The correspondence between characters and indices can be found in `char_to_id`, a
                dictionary in the generation configuration.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, encoder_sequence_length)`):
                Number of characters per text input id.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # create padding mask for character lengths
        char_padding_mask = _compute_new_attention_mask(char_input_ids, char_count_per_id.sum(1))

        # upsample hidden states according to characters sequence lengths
        char_hidden_states = self._hard_upsample(encoder_hidden_states, char_count_per_id)
        # embed char positions
        char_positions = self.pos_emb_alpha_char * self.embed_char_positions(
            inputs_embeds=char_hidden_states
        )
        # update char hidden states with positions and char embeddings
        char_hidden_states = (
            self.embed_char(char_input_ids) * self.embed_scale + char_positions + char_hidden_states
        )

        # predict duration
        log_dur_pred = self.duration_predictor(char_hidden_states, padding_mask=char_padding_mask)
        dur_out = torch.clamp(torch.round((torch.expm1(log_dur_pred))).long(), min=1)
        dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0.0)

        # upsample char hidden states according to predicted duration
        char_hidden_states = self._hard_upsample(char_hidden_states, dur_out)

        positions = self.pos_emb_alpha * self.embed_positions(inputs_embeds=char_hidden_states)
        hidden_states = char_hidden_states + positions

        padding_mask = _compute_new_attention_mask(hidden_states, dur_out.sum(1))
        attention_mask = _prepare_4d_attention_mask(padding_mask, hidden_states.dtype)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    padding_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, padding_mask]
                if v is not None
            )
        return SeamlessM4Tv2TextToUnitDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            padding_mask=padding_mask,
        )


@auto_docstring(
    custom_intro="""
    Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4Tv2Encoder`] without embeddings and the decoder is a [`SeamlessM4Tv2TextToUnitDecoder`].
    """
)
class SeamlessM4Tv2TextToUnitModel(SeamlessM4TTextToUnitModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        r"""
        embed_tokens_decoder (`nn.Embedding`, *optional*):
            input embedding of the decoder.
        """
        super().__init__(config)
        self.decoder = SeamlessM4Tv2TextToUnitDecoder(config, embed_tokens_decoder)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        char_input_ids: Optional[torch.LongTensor] = None,
        char_count_per_id: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn, padding_mask)
        decoder_outputs = self.decoder(
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            padding_mask=decoder_outputs.padding_mask,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4Tv2TextToUnitModel`].
    """
)
class SeamlessM4Tv2TextToUnitForConditionalGeneration(SeamlessM4TTextToUnitForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        char_input_ids: Optional[torch.LongTensor] = None,
        char_count_per_id: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        char_input_ids (`torch.LongTensor` of shape `(batch_size, char_sequence_length)`):
            Character indices. The correspondence between characters and indices can be found in `char_to_id`, a
            dictionary in the generation configuration.
        char_count_per_id (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Number of characters per input id.
        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=lm_logits,
            padding_mask=outputs.padding_mask,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loss=masked_lm_loss,
        )

    def prepare_decoder_input_ids_from_labels(self):
        raise AttributeError("Not needed for SeamlessM4Tv2")

    def _reorder_cache(self):
        raise AttributeError("Not needed for SeamlessM4Tv2")


############ VOCODER related code ################


class HifiGanResidualBlock(HifiGanResidualBlock):
    pass


class SeamlessM4Tv2VariancePredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, kernel_size, var_pred_dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(
            embed_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        # Input: B x T x C; Output: B x T
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv2(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(dim=2)


class SeamlessM4Tv2HifiGan(SeamlessM4THifiGan):
    pass


@auto_docstring(
    custom_intro="""
    Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).
    """
)
class SeamlessM4Tv2CodeHifiGan(SeamlessM4TCodeHifiGan, nn.Module):
    config_class = SeamlessM4Tv2Config
    main_input_name = "input_embeds"
    _no_split_modules = []

    def __init__(self, config):
        nn.Module.__init__(config)

        self.pad_token_id = config.t2u_pad_token_id
        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout
        self.dur_predictor = SeamlessM4Tv2VariancePredictor(
            embed_dim, embed_dim, kernel_size, var_pred_dropout
        )

        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        self.hifi_gan = SeamlessM4Tv2HifiGan(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan.forward with SeamlessM4T->SeamlessM4Tv2, spkr_id->speaker_id
    def forward(
        self, input_ids: torch.LongTensor, speaker_id: torch.Tensor, lang_id: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4Tv2TextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            speaker_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        spkr = self.speaker_embedding(speaker_id).transpose(1, 2)
        lang = self.language_embedding(lang_id).transpose(1, 2)

        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        dur_out = torch.clamp(torch.round((torch.expm1(log_dur_pred))).long(), min=1)
        # B x C x T
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            # if batched sample, need to interleave per sample, and pad -> loss of parallelism
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]

            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)

        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)

        hidden_states = self.hifi_gan(hidden_states)

        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths


############ WHOLE MODEL related code ################


@auto_docstring(
    custom_intro="""
    The text-to-text SeamlessM4Tv2 Model transformer which can be used for T2TT.
    """
)
class SeamlessM4Tv2ForTextToText(SeamlessM4TForTextToText):
    pass


@auto_docstring(
    custom_intro="""
    The speech-to-text SeamlessM4Tv2 Model transformer which can be used for S2TT.
    """
)
class SeamlessM4Tv2ForSpeechToText(SeamlessM4TForSpeechToText):
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]


@auto_docstring(
    custom_intro="""
    The text-to-speech SeamlessM4Tv2 Model transformer which can be used for T2ST.
    """
)
class SeamlessM4Tv2ForTextToSpeech(SeamlessM4TForTextToSpeech):
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, SeamlessM4Tv2GenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.


        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor]]`:
            - If `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
              sequence_length)` and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds"))

        if tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
        else:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {",".join(lang_code_to_id.keys())}. Note that SeamlessM4Tv2 supports
                    more languages for text translation than for speech synthesis."""
                    )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")

        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size, device=self.device)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        text_generation_output = super().generate(input_ids, **kwargs_text)
        sequences = text_generation_output.sequences

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        if attention_mask is not None:
            # repeat attention mask alongside batch dimension
            attention_mask = torch.repeat_interleave(attention_mask, num_return_sequences, dim=0)
        encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        # repeat attention mask alongside batch dimension
        encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, num_return_sequences, dim=0)

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences[:, :-1],  # Manually trim the final EOS token
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences[:, :-1] != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # REMOVE EOS and lang_id
        t2u_input_ids = sequences[:, 2:-1]
        # replace every other EOS
        t2u_input_ids = torch.masked_fill(
            t2u_input_ids, t2u_input_ids == self.generation_config.eos_token_id, pad_token_id
        )

        # compute t2u_char_input_ids
        t2u_subwords = self._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = self._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )

        # Add pads for lang, EOS tokens as per NLLB "source" tokenizer mode.
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
        t2u_char_input_ids = self._get_char_input_ids(
            t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )

        # second pass
        t2u_output = self.t2u_model(
            inputs_embeds=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            **kwargs_speech,
        )

        t2u_logits = t2u_output[0]
        padding_mask = t2u_output[1].bool()

        # The text-to-unit model is non auto-regressive. We keep the ability to use sampling with temperature
        temperature = kwargs_speech.get("temperature", None)
        if (temperature is None or temperature == 1.0) or not kwargs_speech.get("do_sample", False):
            unit_ids = t2u_logits.argmax(dim=-1)
        else:
            t2u_logits = t2u_logits / temperature
            # apply softmax
            probs = nn.functional.softmax(t2u_logits, dim=-1)
            # reshape to 2D: (batch_size, seq_len, t2u_vocab_size) -> (batch_size*seq_len, t2u_vocab_size)
            probs = probs.reshape((-1, probs.shape[2]))
            # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
            unit_ids = torch.multinomial(probs, num_samples=1).view(t2u_logits.shape[0], -1)

        output_unit_ids = unit_ids.detach().clone()

        replace_mask = (unit_ids == self.config.t2u_eos_token_id) | (~padding_mask)
        # replace eos per pad
        unit_ids = unit_ids.masked_fill(replace_mask, self.config.t2u_pad_token_id)

        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids), device=self.device)

        speaker_id = torch.tensor([[speaker_id]] * len(unit_ids), device=self.device)

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, speaker_id=speaker_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return SeamlessM4Tv2GenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths


@auto_docstring(
    custom_intro="""
    The speech-to-speech SeamlessM4Tv2 Model transformer which can be used for S2ST.
    """
)
class SeamlessM4Tv2ForSpeechToSpeech(SeamlessM4TForSpeechToSpeech):
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    @torch.no_grad()
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, SeamlessM4Tv2GenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_features, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.

            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.


        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor]]`:
            - If `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
              sequence_length)` and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_features) if input_features is not None else len(kwargs.get("inputs_embeds"))

        if tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
        else:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {",".join(lang_code_to_id.keys())}. Note that SeamlessM4Tv2 supports
                    more languages for text translation than for speech synthesis."""
                    )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size, device=self.device)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        text_generation_output = super().generate(input_features, **kwargs_text)
        sequences = text_generation_output.sequences

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        # get last_hidden_state from encoder
        encoder_hidden_states = self.speech_encoder(input_features=input_features, attention_mask=attention_mask)[0]

        # input modality = speech so new attention mask for the decoder
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                encoder_hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
            )

            # repeat attention mask alongside batch dimension
            attention_mask = torch.repeat_interleave(attention_mask, num_return_sequences, dim=0)

        # repeat attention mask alongside batch dimension
        encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, num_return_sequences, dim=0)

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences[:, :-1],  # Manually trim the final EOS token
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences[:, :-1] != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # REMOVE EOS and lang_id
        t2u_input_ids = sequences[:, 2:-1]
        # replace every other EOS
        t2u_input_ids = torch.masked_fill(
            t2u_input_ids, t2u_input_ids == self.generation_config.eos_token_id, pad_token_id
        )

        # compute t2u_char_input_ids
        t2u_subwords = self._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = self._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )

        # Add pads for lang, EOS tokens as per NLLB "source" tokenizer mode.
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
        t2u_char_input_ids = self._get_char_input_ids(
            t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )

        # second pass
        t2u_output = self.t2u_model(
            inputs_embeds=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            **kwargs_speech,
        )

        t2u_logits = t2u_output[0]
        padding_mask = t2u_output[1].bool()

        # The text-to-unit model is non auto-regressive. We keep the ability to use sampling with temperature
        temperature = kwargs_speech.get("temperature", None)
        if (temperature is None or temperature == 1.0) or not kwargs_speech.get("do_sample", False):
            unit_ids = t2u_logits.argmax(dim=-1)
        else:
            t2u_logits = t2u_logits / temperature
            # apply softmax
            probs = nn.functional.softmax(t2u_logits, dim=-1)
            # reshape to 2D: (batch_size, seq_len, t2u_vocab_size) -> (batch_size*seq_len, t2u_vocab_size)
            probs = probs.reshape((-1, probs.shape[2]))
            # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
            unit_ids = torch.multinomial(probs, num_samples=1).view(t2u_logits.shape[0], -1)

        output_unit_ids = unit_ids.detach().clone()

        replace_mask = (unit_ids == self.config.t2u_eos_token_id) | (~padding_mask)
        # replace eos per pad
        unit_ids = unit_ids.masked_fill(replace_mask, self.config.t2u_pad_token_id)

        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids), device=self.device)

        speaker_id = torch.tensor([[speaker_id]] * len(unit_ids), device=self.device)

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, speaker_id=speaker_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return SeamlessM4Tv2GenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths


@auto_docstring(
    custom_intro="""
    The original SeamlessM4Tv2 Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).
    """
)
class SeamlessM4Tv2Model(SeamlessM4TModel):
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        generate_speech: Optional[bool] = True,
        **kwargs,
    ) -> Union[torch.Tensor, SeamlessM4Tv2GenerationOutput]:
        """
        Generates translated token ids and/or translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively
        perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>


        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*):
                Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be
                ignored.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            generate_speech (`bool`, *optional*, defaults to `True`):
                If `False`, will only returns the text tokens and won't generate speech.

            kwargs (*optional*):
                Remaining dictioy of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.

        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor], ModelOutput]`:
            - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
            - If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of
              shape `(batch_size, sequence_length)` and `waveform_lengths` which gives the length of each sample.
            - If `generate_speech=False`, it will returns `ModelOutput`.
        """
        if input_ids is None and input_features is None and kwargs.get("inputs_embeds", None) is None:
            raise ValueError(
                "`input_ids`,`input_features` and `inputs_embeds` are all empty. Make sure at least one of them is not."
            )

        if generate_speech and tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")

        if tgt_lang is not None:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            if generate_speech:
                keys_to_check = ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]
            else:
                keys_to_check = ["text_decoder_lang_to_code_id"]
            for key in keys_to_check:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {",".join(lang_code_to_id.keys())}. Note that SeamlessM4Tv2 supports
                    more languages for text translation than for speech synthesis."""
                    )

        batch_size = (
            len(input_features)
            if input_features is not None
            else (len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds")))
        )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            # tgt_lang gets priority over decoder input ids
            text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
            text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size, device=self.device)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        if input_features is not None:
            self.set_modality("speech")
            if input_ids is not None:
                logger.warning(
                    "`input_features` and `input_ids` are both non empty. `input_features` will be used in priority "
                    "through the speech encoder. Make sure `input_features=None` if you want to use the text encoder."
                )
            text_generation_output = super().generate(input_features=input_features, **kwargs_text)
        else:
            self.set_modality("text")
            text_generation_output = super().generate(input_ids=input_ids, input_features=None, **kwargs_text)
        sequences = text_generation_output.sequences

        if not generate_speech:
            return text_generation_output

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        # get encoder last hidden states
        if self.current_modality == "speech":
            # get last_hidden_state from encoder - must do a pass through the speech encoder
            encoder_hidden_states = self.speech_encoder(
                input_features=input_features, attention_mask=attention_mask
            ).last_hidden_state

            # input modality = speech so new attention mask for the decoder
            if attention_mask is not None:
                sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                    encoder_hidden_states.device
                )
                attention_mask = _compute_new_attention_mask(
                    hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
                )
        else:
            encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        if attention_mask is not None:
            # repeat attention mask alongside batch dimension
            attention_mask = torch.repeat_interleave(attention_mask, num_return_sequences, dim=0)

        # repeat attention mask alongside batch dimension
        encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, num_return_sequences, dim=0)

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences[:, :-1],  # Manually trim the final EOS token
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences[:, :-1] != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # REMOVE EOS and lang_id
        t2u_input_ids = sequences[:, 2:-1]
        # replace every other EOS
        t2u_input_ids = torch.masked_fill(
            t2u_input_ids, t2u_input_ids == self.generation_config.eos_token_id, pad_token_id
        )

        # compute t2u_char_input_ids
        t2u_subwords = self._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = self._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )

        # Add pads for lang, EOS tokens as per NLLB "source" tokenizer mode.
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
        t2u_char_input_ids = self._get_char_input_ids(
            t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )

        # second pass
        t2u_output = self.t2u_model(
            inputs_embeds=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            **kwargs_speech,
        )

        t2u_logits = t2u_output[0]
        padding_mask = t2u_output[1].bool()

        # The text-to-unit model is non auto-regressive. We keep the ability to use sampling with temperature
        temperature = kwargs_speech.get("temperature", None)
        if (temperature is None or temperature == 1.0) or not kwargs_speech.get("do_sample", False):
            unit_ids = t2u_logits.argmax(dim=-1)
        else:
            t2u_logits = t2u_logits / temperature
            # apply softmax
            probs = nn.functional.softmax(t2u_logits, dim=-1)
            # reshape to 2D: (batch_size, seq_len, t2u_vocab_size) -> (batch_size*seq_len, t2u_vocab_size)
            probs = probs.reshape((-1, probs.shape[2]))
            # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
            unit_ids = torch.multinomial(probs, num_samples=1).view(t2u_logits.shape[0], -1)

        output_unit_ids = unit_ids.detach().clone()

        replace_mask = (unit_ids == self.config.t2u_eos_token_id) | (~padding_mask)
        # replace eos per pad
        unit_ids = unit_ids.masked_fill(replace_mask, self.config.t2u_pad_token_id)

        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids), device=self.device)

        speaker_id = torch.tensor([[speaker_id]] * len(unit_ids), device=self.device)

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, speaker_id=speaker_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return SeamlessM4Tv2GenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths


__all__ = [
    "SeamlessM4Tv2ForTextToSpeech",
    "SeamlessM4Tv2ForSpeechToSpeech",
    "SeamlessM4Tv2ForTextToText",
    "SeamlessM4Tv2ForSpeechToText",
    "SeamlessM4Tv2Model",
    "SeamlessM4Tv2PreTrainedModel",
]
