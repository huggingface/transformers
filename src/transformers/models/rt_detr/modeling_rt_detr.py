# coding=utf-8
# Copyright 2023 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch RT_DETR model."""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
)
from .configuration_rt_detr import RTDetrConfig


if is_scipy_available():
    pass

if is_timm_available():
    pass

if is_vision_available():
    pass

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RTDetrConfig"
_CHECKPOINT_FOR_DOC = "checkpoing/todo"

RT_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "checkpoing/todo",
    # See all RT_DETR models at https://huggingface.co/models?filter=rt_detr
]


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrDecoderOutput with DETR->RT_DETR,Detr->RT_DETR
class RT_DETRDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the RT_DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrModelOutput with DETR->RT_DETR,Detr->RT_DETR
class RTDetrModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the RT_DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class RTDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`RTDetrForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


RT_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RTDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RT_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DetrImageProcessor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# # Copied from transformers.models.detr.modeling_detr.DetrEncoder with DETR->RT_DETR,Detr->RT_DETR
# class RT_DETREncoder(RTDetrPreTrainedModel):
#     """
#     Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
#     [`RT_DETREncoderLayer`].

#     The encoder updates the flattened feature map through multiple self-attention layers.

#     Small tweak for RT_DETR:

#     - object_queries are added to the forward pass.

#     Args:
#         config: RTDetrConfig
#     """

#     def __init__(self, config: RTDetrConfig):
#         super().__init__(config)

#         self.dropout = config.dropout
#         self.layerdrop = config.encoder_layerdrop

#         self.layers = nn.ModuleList([RT_DETREncoderLayer(config) for _ in range(config.encoder_layers)])

#         # in the original RT_DETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         inputs_embeds=None,
#         attention_mask=None,
#         object_queries=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         **kwargs,
#     ):
#         r"""
#         Args:
#             inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#                 Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

#             attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

#                 - 1 for pixel features that are real (i.e. **not masked**),
#                 - 0 for pixel features that are padding (i.e. **masked**).

#                 [What are attention masks?](../glossary#attention-mask)

#             object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#                 Object queries that are added to the queries in each self-attention layer.

#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_hidden_states (`bool`, *optional*):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more detail.
#             return_dict (`bool`, *optional*):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         """
#         position_embeddings = kwargs.pop("position_embeddings", None)

#         if kwargs:
#             raise ValueError(f"Unexpected arguments {kwargs.keys()}")

#         if position_embeddings is not None and object_queries is not None:
#             raise ValueError(
#                 "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
#             )

#         if position_embeddings is not None:
#             logger.warning_once(
#                 "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
#             )
#             object_queries = position_embeddings

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         hidden_states = inputs_embeds
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

#         # expand attention_mask
#         if attention_mask is not None:
#             # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
#             attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         for i, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             to_drop = False
#             if self.training:
#                 dropout_probability = torch.rand([])
#                 if dropout_probability < self.layerdrop:  # skip the layer
#                     to_drop = True

#             if to_drop:
#                 layer_outputs = (None, None)
#             else:
#                 # we add object_queries as extra input to the encoder_layer
#                 layer_outputs = encoder_layer(
#                     hidden_states,
#                     attention_mask,
#                     object_queries=object_queries,
#                     output_attentions=output_attentions,
#                 )

#                 hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
#         )


# # Copied from transformers.models.detr.modeling_detr.DetrDecoder with DETR->RT_DETR,Detr->RT_DETR
# class RT_DETRDecoder(RTDetrPreTrainedModel):
#     """
#     Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`RT_DETRDecoderLayer`].

#     The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

#     Some small tweaks for RT_DETR:

#     - object_queries and query_position_embeddings are added to the forward pass.
#     - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

#     Args:
#         config: RTDetrConfig
#     """

#     def __init__(self, config: RTDetrConfig):
#         super().__init__(config)
#         self.dropout = config.dropout
#         self.layerdrop = config.decoder_layerdrop

#         self.layers = nn.ModuleList([RT_DETRDecoderLayer(config) for _ in range(config.decoder_layers)])
#         # in RT_DETR, the decoder uses layernorm after the last decoder layer output
#         self.layernorm = nn.LayerNorm(config.d_model)

#         self.gradient_checkpointing = False
#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         inputs_embeds=None,
#         attention_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         object_queries=None,
#         query_position_embeddings=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         **kwargs,
#     ):
#         r"""
#         Args:
#             inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#                 The query embeddings that are passed into the decoder.

#             attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

#                 - 1 for queries that are **not masked**,
#                 - 0 for queries that are **masked**.

#                 [What are attention masks?](../glossary#attention-mask)
#             encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
#                 Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
#                 of the decoder.
#             encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
#                 Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
#                 in `[0, 1]`:

#                 - 1 for pixels that are real (i.e. **not masked**),
#                 - 0 for pixels that are padding (i.e. **masked**).

#             object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#                 Object queries that are added to the queries and keys in each cross-attention layer.
#             query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
#                 , *optional*): Position embeddings that are added to the values and keys in each self-attention layer.

#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_hidden_states (`bool`, *optional*):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more detail.
#             return_dict (`bool`, *optional*):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         """
#         position_embeddings = kwargs.pop("position_embeddings", None)

#         if kwargs:
#             raise ValueError(f"Unexpected arguments {kwargs.keys()}")

#         if position_embeddings is not None and object_queries is not None:
#             raise ValueError(
#                 "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
#             )

#         if position_embeddings is not None:
#             logger.warning_once(
#                 "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
#             )
#             object_queries = position_embeddings

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if inputs_embeds is not None:
#             hidden_states = inputs_embeds
#             input_shape = inputs_embeds.size()[:-1]

#         combined_attention_mask = None

#         if attention_mask is not None and combined_attention_mask is not None:
#             # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
#             combined_attention_mask = combined_attention_mask + _expand_mask(
#                 attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
#             )

#         # expand encoder attention mask
#         if encoder_hidden_states is not None and encoder_attention_mask is not None:
#             # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
#             encoder_attention_mask = _expand_mask(
#                 encoder_attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
#             )

#         # optional intermediate hidden states
#         intermediate = () if self.config.auxiliary_loss else None

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

#         for idx, decoder_layer in enumerate(self.layers):
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#             if self.training:
#                 dropout_probability = torch.rand([])
#                 if dropout_probability < self.layerdrop:
#                     continue

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         return module(*inputs, output_attentions)

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     combined_attention_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     None,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=combined_attention_mask,
#                     object_queries=object_queries,
#                     query_position_embeddings=query_position_embeddings,
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_attention_mask,
#                     output_attentions=output_attentions,
#                 )

#             hidden_states = layer_outputs[0]

#             if self.config.auxiliary_loss:
#                 hidden_states = self.layernorm(hidden_states)
#                 intermediate += (hidden_states,)

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#                 if encoder_hidden_states is not None:
#                     all_cross_attentions += (layer_outputs[2],)

#         # finally, apply layernorm
#         hidden_states = self.layernorm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         # stack intermediate decoder activations
#         if self.config.auxiliary_loss:
#             intermediate = torch.stack(intermediate)

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
#                 if v is not None
#             )
#         return RT_DETRDecoderOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#             cross_attentions=all_cross_attentions,
#             intermediate_hidden_states=intermediate,
#         )


#### TODO: Rafael
from collections import OrderedDict

import torch.nn.functional as F


def get_activation(activation: str, inplace: bool = True):
    activation = activation.lower()
    if activation == "silu":
        activation_func = nn.SiLU()
    elif activation == "relu":
        activation_func = nn.ReLU()
    elif activation == "leaky_relu":
        activation_func = nn.LeakyReLU()
    elif activation == "silu":
        activation_func = nn.SiLU()
    elif activation == "gelu":
        activation_func = nn.GELU()
    elif activation is None:
        activation_func = nn.Identity()
    elif isinstance(activation, nn.Module):
        activation_func = activation
    else:
        raise RuntimeError(f"Not valid activation {activation}")
    if hasattr(activation_func, "inplace"):
        activation_func.inplace = inplace
    return activation_func


class ConvNormLayer(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding=None, bias=False, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(channels_out)
        self.activation = nn.Identity() if activation is None else get_activation(activation)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, channels_in, channels_out, stride, shortcut, activation="relu", variant="b"):
        super().__init__()

        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = channels_out

        self.branch2a = ConvNormLayer(channels_in, width, 1, stride1, activation=activation)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, activation=activation)
        self.branch2c = ConvNormLayer(width, channels_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(channels_in, channels_out * self.expansion, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(channels_in, channels_out * self.expansion, 1, stride)

        self.activation = nn.Identity() if activation is None else get_activation(activation)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.activation(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, channels_in, channels_out, stride, shortcut, activation="relu", variant="b"):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(channels_in, channels_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(channels_in, channels_out, 1, stride)

        self.branch2a = ConvNormLayer(channels_in, channels_out, 3, stride, activation=activation)
        self.branch2b = ConvNormLayer(channels_out, channels_out, 3, 1, activation=None)
        self.activation = nn.Identity() if activation is None else get_activation(activation)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.activation(out)

        return out


class Blocks(nn.Module):
    def __init__(self, block, channels_in, channels_out, count, stage_num, activation="relu", variant="b"):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    channels_in,
                    channels_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    activation=activation,
                )
            )

            if i == 0:
                channels_in = channels_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


# TODO: Rafael! Important: Use Resnet-d from timm library instead of the PResNet (config.userfromtimm=True) -> see how deformable detr does


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->RtDetr
class RtDetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# TODO: Rafael remove it
# class FrozenBatchNorm2d(nn.Module):
#     """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.
#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models than torchvision.models.resnet[18,34,50,101]
#     produce nans.
#     """
#     def __init__(self, num_features, eps=1e-5):
#         super(FrozenBatchNorm2d, self).__init__()
#         n = num_features
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))
#         self.eps = eps
#         self.num_features = n

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, x):
#         # move reshapes to the beginning to make it fuser-friendly
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         scale = w * (rv + self.eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias

#     def extra_repr(self):
#         return (
#             "{num_features}, eps={eps}".format(**self.__dict__)
#         )


def bias_init_with_prob(prior_prob=0.01):
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    batch_size, _, num_head, head_dim = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        new_value_list = value_list[level].flatten(2).permute(0, 2, 1).reshape(batch_size * num_head, head_dim, h, w)
        new_sampling_grid = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        new_sampling_value = F.grid_sample(
            new_value_list, new_sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(new_sampling_value)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_head, 1, len_q, n_levels * n_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(batch_size, num_head * head_dim, len_q)
    )

    return output.permute(0, 2, 1)


class PResNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        block_nums = config.block_nums
        variant = config.variant
        act_presnet = config.act_presnet
        depth = config.depth
        num_stages = config.num_stages
        return_idx = config.return_idx
        freeze_at = config.freeze_at
        freeze_norm = config.freeze_norm
        pretrained = config.pretrained
        channels_in = 64
        if variant in ["c", "d"]:
            conv_def = [
                [3, channels_in // 2, 3, 2, "conv1_1"],
                [channels_in // 2, channels_in // 2, 3, 1, "conv1_2"],
                [channels_in // 2, channels_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, channels_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (_name, ConvNormLayer(c_in, c_out, k, s, activation=act_presnet))
                    for c_in, c_out, k, s, _name in conv_def
                ]
            )
        )

        channels_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in channels_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(
                    block,
                    channels_in,
                    channels_out_list[i],
                    block_nums[i],
                    stage_num,
                    activation=act_presnet,
                    variant=variant,
                )
            )
            channels_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        # TODO Rafael:
        # if pretrained:
        #     state = torch.hub.load_state_dict_from_url(donwload_url[depth])
        #     self.load_state_dict(state)
        #     print(f"Load PResNet{depth} state_dict")

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = RtDetrFrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, num_head, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class RepVggBlock(nn.Module):
    def __init__(self, channels_in, channels_out, activation="relu"):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = ConvNormLayer(channels_in, channels_out, 3, 1, padding=1, activation=None)
        self.conv2 = ConvNormLayer(channels_in, channels_out, 1, 1, padding=0, activation=None)
        self.activation = nn.Identity() if activation is None else get_activation(activation)

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.activation(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.channels_in, self.channels_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, activation="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, activation=activation)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, activation=activation)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(hidden_channels, hidden_channels, activation=activation) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class HybridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.feat_strides = config.feat_strides
        self.hidden_dim = config.hidden_dim
        self.use_encoder_idx = config.use_encoder_idx
        self.num_encoder_layers = config.num_encoder_layers
        self.pe_temperature = config.pe_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.hidden_dim for _ in range(len(self.in_channels))]
        self.out_strides = self.feat_strides
        num_head = config.num_head
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        enc_act = config.enc_act
        expansion = config.expansion
        depth_mult = config.depth_mult
        act_encoder = config.act_encoder
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(self.hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            self.hidden_dim, num_head=num_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act
        )

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), self.num_encoder_layers)
                for _ in range(len(self.use_encoder_idx))
            ]
        )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(self.hidden_dim, self.hidden_dim, 1, 1, activation=act_encoder))
            self.fpn_blocks.append(
                CSPRepLayer(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    round(3 * depth_mult),
                    activation=act_encoder,
                    expansion=expansion,
                )
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(self.hidden_dim, self.hidden_dim, 3, 2, activation=act_encoder))
            self.pan_blocks.append(
                CSPRepLayer(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    round(3 * depth_mult),
                    activation=act_encoder,
                    expansion=expansion,
                )
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride, self.hidden_dim, self.pe_temperature
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        if len(feats) != len(self.in_channels):
            raise "Relation len(feats) != len(self.in_channels) must apply."
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(
                        src_flatten.device
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
    ):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("Relation self.head_dim * num_heads == self.embed_dim does not apply")
        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
        )
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.ms_deformable_attn_core = deformable_attention_core_func
        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        # attention_weights
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        # proj
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        value = self.value_proj(value)

        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
    ):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target,
        reference_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        attention_res, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(attention_res)
        target = self.norm1(target)

        # cross attention
        cross_attention_res = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask
        )
        target = target + self.dropout2(cross_attention_res)
        target = self.norm2(target)

        # ffn
        forward_res = self.forward_ffn(target)
        target = target + self.dropout4(forward_res)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        output = target
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RTDetrTransformer(nn.Module):
    __share__ = ["num_classes"]

    def __init__(self, config):
        super(RTDetrTransformer, self).__init__()

        position_embed_type = config.position_embed_type
        feat_channels = config.feat_channels
        feat_strides = config.feat_strides
        num_levels = config.num_levels

        if position_embed_type not in ["sine", "learned"]:
            raise ValueError(f"position_embed_type not supported {position_embed_type}")
        if len(feat_channels) > num_levels:
            raise ValueError("relation feat_channels <= num_levels does not apply")
        if len(feat_strides) != len(feat_channels):
            raise ValueError("relation len(feat_strides) == len(feat_channels) does not apply")
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_head
        self.feat_strides = config.feat_strides
        self.num_levels = config.num_levels
        self.num_classes = config.num_classes
        self.num_queries = config.num_queries
        self.eps = config.eps
        self.num_decoder_layers = config.num_decoder_layers
        self.eval_spatial_size = config.eval_spatial_size
        self.aux_loss = config.aux_loss
        self.learnt_init_query = config.learnt_init_query
        self.num_denoising = config.num_denoising
        self.label_noise_ratio = config.label_noise_ratio
        self.box_noise_scale = config.box_noise_scale
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        activation = config.act_decoder
        num_decoder_points = config.num_decoder_points
        eval_idx = config.eval_idx
        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            self.hidden_dim, self.num_head, dim_feedforward, dropout, activation, num_levels, num_decoder_points
        )
        self.decoder = TransformerDecoder(self.hidden_dim, decoder_layer, self.num_decoder_layers, eval_idx)

        # denoising part
        if self.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                self.num_classes + 1, self.hidden_dim, padding_idx=self.num_classes
            )

        # decoder embedding
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.query_pos_head = MLP(4, 2 * self.hidden_dim, self.hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(
                self.hidden_dim,
            ),
        )
        self.enc_score_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.enc_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.num_classes) for _ in range(self.num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3) for _ in range(self.num_decoder_layers)]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(cls_.bias, bias)
            nn.init.constant_(reg_.layers[-1].weight, 0)
            nn.init.constant_(reg_.layers[-1].bias, 0)
        # linear_init_(self.enc_output[0])
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            (
                                "norm",
                                nn.BatchNorm2d(
                                    self.hidden_dim,
                                ),
                            ),
                        ]
                    )
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [
            0,
        ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact
        )

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["aux_outputs"].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


class RTDetrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RTDetrConfig
    base_model_prefix = "rt_detr"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initalize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, RtDetrFrozenBatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RT_DETRDecoder): # TODO: <==== what is _set_gradient_checkpointing used for?
            module.gradient_checkpointing = value


@add_start_docstrings(
    """
    The bare RT_DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    RT_DETR_START_DOCSTRING,
)
class RTDetrModel(RTDetrPreTrainedModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        self.backbone = PResNet(config)
        self.encoder = HybridEncoder(config)
        self.decoder = RTDetrTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for _, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for _, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @add_start_docstrings_to_model_forward(RT_DETR_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=RTDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    @replace_return_docstrings(output_type=RTDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        # pixel_mask: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.FloatTensor] = None,
        # encoder_outputs: Optional[torch.FloatTensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RTDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RTDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("checkpoing/todo")
        >>> model = RTDetrModel.from_pretrained("checkpoing/todo")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 100, 256]
        ```"""
        # TODO: Check how other models do these individual steps:

        # First, sent pixel_values through Backbone to obtain the features
        features = self.backbone(pixel_values)
        # TODO: Rafael -> Remove -For the cat image
        # 0 torch.Size([1, 512, 80, 80]) 358398.75
        # 1 torch.Size([1, 1024, 40, 40]) 81166.578125
        # 2 torch.Size([1, 2048, 20, 20]) 15052.0986328125

        # Second: Send features through the encoder
        if features is None:
            pass  # TODO
        else:
            encoder_outputs = self.encoder(features)
        # TODO: Rafael -> Remove -For the cat image
        # Expected
        # 0 torch.Size([1, 256, 80, 80]) 501037.875
        # 1 torch.Size([1, 256, 40, 40]) 103032.84375
        # 2 torch.Size([1, 256, 20, 20]) 34950.58203125
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True

        # Third: Send features through the decoder
        if encoder_outputs is None:
            pass  # TODO
        else:
            decoder_outputs = self.decoder(encoder_outputs)
        # TODO: Rafael -> Remove -For the cat image
        # Expected:
        # pred_logits torch.Size([1, 300, 80]) -160269.53125
        # pred_boxes torch.Size([1, 300, 4]) 622.9794921875
        
        # TODO: The return of RTDetrModel should be a DetrModelOutput object.
        return decoder_outputs

        # # get final feature map and downsampled mask
        # feature_map, mask = features[-1]

        # if mask is None:
        #     raise ValueError("Backbone does not return downsampled pixel mask")

        # # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        # projected_feature_map = self.input_projection(feature_map)

        # # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        # flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        # object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

        # flattened_mask = mask.flatten(1)

        # # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        # if encoder_outputs is None:
        #     encoder_outputs = self.encoder(
        #         inputs_embeds=flattened_features,
        #         attention_mask=flattened_mask,
        #         object_queries=object_queries,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        # # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        # # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        # query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # queries = torch.zeros_like(query_position_embeddings)

        # # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        # decoder_outputs = self.decoder(
        #     inputs_embeds=queries,
        #     attention_mask=None,
        #     object_queries=object_queries,
        #     query_position_embeddings=query_position_embeddings,
        #     encoder_hidden_states=encoder_outputs[0],
        #     encoder_attention_mask=flattened_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # if not return_dict:
        #     return decoder_outputs + encoder_outputs

        # return RTDetrModelOutput(
        #     last_hidden_state=decoder_outputs.last_hidden_state,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        #     intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        # )


@add_start_docstrings(
    """
    RT_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
    such as COCO detection.
    """,
    RT_DETR_START_DOCSTRING,
)
class RTDetrForObjectDetection(RTDetrPreTrainedModel): 
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        # RT_DETR encoder-decoder model
        self.model = RTDetrModel(config)

        # TODO: finish this class
#         # Object detection heads
#         self.class_labels_classifier = nn.Linear(
#             config.d_model, config.num_labels + 1
#         )  # We add one for the "no object" class
#         self.bbox_predictor = RT_DETRMLPPredictionHead(
#             input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
#         )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RT_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RTDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RTDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RTDetrForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("checkpoing/todo")
        >>> model = RTDetrForObjectDetection.from_pretrained("checkpoing/todo")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to COCO API
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
        Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
        Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
        Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
        Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First, sent images through RTDetr base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            # pixel_mask=pixel_mask,
            # decoder_attention_mask=decoder_attention_mask,
            # encoder_outputs=encoder_outputs,
            # inputs_embeds=inputs_embeds,
            # decoder_inputs_embeds=decoder_inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO: model is already returning the logits + pred_boxes, which is wrong
        # model should output the RTDetrModelOutput object, containing the outputs of the model, 
        # not the predictions.
        # The losses, logits, pred_boxes should be computed here
        pred_boxes = outputs["pred_boxes"]

        return RTDetrObjectDetectionOutput(
            loss=None,
            loss_dict=None,
            logits=None,
            pred_boxes=pred_boxes,
            auxiliary_outputs=None,
            last_hidden_state=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
