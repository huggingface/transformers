# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.s and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Mask2Former model."""

import collections.abc
import math
import random
import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from transformers import AutoBackbone, MaskFormerSwinConfig
from transformers.utils import logging

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_torch_cuda_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_ninja_available
from ..deformable_detr import DeformableDetrConfig
from ..deformable_detr.load_custom import load_cuda_kernels
from ..detr import DetrConfig
from .configuration_mask2former import Mask2FormerConfig


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)

# Move this to not compile only when importing, this needs to happen later, like in __init__.
if is_torch_cuda_available() and is_ninja_available():
    logger.info("Loading custom CUDA kernels...")
    try:
        MultiScaleDeformableAttention = load_cuda_kernels()
    except Exception as e:
        logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")
        MultiScaleDeformableAttention = None
else:
    MultiScaleDeformableAttention = None


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        context.im2col_step = im2col_step
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


_CONFIG_FOR_DOC = "Mask2FormerConfig"
_CHECKPOINT_FOR_DOC = "shivi/mask2former-instance-swin-small-coco"
_FEAT_EXTRACTOR_FOR_DOC = "Mask2FormerFeatureExtractor"

MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shivi/mask2former-instance-swin-small-coco",
    # See all mask2former models at https://huggingface.co/models?filter=mask2former
]


@dataclass
class MaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Masked Attention decoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely an optional stack of intermediate decoder activations and predicted
    masks obtained from different decoder layers.

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
            layernorm. This is useful when training the model with auxiliary decoding losses.
        masks_queries_logits (`tuple(torch.FloatTensor)`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns the output of the encoder (optional) and all hidden states
    (multi-scale features) from the `decoder`. By default, the `encoder` is a Swin Backbone and the
    `decoder` is a MultiScaleDeformableAttention Transformer.

    The `decoder_last_hidden_state` are the **per-pixel embeddings** while `decoder_hidden_states` refer to high
    resolution feature maps produced using **multi-scaling strategy** defined in the paper.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Last hidden states (final feature map) of the last stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
        decoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the decoder.
        decoder_hidden_states (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None  # multi_scale_features


class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output. It returns the last hidden state (per-pixel embeddings) and the hidden
    states (multi scale image features).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder.
        hidden_states (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model
            (Mask2FormerPixelDecoder).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model
            (MaskedAttentionDecoder).
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from MaksedAttentionDecoder after the attention softmax, used to
            compute the weighted average in the self-attention heads.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    masks_queries_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Mask2FormerForInstanceSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerForInstanceSegmentation`].

    This output can be directly passed to [`~Mask2FormerFeatureExtractor.post_process_segmentation`] or
    [`~Mask2FormerFeatureExtractor.post_process_panoptic_segmentation`] or
    [`~Mask2FormerFeatureExtractor.post_process_instance_segmentation`] depending on the task. Please, see
    [`~Mask2FormerFeatureExtractor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model
            (Mask2FormerPixelDecoder).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model
            (MaskedAttentionDecoder).
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the transformer decoder at the output
            of each stage.
        hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from MaskedAttentionDecoder after the attention softmax, used to
            compute the weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: torch.FloatTensor = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Adapted from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension
            
    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = F.grid_sample(input_features, 2.0 * point_coordinates - 1.0)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, labels)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss = torch.einsum("nc,mc->nm", cross_entropy_loss_pos, labels) + torch.einsum(
        "nc,mc->nm", cross_entropy_loss_neg, (1 - labels)
    )

    return loss / height_and_width



# Copied from transformers.models.detr.modeling_detr.DetrAttention
class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class MaskedAttentionDecoderLayer(nn.Module):
    """
    The MaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN blocks.

    The cross attention block used as part of `MaskedAttentionDecoderLayer` is actually a `masked attention` block that
    restricts the attention to localized features centered around predicted segments which leads to faster convergence
    and improved performance.

    The order of self and cross (i.e. masked) attention blocks have also been swapped in MaskedAttentionDecoder
    compared to a standard DetrDecoder as an optimization improvement.

    Args:
        config (`DetrConfig`):
            The configuration used to initialize the MaskedAttentionDecoder.
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self.self_attn = DetrAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        # )
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim, config.decoder_attention_heads, config.attention_dropout
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.encoder_attn = DetrAttention(
        #     self.embed_dim,
        #     config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        # )
        self.encoder_attn = nn.MultiheadAttention(
            self.embed_dim, config.decoder_attention_heads, config.attention_dropout
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_embeddings (`torch.FloatTensor`, *optional*):
                position embeddings that are added to the queries and keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # print("encoder_hidden_states[level_index]:", encoder_hidden_states[level_index].shape)
            # print("position_embeddings[level_index]:", position_embeddings[level_index].shape)

            # hidden_states, cross_attn_weights = self.encoder_attn(
            #     hidden_states=hidden_states,
            #     position_embeddings=query_position_embeddings,
            #     key_value_states=encoder_hidden_states[level_index],
            #     attention_mask=encoder_attention_mask,
            #     key_value_position_embeddings=position_embeddings[level_index],
            #     output_attentions=output_attentions,
            # )


            hidden_states = self.encoder_attn(
                query=self.with_pos_embed(hidden_states, query_position_embeddings),
                key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
                value=encoder_hidden_states[level_index],
                attn_mask=encoder_attention_mask,
                key_padding_mask=None,
            )[0]

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states

        # hidden_states, self_attn_weights = self.self_attn(
        #     hidden_states=hidden_states, position_embeddings=query_position_embeddings, attention_mask=None,
        #     output_attentions=output_attentions,
        # )

        query = key = self.with_pos_embed(hidden_states, query_position_embeddings)

        
        hidden_states = self.self_attn(
            query=query, key=key, value=hidden_states, attn_mask=None, key_padding_mask=None
        )[0]


        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Copied from transformers.models.detr.modeling_detr._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
    """
    batch_size, source_len = mask.size()
    target_len = target_len if target_len is not None else source_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class MaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MaskedAttentionDecoderLayer`].
    The decoder updates the query embeddings through multiple cross (masked) and self-attention layers.

    The decoder uses a new **masked attention** mechanism instead of the standard cross-attention, which extracts
    localized features by constraining cross-attention to within the foreground region of the predicted mask for each
    query, instead of attending to the full feature map.

    Args:
        config: DetrConfig
            configuration used to instantiate MaskedAttentionDecoder
        mask_feature_size: (`torch.Tensor`):
            one of the output dimensions of the predicted masks for each query
    """

    def __init__(self, config: DetrConfig, mask_feature_size: torch.Tensor):
        super().__init__()

        self.config = config
        self.mask_feature_size = mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = config.decoder_layers - 1

        self.layers = nn.ModuleList([MaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)])
        self.layernorm = nn.LayerNorm(self.config.d_model)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.decoder_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        pixel_embeddings: torch.Tensor,
        feature_size_list: Optional[List],
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each cross(masked)-attention layer.
            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
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

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]

        combined_attention_mask = None

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        mask_predictions = ()

        inputs_embeds = self.layernorm(inputs_embeds)

        predicted_mask, attention_mask = self.mask_predictor(inputs_embeds, pixel_embeddings, feature_size_list[0])
        mask_predictions += (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            # Add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)

            if self.training and (dropout_probability < self.layerdrop):
                continue

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
            
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )

            else:
                level_index = idx % self.num_feature_levels

                attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    attention_mask=combined_attention_mask,
                    position_embeddings=position_embeddings,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

                decoder_norm_output = self.layernorm(layer_outputs[0])
                predicted_mask, attention_mask = self.mask_predictor(
                    decoder_norm_output, pixel_embeddings, feature_size_list[(idx + 1) % self.num_feature_levels]
                )

                mask_predictions += (predicted_mask,)
                # intermediate += (decoder_norm_output,)
            
            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        # hidden_states = self.layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
                if v is not None
            )
        return MaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=mask_predictions,
        )


# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock with MaskFormer->Mask2Former
class PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # Maintain submodule indexing as if part of a Sequential block
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


# Copied from transformers.models.maskformer.modeling_maskformer.MaskformerMLPPredictionHead with MaskFormer->Mask2Former
class Mask2formerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = PredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor):
        """
        This class is used to get the predicted mask for a given MaskedAttentionDecoder layer. 
        It also generates the binarized attention mask associated with the given predicted mask.

        The attention mask obtained using predicted mask of the (l-1)th decoder layer is fed to the
        cross(masked)-attention block of the next decoder layer as input. 

        Args:
            hidden_size (`int`):
                The feature dimension of the MaskedAttentionDecoder
            num_heads (`int`):
                The number of heads used in the MaskedAttentionDecoder
            mask_feature_size: (`torch.Tensor`):
                one of the output dimensions of the predicted masks for each query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mask_embedder = Mask2formerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):

        outputs = outputs.transpose(0, 1)
        mask_embeddings = self.mask_embedder(outputs)

        # Sum up over the channels
        outputs_mask = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)

        attention_mask = F.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        return outputs_mask, attention_mask


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        num_points: Optional[int] = 12544,
        cost_class: Optional[float] = 1.0,
        cost_mask: Optional[float] = 1.0,
        cost_dice: Optional[float] = 1.0,
    ):
        """Creates the matcher

        Params:
            num_points (int, *optional*, defaults to 12544):
                No. of points to sample on which the mask loss will be calculated. The same set of K points are
                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite
                matching.
            cost_class (float, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> List[Tuple[Tensor]]:
        """
        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor` of dim `batch_size, num_queries, num_labels` with the
                  classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor` of dim `batch_size, num_queries, height, width` with the
                  predicted masks.
            class_labels (`torch.Tensor`):
                A tensor` of dim `num_target_boxes` (where num_target_boxes is the number
                  of ground-truth objects in the target) containing the class labels.
            mask_labels (`torch.Tensor`):
                A tensor` of dim `num_target_boxes, height, width` containing the target
                  masks.

        Returns:
            `List[Tuple[Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
class Mask2FormerLoss(nn.Module):
    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config:
                The configuration for Mask2Former model also containing loss calculation specific parameters
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """

        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        # Weight to apply to the null class
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight, num_points=self.num_points
        )

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # compute finel size
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # target_classes is a (batch_size, num_labels, num_queries), we need to permute pred_logits "b q c -> b c q"
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # Sample point coordinates
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = sample_point(
                target_masks,
                point_coordinates,
                align_corners=False,
            ).squeeze(1)

        point_logits = sample_point(
            pred_masks,
            point_coordinates,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # permute predictions following indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # permute labels following indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (Tensor):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.

        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most uncertain
            locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        # Get uncertainty scores with the most uncertain locations having the highest uncertainty score
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (float):
                logit predictions for P points
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties
            num_points (int):
                The number of points P to sample.
            oversample_ratio (int):
                Oversampling parameter.
            importance_sample_ratio (float):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (torch.Tensor):
                coordinates for P sampled points
        """

        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the MaskedAttentionDecoder.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks: Number = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks_pt = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        return num_masks_pt



# Copied from transformers.models.deformable_detr.modeling_deformable_detr.ms_deform_attn_core_pytorch
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class DeformableDetrMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        dim_per_head = embed_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        try:
            # GPU
            output = MultiScaleDeformableAttentionFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        except Exception:
            # CPU
            output = ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output, attention_weights


class DeformableDetrEncoderLayer(nn.Module):
    def __init__(self, config: DeformableDetrConfig, num_feature_levels=3):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            n_levels=num_feature_levels,
            n_points=config.encoder_n_points,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DeformableDetrEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`DeformableDetrEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DeformableDetrConfig
        num_feature_levels: int
    """

    def __init__(self, config: DeformableDetrConfig, num_feature_levels: int = 3):
        super().__init__()

        self.config = config
        self.dropout = self.config.dropout
        self.layers = nn.ModuleList(
            [DeformableDetrEncoderLayer(self.config, num_feature_levels=3) for _ in range(self.config.encoder_layers)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=inputs_embeds.device)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )

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


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding with MaskFormer->Mask2Former
class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Mask2FormerMSDAEmbeddings(nn.Module):
    def __init__(self, in_channels: int, feature_size: int = 256, num_feature_levels: int = 4):
        """
        Construct the input and position embeddings for the MSDA (Multi-Scale Deformable Attention) model.

        Args:
            in_channels (`int):
                The input features(channels) for the convolution layers
            feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the resulting feature maps.
            num_feature_levels (`int`, *optional*, defaults to 4):
                The number of feature levels that will be used in the Multi-Scale Deformable Detr Attention Encoder
        """
        super().__init__()

        input_projection_list = []
        if num_feature_levels > 1:
            [
                input_projection_list.append(
                    nn.Sequential(nn.Conv2d(in_channel, feature_size, kernel_size=1), nn.GroupNorm(32, feature_size))
                )
                for in_channel in in_channels[::-1]
            ]

        else:
            input_projection_list.append(
                nn.Sequential(nn.Conv2d(in_channels[-1], feature_size, kernel_size=1), nn.GroupNorm(32, feature_size))
            )

        self.input_projection = nn.ModuleList(input_projection_list)

        self.position_embedding_layer = Mask2FormerSinePositionEmbedding(
            num_pos_feats=feature_size // 2, normalize=True
        )

    def forward(self, features: List[torch.Tensor]):
        input_embeds = []
        position_embeddings = []

        for idx, feature in enumerate(features[::-1]):
            input_embedding = self.input_projection[idx](feature)
            input_embeds.append(input_embedding)
            pos_embedding = self.position_embedding_layer(feature)
            position_embeddings.append(pos_embedding)

        return input_embeds, position_embeddings


class Mask2FormerMSDAModel(nn.Module):
    def __init__(
        self,
        config: DeformableDetrConfig,
        feature_size: int = 256,
        feature_channels: List[int] = None,
    ) -> List[torch.Tensor]:
        """
        A wrapper class around `DeformableDetrEncoder` to produce the multi-scale high resolution features
        corresponding to: the features obtained from the backbone network.

        Args:
            config (`DeformableDetrConfig):
                configuration used to initialize DeformableDetrEncoder.
            feature_size (`int`):
                ...
            num_feature_levels (`List[int]`):
                ...

        Returns:
            encoder_features (`List[torch.Tensor]`):
                List of multi-scale high resolution features produced by deformable detr encoder
        """
        super().__init__()

        self.config = config
        self.embed_dim = self.config.d_model
        self.num_head = self.config.encoder_attention_heads
        self.num_feature_levels = len(feature_channels)
        self.embeddings = Mask2FormerMSDAEmbeddings(feature_channels, feature_size, self.num_feature_levels)
        self.encoder = DeformableDetrEncoder(config, self.num_feature_levels)
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, DeformableDetrMultiscaleDeformableAttention):
                module._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_height = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def forward(self, features: List[Tensor], output_hidden_states: bool = False) -> Mask2FormerPixelDecoderOutput:
        input_embeds, position_embeddings = self.embeddings(features[1:])

        masks = [
            torch.zeros((embed.size(0), embed.size(2), embed.size(3)), device=embed.device, dtype=torch.bool)
            for embed in input_embeds
        ]

        spatial_shapes = [(embed.shape[2], embed.shape[3]) for embed in input_embeds]
        input_embeds_flat = torch.cat([embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=input_embeds_flat.device)
        mask_flat = torch.cat([mask.flatten(1) for mask in masks], 1)

        position_embeddings = [embed.flatten(2).transpose(1, 2) for embed in position_embeddings]
        level_pos_embed_flat = [
            embed + self.level_embed[i].view(1, 1, -1) for i, embed in enumerate(position_embeddings)
        ]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1)

        encoder_output = self.encoder(
            input_embeds_flat,
            mask_flat,
            level_pos_embed_flat,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        )

        last_hidden_state = encoder_output.last_hidden_state
        batch_size = last_hidden_state.shape[0]

        split_sizes = [None] * self.num_feature_levels

        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index[i]

        encoder_output = torch.split(last_hidden_state, split_sizes, dim=1)

        # Compute final features
        encoder_features = []
        for i, feature in enumerate(encoder_output):
            height, width = spatial_shapes[i][0], spatial_shapes[i][1]
            feature = feature.transpose(1, 2).view(batch_size, -1, height, width)
            encoder_features.append(feature)

        return encoder_features


class Mask2FormerFPNConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1):
        """
        A basic module that executes conv - norm - in sequence used in Mask2Former.

        Args:
            in_features (`int`):
                The number of input features (channels).
            out_features (`int`):
                The number of outputs features (channels).
        """
        super().__init__()
        self.layers = [
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_features),
            nn.ReLU(inplace=True),
        ]
        for i, layer in enumerate(self.layers):
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerFPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int):
        """
        A Feature Pyramid Network Layer (FPN) layer. It creates a feature map by aggregating features from the previous
        and backbone layer. Due to the spatial mismatch, the tensor coming from the previous layer is upsampled.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_features (`int`):
                The number of lateral features (channels).
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(32, in_features),
        )

        self.block = Mask2FormerFPNConvLayer(in_features, in_features)  # output_conv

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = nn.functional.interpolate(down, size=left.shape[-2:], mode="bilinear", align_corners=False)
        down += left
        down = self.block(down)

        return down


class Mask2FormerFPNModel(nn.Module):
    def __init__(self, feature_channels: List[int], feature_size: int = 256, num_fpn_levels: int = 1):
        """
        Feature Pyramid Network, given an input tensor and a set of feature map of different feature/spatial size, it
        creates a list of feature maps with the same feature size.

        Args:
            in_features (`int`):
                The number of input features (channels).
            feature_channels (`List[int]`):
                A list with the features (channels) size of each lateral connection.
            feature_size (int, *optional*, defaults to 256):
                The features (channels) of the resulting feature maps.
        """
        super().__init__()
        self.layers = nn.Sequential(
            *[
                Mask2FormerFPNLayer(feature_size, in_channels)
                for in_channels in feature_channels[:num_fpn_levels][::-1]
            ]
        )

    def forward(
        self, encoder_features: List[torch.Tensor], features: List[torch.Tensor], num_fpn_levels: int = 3
    ) -> List[Tensor]:
        output = encoder_features[-1]
        fpn_features = []
        for layer, left in zip(self.layers, features[:num_fpn_levels]):  # [::-1]):
            output = layer(output, left)
            fpn_features.append(output)
        return fpn_features


class Mask2FormerPixelDecoder(nn.Module):
    def __init__(
        self,
        config: DeformableDetrConfig,
        feature_channels: List[int],
        feature_size: Optional[int] = 256,
        mask_feature_size: Optional[int] = 256,
    ) -> Mask2FormerPixelDecoderOutput:
        """
        Pixel Decoder Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). The pixel decoder gradually upsamples low resolution features
        of the backbone network to produce high resolution `multi-scale features` with help of Multi Scale Deformable
        Detr Encoder as well as high resolution `pixel embeddings` with help of a Feature Pyramid Network.

        Args:
            config (`DeformableDetrConfig`):
                configuration used to instantiate the pixel decoder.
            feature_channels (`List[int]`):
                A list of input features (channels) for the convolution layers used in the MSDA and FPN models
            feature_size (`int`, *optional*, defaults to 256):
                The feature size (channel dimension) of the feature maps.
            mask_feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the target masks size.
        """
        super().__init__()
        self.stride = min(config.feature_strides[1:])
        self.num_fpn_levels = int(np.log2(self.stride) - np.log2(config.common_stride))

        # Multi-scale deformable attention module
        self.msda_module = Mask2FormerMSDAModel(config, feature_size, feature_channels[1:])
        self.feature_pyramid_network = Mask2FormerFPNModel(feature_channels, feature_size, self.num_fpn_levels)
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=1, stride=1, padding=0)

    def forward(self, backbone_features: List[Tensor]) -> Mask2FormerPixelDecoderOutput:
        print("ours")
        for feat in backbone_features:
            print(feat.shape)
            print(feat)
        multi_scale_features = self.msda_module(backbone_features)

        fpn_features = self.feature_pyramid_network(multi_scale_features, backbone_features)
        
        last_projected_feature = self.mask_projection(fpn_features[-1])

        return Mask2FormerPixelDecoderOutput(
            last_hidden_state=last_projected_feature, hidden_states=tuple(multi_scale_features)
        )


class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        backbone_config = config.backbone_config
        if backbone_config.model_type == "swin":
            # for backwards compatibility
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]

        self.encoder = AutoBackbone.from_config(backbone_config)
        self.decoder = Mask2FormerPixelDecoder(
            config=config.pixel_decoder_config,
            feature_size=config.feature_size,
            mask_feature_size=config.mask_feature_size,
            feature_channels=self.encoder.channels,
        )

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        backbone_output = self.encoder(pixel_values)
        backbone_features = backbone_output.feature_maps
        decoder_output = self.decoder(backbone_features)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1] if output_hidden_states else (),
            decoder_last_hidden_state=decoder_output.last_hidden_state,  # per pixel embedding
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else (),
            decoder_hidden_states=decoder_output.hidden_states,  # multi-scale features
        )


class Mask2FormerTransformerModule(nn.Module):
    """
    Mask2Former's transformer module.

    Args:
        config (`Mask2FormerConfig`):
            configuration used to initialize Mask2former's transformer module
        in_channels (`int`):
            channels of the input features
    """

    def __init__(self, config: Mask2FormerConfig, in_channels: int):
        super().__init__()

        hidden_size = config.decoder_config.hidden_size  # hidden_dim
        should_project = in_channels != hidden_size
        # self.pe_layer
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        # self.query_feat
        self.learnable_queries = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        # self.query_embed
        self.queries_embedder = nn.Embedding(config.decoder_config.num_queries, hidden_size)

        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_size)

        self.input_projection = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_size, kernel_size=1) if should_project else nn.Sequential()
                for _ in range(self.num_feature_levels)
            ]
        )
        
        self.decoder = MaskedAttentionDecoder(config=config.decoder_config, mask_feature_size=config.mask_feature_size)

    def forward(
        self,
        multi_scale_features: torch.Tensor,
        pixel_embeddings: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> MaskedAttentionDecoderOutput:

        input_embeds = []
        position_embeddings = []
        feature_size_list = []

        for i in range(self.num_feature_levels):
            feature_size_list.append(multi_scale_features[i].shape[-2:])
            position_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            input_embeds.append(
                self.input_projection[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )
            position_embeddings[-1] = position_embeddings[-1].permute(2, 0, 1)
            input_embeds[-1] = input_embeds[-1].permute(2, 0, 1)

        _, batch_size, _ = input_embeds[0].shape

        queries_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        learnable_queries = self.learnable_queries.weight.unsqueeze(1).repeat(1, batch_size, 1)
        

        output = self.decoder(
            inputs_embeds=learnable_queries,
            pixel_embeddings=pixel_embeddings,
            feature_size_list=feature_size_list,
            encoder_hidden_states=input_embeds,
            position_embeddings=position_embeddings,
            query_position_embeddings=queries_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return output


MASK2FORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASK2FORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of Detr's decoder attention layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~Mask2FormerModelOutput`] instead of a plain tuple.
"""


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2. Also initializes `module.bias` to 0.
    Caffe2 implementation of XavierFill corresponds to kaiming_uniform_ in PyTorch

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class Mask2FormerPreTrainedModel(PreTrainedModel):
    config_class = Mask2FormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        # Transformer Module
        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projection is not None and isinstance(module.input_projection[0], nn.Conv2d):
                c2_xavier_fill(module.input_projection[0])

        # Pixel Decoder
        elif isinstance(module, Mask2FormerPixelDecoder):
            if module.mask_projection is not None:
                c2_xavier_fill(module.mask_projection)

        elif isinstance(module, Mask2FormerMSDAEmbeddings):
            if module.input_projection is not None:
                for projection in module.input_projection:
                    nn.init.xavier_uniform_(projection[0].weight, gain=xavier_std)
                    nn.init.constant_(projection[0].bias, 0)

        # FPN
        elif isinstance(module, Mask2FormerFPNLayer):
            nn.init.xavier_uniform_(module.proj[0].weight, gain=xavier_std)

        elif isinstance(module, Mask2FormerFPNConvLayer):
            nn.init.xavier_uniform_(module.get_submodule("0").weight, gain=xavier_std)

        # copied from MaskFormer
        # The MLP head
        elif isinstance(module, Mask2formerMLPPredictionHead):
            # I was not able to find the correct initializer in the original implementation
            # we'll use xavier
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=xavier_std)
                    nn.init.constant_(submodule.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # copied from DETR
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Mask2FormerSwinEncoder):
            module.gradient_checkpointing = value
        if isinstance(module, Mask2FormerPixelDecoder):
            module.gradient_checkpointing = value
        if isinstance(module, MaskedAttentionDecoder):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(in_channels=config.feature_size, config=config)

        self.post_init()

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Mask2FormerModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
    )
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerModelOutput:

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        
        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states)

        multi_scale_features = pixel_level_module_output.decoder_hidden_states
        """
        print("our implementation")
        print(len(multi_scale_features))
        for i in multi_scale_features:
            print(i.shape)
            print(i)
        """
        pixel_embeddings = pixel_level_module_output.decoder_last_hidden_state

        # pass multi-scale features from pixel decoder to transformer module
        transformer_module_output = self.transformer_module(
            multi_scale_features, pixel_embeddings, output_hidden_states, output_attentions
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            hidden_states = encoder_hidden_states + pixel_decoder_hidden_states + transformer_decoder_hidden_states

        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_embeddings,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            hidden_states=hidden_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits[-1],
        )

        if not return_dict:
            output = tuple(v for v in output.values())

        return output


class Mask2FormerForInstanceSegmentation(Mask2FormerPreTrainedModel):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)
        hidden_size = config.decoder_config.hidden_size
        # + 1 because we add the "null" class
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)

        self.weight_dict = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        # SetCriterion add-> num_points, oversample_ratio, importance_sample_ratio
        self.criterion = Mask2FormerLoss(config, weight_dict=self.weight_dict)

        self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_logits: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        loss_dict = self.criterion(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
        )
        # Weigh each loss
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[str, Tensor] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForInstanceSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerForInstanceSegmentationOutput:
        r"""
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import Mask2FormerFeatureExtractor, Mask2FormerForInstanceSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> feature_extractor = Mask2FormerFeatureExtractor.from_pretrained(
        ...     "shivi/mask2former-instance-swin-small-coco"
        ... )
        >>> inputs = feature_extractor(images=image, return_tensors="pt")

        >>> model = Mask2FormerForInstanceSegmentation.from_pretrained("shivi/mask2former-instance-swin-small-coco")
        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to feature_extractor for postprocessing
        >>> output = feature_extractor.post_process_segmentation(outputs)
        >>> output = feature_extractor.post_process_semantic_segmentation(outputs)
        >>> output = feature_extractor.post_process_panoptic_segmentation(outputs)
        >>> output = feature_extractor.post_process_instance_segmentation(outputs)
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: Mask2FormerModelOutput = self.model(
            pixel_values,
            pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            return_dict=True,
            output_attentions=output_attentions,
        )

        loss, loss_dict, auxiliary_logits = None, None, None

        if self.config.use_auxiliary_loss:
            transformer_decoder_outputs = torch.stack(outputs.transformer_decoder_hidden_states)
        else:
            transformer_decoder_outputs = outputs.transformer_decoder_last_hidden_state

        class_queries_logits = self.class_predictor(transformer_decoder_outputs.transpose(0, 1))     
        
        masks_queries_logits = outputs.masks_queries_logits

        if self.config.use_auxiliary_loss:
            auxiliary_logits = self.get_auxiliary_logits(class_queries_logits[-1], masks_queries_logits[-1])

        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(
                masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
            )
            loss = self.get_loss(loss_dict)

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = Mask2FormerForInstanceSegmentationOutput(
            loss=loss,
            **outputs,
            class_queries_logits=class_queries_logits,
            auxiliary_logits=auxiliary_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values())
            if loss is not None:
                output = ((loss)) + output

        return output
