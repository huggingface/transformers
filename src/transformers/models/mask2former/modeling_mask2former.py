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
import warnings
import random
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers.utils import logging
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, BaseModelOutput
from ...modeling_utils import ModuleUtilsMixin, PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
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
from ...utils import is_ninja_available
from ..detr import DetrConfig
from ..swin import SwinConfig
from ..deformable_detr import DeformableDetrConfig
from .configuration_mask2former import Mask2FormerConfig
from ..deformable_detr.load_custom import load_cuda_kernels

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

#copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction
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
_CHECKPOINT_FOR_DOC = "shivi/mask2former-segmentation-swin-large-ade"
_FEAT_EXTRACTOR_FOR_DOC = "Mask2FormerFeatureExtractor"

MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shivi/mask2former-segmentation-swin-large-ade",
    # See all mask2former models at https://huggingface.co/models?filter=maskformer
]



@dataclass
# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinModelOutputWithPooling with MaskFormer->Mask2Former
class Mask2FormerSwinModelOutputWithPooling(ModelOutput):
    """
    Class for Mask2FormerSwinModel's outputs that also contains the spatial dimensions of the hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a mean pooling operation.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            A tuple containing the spatial dimension of each `hidden_state` needed to reshape the `hidden_states` to
            `batch, channels, height, width`. Due to padding, their spatial size cannot be inferred before the
            `forward` method.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinBaseModelOutput with MaskFormer->Mask2Former
class Mask2FormerSwinBaseModelOutput(ModelOutput):
    """
    Class for SwinEncoder's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            A tuple containing the spatial dimension of each `hidden_state` needed to reshape the `hidden_states` to
            `batch, channels, height, width`. Due to padding, their spatial size cannot inferred before the `forward`
            method.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrDecoderOutput
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
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
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns both the last and (optionally) the hidden states from the
    `encoder` and `decoder`. By default, the `encoder` is a Mask2FormerSwin Transformer and the `decoder` is a 
    MultiScaleDeformableAttention Transformer.

    The `encoder_last_hidden_state` are referred on the paper as **images features**, while `decoder_last_hidden_state`
    as **pixel embeddings**

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
        decoder_last_hidden_state (`torch.FloatTensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the decoder.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    multi_scale_features: Optional[List[torch.FloatTensor]] = None


class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output. It returns the last hidden state and (optionally) the hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    transformer_encoder_features: torch.FloatTensor = None
    multi_scale_features: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Mask2FormerMSDeformableAttnEncoderOutput(ModelOutput):
    """
    Class for Mask2FormerMSDeformableAttnEncoder's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    spatial_shapes: torch.FloatTensor = None
    level_start_index: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (MultiScaleDeformableAttnTransformer).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
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
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput with MaskFormer->Mask2Former
class Mask2FormerForInstanceSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerForInstanceSegmentation`].

    This output can be directly passed to [`~Mask2FormerFeatureExtractor.post_process_segmentation`] or
    [`~Mask2FormerFeatureExtractor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~Mask2FormerFeatureExtractor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
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
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
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


def upsample_like(pixel_values: Tensor, like: Tensor, mode: str = "bilinear") -> Tensor:
    """
    An utility function that upsamples `pixel_values` to match the dimension of `like`.

    Args:
        pixel_values (`torch.Tensor`):
            The tensor we wish to upsample.
        like (`torch.Tensor`):
            The tensor we wish to use as size target.
        mode (str, *optional*, defaults to `"bilinear"`):
            The interpolation mode.

    Returns:
        `torch.Tensor`: The upsampled tensor
    """
    _, _, height, width = like.shape
    upsampled = nn.functional.interpolate(pixel_values, size=(height, width), mode=mode, align_corners=False)
    return upsampled


# refactored from original implementation
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


# refactored from original implementation
def sigmoid_focal_loss(
    inputs: Tensor, labels: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    r"""
    Focal loss proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) originally used in
    RetinaNet. The loss is computed as follows:

    $$ \mathcal{L}_{\text{focal loss} = -(1 - p_t)^{\gamma}\log{(p_t)} $$

    where \\(CE(p_t) = -\log{(p_t)}}\\), CE is the standard Cross Entropy Loss

    Please refer to equation (1,2,3) of the paper for a better understanding.

    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.
        alpha (float, *optional*, defaults to 0.25):
            Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float, *optional*, defaults to 2.0):
            Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probs = inputs.sigmoid()
    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


# refactored from original implementation
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


# refactored from original implementation
def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    r"""
    A pair wise version of the focal loss, see `sigmoid_focal_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, *optional*, defaults to 0.25):
            Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float, *optional*, defaults to 2.0):
            Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    if alpha < 0:
        raise ValueError("alpha must be positive")

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    prob = inputs.sigmoid()
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    focal_pos = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    focal_neg = (prob**gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    loss = torch.einsum("nc,mc->nm", focal_pos, labels) + torch.einsum("nc,mc->nm", focal_neg, (1 - labels))

    return loss / height_and_width


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    batch_size = math.floor(windows.shape[0] / (height * width / window_size / window_size))
    windows = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return windows


# Copied from transformers.models.swin.modeling_swin.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinEmbeddings with MaskFormer->Mask2Former
class Mask2FormerSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = Mask2FormerSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinPatchEmbeddings with MaskFormer->Mask2Former
class Mask2FormerSwinPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding, including padding.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values):
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings_flat = embeddings.flatten(2).transpose(1, 2)

        return embeddings_flat, output_dimensions


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinPatchMerging with MaskFormer->Mask2Former,maskformer->mask2former
class Mask2FormerSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer for mask2former model.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, width, height):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature, input_dimensions):
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->Mask2FormerSwin
class Mask2FormerSwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.swin.modeling_swin.SwinSelfAttention with Swin->Mask2FormerSwin
class Mask2FormerSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        window_size = config.window_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Mask2FormerSwinModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->Mask2FormerSwin
class Mask2FormerSwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->Mask2FormerSwin
class Mask2FormerSwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads):
        super().__init__()
        self.self = Mask2FormerSwinSelfAttention(config, dim, num_heads)
        self.output = Mask2FormerSwinSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->Mask2FormerSwin
class Mask2FormerSwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput with Swin->Mask2FormerSwin
class Mask2FormerSwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinBlock with MaskFormer->Mask2Former
class Mask2FormerSwinBlock(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = Mask2FormerSwinAttention(config, dim, num_heads)
        self.drop_path = (
            Mask2FormerSwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = Mask2FormerSwinIntermediate(config, dim)
        self.output = Mask2FormerSwinOutput(config, dim)

    def get_attn_mask(self, input_resolution):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            height, width = input_resolution
            img_mask = torch.zeros((1, height, width, 1))
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_left = pad_top = 0
        pad_rigth = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, pad_left, pad_rigth, pad_top, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
        height, width = input_dimensions
        batch_size, dim, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask((height_pad, width_pad))
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        self_attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )  # B height' width' C

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinLayer with MaskFormer->Mask2Former
class Mask2FormerSwinLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                Mask2FormerSwinBlock(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self, hidden_states, input_dimensions, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        all_hidden_states = () if output_hidden_states else None

        height, width = input_dimensions
        for i, block_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            block_hidden_states = block_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = block_hidden_states[0]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        return hidden_states, output_dimensions, all_hidden_states


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinEncoder with MaskFormer->Mask2Former
class Mask2FormerSwinEncoder(nn.Module):
    def __init__(self, config, grid_size):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = nn.ModuleList(
            [
                Mask2FormerSwinLayer(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=Mask2FormerSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        input_dimensions,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_input_dimensions = ()
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_hidden_states, output_dimensions, layer_all_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states, layer_head_mask
                )
            else:
                layer_hidden_states, output_dimensions, layer_all_hidden_states = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    output_hidden_states,
                )

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            if output_hidden_states:
                all_hidden_states += (layer_all_hidden_states,)

            hidden_states = layer_hidden_states

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_all_hidden_states[1],)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return Mask2FormerSwinBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            hidden_states_spatial_dimensions=all_input_dimensions,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinModel with MaskFormer->Mask2Former
class Mask2FormerSwinModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = Mask2FormerSwinEmbeddings(config)
        self.encoder = Mask2FormerSwinEncoder(config, self.embeddings.patch_grid)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        hidden_states_spatial_dimensions = (input_dimensions,) + encoder_outputs.hidden_states_spatial_dimensions

        return Mask2FormerSwinModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            hidden_states_spatial_dimensions=hidden_states_spatial_dimensions,
            attentions=encoder_outputs.attentions,
        )


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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
        bsz, tgt_len, embed_dim = hidden_states.size()

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
            print("key_value_states:",key_value_states.shape)
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

#DetrDecoderLayer with reversed order of cross and self attention layers
class DetrDecoderLayer(nn.Module):
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self.self_attn = DetrAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        # )
        self.self_attn = nn.MultiheadAttention(self.embed_dim, 
                config.decoder_attention_heads,
                config.attention_dropout)
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
        self.encoder_attn = nn.MultiheadAttention(self.embed_dim, 
                config.decoder_attention_heads,
                config.attention_dropout)
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
                position embeddings that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                position embeddings that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            print("encoder_hidden_states[level_index]:",encoder_hidden_states[level_index].shape)
            print("position_embeddings[level_index]:",position_embeddings[level_index].shape)

            # hidden_states, cross_attn_weights = self.encoder_attn(
            #     hidden_states=hidden_states,
            #     position_embeddings=query_position_embeddings,
            #     key_value_states=encoder_hidden_states[level_index],
            #     attention_mask=encoder_attention_mask,
            #     key_value_position_embeddings=position_embeddings[level_index],
            #     output_attentions=output_attentions,
            # )

            hidden_states = self.encoder_attn(query=self.with_pos_embed(hidden_states, query_position_embeddings),
                                   key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
                                   value=encoder_hidden_states[level_index], attn_mask=encoder_attention_mask,
                                   key_padding_mask=None
            )[0]

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)


        # Self Attention
        residual = hidden_states
        # hidden_states, self_attn_weights = self.self_attn(
        #     hidden_states=hidden_states,
        #     position_embeddings=query_position_embeddings,
        #     attention_mask=None, #attention_mask,
        #     output_attentions=output_attentions,
        # )

        query = key = self.with_pos_embed(hidden_states, query_position_embeddings)

        hidden_states = self.self_attn(query=query,
                                   key=key,
                                   value=hidden_states, attn_mask=None,
                                   key_padding_mask=None
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

        print("one pass of decoder layer complete")

        return outputs


# Copied from transformers.models.detr.modeling_detr._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class DetrDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - position_embeddings and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig,mask_feature_size: torch.Tensor):
        super().__init__()
        self.config = config
        self.mask_feature_size = mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.num_feature_levels = 3 #level embedding (3 scales)

        self.layers = nn.ModuleList([DetrDecoderLayer(self.config) for _ in range(self.config.decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(self.config.d_model)

        self.forward_prediction_head = Mask2FormerPredictionHead(self.config, self.mask_feature_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        pixel_embeddings: torch.Tensor,
        feature_size_list: Optional[List],
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each cross-attention layer.
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

        outputs_class, outputs_mask, attention_mask = self.forward_prediction_head(inputs_embeds, pixel_embeddings, feature_size_list[0])

        print("outputs_class:", outputs_class.shape)
        print("outputs_mask:", outputs_mask.shape)
        print("attn_mask:", attention_mask.shape)
        
        for idx, decoder_layer in enumerate(self.layers):

            level_index = idx % self.num_feature_levels
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
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

                outputs_class, outputs_mask, attention_mask = self.forward_prediction_head(layer_outputs[0], pixel_embeddings, feature_size_list[(idx + 1) % self.num_feature_levels])
                print("attn_mask shape aftr decoderlayer:", attention_mask.shape)

            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

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
        return DetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate,
        )


# refactored from original implementation
# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerHungarianMatcher with MaskFormer->Mask2Former
class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        """Creates the matcher

        Params:
            cost_class (float, *optional*, defaults to 1.0):
                This is the relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        """Performs the matching

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

        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        # iterate through batch size
        for pred_probs, pred_mask, target_mask, labels in zip(preds_probs, preds_masks, mask_labels, class_labels):
            # downsample the target mask, save memory
            target_mask = nn.functional.interpolate(target_mask[:, None], size=pred_mask.shape[-2:], mode="nearest")
            pred_probs = pred_probs.softmax(-1)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, labels]
            # flatten spatial dimension "q h w -> q (h w)"
            pred_mask_flat = pred_mask.flatten(1)  # [num_queries, height*width]
            # same for target_mask "c h w -> c (h w)"
            target_mask_flat = target_mask[:, 0].flatten(1)  # [num_total_labels, height*width]
            # compute the focal loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_focal_loss(pred_mask_flat, target_mask_flat)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask_flat, target_mask_flat)
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

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


# copied and adapted from original implementation
# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerLoss with MaskFormer->Mask2Former
class Mask2FormerLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        matcher: Mask2FormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
    ):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we compute
        hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair of
        matched ground-truth / prediction (supervise class and mask)

        Args:
            num_labels (`int`):
                The number of classes.
            matcher (`Mask2FormerHungarianMatcher`):
                A torch module that computes the assigments between the predictions and labels.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
            eos_coef (`float`):
                Weight to apply to the null class.
        """

        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # compute finel size
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # get metadata
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
        idx = self._get_predictions_permutation_indices(indices)
        # shape = (batch_size, num_queries)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        # shape = (batch_size, num_queries)
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
        self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the masks using focal and dice loss.

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
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
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
        # upsample predictions to the target size, we have to add one dim to use interpolate
        pred_masks = nn.functional.interpolate(
            pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        pred_masks = pred_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
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

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
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
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from the
                inner layers of the Detr's Decoder.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains addional losses
            for each auxiliary predictions.
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


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSwinTransformerBackbone with MaskFormer->Mask2Former
class Mask2FormerSwinTransformerBackbone(nn.Module):
    """
    This class uses [`Mask2FormerSwinModel`] to reshape its `hidden_states` from (`batch_size, sequence_length,
    hidden_size)` to (`batch_size, num_channels, height, width)`).

    Args:
        config (`SwinConfig`):
            The configuration used by [`Mask2FormerSwinModel`].
    """

    def __init__(self, config: SwinConfig):
        super().__init__()
        self.model = Mask2FormerSwinModel(config)
        self.hidden_states_norms = nn.ModuleList([nn.LayerNorm(out_shape) for out_shape in self.outputs_shapes])

    def forward(self, *args, **kwargs) -> List[Tensor]:
        output = self.model(*args, **kwargs, output_hidden_states=True)
        hidden_states_permuted: List[Tensor] = []
        # we need to reshape the hidden state to their original spatial dimensions
        # skipping the embeddings
        hidden_states: Tuple[Tuple[Tensor]] = output.hidden_states[1:]
        # spatial dimensions contains all the heights and widths of each stage, including after the embeddings
        spatial_dimensions: Tuple[Tuple[int, int]] = output.hidden_states_spatial_dimensions
        for i, (hidden_state, (height, width)) in enumerate(zip(hidden_states, spatial_dimensions)):
            norm = self.hidden_states_norms[i]
            # the last element corespond to the layer's last block output but before patch merging
            hidden_state_unpolled = hidden_state[-1]
            hidden_state_norm = norm(hidden_state_unpolled)
            # our pixel decoder (FPN) expect 3D tensors (features)
            batch_size, _, hidden_size = hidden_state_norm.shape
            # reshape our tensor "b (h w) d -> b d h w"
            hidden_state_permuted = (
                hidden_state_norm.permute(0, 2, 1).view((batch_size, hidden_size, height, width)).contiguous()
            )
            hidden_states_permuted.append(hidden_state_permuted)
        return hidden_states_permuted

    @property
    def input_resolutions(self) -> List[int]:
        return [layer.input_resolution for layer in self.model.encoder.layers]

    @property
    def outputs_shapes(self) -> List[int]:
        return [layer.dim for layer in self.model.encoder.layers]


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerFPNConvLayer with MaskFormer->Mask2Former
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


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerFPNLayer with MaskFormer->Mask2Former
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

        self.block = Mask2FormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = nn.functional.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerFPNModel with MaskFormer->Mask2Former
class Mask2FormerFPNModel(nn.Module):
    def __init__(self, in_features: int, lateral_widths: List[int], feature_size: int = 256):
        """
        Feature Pyramid Network, given an input tensor and a set of feature map of different feature/spatial size, it
        creates a list of feature maps with the same feature size.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_widths (`List[int]`):
                A list with the features (channels) size of each lateral connection.
            feature_size (int, *optional*, defaults to 256):
                The features (channels) of the resulting feature maps.
        """
        super().__init__()
        self.stem = Mask2FormerFPNConvLayer(in_features, feature_size)
        self.layers = nn.Sequential(
            *[Mask2FormerFPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]]
        )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        fpn_features = []
        last_feature = features[-1]
        other_features = features[:-1]
        output = self.stem(last_feature)
        for layer, left in zip(self.layers, other_features[::-1]):
            output = layer(output, left)
            fpn_features.append(output)
        return fpn_features

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

# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention
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
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
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
    """

    def __init__(self, config: DeformableDetrConfig, num_feature_levels: int = 3):
        super().__init__()
        
        self.config = config
        self.dropout = self.config.dropout
        self.layers = nn.ModuleList([DeformableDetrEncoderLayer(self.config,num_feature_levels=3) for _ in range(self.config.encoder_layers)])

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
            print("hidden_states:",hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        print("hidden_states 2:",hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class MultiScaleDeformAttnEncoderModule(nn.Module):
    def __init__(self, config: DeformableDetrConfig, num_feature_levels: int = 3):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.d_model
        self.num_head = self.config.encoder_attention_heads

        self.encoder = DeformableDetrEncoder(self.config, num_feature_levels)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, self.embed_dim))

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
        valid_height = torch.sum(~mask[:,:,0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_height = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def forward(self, 
        input_embeds, 
        position_embeddings, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
     
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        masks = [torch.zeros((input_embed.size(0), input_embed.size(2), input_embed.size(3)), device=input_embed.device, dtype=torch.bool) for input_embed in input_embeds]

        input_embeds_flatten = []
        mask_flatten = []
        level_pos_embed_flatten = []
        spatial_shapes = []

        for level, (input_embed, mask, position_embedding) in enumerate(zip(input_embeds, masks, position_embeddings)):
            batch_size, channels, height, width = input_embed.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            input_embed = input_embed.flatten(2).transpose(1,2)
            mask = mask.flatten(1)
            position_embedding = position_embedding.flatten(2).transpose(1,2)
            level_pos_embed = position_embedding + self.level_embed[level].view(1,1,-1)
            level_pos_embed_flatten.append(level_pos_embed)
            input_embeds_flatten.append(input_embed)
            mask_flatten.append(mask)

        input_embeds_flatten = torch.cat(input_embeds_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        level_pos_embed_flatten = torch.cat(level_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=input_embeds_flatten.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1)

        encoder_output = self.encoder(input_embeds_flatten, mask_flatten, level_pos_embed_flatten, spatial_shapes,level_start_index, valid_ratios, output_attentions, output_hidden_states)

        output = Mask2FormerMSDeformableAttnEncoderOutput(
                last_hidden_state=encoder_output.last_hidden_state,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index, 
                hidden_states=encoder_output.hidden_states, 
                attentions=encoder_output.attentions)

        if not return_dict:
            output = tuple(value for value in [last_hidden_state, spatial_shapes, level_start_index, encoder_states, attentions] if value is not None)

        return output
        

# copied and adapted from original implementation, also practically equal to DetrSinePositionEmbedding
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
        self.scale = 2 * torch.pi if scale is None else scale

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


class Mask2FormerPixelDecoder(nn.Module):
    def __init__(self,
        config: DeformableDetrConfig,
        in_features: Dict = None,
        feature_size: int = 256, 
        mask_feature_size: int = 256,
        feature_channels: List[int] = None,
        feature_strides: List[int] = None,
        common_stride: int = None):
        """
        Pixel Decoder Module proposed in [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527). It first runs the backbone's feature into a 
        Multi-Scale Deformable Attention Transformer creating a list of feature maps. Then, it projects the last one to the correct `mask_size`.

        Args:
            feature_size (`int`, *optional*, defaults to 256):
                The feature size (channel dimension) of the feature maps.
            mask_feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the target masks size.
        """
        super().__init__()
        
        self.transformer_in_features = in_features
        self.transformer_in_channels = feature_channels[1:] 
        print("self.transformer_in_channels:",self.transformer_in_channels)
        self.transformer_feature_strides =  feature_strides 
        self.transformer_num_feature_levels = len(self.transformer_in_features)

        input_projection_list = []
        if self.transformer_num_feature_levels > 1:
            [input_projection_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, feature_size, kernel_size=1),
                    nn.GroupNorm(32, feature_size)
                )) for in_channels in self.transformer_in_channels[::-1]]

        else:
            input_projection_list.append(nn.Sequential(
                    nn.Conv2d(self.transformer_in_channels[-1], feature_size, kernel_size=1),
                    nn.GroupNorm(32, feature_size)
                ))
        
        self.input_projection = nn.ModuleList(input_projection_list)
        print("self.input_projection:",self.input_projection)

        self.msdeformattn_transformer = MultiScaleDeformAttnEncoderModule(config, self.transformer_num_feature_levels)
        
        self.position_embedding_layer = Mask2FormerSinePositionEmbedding(num_pos_feats = feature_size // 2, normalize=True)
        
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=1, stride=1, padding=0)

        self.mask2former_num_feature_levels = config.num_feature_levels

        ##Extra FPN Levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(common_stride))

        lateral_convs = []
        output_convs = []
        use_bias = False

        for idx, in_channels in enumerate(self.transformer_in_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                    nn.Conv2d(in_channels, feature_size, kernel_size=1),
                    nn.GroupNorm(32, feature_size)
                )
            output_conv = nn.Sequential(
                    nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1,bias=use_bias),
                    nn.GroupNorm(32, feature_size)
                )

            self.add_module("adapter_{}".format(idx+1), lateral_conv)
            self.add_module("layer {}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    
    def forward(self, features: List[Tensor], output_hidden_states: bool = False) -> Mask2FormerPixelDecoderOutput:
        
        input_embeds = []
        position_embeddings = []
        transformer_encoder_features = []
        multi_scale_features = []
        num_cur_levels = 0
        
        features = features[1:] 

        for idx,feature in enumerate(features[::-1]):
            print("features shape:", feature.shape)
            input_embeds.append(self.input_projection[idx](feature))
            position_embeddings.append(self.position_embedding_layer(feature))
        
        encoder_output: Mask2FormerMSDeformableAttnEncoderOutput = self.msdeformattn_transformer(input_embeds, position_embeddings)
        
        final_encoder_output = encoder_output.last_hidden_state
        spatial_shapes = encoder_output.spatial_shapes
        level_start_index = encoder_output.level_start_index
        print('final_encoder_output:',final_encoder_output)
        print("spatial_shapes:",spatial_shapes)
        print("level_start_index:",level_start_index)
        batch_size = final_encoder_output.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels

        for idx in range(self.transformer_num_feature_levels):
            if idx < self.transformer_num_feature_levels -1:
                split_size_or_sections[idx] = level_start_index[idx + 1] - level_start_index[idx]

            else:
                split_size_or_sections[idx] = final_encoder_output.shape[1] - level_start_index[idx]

        final_encoder_output = torch.split(final_encoder_output, split_size_or_sections, dim=1)
        print("final_encoder_output:",final_encoder_output)

        for idx, feature in enumerate(final_encoder_output):
            feature = feature.transpose(1,2).view(batch_size, -1, spatial_shapes[idx][0], spatial_shapes[idx][1])
            transformer_encoder_features.append(feature)

        ##append `transformer_encoder_features` with extra FPN levels

        for idx,feature in enumerate(features[:self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(feature)
            upsampled_feature = nn.functional.interpolate(transformer_encoder_features[-1], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            output = cur_fpn + upsampled_feature
            output = output_conv(output)
            transformer_encoder_features.append(output)

        for feature in transformer_encoder_features:
            if num_cur_levels < self.mask2former_num_feature_levels:
                multi_scale_features.append(feature)
                num_cur_levels += 1

        # we use the last feature map
        last_feature_projected = self.mask_projection(transformer_encoder_features[-1])


        return Mask2FormerPixelDecoderOutput(
            last_hidden_state=last_feature_projected,
            transformer_encoder_features=transformer_encoder_features[0],
            multi_scale_features=multi_scale_features,
            hidden_states=tuple(transformer_encoder_features)
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
class MaskformerMLPPredictionHead(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527). 
        It runs the input image through a backbone and a pixel decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        self.encoder = Mask2FormerSwinTransformerBackbone(config.backbone_config)
        print("self.encoder.outputs_shapes:",self.encoder.outputs_shapes) #[128, 256, 512, 1024]
        self.decoder = Mask2FormerPixelDecoder(
            config=config.pixel_decoder_config,
            in_features=config.encoder_in_features,
            feature_size=config.feature_size,
            mask_feature_size=config.mask_feature_size,
            feature_channels=self.encoder.outputs_shapes, #[256, 512, 1024], #
            feature_strides=config.feature_strides, #[8, 16, 32]
            common_stride=config.common_stride,
        )

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        features: List[Tensor] = self.encoder(pixel_values)
        for feature in features:
            print("pixfeatures:",feature.shape)
        print("Pixel output_hidden_states:",output_hidden_states)
        decoder_output: Mask2FormerPixelDecoderOutput = self.decoder(features, output_hidden_states)
        
        return Mask2FormerPixelLevelModuleOutput(
            # the last feature is actually the output from the last layer
            encoder_last_hidden_state=features[-1],
            decoder_last_hidden_state=decoder_output.last_hidden_state,
            encoder_hidden_states=tuple(features) if output_hidden_states else (),
            decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else (),
            multi_scale_features=decoder_output.multi_scale_features,
        )


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """

    def __init__(self, in_features: int,  config: Mask2FormerConfig):
        super().__init__()
        #in_features -> in_channels
        hidden_size = config.decoder_config.hidden_size #hidden_dim
        print("hidden_size:",hidden_size)
        print("in_features:",in_features)
        should_project = in_features != hidden_size 
        #self.pe_layer
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        #self.query_feat
        self.learnable_queries = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        #self.query_embed
        self.queries_embedder = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_size)

        self.input_projection = nn.ModuleList([
            nn.Conv2d(in_features, hidden_size, kernel_size=1) 
            if should_project else nn.Sequential() for _ in range(self.num_feature_levels)
        ])
        print("self.input_projection:",self.input_projection)
        
        self.decoder = DetrDecoder(config=config.decoder_config, mask_feature_size=config.mask_feature_size)


    def forward(
        self, 
        multi_scale_features: torch.Tensor, 
        pixel_embeddings: torch.Tensor,
        output_hidden_states: bool = False, 
        output_attentions: bool = False
    ) -> DetrDecoderOutput:
        input_embeds = []
        position_embeddings = []
        feature_size_list = []

        for i in range(self.num_feature_levels):
            feature_size_list.append(multi_scale_features[i].shape[-2:])
            position_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            input_embeds.append(self.input_projection[i](multi_scale_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            position_embeddings[-1] = position_embeddings[-1].permute(2,0,1)
            input_embeds[-1] = input_embeds[-1].permute(2,0,1)

        _, batch_size, _ = input_embeds[0].shape

        queries_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        learnable_queries = self.learnable_queries.weight.unsqueeze(1).repeat(1, batch_size, 1)

        print("learnable_queries:",learnable_queries.shape)
        print("queries_embeddings:",queries_embeddings.shape)
        print("feature_size_list:",feature_size_list)

        decoder_output: DetrDecoderOutput = self.decoder(
            inputs_embeds=learnable_queries,
            pixel_embeddings=pixel_embeddings,
            feature_size_list=feature_size_list,
            attention_mask=None,
            encoder_hidden_states=input_embeds,
            encoder_attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=queries_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        return decoder_output


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
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
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
        if isinstance(module, Mask2FormerTransformerModule):
            # print("Mask2FormerTransformerModule:",module)
            if module.input_projection is not None and isinstance(module.input_projection[0], nn.Conv2d):
                # nn.init.xavier_uniform_(module.input_projection[0].weight, gain=xavier_std)
                # nn.init.constant_(module.input_projection[0].bias, 0)
                c2_xavier_fill(module.input_projection[0])
        
        #Pixel Decoder
        elif isinstance(module, Mask2FormerPixelDecoder):
            if module.input_projection is not None:
               for projection in module.input_projection:
                    nn.init.xavier_uniform_(projection[0].weight, gain=xavier_std)
                    nn.init.constant_(projection[0].bias, 0)
                    
            if module.mask_projection is not None:
                c2_xavier_fill(module.mask_projection)

            if module.lateral_convs is not None:
                for lateral_conv in module.lateral_convs:
                    c2_xavier_fill(lateral_conv[0])

            if module.output_convs is not None:
                for output_conv in module.output_convs:
                    c2_xavier_fill(output_conv[0])
            
        # FPN
        # elif isinstance(module, Mask2FormerFPNModel):
        #     nn.init.xavier_uniform_(module.stem.get_submodule("0").weight, gain=xavier_std)

        # elif isinstance(module, Mask2FormerFPNLayer):
        #     nn.init.xavier_uniform_(module.proj[0].weight, gain=xavier_std)

        # elif isinstance(module, Mask2FormerFPNConvLayer):
        #     nn.init.xavier_uniform_(module.get_submodule("0").weight, gain=xavier_std)
        
        # The MLP head
        elif isinstance(module, MaskformerMLPPredictionHead):
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
        if isinstance(module, DetrDecoder):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)

class Mask2FormerModel(Mask2FormerPreTrainedModel):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(
            #in_channels = CONVS_DIM for MSpixeldecoder -> orig implementation
            in_features=config.feature_size, #self.pixel_level_module.encoder.outputs_shapes[-1], 
            config=config
        ) 

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

        pixel_level_module_output: Mask2FormerPixelLevelModuleOutput = self.pixel_level_module(
            pixel_values, output_hidden_states
        )

        multi_scale_features  = pixel_level_module_output.multi_scale_features

        pixel_embeddings = pixel_level_module_output.decoder_last_hidden_state

        #pass multi-scale features from pixel decoder to transformer module
        transformer_module_output: DetrDecoderOutput = self.transformer_module(
            multi_scale_features, pixel_embeddings, output_hidden_states, output_attentions
        )
        queries = transformer_module_output.last_hidden_state

        print("transformer_module_output:",transformer_module_output)

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
            encoder_last_hidden_state=multi_scale_features,
            pixel_decoder_last_hidden_state=pixel_embeddings,
            transformer_decoder_last_hidden_state=queries,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            hidden_states=hidden_states,
            attentions=transformer_module_output.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values())

        return output


class Mask2FormerPredictionHead(nn.Module):
    def __init__(self, config: DetrConfig, mask_feature_size: torch.Tensor):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.decoder_attention_heads
        self.class_predictor = nn.Linear(self.hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor,
            pixel_embeddings: torch.Tensor,
            attention_mask_target_size: Optional[int] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        outputs = outputs.transpose(0,1)
        classes = self.class_predictor(outputs)
        # get the masks
        mask_embeddings = self.mask_embedder(outputs)
        # sum up over the channels
        outputs_mask = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)

        attention_mask = F.interpolate(outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False)
        print("attention_mask1:",attention_mask.shape)
        attention_mask = (attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1,1).flatten(0,1) < 0.5).bool()
        print("attention_mask2:",attention_mask.shape)
        attention_mask = attention_mask.detach()
        print("attention_mask3:",attention_mask.shape)
    
        return classes, outputs_mask, attention_mask


class Mask2FormerForInstanceSegmentation(Mask2FormerPreTrainedModel):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)
        hidden_size = config.decoder_config.hidden_size
        # + 1 because we add the "null" class
        
        self.get_logits = Mask2FormerPredictionHead(config)

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = Mask2FormerLoss( #SetCriterion add-> num_points, oversample_ratio, importance_sample_ratio
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
        )

        self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_logits: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
        )
        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())


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
        >>> feature_extractor = Mask2FormerFeatureExtractor.from_pretrained("shivi/mask2former-segmentation-swin-large-ade")
        >>> inputs = feature_extractor(images=image, return_tensors="pt")

        >>> model = Mask2FormerForInstanceSegmentation.from_pretrained("shivi/mask2former-segmentation-swin-large-ade")
        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to feature_extractor for postprocessing
        >>> output = feature_extractor.post_process_segmentation(outputs)
        >>> output = feature_extractor.post_process_semantic_segmentation(outputs)
        >>> output = feature_extractor.post_process_panoptic_segmentation(outputs)
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

        pixel_embeddings = outputs.pixel_decoder_last_hidden_state

        if self.config.use_auxiliary_loss:
            transformer_decoder_outputs = torch.stack(outputs.transformer_decoder_hidden_states)
        else:
            transformer_decoder_outputs = outputs.transformer_decoder_hidden_states

        classes, output_masks, attention_masks,  = self.get_logits(transformer_decoder_outputs, pixel_embeddings)

        # get the auxiliary predictions (one for each decoder's layer)
        auxiliary_logits: List[str, Tensor] = []

        if self.config.use_auxiliary_loss:
            class_queries_logits = classes[-1]
            masks_queries_logits = output_masks[-1]
            # go till [:-1] because the last one is always used
            for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
                auxiliary_logits.append(
                    {"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes}
                )
        else:
            class_queries_logits = classes
            masks_queries_logits = output_masks

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
            masks_queries_logits=masks_queries_logits,
            auxiliary_logits=auxiliary_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values())
            if loss is not None:
                output = ((loss)) + output
        return output
