# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dino DETR model."""

import copy
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_timm_available,
    is_torch_cuda_available,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ...utils.backbone_utils import load_backbone
from .configuration_dino_detr import DinoDetrConfig


logger = logging.get_logger(__name__)

MultiScaleDeformableAttention = None


def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    global MultiScaleDeformableAttention

    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr"
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    MultiScaleDeformableAttention = load(
        "MultiScaleDeformableAttention",
        src_files,
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )


if is_timm_available():
    from timm import create_model


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DinoDetrConfig"
_CHECKPOINT_FOR_DOC = "dino_detr"


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


def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, d_model = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, d_model
        # -> batch_size, height*width, num_heads*d_model
        # -> batch_size, num_heads*d_model, height*width
        # -> batch_size*num_heads, d_model, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, d_model, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, d_model, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * d_model, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->DinoDetr
class DinoDetrMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: DinoDetrConfig, num_heads: int, n_points: int):
        super().__init__()

        kernel_loaded = MultiScaleDeformableAttention is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DinoDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

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
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        if total_elements != sequence_length:
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
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        if self.disable_custom_kernels or MultiScaleDeformableAttention is None or is_torchdynamo_compiling():
            # PyTorch implementation
            output = multi_scale_deformable_attention(
                value, spatial_shapes_list, sampling_locations, attention_weights
            )
        else:
            try:
                # custom kernel
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention(
                    value, spatial_shapes_list, sampling_locations, attention_weights
                )
        output = self.output_proj(output)

        return output, attention_weights


@dataclass
class DinoDetrEncoderOutput(ModelOutput):
    """
    Base class for outputs of the DinoDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
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
    """

    output: torch.FloatTensor
    intermediate_output: Optional[torch.FloatTensor] = None
    intermediate_ref: Optional[torch.FloatTensor] = None
    encoder_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DinoDetrDecoderOutput(ModelOutput):
    """
    Base class for outputs of the DinoDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
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
    """

    intermediate: List[torch.FloatTensor]
    ref_points: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DinoDetrEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of the DinoDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
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
    """

    hidden_states: torch.FloatTensor
    reference_points: Optional[torch.FloatTensor] = None
    hidden_states_encoder: Optional[torch.FloatTensor] = None
    reference_points_encoder: Optional[torch.FloatTensor] = None
    init_box_proposal: Optional[torch.FloatTensor] = None
    encoder_states: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DinoDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the Dino DETR encoder-decoder model.

    Args:
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[list[torch.FloatTensor]] = None
    references: Optional[list[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_reference: Optional[torch.FloatTensor] = None
    init_box_proposal: Optional[torch.FloatTensor] = None
    denoising_meta: Optional[dict] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DinoDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DinoDetrForObjectDetection`].

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
            possible padding). You can use [`~DinoDetrProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_heads, 4,
            4)`. Attentions weights of the encoder, after the attention softmax, used to compute the weighted average
            in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    last_hidden_state: torch.FloatTensor
    reference: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_reference: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[List[Dict]] = None
    denoising_meta: Optional[dict] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def _get_clones(module: torch.nn.Module, N: int, layer_share: bool = False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x: torch.FloatTensor, eps: float = 1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DinoDetrRandomBoxPerturber:
    def __init__(
        self,
        x_noise_scale: float = 0.2,
        y_noise_scale: float = 0.2,
        w_noise_scale: float = 0.2,
        h_noise_scale: float = 0.2,
    ) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors: torch.FloatTensor) -> torch.FloatTensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->DinoDetr
class DinoDetrFrozenBatchNorm2d(nn.Module):
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


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->DinoDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DinoDetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = DinoDetrFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrConvEncoder with DeformableDetr->DinoDetr
class DinoDetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DinoDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # For backwards compatibility we have to use the timm library directly instead of the AutoBackbone API
        if config.use_timm_backbone:
            # We default to values which were previously hard-coded. This enables configurability from the config
            # using backbone arguments, while keeping the default behavior the same.
            requires_backends(self, ["timm"])
            kwargs = getattr(config, "backbone_kwargs", {})
            kwargs = {} if kwargs is None else kwargs.copy()
            out_indices = kwargs.pop("out_indices", (2, 3, 4) if config.num_feature_levels > 1 else (4,))
            num_channels = kwargs.pop("in_chans", config.num_channels)
            if config.dilation:
                kwargs["output_stride"] = kwargs.get("output_stride", 16)
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=out_indices,
                in_chans=num_channels,
                **kwargs,
            )
        else:
            backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config")

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->DinoDetr
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->DinoDetr
class DinoDetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


def prepare_for_cdn(
    dn_args: torch.FloatTensor,
    training: bool,
    num_queries: int,
    num_classes: int,
    d_model: int,
    label_enc: Callable,
    device: torch.device,
):
    """
    A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param d_model: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["class_labels"])).to(device) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["class_labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["class_labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().to(device).unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().to(device).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).to(device) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, d_model).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True
            else:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True

        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
        }
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(
    outputs_class: torch.FloatTensor,
    outputs_coord: torch.FloatTensor,
    dn_meta: Dict,
    aux_loss: bool,
    _set_aux_loss: Callable,
):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {
            "logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord


class DinoDetrPositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperatureH: int = 10000,
        temperatureW: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: torch.LongTensor):
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pixel_values.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pixel_values.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        position_embeddings = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return position_embeddings


class DinoDetrLearnedPositionEmbedding(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: torch.LongTensor):
        height, width = pixel_values.shape[-2:]
        width_indices = torch.arange(width, device=pixel_values.device)
        height_indices = torch.arange(height, device=pixel_values.device)
        x_embeddings = self.col_embed(width_indices)
        y_embeddings = self.row_embed(height_indices)
        position_embeddings = (
            torch.cat(
                [
                    x_embeddings.unsqueeze(0).repeat(height, 1, 1),
                    y_embeddings.unsqueeze(1).repeat(1, width, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(pixel_values.shape[0], 1, 1, 1)
        )
        return position_embeddings


def build_position_encoding(config):
    N_steps = config.d_model // 2
    if config.position_embedding_type in ("SineHW"):
        position_embeddings = DinoDetrPositionEmbeddingSineHW(
            N_steps,
            temperatureH=config.pe_temperatureH,
            temperatureW=config.pe_temperatureW,
            normalize=True,
        )
    elif config.position_embedding_type in ("Learned"):
        position_embeddings = DinoDetrLearnedPositionEmbedding(N_steps)
    else:
        raise ValueError(f"not supported {config.position_embedding}")

    return position_embeddings


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu/glu/prelu/selu, not {activation}.")


def gen_sineembed_for_position(reference_points: torch.FloatTensor, d_model: int):
    scale = 2 * math.pi
    dim_t = torch.arange(d_model / 2, dtype=torch.float32, device=reference_points.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model / 2))
    x_embed = reference_points[:, :, 0] * scale
    y_embed = reference_points[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if reference_points.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif reference_points.size(-1) == 4:
        w_embed = reference_points[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = reference_points[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown reference_points shape(-1):{}".format(reference_points.size(-1)))
    return pos


def gen_encoder_output_proposals(
    memory: torch.FloatTensor,
    memory_padding_mask: torch.LongTensor,
    spatial_shapes: torch.FloatTensor,
    learned_wh=torch.FloatTensor,
):
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    batch_size, _, _ = memory.shape
    proposals = []
    current_height_width_prod = 0
    for level, (height, width) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[
            :, current_height_width_prod : (current_height_width_prod + height * width)
        ].view(batch_size, height, width, 1)
        valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
            torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # height, width, 2

        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale

        if learned_wh is not None:
            wh = torch.ones_like(grid) * learned_wh.sigmoid() * (2.0**level)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**level)

        proposal = torch.cat((grid, wh), -1).view(batch_size, -1, 4)
        proposals.append(proposal)
        current_height_width_prod += height * width

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


class DinoDetrPreTrainedModel(PreTrainedModel):
    config_class = DinoDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DinoDetrConvEncoder", r"DinoDetrEncoderLayer", r"DinoDetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, DinoDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, DinoDetrMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrEncoderLayer with DeformableDetr->DinoDetr
class DinoDetrEncoderLayer(nn.Module):
    def __init__(self, config: DinoDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DinoDetrMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
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
        spatial_shapes_list=None,
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
            spatial_shapes_list=spatial_shapes_list,
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


class DinoDetrDecoderLayer(nn.Module):
    def __init__(self, config: DinoDetrConfig):
        super().__init__()
        self.module_seq = config.module_seq
        # Cross attention
        self.cross_attn = DinoDetrMultiscaleDeformableAttention(config, config.num_heads, config.decoder_n_points)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)

        # Self attention
        self.self_attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Fully Connected Layer
        self.linear1 = nn.Linear(config.d_model, config.d_ffn)
        self.activation = _get_activation_fn(config.activation)
        self.dropout3 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ffn, config.d_model)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.key_aware_type = config.key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = config.decoder_sa_type

        if config.decoder_sa_type == "ca_content":
            self.self_attn = DinoDetrMultiscaleDeformableAttention(config, config.num_heads, config.decoder_n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor: torch.FloatTensor, pos: torch.FloatTensor):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, pixel_values: torch.FloatTensor):
        transformed_values = self.linear2(self.dropout3(self.activation(self.linear1(pixel_values))))
        output = pixel_values + self.dropout4(transformed_values)
        output = self.norm3(output)
        return output

    def forward_sa(
        self,
        queries: torch.FloatTensor,
        query_position_embeddings: torch.FloatTensor,
        query_reference_points: torch.FloatTensor,
        memory: torch.FloatTensor,
        memory_key_padding_mask: torch.LongTensor,
        memory_level_start_index: torch.FloatTensor,
        memory_spatial_shapes: torch.FloatTensor,
        memory_spatial_shapes_list: List[torch.FloatTensor],
        self_attn_mask: torch.LongTensor,
    ):
        attn_weights = None
        if self.self_attn is not None:
            if self.decoder_sa_type == "sa":
                q = k = self.with_pos_embed(queries, query_position_embeddings)
                transformed_queries, attn_weights = self.self_attn(q, k, queries, attn_mask=self_attn_mask)
                queries = queries + self.dropout2(transformed_queries)
                queries = self.norm2(queries)
            elif self.decoder_sa_type == "ca_label":
                bs = queries.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                transformed_queries, attn_weights = self.self_attn(queries, k, v, attn_mask=self_attn_mask)
                queries = queries + self.dropout2(transformed_queries)
                queries = self.norm2(queries)
            elif self.decoder_sa_type == "ca_content":
                transformed_queries, attn_weights = self.self_attn(
                    hidden_states=self.with_pos_embed(queries, query_position_embeddings).transpose(0, 1),
                    reference_point=query_reference_points.transpose(0, 1).contiguous(),
                    encoder_hidden_states=memory.transpose(0, 1),
                    spatial_shapes=memory_spatial_shapes,
                    spatial_shapes_list=memory_spatial_shapes_list,
                    level_start_index=memory_level_start_index,
                    encoder_attention_mask=memory_key_padding_mask,
                )
                transformed_queries = transformed_queries.transpose(0, 1)
                queries = queries + self.dropout2(transformed_queries)
                queries = self.norm2(queries)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return queries, attn_weights

    def forward_ca(
        self,
        queries: torch.FloatTensor,
        query_position_embeddings: torch.FloatTensor,
        query_reference_points: torch.FloatTensor,
        memory: torch.FloatTensor,
        memory_key_padding_mask: torch.LongTensor,
        memory_level_start_index: torch.FloatTensor,
        memory_spatial_shapes: torch.FloatTensor,
        memory_spatial_shapes_list: List[torch.FloatTensor],
    ):
        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                queries = queries + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                queries = queries + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        transformed_queries, attn_weights = self.cross_attn(
            hidden_states=self.with_pos_embed(queries, query_position_embeddings).transpose(0, 1),
            reference_points=query_reference_points.transpose(0, 1).contiguous(),
            encoder_hidden_states=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            spatial_shapes_list=memory_spatial_shapes_list,
            level_start_index=memory_level_start_index,
            encoder_attention_mask=memory_key_padding_mask,
        )
        transformed_queries = transformed_queries.transpose(0, 1)
        queries = queries + self.dropout1(transformed_queries)
        queries = self.norm1(queries)

        return queries, attn_weights

    def forward(
        self,
        queries: torch.FloatTensor,
        query_position_embeddings: torch.FloatTensor,
        query_reference_points: torch.FloatTensor,
        memory: torch.FloatTensor,
        memory_key_padding_mask: torch.LongTensor,
        memory_level_start_index: torch.FloatTensor,
        memory_spatial_shapes: torch.FloatTensor,
        memory_spatial_shapes_list: List[torch.FloatTensor],
        self_attn_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        attn_weights_total = ()
        for funcname in self.module_seq:
            if funcname == "ffn":
                queries = self.forward_ffn(queries)
            elif funcname == "ca":
                queries, attn_weights = self.forward_ca(
                    queries=queries,
                    query_position_embeddings=query_position_embeddings,
                    query_reference_points=query_reference_points,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=memory_level_start_index,
                    memory_spatial_shapes=memory_spatial_shapes,
                    memory_spatial_shapes_list=memory_spatial_shapes_list,
                )
                attn_weights_total += (attn_weights,)
            elif funcname == "sa":
                queries, attn_weights = self.forward_sa(
                    queries=queries,
                    query_position_embeddings=query_position_embeddings,
                    query_reference_points=query_reference_points,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=memory_level_start_index,
                    memory_spatial_shapes=memory_spatial_shapes,
                    memory_spatial_shapes_list=memory_spatial_shapes_list,
                    self_attn_mask=self_attn_mask,
                )
                attn_weights_total += (attn_weights,)
            else:
                raise ValueError("unknown funcname {}".format(funcname))

        outputs = (queries,)
        if output_attentions:
            outputs += (attn_weights_total,)

        return outputs


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with Detr->DinoDetr
class DinoDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DinoDetrEncoder(DinoDetrPreTrainedModel):
    def __init__(
        self,
        encoder_layer: DinoDetrEncoderLayer,
        norm: torch.nn.Module,
        config: DinoDetrConfig,
    ):
        super().__init__(config)
        if config.num_encoder_layers > 0:
            self.layers = _get_clones(
                encoder_layer,
                config.num_encoder_layers,
                layer_share=config.enc_layer_share,
            )
        else:
            self.layers = []
            del encoder_layer
        self.num_queries = config.num_queries
        self.num_encoder_layers = config.num_encoder_layers
        self.norm = norm
        self.d_model = config.d_model
        self.enc_layer_dropout_prob = config.enc_layer_dropout_prob
        self.two_stage_type = config.two_stage_type
        if config.two_stage_type in ["enceachlayer", "enclayer1"]:
            _proj_layer = nn.Linear(config.d_model, config.d_model)
            _norm_layer = nn.LayerNorm(config.d_model)
            if config.two_stage_type == "enclayer1":
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList(
                    [copy.deepcopy(_norm_layer) for i in range(config.num_encoder_layers - 1)]
                )
                self.enc_proj = nn.ModuleList(
                    [copy.deepcopy(_proj_layer) for i in range(config.num_encoder_layers - 1)]
                )

        self.post_init()

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.FloatTensor,
        valid_ratios: torch.FloatTensor,
        device: torch.device,
    ):
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        input_embeddings: torch.FloatTensor,
        position_embeddings: torch.FloatTensor,
        spatial_shapes: torch.FloatTensor,
        spatial_shapes_list: List[Tuple[int, int]],
        level_start_index: torch.FloatTensor,
        valid_ratios: torch.FloatTensor,
        key_padding_mask: torch.LongTensor,
        ref_token_index: Optional[torch.FloatTensor] = None,
        ref_token_coord: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        """
        Input:
            - input_embeds: [bs, sum(hi*wi), 256]
            - position_embeddings: pos embed for input_embeds. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        all_self_attns = () if output_attentions else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = input_embeddings

        if self.num_encoder_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=input_embeddings.device)

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)
        encoder_states = ()

        for layer_id, layer in enumerate(self.layers):
            encoder_states = encoder_states + (output,)
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                output_layer = layer(
                    hidden_states=output,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_list=spatial_shapes_list,
                    level_start_index=level_start_index,
                    attention_mask=key_padding_mask,
                    output_attentions=output_attentions,
                )

            output = output_layer[0]
            if output_attentions:
                all_self_attns += (output_layer[1],)

            if (
                (layer_id == 0 and self.two_stage_type in ["enceachlayer", "enclayer1"])
                or (self.two_stage_type == "enceachlayer")
            ) and (layer_id != self.num_encoder_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(
                    output, key_padding_mask, spatial_shapes
                )
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            if (layer_id != self.num_encoder_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        encoder_states = encoder_states + (output,)
        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output)
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    output,
                    intermediate_output,
                    intermediate_ref,
                    encoder_states,
                    all_self_attns,
                ]
                # if v is not None
            )
        return DinoDetrEncoderOutput(
            output=output,
            intermediate_output=intermediate_output,
            intermediate_ref=intermediate_ref,
            encoder_states=encoder_states,
            attentions=all_self_attns,
        )


class DinoDetrDecoder(DinoDetrPreTrainedModel):
    def __init__(
        self,
        decoder_layer: DinoDetrDecoderLayer,
        norm: torch.nn.Module,
        decoder_query_perturber: Callable[[torch.Tensor], torch.Tensor],
        config: DinoDetrConfig,
    ):
        super().__init__(config)
        if config.num_decoder_layers > 0:
            self.layers = _get_clones(
                decoder_layer,
                config.num_decoder_layers,
                layer_share=config.dec_layer_share,
            )
        else:
            self.layers = []
        self.num_decoder_layers = config.num_decoder_layers
        self.norm = norm
        self.num_feature_levels = config.num_feature_levels
        self.use_detached_boxes_dec_out = config.use_detached_boxes_dec_out
        self.ref_point_head = DinoDetrMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.query_pos_sine_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.d_model = config.d_model
        self.ref_anchor_head = None
        self.decoder_query_perturber = decoder_query_perturber
        self.dec_layer_number = config.dec_layer_number
        self.dec_layer_dropout_prob = config.dec_layer_dropout_prob
        self.dec_detach = config.dec_detach

        self.post_init()

    def forward(
        self,
        queries: torch.FloatTensor,
        memory: torch.FloatTensor,
        refpoints_unsigmoid: torch.FloatTensor,
        spatial_shapes_list: List[Tuple[int, int]],
        self_attn_mask: Optional[torch.LongTensor] = None,
        memory_key_padding_mask: Optional[torch.LongTensor] = None,
        level_start_index: Optional[torch.FloatTensor] = None,
        spatial_shapes: Optional[torch.FloatTensor] = None,
        valid_ratios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        all_attns = () if output_attentions else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output = queries

        intermediate = [output]
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            elif reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], d_model=self.d_model
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = 1
            query_pos = pos_scale * raw_query_pos

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output_layer = layer(
                    queries=output,
                    query_position_embeddings=query_pos,
                    query_reference_points=reference_points_input,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_spatial_shapes_list=spatial_shapes_list,
                    self_attn_mask=self_attn_mask,
                    output_attentions=output_attentions,
                )
                output = output_layer[0]
                if output_attentions:
                    all_attns += (output_layer[1],)

            # iter update
            if self.bbox_embed is not None:
                new_reference_points_unsigmoid = self.bbox_embed[layer_id](output) + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points_unsigmoid.sigmoid()

                # select # ref points
                if self.dec_layer_number is not None and layer_id != self.num_decoder_layers - 1:
                    new_reference_points_number = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if new_reference_points_number != select_number:
                        class_unselected = self.class_embed[layer_id](output)  # nq, bs, 91
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1]  # new_nq, bs
                        new_reference_points = torch.gather(
                            new_reference_points,
                            0,
                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
                        )  # unsigmoid

                if self.dec_detach:
                    reference_points = new_reference_points.detach()
                else:
                    reference_points = new_reference_points
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if self.dec_layer_number is not None and layer_id != self.num_decoder_layers - 1:
                if new_reference_points_number != select_number:
                    output = torch.gather(
                        output,
                        0,
                        topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
                    )  # unsigmoid

        if not return_dict:
            return (
                [itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
                all_attns,
            )
        return DinoDetrDecoderOutput(
            intermediate=[itm_out.transpose(0, 1) for itm_out in intermediate],
            ref_points=[itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            attentions=all_attns,
        )


class DinoDetrEncoderDecoder(DinoDetrPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.decoder_layer_noise:
            self.decoder_query_perturber = DinoDetrRandomBoxPerturber(
                x_noise_scale=config.dln_xy_noise,
                y_noise_scale=config.dln_xy_noise,
                w_noise_scale=config.dln_hw_noise,
                h_noise_scale=config.dln_hw_noise,
            )
        else:
            self.decoder_query_perturber = None

        self.num_feature_levels = config.num_feature_levels
        self.two_stage_keep_all_tokens = config.two_stage_keep_all_tokens
        self.num_queries = config.num_queries
        self.random_refpoints_xy = config.random_refpoints_xy
        encoder_layer = DinoDetrEncoderLayer(config)
        encoder_norm = nn.LayerNorm(config.d_model) if config.normalize_before else None
        self.encoder = DinoDetrEncoder(encoder_layer, encoder_norm, config)
        decoder_layer = DinoDetrDecoderLayer(config)
        decoder_norm = nn.LayerNorm(config.d_model)
        self.decoder = DinoDetrDecoder(
            decoder_layer=decoder_layer,
            norm=decoder_norm,
            decoder_query_perturber=self.decoder_query_perturber,
            config=config,
        )
        self.d_model = config.d_model
        self.num_patterns = config.num_patterns
        if config.num_feature_levels > 1:
            if config.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))
            else:
                self.level_embed = None
        self.embed_init_tgt = config.embed_init_tgt
        if (config.two_stage_type != "no" and config.embed_init_tgt) or (config.two_stage_type == "no"):
            self.content_query_embeddings = nn.Embedding(self.num_queries, config.d_model)
            nn.init.normal_(self.content_query_embeddings.weight.data)
        else:
            self.content_query_embeddings = None
        # for two stage
        self.two_stage_type = config.two_stage_type
        self.two_stage_pat_embed = config.two_stage_pat_embed
        self.two_stage_add_query_num = config.two_stage_add_query_num
        self.two_stage_learn_wh = config.two_stage_learn_wh
        if config.two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)

            if config.two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(config.two_stage_pat_embed, config.d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if config.two_stage_add_query_num > 0:
                self.content_query_embeddings = nn.Embedding(self.two_stage_add_query_num, config.d_model)

            if config.two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if config.two_stage_type == "no":
            self.init_ref_points(config.num_queries)  # init self.refpoint_embed
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        if config.rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(config.rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in config.rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.post_init()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DinoDetrMultiscaleDeformableAttention):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask: torch.FloatTensor):
        _, height, width = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)
        valid_W = torch.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / height
        valid_ratio_w = valid_W.float() / width
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries: int):
        self.content_query_reference_points = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.content_query_reference_points.weight.data[:, :2].uniform_(0, 1)
            self.content_query_reference_points.weight.data[:, :2] = inverse_sigmoid(
                self.content_query_reference_points.weight.data[:, :2]
            )
            self.content_query_reference_points.weight.data[:, :2].requires_grad = False

    def forward(
        self,
        pixel_values: List[torch.FloatTensor],
        pixel_masks: List[torch.LongTensor],
        pixel_position_embeddings: List[torch.FloatTensor],
        query_reference_points: torch.FloatTensor,
        queries: torch.FloatTensor,
        attn_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        """
        Input:
            - pixel_values: List of multi features [batch_size, ci, hi, wi]
            - pixel_masks: List of multi pixel_masks [batch_size, hi, wi]
            - query_reference_points: [batch_size, num_dn, 4]. None in infer
            - query_positional_embeddings: List of multi pos embeds [batch_size, ci, hi, wi]
            - queries: [batch_size, num_dn, d_model]. None in infer

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        encoder_attentions = None
        decoder_attentions = None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare input for encoder
        src_flatten = []
        mask_flatten = []
        level_pos_embed_flatten = []
        spatial_shapes_list = []
        for level, (src, mask, pos_embed) in enumerate(zip(pixel_values, pixel_masks, pixel_position_embeddings)):
            batch_size, c, height, width = src.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                level_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            else:
                level_pos_embed = pos_embed
            level_pos_embed_flatten.append(level_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        level_pos_embed_flatten = torch.cat(level_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in pixel_masks], 1)

        encoder_topk_proposals = encoder_query_reference_points = None

        # Begin Encoder
        outputs_encoder_part = self.encoder(
            input_embeddings=src_flatten,
            position_embeddings=level_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=encoder_topk_proposals,
            ref_token_coord=encoder_query_reference_points,
            return_dict=return_dict,
            output_attentions=output_attentions,
        )
        if not return_dict:
            (
                memory,
                # enc_intermediate_output,
                # enc_intermediate_refpoints,
                encoder_states,
            ) = (
                outputs_encoder_part[0],
                # outputs_encoder_part[1],
                # outputs_encoder_part[2],
                outputs_encoder_part[3],
            )
            if output_attentions:
                encoder_attentions = outputs_encoder_part[-1]
        else:
            (
                memory,
                # enc_intermediate_output,
                # enc_intermediate_refpoints,
                encoder_states,
            ) = (
                outputs_encoder_part["output"],
                # outputs_encoder_part["intermediate_output"],
                # outputs_encoder_part["intermediate_ref"],
                outputs_encoder_part["encoder_states"],
            )
            if output_attentions:
                encoder_attentions = outputs_encoder_part["attentions"]

        # Prepare queries
        mask_flatten = ~mask_flatten
        if self.two_stage_type == "standard":
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                batch_size, nhw, _ = output_memory.shape
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                output_memory = torch.cat((output_memory, queries), dim=1)
                output_proposals = torch.cat((output_proposals, query_reference_points), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

            query_reference_points_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            content_query_reference_points = query_reference_points_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()

            queries_undetach = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )
            if self.embed_init_tgt:
                content_queries = (
                    self.content_query_embeddings.weight[:, None, :].repeat(1, batch_size, 1).transpose(0, 1)
                )
            else:
                content_queries = queries_undetach.detach()

            if query_reference_points is not None:
                query_reference_points = torch.cat([query_reference_points, content_query_reference_points], dim=1)
                queries = torch.cat([queries, content_queries], dim=1)
            else:
                query_reference_points, queries = (
                    content_query_reference_points,
                    content_queries,
                )

        elif self.two_stage_type == "no":
            content_queries = self.content_query_embeddings.weight[:, None, :].repeat(1, batch_size, 1).transpose(0, 1)
            content_query_reference_points = (
                self.content_query_reference_points.weight[:, None, :].repeat(1, batch_size, 1).transpose(0, 1)
            )

            if query_reference_points is not None:
                query_reference_points = torch.cat([query_reference_points, content_query_reference_points], dim=1)
                queries = torch.cat([queries, content_queries], dim=1)
            else:
                query_reference_points, queries = (
                    content_query_reference_points,
                    content_queries,
                )

            if self.num_patterns > 0:
                queries_embed = queries.repeat(1, self.num_patterns, 1)
                query_reference_points = query_reference_points.repeat(1, self.num_patterns, 1)
                queries_patterns = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1)
                queries = queries_embed + queries_patterns

            init_box_proposal = content_query_reference_points.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))

        # Decoder
        outputs_decoder_part = self.decoder(
            queries=queries.transpose(0, 1),
            memory=memory.transpose(0, 1),
            self_attn_mask=attn_mask,
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=query_reference_points.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            return_dict=return_dict,
            output_attentions=output_attentions,
        )
        if not return_dict:
            hidden_states, reference_points = (
                outputs_decoder_part[0],
                outputs_decoder_part[1],
            )
            if output_attentions:
                decoder_attentions = outputs_decoder_part[-1]
        else:
            hidden_states, reference_points = (
                outputs_decoder_part["intermediate"],
                outputs_decoder_part["ref_points"],
            )
            if output_attentions:
                decoder_attentions = outputs_decoder_part["attentions"]

        # Postprocess
        if self.two_stage_type == "standard":
            if self.two_stage_keep_all_tokens:
                hidden_states_encoder = output_memory.unsqueeze(0)
                reference_points_encoder = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:
                hidden_states_encoder = queries_undetach.unsqueeze(0)
                reference_points_encoder = query_reference_points_undetach.sigmoid().unsqueeze(0)
        else:
            hidden_states_encoder = reference_points_encoder = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    reference_points,
                    hidden_states_encoder,
                    reference_points_encoder,
                    init_box_proposal,
                    encoder_states,
                    encoder_attentions,
                    decoder_attentions,
                ]
                # if v is not None
            )
        return DinoDetrEncoderDecoderOutput(
            hidden_states=hidden_states,
            reference_points=reference_points,
            hidden_states_encoder=hidden_states_encoder,
            reference_points_encoder=reference_points_encoder,
            init_box_proposal=init_box_proposal,
            encoder_states=encoder_states,
            encoder_attentions=encoder_attentions,
            decoder_attentions=decoder_attentions,
        )


DINO_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DinoDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DINO_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DinoDetrImageProcessor.__call__`]
            for details.

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
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """
    The bare Dino DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    DINO_DETR_START_DOCSTRING,
)
class DinoDetrModel(DinoDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [
        "bbox_embed",
        "class_embed",
        r"bbox_embed\.[1-9]\d*",
        r"class_embed\.[1-9]\d*",
        r"transformer\.decoder\.bbox_embed\.[1-9]\d*",
        r"transformer\.decoder\.class_embed\.[1-9]\d*",
    ]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: DinoDetrConfig):
        super().__init__(config)
        # create deformable transformer
        self.transformer = DinoDetrEncoderDecoder(config=config)
        self.label_enc = nn.Embedding(config.dn_labelbook_size + 1, config.d_model)

        # Create backbone + positional encoding
        backbone = DinoDetrConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = DinoDetrConvModel(backbone, position_embeddings)
        self.output_hidden_states = config.output_hidden_states
        d_model = config.d_model

        # Prepare input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.conv_encoder.intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.conv_encoder.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            d_model,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(32, d_model),
                    )
                )
                in_channels = d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.backbone.conv_encoder.intermediate_channel_sizes[-1], d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    )
                ]
            )

        # Prepare class & box embed
        self.class_embed = nn.Linear(config.d_model, config.num_classes)
        self.bbox_embed = DinoDetrMLPPredictionHead(d_model, d_model, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if config.dec_pred_bbox_embed_share:
            self.bbox_embed = _get_clones(self.bbox_embed, config.num_decoder_layers, layer_share=True)
        else:
            self.bbox_embed = [copy.deepcopy(self.bbox_embed) for _ in range(config.num_decoder_layers)]
        if config.dec_pred_class_embed_share:
            self.class_embed = _get_clones(self.class_embed, config.num_decoder_layers, layer_share=True)
        else:
            self.class_embed = [copy.deepcopy(self.class_embed) for _ in range(config.num_decoder_layers)]
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # Adjust embeddings based on two stage approach
        if config.two_stage_type != "no":
            if config.two_stage_bbox_embed_share:
                self.transformer.enc_out_bbox_embed = self.bbox_embed[0]
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(self.bbox_embed[0])

            if config.two_stage_class_embed_share:
                self.transformer.enc_out_class_embed = self.class_embed[0]
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(self.class_embed[0])

        if config.decoder_sa_type == "ca_label":
            self.label_embedding = nn.Embedding(config.num_classes, d_model)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        # self._reset_parameters()
        self.post_init()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    @add_start_docstrings_to_model_forward(DINO_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DinoDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[List[dict]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DinoDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("dino_detr")
        >>> model = DinoDetrModel.from_pretrained("dino_detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        encoder_attentions = None
        decoder_attentions = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        features, position_embeddings = self.backbone(pixel_values, pixel_mask)
        srcs = []
        masks = []
        for level, (src, mask) in enumerate(features):
            srcs.append(self.input_proj[level](src))
            masks.append(mask)
        if self.config.num_feature_levels > len(srcs):
            srcs_length = len(srcs)
            for additional_level in range(srcs_length, self.config.num_feature_levels):
                if additional_level == srcs_length:
                    src = self.input_proj[additional_level](features[-1][0])
                else:
                    src = self.input_proj[additional_level](srcs[-1])
                mask = F.interpolate(pixel_mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                additional_position_embedding = self.backbone.position_embedding(src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                position_embeddings.append(additional_position_embedding)

        if self.config.dn_number > 0 and labels is not None:
            queries, query_reference_points, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(
                    labels,
                    self.config.dn_number,
                    self.config.dn_label_noise_ratio,
                    self.config.dn_box_noise_scale,
                ),
                training=self.training,
                num_queries=self.config.num_queries,
                num_classes=self.config.num_classes,
                d_model=self.config.d_model,
                label_enc=self.label_enc,
                device=device,
            )
        else:
            queries = query_reference_points = attn_mask = dn_meta = None

        outputs_transformer_part = self.transformer(
            pixel_values=srcs,
            pixel_masks=masks,
            pixel_position_embeddings=position_embeddings,
            query_reference_points=query_reference_points,
            queries=queries,
            attn_mask=attn_mask,
            return_dict=return_dict,
            output_attentions=output_attentions,
        )
        if not return_dict:
            (
                hidden_states,
                reference_points,
                hidden_states_encoder,
                reference_points_encoder,
                init_box_proposal,
                encoder_states,
            ) = (
                outputs_transformer_part[0],
                outputs_transformer_part[1],
                outputs_transformer_part[2],
                outputs_transformer_part[3],
                outputs_transformer_part[4],
                outputs_transformer_part[5],
            )
            if output_attentions:
                encoder_attentions = outputs_transformer_part[-2]
                decoder_attentions = outputs_transformer_part[-1]
        else:
            (
                hidden_states,
                reference_points,
                hidden_states_encoder,
                reference_points_encoder,
                init_box_proposal,
                encoder_states,
            ) = (
                outputs_transformer_part["hidden_states"],
                outputs_transformer_part["reference_points"],
                outputs_transformer_part["hidden_states_encoder"],
                outputs_transformer_part["reference_points_encoder"],
                outputs_transformer_part["init_box_proposal"],
                outputs_transformer_part["encoder_states"],
            )
            if output_attentions:
                encoder_attentions = outputs_transformer_part["encoder_attentions"]
                decoder_attentions = outputs_transformer_part["decoder_attentions"]

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states[-1],
                    hidden_states,
                    reference_points,
                    hidden_states_encoder,
                    reference_points_encoder,
                    init_box_proposal,
                    dn_meta,
                    (hidden_states if output_hidden_states or self.config.output_hidden_states else None),
                    (encoder_states if output_hidden_states or self.config.output_hidden_states else None),
                    encoder_attentions,
                    decoder_attentions,
                ]
                if v is not None
            )
        return DinoDetrModelOutput(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
            references=reference_points,
            encoder_last_hidden_state=hidden_states_encoder,
            encoder_reference=reference_points_encoder,
            init_box_proposal=init_box_proposal,
            denoising_meta=dn_meta,
            decoder_hidden_states=(
                hidden_states if output_hidden_states or self.config.output_hidden_states else None
            ),
            encoder_hidden_states=(
                encoder_states if output_hidden_states or self.config.output_hidden_states else None
            ),
            encoder_attentions=encoder_attentions,
            decoder_attentions=decoder_attentions,
        )


@add_start_docstrings(
    """
    Dino DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DINO_DETR_START_DOCSTRING,
)
class DinoDetrForObjectDetection(DinoDetrPreTrainedModel):
    _tied_weights_keys = [
        "bbox_embed",
        "class_embed",
        r"bbox_embed\.[1-9]\d*",
        r"class_embed\.[1-9]\d*",
        r"transformer\.decoder\.bbox_embed\.[1-9]\d*",
        r"transformer\.decoder\.class_embed\.[1-9]\d*",
    ]

    def __init__(self, config: DinoDetrConfig):
        super().__init__(config)

        # Dino DETR encoder-decoder model
        self.model = DinoDetrModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DINO_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DinoDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[List[dict]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DinoDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("dino_detr")
        >>> model = DinoDetrForObjectDetection.from_pretrained("dino_detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected cat with confidence 0.8 at location [16.5, 52.84, 318.25, 470.78]
        Detected cat with confidence 0.789 at location [342.19, 24.3, 640.02, 372.25]
        Detected remote with confidence 0.633 at location [40.79, 72.78, 176.76, 117.25]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        encoder_attentions = None
        decoder_attentions = None

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        denoising_meta = None

        # Apply base model to inputs
        outputs_model_part = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if not return_dict:
            (
                last_hidden_state,
                hidden_states,
                reference_points,
                hidden_states_encoder,
                reference_points_encoder,
                init_box_proposal,
            ) = (
                outputs_model_part[0],
                outputs_model_part[1][1:],
                outputs_model_part[2],
                outputs_model_part[3],
                outputs_model_part[4],
                outputs_model_part[5],
            )
            if self.training:
                denoising_meta = outputs_model_part[6]
            if self.training and (output_hidden_states or self.model.output_hidden_states):
                decoder_hidden_states = outputs_model_part[7]
                encoder_hidden_states = outputs_model_part[8]
            if not self.training and (output_hidden_states or self.model.output_hidden_states):
                decoder_hidden_states = outputs_model_part[6]
                encoder_hidden_states = outputs_model_part[7]
            if output_attentions:
                encoder_attentions = outputs_model_part[-2]
                decoder_attentions = outputs_model_part[-1]

        else:
            last_hidden_state = outputs_model_part.last_hidden_state
            hidden_states = outputs_model_part.hidden_states[1:]
            reference_points = outputs_model_part.references
            hidden_states_encoder = outputs_model_part.encoder_last_hidden_state
            reference_points_encoder = outputs_model_part.encoder_reference
            init_box_proposal = outputs_model_part.init_box_proposal
            if self.training:
                denoising_meta = outputs_model_part.denoising_meta
            if output_hidden_states or self.model.output_hidden_states:
                decoder_hidden_states = outputs_model_part.decoder_hidden_states
                encoder_hidden_states = outputs_model_part.encoder_hidden_states
            if output_attentions:
                encoder_attentions = outputs_model_part.encoder_attentions
                decoder_attentions = outputs_model_part.decoder_attentions

        # Convert hidden states to bounding boxes
        hidden_states[0] += self.model.label_enc.weight[0, 0] * 0.0
        outputs_coord_list = []
        for _, (
            layer_reference_points_sigmoid,
            layer_bbox_embed,
            layer_hidden_states,
        ) in enumerate(
            zip(
                reference_points[:-1],
                self.model.transformer.decoder.bbox_embed,
                hidden_states,
            )
        ):
            layer_outputs_unsigmoid = layer_bbox_embed(layer_hidden_states) + inverse_sigmoid(
                layer_reference_points_sigmoid
            )
            outputs_coord_list.append(layer_outputs_unsigmoid.sigmoid())
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hidden_states)
                for layer_cls_embed, layer_hidden_states in zip(
                    self.model.transformer.decoder.class_embed, hidden_states
                )
            ]
        )

        # Apply post processing and compute loss
        if self.config.dn_number > 0 and denoising_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                denoising_meta,
                self.config.auxiliary_loss,
                self._set_aux_loss,
            )

        if self.config.auxiliary_loss:
            out_aux_loss = self._set_aux_loss(outputs_class, outputs_coord_list)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits=outputs_class[-1],
                labels=labels,
                device=self.device,
                pred_boxes=outputs_coord_list[-1],
                dn_meta=denoising_meta,
                outputs_class=outputs_class,
                outputs_coord=outputs_coord_list,
                class_cost=self.config.class_cost,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
                num_labels=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                auxiliary_loss=self.config.auxiliary_loss,
                cls_loss_coefficient=self.config.cls_loss_coefficient,
                bbox_loss_coefficient=self.config.bbox_loss_coefficient,
                giou_loss_coefficient=self.config.giou_loss_coefficient,
                mask_loss_coefficient=self.config.mask_loss_coefficient,
                use_dn=self.config.use_dn,
                use_masks=self.config.use_masks,
                dice_loss_coefficient=self.config.dice_loss_coefficient,
                num_decoder_layers=self.config.num_decoder_layers,
                two_stage_type=self.config.two_stage_type,
                no_interm_box_loss=self.config.no_interm_box_loss,
                interm_loss_coef=self.config.interm_loss_coef,
            )

        # Remove?
        out = {}
        # Prepare encoder output
        if hidden_states_encoder is not None:
            # Prepare intermediate outputs
            interm_coord = reference_points_encoder[-1]
            interm_class = self.model.transformer.enc_out_class_embed(hidden_states_encoder[-1])
            out["interm_outputs"] = {
                "logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

            # Prepare enc outputs
            if hidden_states_encoder.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (
                    layer_box_embed,
                    layer_class_embed,
                    layer_hs_enc,
                    layer_ref_enc,
                ) in enumerate(
                    zip(
                        self.enc_bbox_embed,
                        self.enc_class_embed,
                        hidden_states_encoder[:-1],
                        reference_points_encoder[:-1],
                    )
                ):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out["enc_outputs"] = [
                    {"logits": a, "pred_boxes": b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]
        # End remove

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    reference_points[-1],
                    hidden_states_encoder,
                    reference_points_encoder,
                    loss,
                    loss_dict,
                    outputs_class[-1],
                    outputs_coord_list[-1],
                    out_aux_loss,
                    denoising_meta,
                    (encoder_hidden_states if output_hidden_states or self.model.output_hidden_states else None),
                    (decoder_hidden_states if output_hidden_states or self.model.output_hidden_states else None),
                    encoder_attentions,
                    decoder_attentions,
                ]
                if v is not None
            )
        dict_outputs = DinoDetrObjectDetectionOutput(
            last_hidden_state=last_hidden_state,
            reference=reference_points[-1],
            encoder_last_hidden_state=hidden_states_encoder,
            encoder_reference=reference_points_encoder,
            loss=loss,
            loss_dict=loss_dict,
            logits=outputs_class[-1],
            pred_boxes=outputs_coord_list[-1],
            auxiliary_outputs=out_aux_loss,
            denoising_meta=denoising_meta,
            encoder_hidden_states=(
                encoder_hidden_states if output_hidden_states or self.model.output_hidden_states else None
            ),
            decoder_hidden_states=(
                decoder_hidden_states if output_hidden_states or self.model.output_hidden_states else None
            ),
            encoder_attentions=encoder_attentions,
            decoder_attentions=decoder_attentions,
        )
        return dict_outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


__all__ = [
    "DinoDetrForObjectDetection",
    "DinoDetrModel",
    "DinoDetrPreTrainedModel",
]
