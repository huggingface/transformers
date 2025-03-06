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
import random
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
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

from torch.nn.init import constant_, xavier_uniform_

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
_CHECKPOINT_FOR_DOC = "dino-detr"


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
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
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
        grad_value, grad_sampling_loc, grad_attn_weight = (
            MultiScaleDeformableAttention.ms_deform_attn_backward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                grad_output,
                context.im2col_step,
            )
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
    value_list = value.split(
        [height * width for height, width in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, d_model
        # -> batch_size, height*width, num_heads*d_model
        # -> batch_size, num_heads*d_model, height*width
        # -> batch_size*num_heads, d_model, height, width
        value_l_ = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, d_model, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        )
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
                logger.warning(
                    f"Could not load the custom kernel for multi-scale deformable attention: {e}"
                )

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
        self.num_feature_levels = config.num_feature_levels
        self.num_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            config.d_model, num_heads * self.num_feature_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            config.d_model, num_heads * self.num_feature_levels * n_points
        )
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]
    ):
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
        value = value.view(
            batch_size, sequence_length, self.num_heads, self.d_model // self.num_heads
        )
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_feature_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_feature_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_feature_levels,
            self.n_points,
        )
        # batch_size, num_queries, num_heads, num_feature_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        if (
            self.disable_custom_kernels
            or MultiScaleDeformableAttention is None
            or is_torchdynamo_compiling()
        ):
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

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_feature_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)


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

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


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

    hs: torch.FloatTensor = None
    reference: torch.FloatTensor = None
    hs_enc: torch.FloatTensor = None
    ref_enc: torch.FloatTensor = None
    init_box_proposal: torch.FloatTensor = None
    dn_meta: dict = None


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

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    pred_logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class RandomBoxPerturber:
    def __init__(
        self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2
    ) -> None:
        self.noise_scale = torch.Tensor(
            [x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale]
        )

    def __call__(self, refanchors: Tensor) -> Tensor:
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
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
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
            out_indices = kwargs.pop(
                "out_indices", (2, 3, 4) if config.num_feature_levels > 1 else (4,)
            )
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
            self.model.feature_info.channels()
            if config.use_timm_backbone
            else self.model.channels
        )

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError(
                "Either `backbone` or `backbone_config` should be provided in the config"
            )

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if (
                        "layer2" not in name
                        and "layer3" not in name
                        and "layer4" not in name
                    ):
                        parameter.requires_grad_(False)
                else:
                    if (
                        "stage.1" not in name
                        and "stage.2" not in name
                        and "stage.3" not in name
                    ):
                        parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->DinoDetr
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = (
            self.model(pixel_values)
            if self.config.use_timm_backbone
            else self.model(pixel_values).feature_maps
        )

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(
                pixel_mask[None].float(), size=feature_map.shape[-2:]
            ).to(torch.bool)[0]
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
        return_interm_indices = [1, 2, 3]  # [[0,1,2,3], [1,2,3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        self.num_channels = num_channels_all[4 - len(return_interm_indices) :]

    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, d_model, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [d_model] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(
            tensor.dim()
        )
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    def _max_by_axis(the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


def prepare_for_cdn(dn_args, training, num_queries, num_classes, d_model, label_enc):
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
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
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
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

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
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes)))
            .long()
            .cuda()
            .unsqueeze(0)
            .repeat(dn_number, 1)
        )
        positive_idx += (
            (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32)
                * 2.0
                - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = (
                known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            )
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, d_model).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2
                ] = True
            else:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i
                ] = True

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


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
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
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord


class DinoDetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, embedding_dim=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=pixel_values.dtype)
        x_embed = pixel_mask.cumsum(2, dtype=pixel_values.dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.embedding_dim, dtype=pixel_values.dtype, device=pixel_values.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding
class DinoDetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(height, 1, 1),
                y_emb.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


# Copied from transformers.models.detr.modeling_detr.build_position_encoding with Detr->DinoDetr
def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = DinoDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = DinoDetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


def _get_activation_fn(activation, d_model=256, batch_dim=0):
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

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(
    memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
            N_, H_, W_, 1
        )
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float("inf")
    )
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float("inf")
    )

    output_memory = memory
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0)
    )
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


class DinoDetrMultiheadAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the Dino DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
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
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]
    ):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, target_len, embed_dim = hidden_states.size()
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # get queries, keys and values
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(
            *proj_shape
        )
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(batch_size, self.num_heads, target_len, source_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, target_len, source_len
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_heads, target_len, source_len
            )
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_heads, target_len, source_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (
            batch_size * self.num_heads,
            target_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            batch_size, self.num_heads, target_len, self.head_dim
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class DinoDetrEncoderLayer(nn.Module):
    def __init__(self, config: DinoDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DinoDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.num_heads,
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

        # Apply Multi-scale Dino Attention Module on the multi-scale feature maps.
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

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DinoDetrDecoderLayer(nn.Module):
    def __init__(self, config: DinoDetrConfig):
        """
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        num_heads=8,
        n_points=4,
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        key_aware_type=None,
        decoder_sa_type="ca",
        module_seq=["sa", "ca", "ffn"],
        """
        super().__init__()
        self.module_seq = config.module_seq
        assert sorted(config.module_seq) == ["ca", "ffn", "sa"]
        # cross attention
        self.cross_attn = DinoDetrMultiscaleDeformableAttention(
            config, config.num_heads, config.decoder_n_points
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout
        )
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)

        # ffn
        self.linear1 = nn.Linear(config.d_model, config.d_ffn)
        self.activation = _get_activation_fn(
            config.activation, d_model=config.d_ffn, batch_dim=1
        )
        self.dropout3 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ffn, config.d_model)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.key_aware_type = config.key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = config.decoder_sa_type
        assert config.decoder_sa_type in ["sa", "ca_label", "ca_content"]

        if config.decoder_sa_type == "ca_content":
            self.self_attn = DinoDetrMultiscaleDeformableAttention(
                config, config.num_heads, config.decoder_n_points
            )

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_spatial_shapes_list: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == "sa":
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == "ca_label":
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == "ca_content":
                tgt2, attn_weights = self.self_attn(
                    hidden_states=self.with_pos_embed(tgt, tgt_query_pos).transpose(
                        0, 1
                    ),
                    reference_point=tgt_reference_points.transpose(0, 1).contiguous(),
                    encoder_hidden_states=memory.transpose(0, 1),
                    spatial_shapes=memory_spatial_shapes,
                    spatial_shapes_list=memory_spatial_shapes_list,
                    level_start_index=memory_level_start_index,
                    encoder_attention_mask=memory_key_padding_mask,
                )
                tgt2 = tgt2.transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError(
                    "Unknown decoder_sa_type {}".format(self.decoder_sa_type)
                )

        return tgt

    def forward_ca(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_spatial_shapes_list: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(
                    "Unknown key_aware_type: {}".format(self.key_aware_type)
                )
        tgt2, attn_weights = self.cross_attn(
            hidden_states=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            encoder_hidden_states=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            spatial_shapes_list=memory_spatial_shapes_list,
            level_start_index=memory_level_start_index,
            encoder_attention_mask=memory_key_padding_mask,
        )
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_spatial_shapes_list: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        for funcname in self.module_seq:
            if funcname == "ffn":
                tgt = self.forward_ffn(tgt)
            elif funcname == "ca":
                tgt = self.forward_ca(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_spatial_shapes_list,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            elif funcname == "sa":
                tgt = self.forward_sa(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes_list,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            else:
                raise ValueError("unknown funcname {}".format(funcname))

        return tgt


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class DinoDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, d_model, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [d_model] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DinoDetrEncoder(nn.Module):
    def __init__(self, encoder_layer, norm, config):
        """
        encoder_layer,
        num_encoder_layers,
        norm=None,
        d_model=256,
        num_queries=300,
        deformable_encoder=False,
        enc_layer_share=False,
        enc_layer_dropout_prob=None,
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        """
        super().__init__()
        # prepare layers
        if config.num_encoder_layers > 0:
            self.layers = _get_clones(
                encoder_layer,
                config.num_encoder_layers,
                layer_share=config.enc_layer_share,
            )
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = config.num_queries
        self.deformable_encoder = config.deformable_encoder
        self.num_encoder_layers = config.num_encoder_layers
        self.norm = norm
        self.d_model = config.d_model

        self.enc_layer_dropout_prob = config.enc_layer_dropout_prob
        if config.enc_layer_dropout_prob is not None:
            assert isinstance(config.enc_layer_dropout_prob, list)
            assert len(config.enc_layer_dropout_prob) == config.num_encoder_layers
            for i in config.enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = config.two_stage_type
        if config.two_stage_type in ["enceachlayer", "enclayer1"]:
            _proj_layer = nn.Linear(config.d_model, config.d_model)
            _norm_layer = nn.LayerNorm(config.d_model)
            if config.two_stage_type == "enclayer1":
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList(
                    [
                        copy.deepcopy(_norm_layer)
                        for i in range(config.num_encoder_layers - 1)
                    ]
                )
                self.enc_proj = nn.ModuleList(
                    [
                        copy.deepcopy(_proj_layer)
                        for i in range(config.num_encoder_layers - 1)
                    ]
                )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: List,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        ref_token_index: Optional[Tensor] = None,
        ref_token_coord: Optional[Tensor] = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
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
        if self.two_stage_type in ["no", "standard", "enceachlayer", "enclayer1"]:
            assert ref_token_index is None

        output = src
        # preparation and reshape
        if self.num_encoder_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(
                    spatial_shapes, valid_ratios, device=src.device
                )

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(
                output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                if self.deformable_encoder:
                    output = layer(
                        hidden_states=output,
                        position_embeddings=pos,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        spatial_shapes_list=spatial_shapes_list,
                        level_start_index=level_start_index,
                        attention_mask=key_padding_mask,
                    )
                else:
                    output = layer(
                        src=output.transpose(0, 1),
                        pos=pos.transpose(0, 1),
                        key_padding_mask=key_padding_mask,
                    ).transpose(0, 1)

            output = output[0]

            if (
                (layer_id == 0 and self.two_stage_type in ["enceachlayer", "enclayer1"])
                or (self.two_stage_type == "enceachlayer")
            ) and (layer_id != self.num_encoder_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(
                    output, key_padding_mask, spatial_shapes
                )
                output_memory = self.enc_norm[layer_id](
                    self.enc_proj[layer_id](output_memory)
                )

                # gather boxes
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[
                    1
                ]  # bs, nq
                ref_token_coord = torch.gather(
                    output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4)
                )

                output = output_memory

            # aux loss
            if (
                layer_id != self.num_encoder_layers - 1
            ) and ref_token_index is not None:
                out_i = torch.gather(
                    output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model)
                )
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(
                intermediate_output
            )  # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class DinoDetrDecoder(nn.Module):
    def __init__(self, decoder_layer, norm, decoder_query_perturber, config):
        """
        decoder_layer,
        num_decoder_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        modulate_hw_attn=False,
        num_feature_levels=1,
        deformable_decoder=False,
        decoder_query_perturber=None,
        dec_layer_number=None,  # number of queries each layer in decoder
        rm_dec_query_scale=False,
        dec_layer_share=False,
        dec_layer_dropout_prob=None,
        use_detached_boxes_dec_out=False,
        """
        super().__init__()
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
        self.return_intermediate = config.return_intermediate
        assert config.return_intermediate, "support return_intermediate only"
        self.query_dim = config.query_dim
        assert config.query_dim in [2, 4], "query_dim should be 2/4 but {}".format(
            config.query_dim
        )
        self.num_feature_levels = config.num_feature_levels
        self.use_detached_boxes_dec_out = config.use_detached_boxes_dec_out

        self.ref_point_head = MLP(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        if not config.deformable_decoder:
            self.query_pos_sine_scale = MLP(
                config.d_model, config.d_model, config.d_model, 2
            )
        else:
            self.query_pos_sine_scale = None

        if config.rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(config.d_model, config.d_model, config.d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = config.d_model
        self.modulate_hw_attn = config.modulate_hw_attn
        self.deformable_decoder = config.deformable_decoder

        if not config.deformable_decoder and config.modulate_hw_attn:
            self.ref_anchor_head = MLP(config.d_model, config.d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = config.dec_layer_number
        if config.dec_layer_number is not None:
            assert isinstance(config.dec_layer_number, list)
            assert len(config.dec_layer_number) == config.num_decoder_layers

        self.dec_layer_dropout_prob = config.dec_layer_dropout_prob
        if config.dec_layer_dropout_prob is not None:
            assert isinstance(config.dec_layer_dropout_prob, list)
            assert len(config.dec_layer_dropout_prob) == config.num_decoder_layers
            for i in config.dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        spatial_shapes_list: Optional[List] = None,
        valid_ratios: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if (
                self.training
                and self.decoder_query_perturber is not None
                and layer_id != 0
            ):
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                    )  # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (
                        reference_points[:, :, None] * valid_ratios[None, :]
                    )
                query_sine_embed = gen_sineembed_for_position(
                    reference_points_input[:, :, 0, :]
                )  # nq, bs, 256*2
            else:
                query_sine_embed = gen_sineembed_for_position(
                    reference_points
                )  # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[
                    ..., : self.d_model
                ] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2 :] *= (
                    refHW_cond[..., 0] / reference_points[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.d_model // 2] *= (
                    refHW_cond[..., 1] / reference_points[..., 3]
                ).unsqueeze(-1)

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output = layer(
                    tgt=output,
                    tgt_query_pos=query_pos,
                    tgt_query_sine_embed=query_sine_embed,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=reference_points_input,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_spatial_shapes_list=spatial_shapes_list,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask,
                )

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # select # ref points
                if (
                    self.dec_layer_number is not None
                    and layer_id != self.num_decoder_layers - 1
                ):
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](
                            output
                        )  # nq, bs, 91
                        topk_proposals = torch.topk(
                            class_unselected.max(-1)[0], select_number, dim=0
                        )[
                            1
                        ]  # new_nq, bs
                        new_reference_points = torch.gather(
                            new_reference_points,
                            0,
                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
                        )  # unsigmoid

                if self.rm_detach and "dec" in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if (
                self.dec_layer_number is not None
                and layer_id != self.num_decoder_layers - 1
            ):
                if nq_now != select_number:
                    output = torch.gather(
                        output,
                        0,
                        topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
                    )  # unsigmoid

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DinoDeformableTransformer(nn.Module):
    def __init__(self, config):
        """
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        modulate_hw_attn=False,
        # for deformable encoder
        deformable_encoder=False,
        deformable_decoder=False,
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        # init query
        learnable_tgt_init=False,
        decoder_query_perturber=None,
        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        # evo of #anchors
        dec_layer_number=None,
        rm_enc_query_scale=True,
        rm_dec_query_scale=True,
        rm_self_attn_layers=None,
        key_aware_type=None,
        # layer share
        layer_share_type=None,
        # for detach
        rm_detach=None,
        decoder_sa_type="ca",
        module_seq=["sa", "ca", "ffn"],
        # for dn
        embed_init_tgt=False,
        use_detached_boxes_dec_out=False,
        """
        super().__init__()
        if config.decoder_layer_noise:
            self.decoder_query_perturber = RandomBoxPerturber(
                x_noise_scale=config.dln_xy_noise,
                y_noise_scale=config.dln_xy_noise,
                w_noise_scale=config.dln_hw_noise,
                h_noise_scale=config.dln_hw_noise,
            )
        else:
            self.decoder_query_perturber = None

        self.num_feature_levels = config.num_feature_levels
        self.num_encoder_layers = config.num_encoder_layers
        self.num_unicoder_layers = config.num_unicoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.deformable_encoder = config.deformable_encoder
        self.deformable_decoder = config.deformable_decoder
        self.two_stage_keep_all_tokens = config.two_stage_keep_all_tokens
        self.num_queries = config.num_queries
        self.random_refpoints_xy = config.random_refpoints_xy
        self.use_detached_boxes_dec_out = config.use_detached_boxes_dec_out
        assert config.query_dim == 4

        if config.num_feature_levels > 1:
            assert (
                config.deformable_encoder
            ), "only support deformable_encoder for num_feature_levels > 1"
        if config.use_deformable_box_attn:
            assert config.deformable_encoder or config.deformable_encoder

        assert config.layer_share_type in [None, "encoder", "decoder", "both"]
        if config.layer_share_type in ["encoder", "both"]:
            config.enc_layer_share = True
        else:
            config.enc_layer_share = False
        if config.layer_share_type in ["decoder", "both"]:
            config.dec_layer_share = True
        else:
            config.dec_layer_share = False
        assert config.layer_share_type is None

        self.decoder_sa_type = config.decoder_sa_type
        assert config.decoder_sa_type in ["sa", "ca_label", "ca_content"]

        # choose encoder layer type
        if config.deformable_encoder:
            encoder_layer = DinoDetrEncoderLayer(config)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(config.d_model) if config.normalize_before else None
        self.encoder = DinoDetrEncoder(encoder_layer, encoder_norm, config)

        # choose decoder layer type
        if config.deformable_decoder:
            decoder_layer = DinoDetrDecoderLayer(config)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(config.d_model)
        self.decoder = DinoDetrDecoder(
            decoder_layer=decoder_layer,
            norm=decoder_norm,
            decoder_query_perturber=self.decoder_query_perturber,
            config=config,
        )

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.dec_layers = config.num_decoder_layers
        self.num_queries = config.num_queries  # useful for single stage model only
        self.num_patterns = config.num_patterns
        if not isinstance(config.num_patterns, int):
            Warning(
                "num_patterns should be int but {}".format(type(config.num_patterns))
            )
            self.num_patterns = 0

        if config.num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(
                    torch.Tensor(config.num_feature_levels, config.d_model)
                )
            else:
                self.level_embed = None

        self.learnable_tgt_init = config.learnable_tgt_init
        assert config.learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = config.embed_init_tgt
        if (config.two_stage_type != "no" and config.embed_init_tgt) or (
            config.two_stage_type == "no"
        ):
            self.tgt_embed = nn.Embedding(self.num_queries, config.d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = config.two_stage_type
        self.two_stage_pat_embed = config.two_stage_pat_embed
        self.two_stage_add_query_num = config.two_stage_add_query_num
        self.two_stage_learn_wh = config.two_stage_learn_wh
        assert config.two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(config.two_stage_type)
        if config.two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)

            if config.two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(
                    torch.Tensor(config.two_stage_pat_embed, config.d_model)
                )
                nn.init.normal_(self.pat_embed_for_2stage)

            if config.two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(
                    self.two_stage_add_query_num, config.d_model
                )

            if config.two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if config.two_stage_type == "no":
            self.init_ref_points(config.num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = config.dec_layer_number
        if config.dec_layer_number is not None:
            if self.two_stage_type != "no" or config.num_patterns == 0:
                assert (
                    config.dec_layer_number[0] == config.num_queries
                ), f"dec_layer_number[0]({config.dec_layer_number[0]}) != num_queries({config.num_queries})"
            else:
                assert (
                    config.dec_layer_number[0]
                    == config.num_queries * config.num_patterns
                ), f"dec_layer_number[0]({config.dec_layer_number[0]}) != num_queries({config.num_queries}) * num_patterns({config.num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = config.rm_self_attn_layers
        if config.rm_self_attn_layers is not None:
            print(
                "Removing the self-attn in {} decoder layers".format(
                    config.rm_self_attn_layers
                )
            )
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in config.rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = config.rm_detach
        if self.rm_detach:
            assert isinstance(config.rm_detach, list)
            assert any([i in ["enc_ref", "enc_tgt", "dec"] for i in config.rm_detach])
        self.decoder.rm_detach = config.rm_detach

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
            nn.init.constant_(
                self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05))
            )

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)
        valid_W = torch.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes_list = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_list.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

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
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(
                    1, self.two_stage_pat_embed, 1
                )

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1
            )[
                1
            ]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(self.two_stage_type)
            )
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


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


class DinoDetrPreTrainedModel(PreTrainedModel):
    config_class = DinoDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        r"DinoDetrConvEncoder",
        r"DinoDetrEncoderLayer",
        r"DinoDetrDecoderLayer",
    ]

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, DinoDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, DinoDetrMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.num_heads, dtype=torch.int64).to(
                default_dtype
            ) * (2.0 * math.pi / module.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.num_heads, 1, 1, 2)
                .repeat(1, module.num_feature_levels, module.n_points, 1)
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


@add_start_docstrings(
    """
    The bare Dino DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    DINO_DETR_START_DOCSTRING,
)
class DinoDetrModel(DinoDetrPreTrainedModel):
    def __init__(self, config: DinoDetrConfig):
        """
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=1,
        num_heads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        two_stage_add_query_num=0,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        decoder_sa_type="sa",
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        """
        super().__init__(config)
        # create deformable transformer
        self.transformer = DinoDeformableTransformer(config=config)
        self.num_queries = config.num_queries
        self.num_classes = config.num_classes
        self.d_model = d_model = self.transformer.d_model
        self.num_feature_levels = config.num_feature_levels
        self.num_heads = config.num_heads
        self.label_enc = nn.Embedding(config.dn_labelbook_size + 1, config.d_model)

        # setting query dim
        self.query_dim = config.query_dim
        assert config.query_dim == 4
        self.random_refpoints_xy = config.random_refpoints_xy
        self.fix_refpoints_hw = config.fix_refpoints_hw

        # for dn training
        self.num_patterns = config.num_patterns
        self.dn_number = config.dn_number
        self.dn_box_noise_scale = config.dn_box_noise_scale
        self.dn_label_noise_ratio = config.dn_label_noise_ratio
        self.dn_labelbook_size = config.dn_labelbook_size

        # Create backbone + positional encoding
        backbone = DinoDetrConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = DinoDetrConvModel(backbone, position_embeddings)

        # prepare input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
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
            assert (
                config.two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            self.backbone.num_channels[-1], d_model, kernel_size=1
                        ),
                        nn.GroupNorm(32, d_model),
                    )
                ]
            )

        self.aux_loss = config.aux_loss
        self.box_pred_damping = None

        self.iter_update = config.iter_update
        assert config.iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = config.dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = config.dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(config.d_model, config.num_classes)
        _bbox_embed = MLP(d_model, d_model, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if config.dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for _ in range(self.transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for _ in range(self.transformer.num_decoder_layers)
            ]
        if config.dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for _ in range(self.transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for _ in range(self.transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = config.two_stage_type
        self.two_stage_add_query_num = config.two_stage_add_query_num
        assert config.two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(config.two_stage_type)
        if config.two_stage_type != "no":
            if config.two_stage_bbox_embed_share:
                assert (
                    config.dec_pred_class_embed_share
                    and config.dec_pred_bbox_embed_share
                )
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if config.two_stage_class_embed_share:
                assert (
                    config.dec_pred_class_embed_share
                    and config.dec_pred_bbox_embed_share
                )
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(config.two_stage_add_query_num)

        self.decoder_sa_type = config.decoder_sa_type
        assert config.decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if config.decoder_sa_type == "ca_label":
            self.label_embedding = nn.Embedding(config.num_classes, d_model)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, 2:]
            )
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print("learn a shared h and w")
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError(
                "Unknown fix_refpoints_hw {}".format(self.fix_refpoints_hw)
            )

    @add_start_docstrings_to_model_forward(DINO_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=DinoDetrModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(self, pixel_values, pixel_mask, targets: List = None):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DinoDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("dino-detr")
        >>> model = DinoDetrModel.from_pretrained("dino-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        """

        features, poss = self.backbone(pixel_values, pixel_mask)

        srcs = []
        masks = []
        for l, (src, mask) in enumerate(features):
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1][0])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = pixel_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone.position_embedding(src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(
                    targets,
                    self.dn_number,
                    self.dn_label_noise_ratio,
                    self.dn_box_noise_scale,
                ),
                training=self.training,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                d_model=self.d_model,
                label_enc=self.label_enc,
            )
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask
        )
        return DinoDetrModelOutput(
            hs, reference, hs_enc, ref_enc, init_box_proposal, dn_meta
        )


@add_start_docstrings(
    """
    Dino DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DINO_DETR_START_DOCSTRING,
)
class DinoDetrForObjectDetection(DinoDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"class_embed\.[1-9]\d*"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: DinoDetrConfig):
        super().__init__(config)

        # Dino DETR encoder-decoder model
        self.model = DinoDetrModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DINO_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=DinoDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(self, pixel_values, pixel_mask, targets: List = None):
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

        >>> image_processor = AutoImageProcessor.from_pretrained("dino-detr")
        >>> model = DinoDetrForObjectDetection.from_pretrained("dino-detr")

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

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(
                ((batch_size, height, width)), dtype=torch.long, device=device
            )

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, targets=targets
        )
        hs = outs.hs
        reference = outs.reference
        hs_enc = outs.hs_enc
        ref_enc = outs.ref_enc
        init_box_proposal = outs.init_box_proposal
        dn_meta = outs.dn_meta

        # In case num object=0
        hs[0] += self.model.label_enc.weight[0, 0] * 0.0
        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for _, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.model.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs)
                for layer_cls_embed, layer_hs in zip(self.model.class_embed, hs)
            ]
        )
        if self.model.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                self.model.aux_loss,
                self._set_aux_loss,
            )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        if self.model.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.model.transformer.enc_out_class_embed(hs_enc[-1])
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
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
                        hs_enc[:-1],
                        ref_enc[:-1],
                    )
                ):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = (
                        layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    )
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out["enc_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out["dn_meta"] = dn_meta

        loss, loss_dict, auxiliary_outputs = None, None, None
        if targets is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                out["pred_logits"],  #
                targets,
                self.device,
                out["pred_boxes"],  # out["pred_boxes"],
                self.config,
                outputs_class,
                outputs_coord_list,
            )

        out["loss"] = loss
        out["loss_dict"] = loss_dict
        dict_outputs = DinoDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            pred_logits=out["pred_logits"],
            pred_boxes=out["pred_boxes"],
            auxiliary_outputs=out["aux_outputs"],
        )

        return dict_outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


__all__ = [
    "DinoDetrForObjectDetection",
    "DinoDetrModel",
    "DinoDetrPreTrainedModel",
]
