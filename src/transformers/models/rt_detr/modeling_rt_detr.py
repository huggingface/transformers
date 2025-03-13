# coding=utf-8
# Copyright 2024 Baidu Inc and The HuggingFace Inc. team.
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
"""PyTorch RT-DETR model."""

import math
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
from ...image_transforms import center_to_corners_format, corners_to_center_format
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ...utils.backbone_utils import load_backbone
from .configuration_rt_detr import RTDetrConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RTDetrConfig"
_CHECKPOINT_FOR_DOC = "PekingU/rtdetr_r50vd"
_DETECTION_OUTPUT_FOR_DOC = """
    Detected 'sofa' (0.97) at [0.14, 0.38, 640.13, 476.21]
    Detected 'cat' (0.96) at [343.38, 24.28, 640.14, 371.5]
    Detected 'cat' (0.96) at [13.23, 54.18, 318.98, 472.22]
    Detected 'remote' (0.95) at [40.11, 73.44, 175.96, 118.48]
    Detected 'remote' (0.92) at [333.73, 76.58, 369.97, 186.99]
"""

# Global module placeholder for custom cuda kernel, will be loaded lazily in DeformableAttention
# in case of first usage (if no disabled in config).
MultiScaleDeformableAttention = None


@dataclass
class RTDetrDecoderOutput(ModelOutput):
    """
    Base class for outputs of the RTDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        class_outputs (`torch.Tensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
            Stacked class head outputs of each layer of the decoder.
        bbox_outputs (`torch.Tensor` of shape `(batch_size, config.decoder_layers, sequence_length, 4)`):
            Stacked bbox head outputs of each layer of the decoder.
        last_hidden_state (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The last hidden state of the decoder.
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.Tensor = None
    class_outputs: Optional[torch.Tensor] = None
    bbox_outputs: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    self_attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None


@dataclass
class RTDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the RT-DETR encoder-decoder model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        class_outputs (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
            Stacked class head outputs of each layer of the decoder.
        bbox_outputs (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked bbox head outputs of each layer of the decoder.
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
        enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the encoder stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`):
            Logits of predicted bounding boxes coordinates in the encoder stage.
        denoising_meta_values (`dict`):
            Extra dictionary for the denoising related values
    """

    last_hidden_state: torch.FloatTensor = None
    class_outputs: torch.FloatTensor = None
    bbox_outputs: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    enc_topk_logits: Optional[torch.FloatTensor] = None
    enc_topk_bboxes: Optional[torch.FloatTensor] = None
    denoising_meta_values: Optional[Dict] = None


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
            possible padding). You can use [`~RTDetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized (absolute) bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        class_outputs (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
            Stacked class head outputs of each layer of the decoder.
        bbox_outputs (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked bbox head outputs of each layer of the decoder.
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
        enc_topk_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the encoder.
        enc_topk_bboxes (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the encoder.
        denoising_meta_values (`dict`):
            Extra dictionary for the denoising related values
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: torch.FloatTensor = None
    class_outputs: torch.FloatTensor = None
    bbox_outputs: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    enc_topk_logits: Optional[torch.FloatTensor] = None
    enc_topk_bboxes: Optional[torch.FloatTensor] = None
    denoising_meta_values: Optional[Dict] = None


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.load_cuda_kernels
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


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    batched_num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    num_queries = batched_num_queries // batch_size

    sampling_grids = 2 * sampling_locations - 1
    value_levels = value.split([height * width for height, width in value_spatial_shapes], dim=1)

    sampled_values = []
    for idx, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height * width, num_heads, hidden_dim
        # -> batch_size, num_heads * hidden_dim, height * width
        # -> batch_size * num_heads, hidden_dim, height, width
        value_i = value_levels[idx]
        value_i = value_i.flatten(2).transpose(1, 2)
        value_i = value_i.reshape(batch_size * num_heads, hidden_dim, height, width)

        # batch_size * num_queries, num_heads, num_points, 2
        # -> batch_size * num_heads, num_queries, num_points, 2
        sampling_grid_i = sampling_grids[:, :, idx]
        sampling_grid_i = sampling_grid_i.view(batch_size, num_queries, num_heads, num_points, 2)
        sampling_grid_i = sampling_grid_i.transpose(1, 2).flatten(0, 1)

        # batch_size * num_heads, hidden_dim, num_queries, num_points
        sampled_value_i = nn.functional.grid_sample(
            value_i, sampling_grid_i, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampled_values.append(sampled_value_i)
    sampled_values = torch.stack(sampled_values, dim=-2)

    # (batch_size, num_queries, num_heads, ...) -> (batch_size, num_heads, num_queries, ...)
    attention_weights = attention_weights.transpose(1, 2)
    attention_weights = attention_weights.reshape(batch_size * num_heads, 1, num_queries, num_levels * num_points)

    output = attention_weights * sampled_values.flatten(-2)
    output = output.sum(-1)

    output = output.view(batch_size, num_heads * hidden_dim, num_queries)
    output = output.transpose(1, 2).contiguous()

    return output


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->RTDetr
class RTDetrMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: RTDetrConfig, num_heads: int, num_points: int):
        super().__init__()

        # Doing basic sanity checks
        if config.hidden_size % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.hidden_size} and {num_heads}"
            )

        # check if head_dim is power of 2
        head_dim = config.hidden_size // num_heads
        if not ((head_dim & (head_dim - 1) == 0) and head_dim != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in RTDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64
        self.hidden_size = config.hidden_size
        self.num_levels = config.num_feature_levels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_points = num_points
        self.disable_custom_kernels = config.disable_custom_kernels

        # Load custom deformable attention kernel if needed,
        # otherwise use the pytorch implementation would be used
        if (
            not config.disable_custom_kernels  # manually disabled in config
            and is_torch_cuda_available()  # only CUDA is supported for now
            and is_ninja_available()  # we need Ninja to compile the kernel
            and MultiScaleDeformableAttention is not None  # only if the kernel is previously loaded
        ):
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        self.sampling_offsets = nn.Linear(self.hidden_size, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(self.hidden_size, self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

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
            value = value.masked_fill(~attention_mask[..., None], 0.0)

        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        sampling_offsets = self.sampling_offsets(hidden_states)
        sampling_offsets = sampling_offsets.view(
            batch_size * num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        batch_size, num_reference_points, _, num_coordinates = reference_points.shape
        reference_points = reference_points.view(batch_size * num_reference_points, 1, -1, 1, num_coordinates)

        if num_coordinates == 2:
            height, width = spatial_shapes[..., 0], spatial_shapes[..., 1]
            offset_normalizer = torch.stack([width, height], -1)
            normalized_sampling_offsets = sampling_offsets / offset_normalizer[None, None, :, None, :]
            sampling_locations = reference_points + normalized_sampling_offsets

        elif num_coordinates == 4:
            reference_points_xy = reference_points[..., :2]
            offset = sampling_offsets / self.num_points * reference_points[..., 2:] * 0.5
            sampling_locations = reference_points_xy + offset

        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_coordinates}")

        if (
            self.disable_custom_kernels  # manually disabled in config
            or MultiScaleDeformableAttention is None  # error while loading the kernel
            or is_torchdynamo_compiling()  # torch.compile / torch.export mode
        ):
            # PyTorch implementation
            output = multi_scale_deformable_attention(
                value, spatial_shapes_list, sampling_locations, attention_weights
            )
        else:
            try:
                # Calling custom kernel
                # Note: for custom kernel we pass sampling locations as 6D tensor,
                #       but for torch implementation we keep it as 5D tensor (for CoreML compat)
                kernel_sampling_locations = sampling_locations.view(
                    batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
                )
                level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    kernel_sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention(
                    value, spatial_shapes_list, sampling_locations, attention_weights
                )
        output = self.output_proj(output)

        if not output_attentions:
            attention_weights = None

        return output, attention_weights


# Copied from transformers.models.conditional_detr.modeling_conditional_detr.inverse_sigmoid
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->RTDetr
class RTDetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape for broadcasting
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        running_var = self.running_var.view(1, -1, 1, 1)
        running_mean = self.running_mean.view(1, -1, 1, 1)
        # compute batchnorm
        scale = weight * (running_var + self.eps).rsqrt()
        return (x - running_mean) * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->RTDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `RTDetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = RTDetrFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising_queries=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """
    Creates a contrastive denoising training group using ground-truth samples. It adds noise to labels and boxes.

    Args:
        targets (`List[dict]`):
            The target objects, each containing 'class_labels' and 'boxes' for objects in an image.
        num_classes (`int`):
            Total number of classes in the dataset.
        num_queries (`int`):
            Number of query slots in the transformer.
        class_embed (`callable`):
            A function or a model layer to embed class labels.
        num_denoising_queries (`int`, *optional*, defaults to 100):
            Number of denoising queries.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            Ratio of noise applied to labels.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale of noise applied to bounding boxes.
    Returns:
        `tuple` comprising various elements:
        - **input_query_class** (`torch.FloatTensor`) --
          Class queries with applied label noise.
        - **input_query_bbox** (`torch.FloatTensor`) --
          Bounding box queries with applied box noise.
        - **attn_mask** (`torch.FloatTensor`) --
           Attention mask for separating denoising and reconstruction queries.
        - **denoising_meta_values** (`dict`) --
          Metadata including denoising positive indices, number of groups, and split sizes.
    """

    if num_denoising_queries <= 0:
        return None, None, None, None

    num_ground_truths = [len(t["class_labels"]) for t in targets]
    device = targets[0]["class_labels"].device

    max_gt_num = max(num_ground_truths)
    if max_gt_num == 0:
        return None, None, None, None

    num_groups_denoising_queries = num_denoising_queries // max_gt_num
    num_groups_denoising_queries = 1 if num_groups_denoising_queries == 0 else num_groups_denoising_queries
    # pad gt to max_num of a batch
    batch_size = len(num_ground_truths)

    input_query_class = torch.full([batch_size, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([batch_size, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool, device=device)

    for i in range(batch_size):
        num_gt = num_ground_truths[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["class_labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_groups_denoising_queries])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_groups_denoising_queries, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_groups_denoising_queries])
    # positive and negative mask
    negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_groups_denoising_queries, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    denoise_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    denoise_positive_idx = torch.split(
        denoise_positive_idx, [n * num_groups_denoising_queries for n in num_ground_truths]
    )
    # total denoising queries
    num_denoising_queries = torch_int(max_gt_num * 2 * num_groups_denoising_queries)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = center_to_corners_format(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = corners_to_center_format(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    target_size = num_denoising_queries + num_queries
    attn_mask = torch.full([target_size, target_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising_queries:, :num_denoising_queries] = True

    # reconstructions cannot see each other
    for i in range(num_groups_denoising_queries):
        idx_block_start = max_gt_num * 2 * i
        idx_block_end = max_gt_num * 2 * (i + 1)
        attn_mask[idx_block_start:idx_block_end, :idx_block_start] = True
        attn_mask[idx_block_start:idx_block_end, idx_block_end:num_denoising_queries] = True

    denoising_meta_values = {
        "dn_positive_idx": denoise_positive_idx,
        "dn_num_group": num_groups_denoising_queries,
        "dn_num_split": [num_denoising_queries, num_queries],
    }

    return input_query_class, input_query_bbox, attn_mask, denoising_meta_values


class RTDetrConvEncoder(nn.Module):
    """
    Convolutional backbone using the modeling_rt_detr_resnet.py.

    nn.BatchNorm2d layers are replaced by RTDetrFrozenBatchNorm2d as defined above.
    https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/nn/backbone/presnet.py#L142
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        self.model = load_backbone(config)
        self.intermediate_channel_sizes = self.model.channels

        # replace batch norm by frozen batch norm
        if config.freeze_backbone_batch_norms:
            replace_batch_norm(self.model)

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        feature_maps = self.model(pixel_values).feature_maps
        return feature_maps


class RTDetrConvNormLayer(nn.Module):
    def __init__(
        self,
        config: RTDetrConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Feature map of shape (batch_size, embed_dim, height, width)"""
        feature_map = self.conv(feature_map)
        feature_map = self.norm(feature_map)
        feature_map = self.activation(feature_map)
        return feature_map


class RTDetrRepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDetrConvNormLayer(config, hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = RTDetrConvNormLayer(config, hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Feature map of shape (batch_size, embed_dim, height, width)"""
        feature_map = self.conv1(feature_map) + self.conv2(feature_map)
        return self.activation(feature_map)


class RTDetrCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        hidden_channels = int(out_channels * config.hidden_expansion)
        params = {"kernel_size": 1, "stride": 1, "activation": config.activation_function}

        # branch 1
        self.conv1 = RTDetrConvNormLayer(config, in_channels, hidden_channels, **params)
        self.bottlenecks = nn.Sequential(*[RTDetrRepVggBlock(config) for _ in range(3)])

        # branch 2
        self.conv2 = RTDetrConvNormLayer(config, in_channels, hidden_channels, **params)

        # fuse step
        if hidden_channels != out_channels:
            self.conv3 = RTDetrConvNormLayer(config, hidden_channels, out_channels, **params)
        else:
            self.conv3 = nn.Identity()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Feature map of shape (batch_size, embed_dim, height, width)"""

        # branch 1
        feature_map_1 = self.conv1(feature_map)
        feature_map_1 = self.bottlenecks(feature_map_1)

        # branch 2
        feature_map_2 = self.conv2(feature_map)

        # fuse step
        feature_map = self.conv3(feature_map_1 + feature_map_2)

        return feature_map


class RTDetrMultiheadAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the Deformable DETR paper).
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
        self.scaling = self.head_dim**-0.5

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _reshape_for_multihead_attention(self, tensor: torch.Tensor):
        """Reshape tensor (batch_size, seq_length, embed_dim) to (batch_size, num_heads, seq_length, head_dim)"""
        batch_size, seq_length, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_length, self.num_heads, self.head_dim)
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, embed_dim = hidden_states.shape
        query_states, key_states, value_states = hidden_states, hidden_states, hidden_states

        if position_embeddings is not None:
            query_states = query_states + position_embeddings
            key_states = query_states  # avoid recomputing, they are the same

        # Apply projection and scaling
        query_states = self.q_proj(query_states) * self.scaling
        key_states = self.k_proj(key_states)
        value_states = self.v_proj(value_states)

        # (batch_size, seq_length, embed_dim) -> (batch_size, num_heads, seq_length, head_dim)
        query_states = self._reshape_for_multihead_attention(query_states)
        key_states = self._reshape_for_multihead_attention(key_states)
        value_states = self._reshape_for_multihead_attention(value_states)

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_length, seq_length)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)

        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class RTDetrEncoderLayer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.normalize_before = config.normalize_before

        # self-attention
        self.self_attn = RTDetrMultiheadAttention(
            embed_dim=config.encoder_hidden_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_dropout = nn.Dropout(config.dropout)

        # feed-forward
        self.fc1 = nn.Linear(config.encoder_hidden_dim, config.encoder_ffn_dim)
        self.fc1_activation = ACT2FN[config.encoder_activation_function]
        self.fc1_dropout = nn.Dropout(config.activation_dropout)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.encoder_hidden_dim)
        self.fc2_dropout = nn.Dropout(config.dropout)

        # norms
        self.identity = nn.Identity()
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.Tensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            position_embeddings (`torch.Tensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        if self.normalize_before:
            norm1 = self.self_attn_layer_norm
            norm2 = self.final_layer_norm
            norm3 = self.identity
        else:
            norm1 = self.identity
            norm2 = self.self_attn_layer_norm
            norm3 = self.final_layer_norm

        # self-attention step
        residual = hidden_states
        hidden_states = norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = norm2(hidden_states)

        # feed-forward step
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc1_activation(hidden_states)
        hidden_states = self.fc1_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.fc2_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = norm3(hidden_states)

        # clamp values if training
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


class RTDetrEncoder(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.layers = nn.ModuleList([RTDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
        return hidden_states


class RTDetrHybridEncoder(nn.Module):
    """
    Decoder consisting of a projection layer, a set of `RTDetrEncoder`, a top-down Feature Pyramid Network
    (FPN) and a bottom-up Path Aggregation Network (PAN). More details on the paper: https://arxiv.org/abs/2304.08069

    Args:
        config: RTDetrConfig
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1
        activation = config.activation_function

        # encoder transformer
        self.encoder = nn.ModuleList([RTDetrEncoder(config) for _ in self.encode_proj_layers])

        # top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = RTDetrConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=activation,
            )
            fpn_block = RTDetrCSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_pan_stages):
            downsample_conv = RTDetrConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=activation,
            )
            pan_block = RTDetrCSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

    @compile_compatible_method_lru_cache(maxsize=32)
    def build_2d_sincos_position_embedding(
        self, width, height, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ):
        grid_w = torch.arange(torch_int(width), device=device).to(dtype)
        grid_h = torch.arange(torch_int(height), device=device).to(dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        r"""
        Apply the transformer encoder to the feature maps. Then apply the FPN and PAN to the feature maps.

        Args:
            feature_maps (`List[torch.FloatTensor]` of shape `(batch_size, embed_dim, height, width)`):
                List of feature maps from different stages of the backbone. For example, for RT-DETR-R50
                `[torch.Size([1, 256, 80, 80]), torch.Size([1, 256, 40, 40]), torch.Size([1, 256, 20, 20])]`.
            output_hidden_states (`bool`, default `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            output_attentions (`bool`, default `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        encoder_intermediates = () if output_hidden_states else None
        encoder_attentions = () if output_attentions else None

        # Step 1: Apply transformer encoder
        for i, feature_map_idx in enumerate(self.encode_proj_layers):
            feature_map = feature_maps[feature_map_idx]

            if output_hidden_states:
                encoder_intermediates = encoder_intermediates + (feature_map,)

            # 1. flatten [batch, channel, height, width] to [batch, height * width, channel]
            height, width = feature_map.shape[2:]
            hidden_state = feature_map.flatten(2).permute(0, 2, 1)

            # build position embeddings
            position_embeddings = self.build_2d_sincos_position_embedding(
                width,
                height,
                self.encoder_hidden_dim,
                self.positional_encoding_temperature,
                device=hidden_state.device,
                dtype=hidden_state.dtype,
            )

            # 2. Apply transformer encoder layer
            layer_outputs = self.encoder[i](
                hidden_state,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
            hidden_state = layer_outputs[0]

            # 3. Reshape back to [batch, height, width, channel]
            hidden_state = hidden_state.permute(0, 2, 1)
            feature_map = hidden_state.reshape(-1, self.encoder_hidden_dim, height, width)
            feature_maps[feature_map_idx] = feature_map.contiguous()

            if output_attentions:
                encoder_attentions += (layer_outputs[1],)

        if output_hidden_states:
            encoder_intermediates += (feature_maps[feature_map_idx],)

        # Step 2: Apply FPN (conv part)
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            # apply lateral block
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            # apply fpn block
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps = fpn_feature_maps[::-1]

        # Step 3: Apply PAN (conv part)
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return pan_feature_maps, encoder_intermediates, encoder_attentions


class RTDetrDecoderLayer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

        # Self-Attention
        self.self_attn = RTDetrMultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Cross-Attention
        self.encoder_attn = RTDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            num_points=config.decoder_n_points,
        )
        self.encoder_attn_dropout = nn.Dropout(config.dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Feedforward
        self.fc1 = nn.Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc1_activation = ACT2FN[config.decoder_activation_function]
        self.fc1_dropout = nn.Dropout(config.activation_dropout)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.hidden_size)
        self.fc2_dropout = nn.Dropout(config.dropout)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_state: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        position_embeddings: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_state (`torch.Tensor`):
                Input to the layer of shape `(batch_size, num_queries, hidden_size)`.
            encoder_hidden_states (`torch.Tensor`):
                Flattened and concatenated encoder feature maps of shape `(batch_size, sequence_length, hidden_size)`.
                Where `sequence_length` is the sum of the product of all spatial shapes of the encoder feature maps.
                For example, spatial_shapes = [(80, 80), (40, 40), (20, 20)] ->
                sequence_length = 80 * 80 + 40 * 40 + 20 * 20 = 6400 + 1600 + 400 = 8400.
            reference_points (`torch.Tensor`, *optional*):
                Reference points of shape `(batch_size, num_queries, 1, 2)`.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of encoder feature maps of shape `(num_feature_levels, 2)`.
            spatial_shapes_list (`List[Tuple[int, int]]`, *optional*):
                Spatial shapes of encoder feature maps of shape `(num_feature_levels, 2)`.
                Same as `spatial_shapes` but as a list of tuples for `torch.compile` compatibility.
            position_embeddings (`torch.Tensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer of
                shape `(batch_size, num_queries, hidden_size)`.
            encoder_attention_mask (`torch.Tensor`):
                4d encoder attention mask of size `(batch, 1, target_len, source_len)` where padding elements
                are indicated by very large negative values.
            output_attentions (`bool`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # Self-Attention
        self_attn_residual = hidden_state
        hidden_state, self_attn_weights = self.self_attn(
            hidden_states=hidden_state,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_state = self.self_attn_dropout(hidden_state)
        hidden_state = self_attn_residual + hidden_state
        hidden_state = self.self_attn_layer_norm(hidden_state)

        # Cross-Attention
        cross_attn_residual = hidden_state
        hidden_state, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            output_attentions=output_attentions,
        )
        hidden_state = self.encoder_attn_dropout(hidden_state)
        hidden_state = cross_attn_residual + hidden_state
        hidden_state = self.encoder_attn_layer_norm(hidden_state)

        # Feedforward
        ffn_residual = hidden_state
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.fc1_activation(hidden_state)
        hidden_state = self.fc1_dropout(hidden_state)
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.fc2_dropout(hidden_state)
        hidden_state = ffn_residual + hidden_state
        hidden_state = self.final_layer_norm(hidden_state)

        if not output_attentions:
            self_attn_weights, cross_attn_weights = None, None

        return hidden_state, self_attn_weights, cross_attn_weights


class RTDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        input_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        output_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(input_dims, output_dims))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_state = layer(hidden_state)
            if i < self.num_layers - 1:
                hidden_state = nn.functional.relu(hidden_state)
        return hidden_state


class RTDetrDecoder(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        self.dropout = config.dropout
        self.layers = nn.ModuleList([RTDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = RTDetrMLPPredictionHead(
            input_dim=4, hidden_dim=2 * config.hidden_size, output_dim=config.hidden_size, num_layers=2
        )

        # hack implementation for iterative bounding box refinement and two-stage RT-DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        hidden_state: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        encoder_attention_mask: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        r"""
        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            output_attentions (`bool`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        all_bbox_outputs = () if self.bbox_embed is not None else None
        all_class_outputs = () if self.class_embed is not None else None

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            position_embeddings = self.query_pos_head(reference_points)

            if output_hidden_states:
                all_hidden_states += (hidden_state,)

            hidden_state, self_attn_weights, cross_attn_weights = decoder_layer(
                hidden_state,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            if output_attentions and self_attn_weights is not None:
                all_self_attentions += (self_attn_weights,)

            if output_attentions and cross_attn_weights is not None:
                all_cross_attentions += (cross_attn_weights,)

            # Iterative bounding box and class refinement
            if self.bbox_embed is not None:
                box_refinement = self.bbox_embed[idx](hidden_state)
                reference_points = F.sigmoid(inverse_sigmoid(reference_points) + box_refinement)
                all_bbox_outputs += (reference_points,)
                reference_points = reference_points.detach()

            if self.class_embed is not None:
                class_output = self.class_embed[idx](hidden_state)
                all_class_outputs += (class_output,)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_state,)

        if self.bbox_embed is not None:
            all_bbox_outputs = torch.stack(all_bbox_outputs, dim=1)

        if self.class_embed is not None:
            all_class_outputs = torch.stack(all_class_outputs, dim=1)

        outputs = RTDetrDecoderOutput(
            last_hidden_state=hidden_state,
            class_outputs=all_class_outputs,
            bbox_outputs=all_bbox_outputs,
            hidden_states=all_hidden_states,
            self_attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

        if not return_dict:
            outputs = outputs.to_tuple()
        return outputs


class RTDetrPreTrainedModel(PreTrainedModel):
    config_class = RTDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"RTDetrHybridEncoder", r"RTDetrDecoderLayer"]

    def _init_weights(self, module):
        """Initalize the weights"""

        # initialize linear layer bias value according to a given probability value
        if isinstance(module, (RTDetrForObjectDetection, RTDetrDecoder)):
            if module.class_embed is not None:
                for layer in module.class_embed:
                    prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
                    bias = float(-math.log((1 - prior_prob) / prior_prob))
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, bias)

            if module.bbox_embed is not None:
                for layer in module.bbox_embed:
                    nn.init.constant_(layer.layers[-1].weight, 0)
                    nn.init.constant_(layer.layers[-1].bias, 0)

        if isinstance(module, RTDetrMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.num_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.num_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.num_heads, 1, 1, 2)
                .repeat(1, module.num_levels, module.num_points, 1)
            )
            for i in range(module.num_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        if isinstance(module, RTDetrModel):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            nn.init.xavier_uniform_(module.enc_score_head.weight)
            nn.init.constant_(module.enc_score_head.bias, bias)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        if hasattr(module, "weight_embedding") and self.config.learn_initial_query:
            nn.init.xavier_uniform_(module.weight_embedding.weight)
        if hasattr(module, "denoising_class_embed") and self.config.num_denoising > 0:
            nn.init.xavier_uniform_(module.denoising_class_embed.weight)


RTDETR_START_DOCSTRING = r"""
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


RTDETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`RTDetrImageProcessor.__call__`] for details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
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
    """
    RT-DETR Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.
    """,
    RTDETR_START_DOCSTRING,
)
class RTDetrModel(RTDetrPreTrainedModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self.learn_initial_query = config.learn_initial_query

        # Create backbone
        self.backbone = RTDetrConvEncoder(config)

        # Create encoder input projection layers
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes
        num_backbone_outs = len(intermediate_channel_sizes)
        self.encoder_input_proj = nn.ModuleList()
        for idx in range(num_backbone_outs):
            in_channels = intermediate_channel_sizes[idx]
            conv = nn.Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False)
            batchnorm = nn.BatchNorm2d(config.encoder_hidden_dim)
            self.encoder_input_proj.append(nn.Sequential(conv, batchnorm))

        # Create encoder
        self.encoder = RTDetrHybridEncoder(config)

        # De-noising part
        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                config.num_labels + 1, config.hidden_size, padding_idx=config.num_labels
            )

        # Decoder embedding
        if self.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.hidden_size)

        # Encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.hidden_size, config.num_labels)
        self.enc_bbox_head = RTDetrMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
        )

        # Init encoder output anchors and valid_mask
        if config.anchor_image_size:
            self.anchors, self.valid_mask = self.generate_anchors(dtype=self.dtype)

        # Create decoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        num_backbone_outs = len(config.decoder_in_channels)
        self.decoder_input_proj = nn.ModuleList()

        for idx in range(num_backbone_outs):
            in_channels = config.decoder_in_channels[idx]
            conv = nn.Conv2d(in_channels, config.hidden_size, kernel_size=1, bias=False)
            batchnorm = nn.BatchNorm2d(config.hidden_size, eps=config.batch_norm_eps)
            self.decoder_input_proj.append(nn.Sequential(conv, batchnorm))

        for _ in range(config.num_feature_levels - num_backbone_outs):
            conv = nn.Conv2d(in_channels, config.hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
            batchnorm = nn.BatchNorm2d(config.hidden_size, eps=config.batch_norm_eps)
            self.decoder_input_proj.append(nn.Sequential(conv, batchnorm))
            in_channels = config.hidden_size

        # Decoder
        self.decoder = RTDetrDecoder(config)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    @compile_compatible_method_lru_cache(maxsize=32)
    def generate_anchors(
        self,
        spatial_shapes: Optional[Tuple[Tuple[int, int], ...]] = None,
        grid_size: float = 0.05,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if spatial_shapes is None:
            height, width = self.config.anchor_image_size
            spatial_shapes = tuple(
                [(int(height / stride), int(width / stride)) for stride in self.config.feat_strides]
            )

        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            # Generate normalized grid coordinates from the center of the first pixel
            # to the center of the last pixel (e.g. 0.5 / width, 1 - 0.5 / width)
            grid_x = torch.linspace(0.5 / width, 1 - 0.5 / width, width, device=device, dtype=dtype)
            grid_y = torch.linspace(0.5 / height, 1 - 0.5 / height, height, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)

            grid_wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            level_anchors = torch.concat([grid_xy, grid_wh], dim=-1).reshape(height * width, 4)
            anchors.append(level_anchors)
        anchors = torch.concat(anchors).unsqueeze(0)

        # define the valid range for anchor coordinates
        eps = 1e-2
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))

        return anchors, valid_mask

    @add_start_docstrings_to_model_forward(RTDETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RTDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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

        >>> image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        >>> model = RTDetrModel.from_pretrained("PekingU/rtdetr_r50vd")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device

        # Stage 1: Convolutional backbone
        backbone_outputs = self.backbone(pixel_values)

        # Stage 2: Hybrid encoder (transformer -> FPN -> PAN)
        projected_feature_maps = [
            layer(feature_map) for layer, feature_map in zip(self.encoder_input_proj, backbone_outputs)
        ]
        encoder_outputs = self.encoder(
            projected_feature_maps,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        feature_maps = encoder_outputs[0]
        last_encoder_feature_map = feature_maps[-1]

        # Apply projection to each feature map
        feature_maps = [proj(feature_map) for proj, feature_map in zip(self.decoder_input_proj, feature_maps)]
        for i in range(len(feature_maps), self.config.num_feature_levels):
            feature_maps.append(self.decoder_input_proj[i](last_encoder_feature_map))

        # Flatten decoder inputs
        hidden_states = [feature_map.flatten(2).transpose(1, 2) for feature_map in feature_maps]
        hidden_states = torch.cat(hidden_states, dim=1)

        # Prepare spatial shapes, we have to keep both: the list and the tensor
        # for torch.compile fullgraph=True and ONNX dynamic export
        spatial_shapes_list = [(feat.shape[-2], feat.shape[-1]) for feat in feature_maps]
        spatial_shapes = torch.empty((len(feature_maps), 2), device=device, dtype=torch.long)
        for level, source in enumerate(feature_maps):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width

        # prepare denoising training
        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        batch_size = len(hidden_states)
        device = hidden_states.device
        dtype = hidden_states.dtype

        # prepare input for decoder
        if self.training or self.config.anchor_image_size is None:
            # Pass spatial_shapes as tuple to make it hashable and make sure
            # lru_cache is working for generate_anchors()
            spatial_shapes_tuple = tuple(spatial_shapes_list)
            anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, device=device, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            anchors, valid_mask = anchors.to(device, dtype), valid_mask.to(device, dtype)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        masked_hidden_states = hidden_states * valid_mask.to(hidden_states.dtype)
        proj_hidden_states = self.enc_output(masked_hidden_states)

        class_logits = self.enc_score_head(proj_hidden_states)
        bboxes_logits = self.enc_bbox_head(proj_hidden_states) + anchors

        # gather encoder logits and boxes based on top K confident logits
        _, topk_ind = torch.topk(class_logits.max(-1).values, self.config.num_queries, dim=1)
        topk_ind = topk_ind.unsqueeze(-1)

        topk_logits = class_logits.gather(dim=1, index=topk_ind.repeat(1, 1, class_logits.shape[-1]))
        topk_bboxes_logits = bboxes_logits.gather(dim=1, index=topk_ind.repeat(1, 1, bboxes_logits.shape[-1]))
        if denoising_bbox_unact is not None:
            topk_bboxes_logits = torch.concat([denoising_bbox_unact, topk_bboxes_logits], 1)

        # extract region features
        if self.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = proj_hidden_states.gather(dim=1, index=topk_ind.repeat(1, 1, proj_hidden_states.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        topk_bboxes = F.sigmoid(topk_bboxes_logits)

        # decoder
        decoder_outputs: RTDetrDecoderOutput = self.decoder(
            hidden_state=target,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            reference_points=topk_bboxes,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        outputs = RTDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            class_outputs=decoder_outputs.class_outputs,  # + for loss and output logits
            bbox_outputs=decoder_outputs.bbox_outputs,  # + for loss and output boxes
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.self_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs[0],
            encoder_hidden_states=encoder_outputs[1],
            encoder_attentions=encoder_outputs[2],
            enc_topk_logits=topk_logits,  # + for loss
            enc_topk_bboxes=topk_bboxes,  # + for loss
            denoising_meta_values=denoising_meta_values,  # + for loss
        )

        if not return_dict:
            outputs = outputs.to_tuple()
        return outputs


@add_start_docstrings(
    """
    RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further
    decoded into scores and classes.
    """,
    RTDETR_START_DOCSTRING,
)
class RTDetrForObjectDetection(RTDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = ["bbox_embed", "class_embed"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        # RTDETR encoder-decoder model
        self.model = RTDetrModel(config)

        # Detection heads on top
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_layers = config.decoder_layers
        self.class_embed = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for _ in range(num_layers)])
        self.bbox_embed = nn.ModuleList(
            [
                RTDetrMLPPredictionHead(
                    input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
                )
                for _ in range(num_layers)
            ]
        )

        # hack implementation for iterative bounding box refinement
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RTDETR_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RTDetrObjectDetectionOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_DETECTION_OUTPUT_FOR_DOC,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **loss_kwargs,
    ) -> Union[Tuple[torch.FloatTensor], RTDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: RTDetrModelOutput = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        denoising_meta_values = outputs.denoising_meta_values if self.training else None
        class_outputs = outputs.class_outputs
        bbox_outputs = outputs.bbox_outputs

        logits = class_outputs[:, -1]
        pred_boxes = bbox_outputs[:, -1]

        # Conpute loss if labels are provided
        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            enc_topk_logits = outputs.enc_topk_logits
            enc_topk_bboxes = outputs.enc_topk_bboxes
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits=logits,
                labels=labels,
                pred_boxes=pred_boxes,
                config=self.config,
                outputs_class=class_outputs,
                outputs_coord=bbox_outputs,
                enc_topk_logits=enc_topk_logits,
                enc_topk_bboxes=enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values,
                device=self.device,
                **loss_kwargs,
            )

        detection_outputs = RTDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            **outputs,
        )

        if not return_dict:
            detection_outputs = detection_outputs.to_tuple()
        return detection_outputs


__all__ = [
    "RTDetrForObjectDetection",
    "RTDetrModel",
    "RTDetrPreTrainedModel",
]
