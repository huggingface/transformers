# coding=utf-8
# Copyright 2024 IDEA Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch OmDet-Turbo model."""

import math
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import xavier_uniform_

from ...activations import ACT2CLS, ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_timm_available,
    is_torch_cuda_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_omdet_turbo import OmDetTurboConfig


MultiScaleDeformableAttention = None

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "OmDetTurboConfig"


if is_timm_available():
    pass


# Copied from models.deformable_detr.load_cuda_kernels
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


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
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
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class OmDetTurboLRUCache:
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def has(self, key) -> bool:
        return key in self.cache

    # we return the value of the key
    # that is queried in O(1) and return None if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class OmDetTurboLanguageBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.model = AutoModel.from_config(config.text_config, attn_implementation=config._attn_implementation)
        self.text_projection = nn.Parameter(
            torch.empty(config.text_projection_in_features, config.text_projection_out_features)
        )

    def forward(self, hidden_states, mask=None, encode_type="task"):
        text_outputs = self.model(hidden_states)
        pooled_output = text_outputs[0]
        if encode_type == "task":
            if mask is None:
                raise ValueError("mask is required for task encoding")
            max_len = (mask != 0).sum(1).max().item()
            truncated_mask = mask[:, :max_len]
            truncated_output = pooled_output[:, :max_len, :]
            return truncated_output.transpose(0, 1), truncated_mask
        else:
            out = (
                pooled_output[torch.arange(pooled_output.shape[0]), hidden_states.argmax(dim=-1)]
                @ self.text_projection
            )
            return out


class OmDetTurboVisionBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.use_timm_backbone = config.use_timm_backbone
        self.num_features = [int(config.backbone_embed_dim * 2**i) for i in range(len(config.backbone_depths))]
        self.out_indices = config.backbone_out_indices
        if self.use_timm_backbone:
            requires_backends(self, ["timm"])
            vision_backbone = load_backbone(config)._backbone

            if "swin" in config.backbone:
                OmDetTurboVisionBackbone._force_partition_timm_backbone(config, vision_backbone)
        else:
            vision_backbone = load_backbone(config.vision_config)

        self.vision_backbone = vision_backbone
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.num_features[i_layer]) for i_layer in self.out_indices])

    # temporary while timm doesn't support always partition for swin
    @staticmethod
    def _window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (torch.Tensor): input tokens with [batch_size, height, width, num_channels].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, num_channels].
            (Hp, Wp): padded height and width before partition
        """
        batch_size, height, width, num_channels = x.shape
        x = x.view(
            batch_size, height // window_size[0], window_size[0], width // window_size[1], window_size[1], num_channels
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], num_channels)
        return windows

    # temporary while timm doesn't support always partition for swin
    @staticmethod
    def _compute_attention_mask(block_to_fix):
        H, W = block_to_fix.input_resolution
        H = math.ceil(H / block_to_fix.window_size[0]) * block_to_fix.window_size[0]
        W = math.ceil(W / block_to_fix.window_size[1]) * block_to_fix.window_size[1]
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        cnt = 0
        for h in (
            slice(0, -block_to_fix.window_size[0]),
            slice(-block_to_fix.window_size[0], -block_to_fix.shift_size[0]),
            slice(-block_to_fix.shift_size[0], None),
        ):
            for w in (
                slice(0, -block_to_fix.window_size[1]),
                slice(-block_to_fix.window_size[1], -block_to_fix.shift_size[1]),
                slice(-block_to_fix.shift_size[1], None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = OmDetTurboVisionBackbone._window_partition(
            img_mask, block_to_fix.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, block_to_fix.window_area)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    # temporary while timm doesn't support always partition for swin
    @staticmethod
    def _force_partition_timm_backbone(config, model):
        target_shift_size = tuple([config.backbone_window_size // 2] * 2)
        for layer, depth in enumerate(config.backbone_depths):
            for block_num in range(depth):
                if block_num % 2 != 0:
                    block = getattr(model, f"layers_{layer}").blocks[block_num]
                    if block.shift_size != target_shift_size:
                        block.shift_size = target_shift_size
                        attention_mask = OmDetTurboVisionBackbone._compute_attention_mask(block)
                        block.register_buffer("attn_mask", attention_mask, persistent=False)

    def forward(self, pixel_values):
        outputs_before_norm = self.vision_backbone(pixel_values)
        if not self.use_timm_backbone:
            hidden_states = outputs_before_norm.hidden_states
            outputs_before_norm = [
                hidden_states[i + 1].view(
                    1,
                    int(hidden_states[i + 1].shape[1] ** (1 / 2)),
                    int(hidden_states[i + 1].shape[1] ** (1 / 2)),
                    -1,
                )
                for i in self.out_indices
            ]

        out = [
            layer_norm(output).permute(0, 3, 1, 2).contiguous()
            for layer_norm, output in zip(self.layer_norms, outputs_before_norm)
        ]

        return out


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


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->OmDetTurbo, Deformable DETR->OmDet-Turbo
class OmDetTurboMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: OmDetTurboConfig, num_heads: int, n_points: int):
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
                "You'd better set embed_dim (d_model) in OmDetTurboMultiscaleDeformableAttention to make the"
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

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        default_dtype = torch.get_default_dtype()
        thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * (2.0 * math.pi / self.n_heads)
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

        if self.disable_custom_kernels:
            # PyTorch implementation
            output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
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
                output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output, attention_weights


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrConvNormLayer with RTDetr->OmDetTurbo
class OmDetTurboConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
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
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrRepVggBlock with RTDetr->OmDetTurbo
class OmDetTurboRepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        activation = config.csp_activation
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = OmDetTurboConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = OmDetTurboConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrCSPRepLayer with RTDetr->OmDetTurbo
class OmDetTurboCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.csp_activation

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = OmDetTurboConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = OmDetTurboConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[OmDetTurboRepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = OmDetTurboConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        device = hidden_state.device
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1).to(device)
        hidden_state_2 = self.conv2(hidden_state).to(device)
        return self.conv3(hidden_state_1 + hidden_state_2)


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrMultiheadAttention with RTDetr->OmDetTurbo
class OmDetTurboMultiheadAttention(nn.Module):
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

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
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
        key_states = self._reshape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._reshape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._reshape(query_states, target_len, batch_size).view(*proj_shape)
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
            # [seq_len, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = attention_mask.expand(batch_size, 1, *attention_mask.size())

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


class OmDetTurboEncoderLayer(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.self_attn = torch.nn.MultiheadAttention(
            config.encoder_hidden_dim, config.num_attention_heads, config.dropout
        )

        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.ffn_encoder_activation]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(config.encoder_hidden_dim, config.encoder_dim_feedforward)
        self.fc2 = nn.Linear(config.encoder_dim_feedforward, config.encoder_hidden_dim)
        self.final_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        query = key = self.with_pos_embed(hidden_states, position_embeddings)
        attentions = self.self_attn(query, key, value=hidden_states, attn_mask=attention_mask)[0]

        hidden_states = nn.functional.dropout(attentions, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrEncoder with RTDetr->OmDetTurbo
class OmDetTurboEncoder(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        self.layers = nn.ModuleList([OmDetTurboEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, src, src_mask=None, pos_embed=None, output_attentions: bool = False) -> torch.Tensor:
        hidden_states = src
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=src_mask,
                position_embeddings=pos_embed,
                output_attentions=output_attentions,
            )
        return hidden_states


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrHybridEncoder with RTDetr->OmDetTurbo
class OmDetTurboHybridEncoder(nn.Module):
    """
    Decoder consisting of a projection layer, a set of `OmDetTurboEncoder`, a top-down Feature Pyramid Network
    (FPN) and a bottom-up Path Aggregation Network (PAN). More details on the paper: https://arxiv.org/abs/2304.08069

    Args:
        config: OmDetTurboConfig
    """

    # Ignore copy
    def __init__(self, config: OmDetTurboConfig):
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
        conv_norm_activation = config.conv_norm_activation

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.encoder_hidden_dim, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(self.encoder_hidden_dim),
                )
            )

        # encoder transformer
        self.encoder = nn.ModuleList([OmDetTurboEncoder(config) for _ in range(len(self.encode_proj_layers))])
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                OmDetTurboConvNormLayer(
                    config, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1, activation=conv_norm_activation
                )
            )
            self.fpn_blocks.append(OmDetTurboCSPRepLayer(config))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                OmDetTurboConvNormLayer(
                    config, self.encoder_hidden_dim, self.encoder_hidden_dim, 3, 2, activation=conv_norm_activation
                )
            )
            self.pan_blocks.append(OmDetTurboCSPRepLayer(config))

    @staticmethod
    def build_2d_sincos_position_embedding(width, height, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(width), dtype=torch.float32)
        grid_h = torch.arange(int(height), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    # Ignore copy
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

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # get projection features
        projection_features = [self.input_proj[i](feature) for i, feature in enumerate(hidden_states)]
        # encoder
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (projection_features[enc_ind],)
                height, width = projection_features[enc_ind].shape[2:]
                # flatten [batch, channel, height, width] to [batch, height*width, channel]
                src_flatten = projection_features[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width, height, self.encoder_hidden_dim, self.positional_encoding_temperature
                    ).to(src_flatten.device, src_flatten.dtype)
                else:
                    pos_embed = None

                layer_outputs = self.encoder[i](
                    src_flatten,
                    pos_embed=pos_embed,
                    output_attentions=output_attentions,
                )
                projection_features[enc_ind] = (
                    layer_outputs[0].permute(0, 2, 1).reshape(-1, self.encoder_hidden_dim, height, width).contiguous()
                )

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (projection_features[enc_ind],)

        # broadcasting and fusion
        fpn_feature_maps = [projection_features[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = fpn_feature_maps[0]
            feat_low = projection_features[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            fpn_feature_maps[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fps_map = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            fpn_feature_maps.insert(0, fps_map)

        fpn_states = [fpn_feature_maps[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = fpn_states[-1]
            feat_high = fpn_feature_maps[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            hidden_states = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_high.to(downsample_feat.device)], dim=1)
            )
            fpn_states.append(hidden_states)

        if not return_dict:
            return tuple(v for v in [fpn_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=fpn_states, hidden_states=encoder_states, attentions=all_attentions)


class OmDetTurboMLPWithDropout(nn.Module):
    def __init__(self, config):
        d_input = config.label_dim
        d_output = config.label_dim
        dropout = config.decoder_dropout
        activation = config.decoder_activation
        d_hidden = config.decoder_encoder_dim_feedforward
        super(OmDetTurboMLPWithDropout, self).__init__()
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.activation = ACT2FN[activation]
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class OmDetTurboMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class OmDetTurboResidualLayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, config):
        size = config.label_dim
        dropout = config.decoder_dropout
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        "Apply residual connection to any sublayer with the same size."
        return self.norm1(x + self.dropout(y))


class OmDetTurboResidualMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = OmDetTurboMLPWithDropout(config)
        self.res1 = OmDetTurboResidualLayer(config)

    def forward(self, x):
        mlp_out = self.mlp(x)
        x = self.res1(x, mlp_out)
        return x


class OmDetTurboDeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, config):
        super().__init__()

        encoder_hidden_dim = config.decoder_hidden_dim
        n_heads = config.decoder_num_heads
        d_ffn = config.decoder_dim_feedforward
        dropout = config.decoder_dropout
        act = ACT2FN[config.decoder_activation]
        n_points = config.decoder_num_points
        fuse_type = config.fuse_type

        # self attention
        self.self_attn = nn.MultiheadAttention(encoder_hidden_dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(encoder_hidden_dim)

        # cross attention
        self.cross_attn = OmDetTurboMultiscaleDeformableAttention(config, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(encoder_hidden_dim)

        # ffn
        self.linear1 = nn.Linear(encoder_hidden_dim, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, encoder_hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(encoder_hidden_dim)

        self.fuse_type = fuse_type

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, embed, task_feats, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        origin_emb_len = embed.shape[1]

        # self attention
        q = k = self.with_pos_embed(embed, query_pos)
        # combine task_emb with q, k, v
        if self.fuse_type == "merged_attn":
            task_feats = task_feats.transpose(0, 1)  # [bs, token_len, hidden]
            q = torch.cat((q, task_feats), dim=1)  # [bs, dn+num_query+token_len, hidden]
            k = torch.cat((k, task_feats), dim=1)  # [bs, dn+num_query+token_len, hidden]
            embed = torch.cat((embed, task_feats), dim=1)  # [bs, dn+num_query+token_len, hidden]

        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # cut fused embedd to vision emb and task emb         todo  spilt here or split before
        task_feats = embed[:, origin_emb_len:, :].transpose(0, 1)  # [token_len, bs, hidden]
        embed = embed[:, :origin_emb_len, :]  # [bs, dn+num_query, hidden]

        # cross attention
        tgt = self.cross_attn(
            hidden_states=self.with_pos_embed(embed, query_pos),
            attention_mask=padding_mask,
            encoder_hidden_states=feats,
            reference_points=refer_bbox.unsqueeze(2),
            spatial_shapes=torch.tensor(shapes, dtype=torch.int64),
            # self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )[0]
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # ffn
        embed = self.forward_ffn(embed)

        return embed, task_feats


class OmDetTurboDeformableTransformerDecoder(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [OmDetTurboDeformableTransformerDecoderLayer(config) for _ in range(config.decoder_num_layers)]
        )
        self.num_layers = config.decoder_num_layers
        self.eval_idx = (
            config.decoder_eval_idx
            if config.decoder_eval_idx >= 0
            else config.decoder_num_layers + config.decoder_eval_idx
        )
        self.class_distance_type = config.class_distance_type
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

    @staticmethod
    def _norm(f, dim=-1):
        return f / f.norm(dim=dim, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _b_cosine(a, b, logit_scale):
        """
        a: B x K x H
        b: B x H x K
        """
        a = OmDetTurboDeformableTransformerDecoder._norm(a, dim=2)
        b = OmDetTurboDeformableTransformerDecoder._norm(b, dim=1)
        # Calculating the Loss
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * torch.bmm(a, b)
        return logits_per_image

    @staticmethod
    def get_class_similarity(class_distance_type, cls_feature, class_proj, logit_scale):
        if class_distance_type == "cosine":
            class_logits = OmDetTurboDeformableTransformerDecoder._b_cosine(
                cls_feature, class_proj, logit_scale
            )  # 4 100 256 4 256 20
        elif class_distance_type == "dot":
            class_logits = torch.bmm(cls_feature, class_proj)  # 4 100 20
        else:
            raise Exception("Unknown cls type {}".format(class_distance_type))
        return class_logits

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def forward(
        self,
        embed,  # decoder embeddings
        refer_bbox,  # anchor
        feats,  # image features
        shapes,  # feature shapes
        label_feats,  # label features
        task_feats,
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        output = embed
        decoder_bboxes = []
        decoder_classes = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output, task_feats = layer(
                output, task_feats, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox)
            )

            # refine bboxes, (bs, num_queries+num_denoising, 4)
            refined_bbox = torch.sigmoid(bbox_head[i](output) + self._inverse_sigmoid(refer_bbox))

            clas_proj = score_head[i](label_feats).permute(1, 2, 0)

            if self.training:
                decoder_classes.append(
                    self.get_class_similarity(self.class_distance_type, output, clas_proj, self.logit_scale)
                )
                if i == 0:
                    decoder_bboxes.append(refined_bbox)
                else:
                    decoder_bboxes.append(
                        torch.sigmoid(bbox_head[i](output) + self._inverse_sigmoid(last_refined_bbox))
                    )
            elif i == self.eval_idx:
                decoder_classes.append(
                    self.get_class_similarity(self.class_distance_type, output, clas_proj, self.logit_scale)
                )
                decoder_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(decoder_bboxes), torch.stack(decoder_classes)


class OmDetTurboPreTrainedModel(PreTrainedModel):
    config_class = OmDetTurboConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _get_cache_key_at_index(inputs, index):
        input_id = inputs["input_ids"][index]
        input_mask = inputs["attention_mask"][index]
        cache_key = "-".join([str(input_id[i]) for i in range(len(input_id)) if input_mask[i] != 0])

        return cache_key

    def get_cached_label_emb(self, labels):
        not_cached_index = []
        not_cached_labels = []
        total_embs = []
        for idx, _ in enumerate(labels["input_ids"]):
            cache_key = OmDetTurboModel._get_cache_key_at_index(labels, idx)
            if self.language_cache_label.has(cache_key):
                total_embs.append(self.language_cache_label.get(cache_key))
            else:
                total_embs.append(None)
                not_cached_index.append(idx)
                not_cached_labels.append(cache_key)

        logger.info(
            "cached label emb num: {}, not cached num: {}".format(
                len(total_embs) - len(not_cached_labels), len(not_cached_labels)
            )
        )

        if not_cached_labels:
            not_cached_labels_ids = torch.stack([labels["input_ids"][idx] for idx in not_cached_index])
            embeddings = self.language_backbone(not_cached_labels_ids, encode_type="label")
            for idx, emb in enumerate(embeddings):
                idx_to_put = not_cached_index[idx]
                total_embs[idx_to_put] = emb
                self.language_cache_label.put(not_cached_labels[idx], emb)

        total_label_embs = torch.stack(total_embs).to(self.device)
        return total_label_embs

    def get_cached_prompt_emb(self, batched_tasks):
        not_cached_index = []
        not_cached_tasks = []
        total_task_features = []
        total_task_masks = []
        for idx, _ in enumerate(batched_tasks["input_ids"]):
            cache_key = OmDetTurboModel._get_cache_key_at_index(batched_tasks, idx)
            if self.language_cache_prompt.has(cache_key):
                task_feature, task_mask = self.language_cache_prompt.get(cache_key)
                total_task_features.append(task_feature)
                total_task_masks.append(task_mask)
            else:
                total_task_features.append(None)
                total_task_masks.append(None)
                not_cached_index.append(idx)
                not_cached_tasks.append(cache_key)

        logger.info(
            "cached prompt emb num: {}, not cached num: {}".format(
                len(total_task_features) - len(not_cached_tasks), len(not_cached_tasks)
            )
        )
        if not_cached_tasks:
            not_cached_index_ids = torch.stack([batched_tasks["input_ids"][idx] for idx in not_cached_index])
            not_cached_mask = torch.stack([batched_tasks["attention_mask"][idx] for idx in not_cached_index])
            embeddings, masks = self.language_backbone(not_cached_index_ids, mask=not_cached_mask, encode_type="task")
            for idx in range(embeddings.shape[1]):
                emb = embeddings[:, [idx], :]
                idx_to_put = not_cached_index[idx]
                cur_mask = torch.unsqueeze(masks[idx], dim=0).to(self.device)
                total_task_features[idx_to_put] = emb
                total_task_masks[idx_to_put] = cur_mask
                self.language_cache_prompt.put(not_cached_tasks[idx], (emb, cur_mask))

        total_prompt_features = torch.cat(total_task_features, dim=1)
        total_prompt_masks = torch.cat(total_task_masks, dim=0).to(self.device)

        return total_prompt_features, total_prompt_masks

    def get_language_embedding(self, batched_labels, batched_tasks):
        max_label_size = max([len(label["input_ids"]) for label in batched_labels])
        label_features = []
        for labels in batched_labels:
            pad_size = max_label_size - len(labels["input_ids"])
            label_emb = self.get_cached_label_emb(labels)
            label_features.append(F.pad(label_emb, (0, 0, 0, pad_size)).unsqueeze(1).to(self.device))

        label_features = torch.cat(label_features, dim=1)  # num_label x batch_size x dim_size

        # Task Features
        # prompt_features: max_task_len x batch_size x dim_size
        # prompt_mask: batch_size x max_task_len
        # batched_tasks = ['detect a person', 'detect dog and cat']
        prompt_features, prompt_mask = self.get_cached_prompt_emb(batched_tasks)

        return label_features, prompt_features, prompt_mask


OMDET_TURBO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OmDetTurboConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

OMDET_TURBO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DetrImageProcessor.__call__`] for
            details.

        labels (BatchEmbedding):
            Tokenized labels. Several labels can be provided for each tasks, thus each element of labels is itself a
            BatchEmbedding corresponding to labels for a task. Each labels has the attributes `input_ids` and
            `attention_mask`.

        tasks (BatchEmbedding):
            Tokenized tasks. Each tasks has the attributes `input_ids` and `attention_mask`.
"""


@dataclass
class OmDetTurboDecoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoDecoder. This class adds two attributes to
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
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
    """

    decoder_bboxes: torch.FloatTensor = None
    decoder_cls: torch.FloatTensor = None
    encoder_bboxes: torch.FloatTensor = None
    encoder_cls: Optional[Tuple[torch.FloatTensor]] = None
    intermediate_reference_points: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class OmDetTurboDecoder(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)
        hidden_dim = config.decoder_hidden_dim
        self.num_head = config.decoder_num_heads
        self.num_queries = config.num_queries
        self.class_distance_type = config.class_distance_type
        self.logit_scale = torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        self.learn_init_query = config.learn_init_query

        # backbone feature projection
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim))
            for x in config.backbone_feat_channels
        )

        self.task_encoder = None
        self.fuse_type = config.fuse_type

        if self.fuse_type is not None:
            self.task_encoder = OmDetTurboResidualMLP(config)

            if self.fuse_type == "merged_attn" and config.label_dim != hidden_dim:
                self.task_project = nn.Linear(config.label_dim, hidden_dim)

        # Transformer module
        self.decoder = OmDetTurboDeformableTransformerDecoder(config)

        # TODO add denoising for training
        # self.denoising_class_embed = nn.Embedding(self.nc, hd)
        # self.num_denoising = nd
        # self.label_noise_ratio = label_noise_ratio
        # self.box_noise_scale = box_noise_scale
        # self.denoising_embed_proj = nn.Linear(config.label_dim, hidden_dim)

        # decoder embedding
        if self.learn_init_query:
            self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.query_pos_head = OmDetTurboMLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(config.label_dim, hidden_dim)
        self.enc_bbox_head = OmDetTurboMLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(config.label_dim, hidden_dim) for _ in range(config.decoder_num_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [OmDetTurboMLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(config.decoder_num_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, label_feats, task_feats, task_mask, batch=None):
        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # TODO add denoising for training
        # num_classes = label_feats.shape[0]
        # new_batch = {}
        # if self.training:
        #     #new_batch["gt_groups"] = [x["groups"] for x in batch]
        #     new_batch["cls"] = torch.cat([b["labels"] for b in batch], dim=0)
        #     new_batch["bboxes"] = torch.cat([b["boxes"] for b in batch], dim=0)
        #     new_batch["gt_groups"] = [b["groups"] for b in batch]
        #     batch_idx = torch.tensor([], dtype=torch.int32)
        #     for i, idx in enumerate(new_batch["gt_groups"]):
        #         x = torch.tensor([i] * idx, dtype=torch.int32)
        #         batch_idx = torch.cat((batch_idx, x), 0)

        #     new_batch["batch_idx"] = batch_idx.to("cuda")

        # prepare denoising training
        # dn_embed, dn_bbox, attn_mask, dn_meta = \
        #     get_cdn_group(new_batch,
        #                   num_classes,
        #                   self.num_queries,
        #                   self.denoising_embed_proj(label_feats),
        #                   self.num_denoising,
        #                   self.label_noise_ratio,
        #                   self.box_noise_scale,
        #                   self.training,
        #                   self.amp)
        dn_embed, dn_bbox, attn_mask, dn_meta = None, None, None, None
        bs = task_mask.shape[0]

        # compose attn_mask for vision_emb and task_emb fusion
        if self.fuse_type == "merged_attn":
            if self.task_encoder is not None:
                task_feats = self.task_encoder(task_feats)

            if self.task_project is not None:
                task_feats = self.task_project(task_feats)

            src_key_mask = (task_mask == 0).detach()
            # if self.training and attn_mask is not None:
            #     attn_mask_len = attn_mask.shape[0]

            #     fusion_size = attn_mask.shape[0]+task_feats.shape[0]
            #     new_attn_mask = torch.zeros([bs, fusion_size, fusion_size], dtype=torch.bool)
            #     new_attn_mask[:, :attn_mask_len, :attn_mask_len] = attn_mask.unsqueeze(0).expand(bs, -1, -1)

            #     new_attn_mask[:, attn_mask_len:, :dn_embed.shape[2]] = True
            #     new_attn_mask[:, :, attn_mask_len:] = src_key_mask.unsqueeze(1)
            #     new_attn_mask = new_attn_mask.repeat(self.nhead, 1, 1)
            #     attn_mask = new_attn_mask.to(attn_mask.device)  # [bs, dn+num_query+task_token_len, dn+num_query+task_token_len]
            # else:
            attn_mask_len = self.num_queries
            fusion_size = attn_mask_len + task_feats.shape[0]
            new_attn_mask = torch.zeros([bs, fusion_size, fusion_size], dtype=torch.bool)
            new_attn_mask[:, :, attn_mask_len:] = src_key_mask.unsqueeze(1)
            new_attn_mask = new_attn_mask.repeat(self.num_head, 1, 1)
            attn_mask = new_attn_mask.to(task_mask.device)

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, label_feats, dn_embed, dn_bbox
        )

        # decoder
        decoder_bboxes, decoder_classes = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            label_feats,
            task_feats,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        return OmDetTurboDecoderOutput(
            decoder_bboxes=decoder_bboxes[-1],
            decoder_cls=decoder_classes[-1],
            encoder_bboxes=enc_bboxes,
            encoder_cls=enc_scores,
            intermediate_reference_points=refer_bbox,
        )

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype, device=device),
                torch.arange(end=w, dtype=dtype, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.inf, dtype=torch.float32))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, label_feats, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(
            torch.where(valid_mask, feats, torch.tensor(0.0, dtype=torch.float32))
        )  # bs, h*w, 256

        # enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        clas_proj = self.enc_score_head(label_feats).permute(1, 2, 0)  #
        enc_outputs_scores = OmDetTurboDeformableTransformerDecoder.get_class_similarity(
            self.class_distance_type, features, clas_proj, self.logit_scale
        )

        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # Unsigmoided
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # refer_bbox = torch.gather(enc_outputs_bboxes, 1, topk_ind.reshape(bs, self.num_queries).unsqueeze(-1).repeat(1, 1, 4))

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        if self.learn_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores


@dataclass
class OmDetTurboModelOutput(ModelOutput):
    """
    Base class for outputs of the OmDetTurbo model.

    Args:
        decoder_bboxes (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes, 4)`):
            The predicted bounding boxes of the objects.
        decoder_cls (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            The predicted classes of the objects.
        encoder_bboxes (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes, 4)`):
            The predicted bounding boxes of the objects from the encoder.
        encoder_cls (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            The predicted classes of the objects from the encoder.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            The last hidden state of the encoder.
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 2)`:
            The intermediate reference
    """

    decoder_bboxes: torch.FloatTensor = None
    decoder_cls: torch.FloatTensor = None
    encoder_bboxes: torch.FloatTensor = None
    encoder_cls: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@add_start_docstrings(
    """
    OmDetTurbo Model (consisting of a vision and a text backbone, and encoder-decoder architecture).
    Same as `OmDetTurboForObjectDetection` as OmDetTurbo's decoder outputs bounding boxes and classes score directly.
    """,
    OMDET_TURBO_START_DOCSTRING,
)
class OmDetTurboModel(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)
        # Create backbone
        self.backbone = OmDetTurboVisionBackbone(config)
        self.language_backbone = OmDetTurboLanguageBackbone(config=config)
        self.encoder = OmDetTurboHybridEncoder(config)
        self.decoder = OmDetTurboDecoder(config)
        self.num_queries = config.num_queries

        self.language_cache_label = OmDetTurboLRUCache(config.cache_size)
        self.language_cache_prompt = OmDetTurboLRUCache(config.cache_size)
        self.post_init()

    @add_start_docstrings_to_model_forward(OMDET_TURBO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OmDetTurboModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, labels: Tensor, tasks: Tensor, ground_truths: Optional[Tensor] = None):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> labels = ["cat", "remote"]
        >>> task = "Detect {}.".format(",".join(labels))

        >>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-tiny")
        >>> model = AutoModel.from_pretrained("omlab/omdet-turbo-tiny")

        >>> inputs = processor(image, tasks=task, labels=labels, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        if ground_truths is not None:
            raise NotImplementedError("Training is not implemented yet")
        labels = [labels[str(i)] for i in range(len(labels))]

        image_features = self.backbone(pixel_values)
        encoder_features = self.encoder(image_features)[0]
        label_features, prompt_features, prompt_mask = self.get_language_embedding(labels, tasks)
        decoder_outputs = self.decoder(encoder_features, label_features, prompt_features, prompt_mask)

        return OmDetTurboModelOutput(
            decoder_bboxes=decoder_outputs.decoder_bboxes,
            decoder_cls=decoder_outputs.decoder_cls,
            encoder_bboxes=decoder_outputs.encoder_bboxes,
            encoder_cls=decoder_outputs.encoder_cls,
            encoder_last_hidden_state=encoder_features,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
        )


@dataclass
class OmDetTurboObjectDetectionOutput(ModelOutput):
    """
    Base class for outputs of the OmDetTurbo model.

    Args:
        decoder_bboxes (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes, 4)`):
            The predicted bounding boxes of the objects.
        decoder_cls (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            The predicted classes of the objects.
        encoder_bboxes (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes, 4)`):
            The predicted bounding boxes of the objects from the encoder.
        encoder_cls (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            The predicted classes of the objects from the encoder.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            The last hidden state of the encoder.
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 2)`:
            The intermediate reference
    """

    decoder_bboxes: torch.FloatTensor = None
    decoder_cls: torch.FloatTensor = None
    encoder_bboxes: torch.FloatTensor = None
    encoder_cls: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@add_start_docstrings(
    """
    OmDetTurbo Model (consisting of a vision and a text backbone, and encoder-decoder architecture) outputting
    bounding boxes and classes scores for tasks such as COCO detection.
    """,
    OMDET_TURBO_START_DOCSTRING,
)
class OmDetTurboForObjectDetection(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)
        # Create backbone
        self.backbone = OmDetTurboVisionBackbone(config)
        self.language_backbone = OmDetTurboLanguageBackbone(config=config)
        self.encoder = OmDetTurboHybridEncoder(config)
        self.decoder = OmDetTurboDecoder(config)
        self.num_queries = config.num_queries

        self.language_cache_label = OmDetTurboLRUCache(config.cache_size)
        self.language_cache_prompt = OmDetTurboLRUCache(config.cache_size)
        self.post_init()

    @add_start_docstrings_to_model_forward(OMDET_TURBO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OmDetTurboObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, labels: Tensor, tasks: Tensor, ground_truths: Optional[Tensor] = None):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> labels = ["cat", "remote"]
        >>> task = "Detect {}.".format(",".join(labels))

        >>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-tiny")
        >>> model = AutoModel.from_pretrained("omlab/omdet-turbo-tiny")

        >>> inputs = processor(image, tasks=task, labels=labels, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        labels = [labels[str(i)] for i in range(len(labels))]
        if ground_truths is not None:
            raise NotImplementedError("Training is not implemented yet")
        image_features = self.backbone(pixel_values)
        encoder_features = self.encoder(image_features)[0]

        label_features, prompt_features, prompt_mask = self.get_language_embedding(labels, tasks)

        decoder_outputs = self.decoder(encoder_features, label_features, prompt_features, prompt_mask)

        return OmDetTurboObjectDetectionOutput(
            decoder_bboxes=decoder_outputs.decoder_bboxes,
            decoder_cls=decoder_outputs.decoder_cls,
            encoder_bboxes=decoder_outputs.encoder_bboxes,
            encoder_cls=decoder_outputs.encoder_cls,
            encoder_last_hidden_state=encoder_features,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
        )
