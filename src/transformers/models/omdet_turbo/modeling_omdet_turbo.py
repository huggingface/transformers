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

import copy
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import uniform_

from ...activations import ACT2CLS, ACT2FN
from ...file_utils import (
    is_torch_cuda_available,
)
from ...modeling_utils import PreTrainedModel
from ...utils import is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_omdet_turbo import OmDetTurboConfig


MultiScaleDeformableAttention = None

logger = logging.get_logger(__name__)


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

        if config.encoder_hidden_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (encoder_hidden_dim) must be divisible by num_heads, but got {config.encoder_hidden_dim} and {num_heads}"
            )
        dim_per_head = config.encoder_hidden_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (encoder_hidden_dim) in OmDetTurboMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.n_levels = len(config.backbone_feat_channels)
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.encoder_hidden_dim, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.encoder_hidden_dim, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim)
        self.output_proj = nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim)

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
        value = value.view(batch_size, sequence_length, self.n_heads, self.encoder_hidden_dim // self.n_heads)
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


class OmDetTurboPreTrainedModel(PreTrainedModel):
    config_class = OmDetTurboConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.config.init_std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)


# class OmDetTurboVisionBackbone(nn.Module):
#     def __init__(self, config: OmDetTurboConfig):
#         super().__init__()
#         self.backbone = load_backbone(config.vision_config)
#         self.intermediate_out_indices = config.intermediate_out_indices  # stage2, stage3
#         self.norm_layer1 = nn.LayerNorm(
#             self.backbone.out_feature_channels["stage{}".format(self.intermediate_out_indices[0])]
#         )
#         self.norm_layer2 = nn.LayerNorm(
#             self.backbone.out_feature_channels["stage{}".format(self.intermediate_out_indices[1])]
#         )

#     def forward(self, pixel_values: Tensor):
#         outputs = self.backbone(pixel_values, output_hidden_states=True)
#         feature_map3 = outputs["feature_maps"]
#         hidden_states = outputs["hidden_states"]
#         feature_map1 = self.norm_layer1(hidden_states[self.intermediate_out_indices[0]])
#         # square root of the feature map size
#         H = W = int(math.sqrt(feature_map1.shape[-1]))
#         feature_map1 = (
#             feature_map1.view(-1, H, W, self.norm_layer1.normalized_shape[0]).permute(0, 3, 1, 2).contiguous()
#         )
#         feature_map2 = self.norm_layer2(hidden_states[self.intermediate_out_indices[1]])
#         H = W = int(math.sqrt(feature_map2.shape[-1]))
#         feature_map2 = (
#             feature_map2.view(-1, H, W, self.norm_layer2.normalized_shape[0]).permute(0, 3, 1, 2).contiguous()
#         )
#         return feature_map1, feature_map2, feature_map3


class OmDetTurboLanguageBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.model = AutoModel.from_config(config.text_config)
        self.text_projection = nn.Linear(
            config.text_projection_in_features, config.text_projection_out_features, bias=False
        )

    def forward(self, hidden_states, pos_embed=None):
        text_outputs = self.model(hidden_states)
        pooled_output = text_outputs[1]
        text_embeds = self.text_projection(pooled_output)

        return text_embeds


class OmDetTurboModel(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)

        # Create backbone
        self.vision_backbone = load_backbone(config.vision_config)
        # self.language_backbone = AutoModel.from_config(config.text_config)
        self.language_backbone = OmDetTurboLanguageBackbone(config=config)
        self.encoder = OmDetTurboEncoder(config)
        self.decoder = OmDetTurboDecoder(config)
        self.num_queries = config.num_queries

        self.language_cache_label = LRUCache(100)
        self.language_cache_prompt = LRUCache(100)
        self.post_init()

    # @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=OmDetTurboModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, labels_ids: Tensor, tasks_ids: Tensor, labels: Optional[Tensor] = None):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-tiny")
        >>> model = AutoModel.from_pretrained("omlab/omdet-turbo-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        body_feats = self.vision_backbone(pixel_values)
        encoder_feats = self.encoder(body_feats)
        label_feats = self.language_backbone(labels_ids, return_tokens=False)
        prompt_feats, prompt_mask = self.language_backbone(tasks_ids, return_tokens=True)
        decoder_feats = self.decoder(encoder_feats, label_feats, prompt_feats, prompt_mask)

        return decoder_feats


def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


class OmDetTurboBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(
            out_channels,
            # epsilon=1e-3,  # for amp(fp16), set in ppdet/engine/trainer.py
            # momentum=0.97,
            # weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            # bias_attr=ParamAttr(regularizer=L2Decay(0.0))
        )

        self.act = ACT2FN[act]

    #     self._init_weights()
    #
    # def _init_weights(self):
    #     conv_init_(self.conv)

    def forward(self, x):
        x = self.bn(self.conv(x))
        y = self.act(x)

        return y


# Copied from models.rt_detr.modeling_rt_detr.RTDetrConvNormLayer with RTDetr -> OmDetTurbo
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


# Copied from models.rt_detr.modeling_rt_detr.OmDetTurboRepVggBlock with RTDetr -> OmDetTurbo
class OmDetTurboRepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = OmDetTurboConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = OmDetTurboConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


# Copied from models.rt_detr.modeling_rt_detr.RTDetrCSPRepLayer with RTDetr -> OmDetTurbo
class OmDetTurboCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

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


class OmDetTurboEncoderLayer(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
    ):
        super(OmDetTurboEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = torch.nn.MultiheadAttention(encoder_hidden_dim, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(encoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, encoder_hidden_dim)

        self.norm1 = nn.LayerNorm(encoder_hidden_dim)
        self.norm2 = nn.LayerNorm(encoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)
        # print(src[1].shape, src[0].shape)
        src = src[0]
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


class OmDetTurboEncoder(OmDetTurboPreTrainedModel):
    """
    ELA Encoder

    --------------------------
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`OmDetTurboEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: OmDetTurboConfig
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)

        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.encoder_feat_strides
        self.hidden_dim = config.encoder_hidden_dim
        self.use_encoder_idx = config.use_encoder_idx
        self.num_encoder_layers = config.num_encoder_layers
        self.pe_temperature = config.pe_temperature
        activation = config.encoder_activation
        dim_feedforward = config.encoder_dim_feedforward

        self.encoder_layer = OmDetTurboEncoderLayer(dim_feedforward=dim_feedforward)

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.hidden_dim, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(self.hidden_dim),
                )
            )
        # encoder transformer
        # self.encoder = nn.ModuleList([
        #     TransformerEncoder(self.encoder_layer, self.num_encoder_layers)
        #     for _ in range(len(self.use_encoder_idx))
        # ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(OmDetTurboBaseConv(self.hidden_dim, self.hidden_dim, 1, 1, act=activation))
            self.fpn_blocks.append(OmDetTurboCSPRepLayer(config=config))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                OmDetTurboBaseConv(self.hidden_dim, self.hidden_dim, 3, stride=2, act=activation)
            )
            self.pan_blocks.append(OmDetTurboCSPRepLayer(config=config))

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(start_dim=2).transpose(1, 2)
                pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(
                    src_flatten.device
                )
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose(1, 2).reshape((-1, self.hidden_dim, h, w))

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


class OmDetTurboMLPWithDropout(nn.Module):
    def __init__(self, d_input, d_output, d_hidden=1024, dropout=0.1, activation="relu"):
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

    def __init__(self, size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        "Apply residual connection to any sublayer with the same size."
        return self.norm1(x + self.dropout(y))


class OmDetTurboResidualMLP(nn.Module):
    def __init__(self, d_m, dropout, d_hidden=1024, activation="relu"):
        super().__init__()
        self.mlp = OmDetTurboMLPWithDropout(d_m, d_m, d_hidden, dropout, activation)
        self.res1 = OmDetTurboResidualLayer(d_m, dropout)

    def forward(self, x):
        mlp_out = self.mlp(x)
        x = self.res1(x, mlp_out)
        return x


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def cls_score(cls_type, cls_feature, class_proj, logit_scale):
    if cls_type == "cosine":
        class_logits = _b_cosine(cls_feature, class_proj, logit_scale)  # 4 100 256 4 256 20
    elif cls_type == "dot":
        class_logits = torch.bmm(cls_feature, class_proj)  # 4 100 20
    else:
        raise Exception("Unknown cls type {}".format(cls_type))
    return class_logits


def _norm(f, dim=-1):
    return f / f.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def _b_cosine(a, b, logit_scale):
    """
    a: B x K x H
    b: B x H x K
    """
    a = _norm(a, dim=2)
    b = _norm(b, dim=1)
    # Calculating the Loss
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * torch.bmm(a, b)
    return logits_per_image


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        config,
        encoder_hidden_dim=256,
        n_heads=8,
        d_ffn=1024,
        dropout=0.0,
        act=nn.ReLU(),
        n_levels=4,
        n_points=4,
        fuse_type="merged_attn",
    ):
        super().__init__()

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
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # ffn
        embed = self.forward_ffn(embed)

        return embed, task_feats


class DeformableTransformerDecoderV2(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, cls_type="cosine"):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.cls_type = cls_type
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

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
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output, task_feats = layer(
                output, task_feats, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox)
            )

            # refine bboxes, (bs, num_queries+num_denoising, 4)
            refined_bbox = torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(refer_bbox))

            clas_proj = score_head[i](label_feats).permute(1, 2, 0)

            if self.training:
                dec_cls.append(cls_score(self.cls_type, output, clas_proj, self.logit_scale))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(cls_score(self.cls_type, output, clas_proj, self.logit_scale))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class OmDetTurboDecoder(OmDetTurboPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`OmDetTurboDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for OmDet-Turbo:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: OmDetTurboConfig
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)
        self.hidden_dim = config.decoder_hidden_dim
        self.num_head = config.decoder_num_heads
        self.num_levels = len(config.backbone_feat_channels)  # num level
        self.nc = 80
        self.num_queries = config.num_queries
        self.num_decoder_layers = config.decoder_num_layers
        self.label_dim = config.label_dim
        self.cls_type = config.cls_type
        self.logit_scale = torch.tensor(np.log(1 / 0.07), dtype=torch.float32)

        activation = ACT2FN[config.decoder_activation]
        dim_feedforward = config.decoder_dim_feedforward
        num_points = config.decoder_num_points
        dropout = config.decoder_dropout
        eval_idx = config.decoder_eval_idx
        # backbone feature projection
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, self.hidden_dim, 1, bias=False), nn.BatchNorm2d(self.hidden_dim))
            for x in config.backbone_feat_channels
        )
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        self.task_encoder = None
        self.fuse_type = config.fuse_type

        if self.fuse_type is not None:
            self.task_encoder = OmDetTurboResidualMLP(self.label_dim, dropout=dropout)

            if self.fuse_type == "merged_attn" and self.label_dim != self.hidden_dim:
                self.task_project = nn.Linear(self.label_dim, self.hidden_dim)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(
            config,
            self.hidden_dim,
            self.num_head,
            dim_feedforward,
            dropout,
            activation,
            self.num_levels,
            num_points,
            self.fuse_type,
        )
        self.decoder = DeformableTransformerDecoderV2(
            self.hidden_dim, decoder_layer, self.num_decoder_layers, eval_idx, cls_type=self.cls_type
        )

        # denoising part
        # self.denoising_class_embed = nn.Embedding(self.nc, hd)
        # self.num_denoising = nd
        # self.label_noise_ratio = label_noise_ratio
        # self.box_noise_scale = box_noise_scale
        # self.denoising_embed_proj = nn.Linear(self.label_dim, self.hidden_dim)

        # decoder embedding
        self.query_pos_head = OmDetTurboMLP(4, 2 * self.hidden_dim, self.hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        # self.enc_score_head = nn.Linear(hd, self.nc)
        self.enc_score_head = nn.Linear(self.label_dim, self.hidden_dim)
        self.enc_bbox_head = OmDetTurboMLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(self.label_dim, self.hidden_dim) for _ in range(self.num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [OmDetTurboMLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3) for _ in range(self.num_decoder_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, label_feats, task_feats, task_mask, batch=None):
        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)
        num_classes = label_feats.shape[0]

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
            new_attn_mask = new_attn_mask.repeat(self.nhead, 1, 1)
            attn_mask = new_attn_mask.to(task_mask.device)

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, label_feats, dn_embed, dn_bbox
        )

        # decoder
        dec_bboxes, dec_scores = self.decoder(
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
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        return x

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
        enc_outputs_scores = cls_score(self.cls_type, features, clas_proj, self.logit_scale)

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

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores


class LRUCache:
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def has(self, key) -> bool:
        return key in self.cache

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
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

    def pop(self, key, value):
        self.cache.pop(key, None)


################################################## END NEW ##################################################################
