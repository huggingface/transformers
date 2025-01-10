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
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..rt_detr.configuration_rt_detr import RTDetrConfig
from ..rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig
from ..rt_detr.modeling_rt_detr import (
    MultiScaleDeformableAttentionFunction,
    RTDetrDecoder,
    RTDetrDecoderLayer,
    RTDetrForObjectDetection,
    RTDetrModel,
    RTDetrMultiscaleDeformableAttention,
    RTDetrPreTrainedModel,
)


class RtDetrV2ResNetConfig(RTDetrResNetConfig):
    def __init__(
        self,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class RtDetrV2Config(RTDetrConfig):
    r"""
    decoder_n_levels (`int`, *optional*, defaults to 3): The number of feature levels used by the decoder.
    decoder_offset_scale (`float`, *optional*, defaults to 0.5): Scaling factor applied to the attention offsets in the decoder.
    """

    model_type = "rt_detr_v2"

    def __init__(
        self,
        decoder_n_levels=3,  # default value
        decoder_offset_scale=0.5,  # default value
        d_model=256,
        encoder_attention_heads=8,
        **super_kwargs,
    ):
        # init the base RTDetrConfig class with additional parameters
        super().__init__(**super_kwargs)
        del self.d_model
        del self.encoder_attention_heads

        if not hasattr(self, "d_model"):
            self.d_model = d_model

        if not hasattr(self, "encoder_attention_heads"):
            self.encoder_attention_heads = encoder_attention_heads
        # add the new attributes with the given values or defaults
        self.decoder_n_levels = decoder_n_levels
        self.decoder_offset_scale = decoder_offset_scale


def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: List[int],
    method="default",
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points = sampling_locations.shape
    value_list = (
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height.item() * width.item() for height, width in value_spatial_shapes], dim=-1)
    )
    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[level_id]
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        if method == "default":
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (sampling_grid_l_ * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                torch.int64
            )

            # Separate clamping for x and y coordinates
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)

            # Combine the clamped coordinates
            sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
            sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
            sampling_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l_ = value_l_[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l_ = sampling_value_l_.permute(0, 2, 1).reshape(
                batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
            )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttentionFunctionV2(MultiScaleDeformableAttentionFunction):
    pass


# the main change
class RtDetrV2MultiscaleDeformableAttention(RTDetrMultiscaleDeformableAttention):
    """
    RTDETRV2 version of multiscale deformable attention, extending the base implementation
    with improved offset handling and initialization.
    """

    def __init__(self, config: RtDetrV2Config, num_heads: int = None, n_points: int = None):
        num_heads = num_heads if num_heads is not None else config.decoder_attention_heads
        n_points = n_points if n_points is not None else config.decoder_n_points
        # Initialize parent class with config parameters
        super().__init__(config=config, num_heads=num_heads, n_points=n_points)

        # V2-specific attributes
        self.n_levels = config.decoder_n_levels
        self.offset_scale = config.decoder_offset_scale
        # Initialize n_points list and scale
        n_points_list = [self.n_points for _ in range(self.n_levels)]
        self.n_points_list = n_points_list
        n_points_scale = [1 / n for n in n_points_list for _ in range(n)]
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

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
        **kwargs,
    ):
        # Process inputs up to sampling locations calculation using parent class logic
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
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        # V2-specific sampling offsets shape
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )

        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        # V2-specific sampling locations calculation
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        # V2-specific attention implementation choice
        if self.disable_custom_kernels:
            output = multi_scale_deformable_attention_v2(
                value, spatial_shapes, sampling_locations, attention_weights, self.n_points_list
            )
        else:
            try:
                output = MultiScaleDeformableAttentionFunctionV2.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                output = multi_scale_deformable_attention_v2(
                    value, spatial_shapes, sampling_locations, attention_weights, self.n_points_list
                )

        output = self.output_proj(output)
        return output, attention_weights


class RtDetrV2DecoderLayer(RTDetrDecoderLayer):
    def __init__(self, config: RtDetrV2Config):
        # initialize parent class
        super().__init__(config)
        # override only the encoder attention module with v2 version
        self.encoder_attn = RtDetrV2MultiscaleDeformableAttention(config)


class RtDetrV2PreTrainedModel(RTDetrPreTrainedModel):
    def _init_weights(self, module):
        # this could be simplified like calling the base class function first then adding new stuff
        """initialize linear layer biasreturn value according to a given probability value."""
        if isinstance(module, (RtDetrV2ForObjectDetection, RtDetrV2Decoder)):
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

        if isinstance(module, RtDetrV2MultiscaleDeformableAttention):
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

        if isinstance(module, RtDetrV2Model):
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


class RtDetrV2Decoder(RTDetrDecoder):
    def __init__(self, config: RtDetrV2Config):
        super().__init__(config)
        self.layers = nn.ModuleList([RtDetrV2DecoderLayer(config) for _ in range(config.decoder_layers)])


class RtDetrV2Model(RTDetrModel):
    def __init__(self, config: RtDetrV2Config):
        super().__init__(config)
        # decoder
        self.decoder = RtDetrV2Decoder(config)


def _get_clones(partial_module, N):
    return nn.ModuleList([partial_module() for i in range(N)])


class RtDetrV2ForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RtDetrV2Config):
        super().__init__(config)
        # RTDETR encoder-decoder model
        self.model = RtDetrV2Model(config)
        # here self.model.decoder.bbox_embed is null, but not self.bbox_embed
        # fix bug
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed


__all__ = [
    "RtDetrV2Config",
    "RtDetrV2ResNetConfig",
    "RtDetrV2Model",
    "RtDetrV2PreTrainedModel",
    "RtDetrV2ForObjectDetection",
]
