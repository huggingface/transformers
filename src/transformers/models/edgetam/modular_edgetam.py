# coding=utf-8
# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch SAM 2 model."""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.models.sam2.configuration_sam2 import Sam2Config, Sam2MaskDecoderConfig, Sam2PromptEncoderConfig
from transformers.models.sam2.modeling_sam2 import (
    Sam2Attention,
    Sam2FeedForward,
    Sam2LayerNorm,
    Sam2Model,
    Sam2PreTrainedModel,
    Sam2TwoWayAttentionBlock,
    Sam2VisionEncoderOutput,
    Sam2VisionModel,
)
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from ...configuration_utils import PretrainedConfig
from ...processing_utils import Unpack
from ...utils import (
    auto_docstring,
)
from ..auto import CONFIG_MAPPING, AutoConfig


# fix this in modular
if True:
    from transformers.models.timm_wrapper.modeling_timm_wrapper import TimmWrapperModel


class EdgeTamVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EdgeTamVisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/EdgeTAM](https://huggingface.co/facebook/EdgeTAM) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PretrainedConfig"]`, *optional*):
            Configuration for the vision backbone. This is used to instantiate the backbone using
            `AutoModel.from_config`.
        backbone_channel_list (`List[int]`, *optional*, defaults to `[384, 192, 96, 48]`):
            The list of channel dimensions for the backbone.
        backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[256, 256], [128, 128], [64, 64]]`):
            The spatial sizes of the feature maps from the backbone.
        fpn_hidden_size (`int`, *optional*, defaults to 256):
            The hidden dimension of the FPN.
        fpn_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size for the convolutions in the neck.
        fpn_stride (`int`, *optional*, defaults to 1):
            The stride for the convolutions in the neck.
        fpn_padding (`int`, *optional*, defaults to 0):
            The padding for the convolutions in the neck.
        fpn_top_down_levels (`List[int]`, *optional*, defaults to `[2, 3]`):
            The levels for the top-down FPN connections.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels from the FPN to use.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "vision_config"
    model_type = "edgetam_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    def __init__(
        self,
        backbone_config=None,
        backbone_channel_list=None,
        backbone_feature_sizes=None,
        fpn_hidden_size=256,
        fpn_kernel_size=1,
        fpn_stride=1,
        fpn_padding=0,
        fpn_top_down_levels=None,
        num_feature_levels=3,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        backbone_channel_list = [384, 192, 96, 48] if backbone_channel_list is None else backbone_channel_list
        backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if backbone_feature_sizes is None else backbone_feature_sizes
        )
        fpn_top_down_levels = [2, 3] if fpn_top_down_levels is None else fpn_top_down_levels

        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = backbone_config.get("model_type", "timm_wrapper")
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif isinstance(backbone_config, AutoConfig):
            backbone_config = backbone_config
        elif backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(
                "timm/repvit_m1.dist_in1k",
                model_args={"in_chans": 3, "features_only": True, "out_indices": [0, 1, 2, 3]},
            )

        self.backbone_config = backbone_config

        # Neck
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_kernel_size = fpn_kernel_size
        self.fpn_stride = fpn_stride
        self.fpn_padding = fpn_padding
        self.fpn_top_down_levels = fpn_top_down_levels
        self.num_feature_levels = num_feature_levels

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class EdgeTamPromptEncoderConfig(Sam2PromptEncoderConfig):
    pass


class EdgeTamMaskDecoderConfig(Sam2MaskDecoderConfig):
    pass


class EdgeTamConfig(Sam2Config):
    pass


class EdgeTamLayerNorm(Sam2LayerNorm):
    pass


class EdgeTamVisionEncoderOutput(Sam2VisionEncoderOutput):
    pass


class EdgeTamAttention(Sam2Attention):
    pass


class EdgeTamTwoWayAttentionBlock(Sam2TwoWayAttentionBlock):
    pass


class EdgeTamFeedForward(Sam2FeedForward):
    pass


@auto_docstring
class EdgeTamPreTrainedModel(Sam2PreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, EdgeTamLayerNorm)):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        if isinstance(module, EdgeTamModel):
            if module.no_memory_embedding is not None:
                module.no_memory_embedding.data.zero_()


@auto_docstring(
    custom_intro="""
    The vision model from EdgeTAM without any head or projection on top.
    """
)
class EdgeTamVisionModel(Sam2VisionModel):
    config_class = EdgeTamVisionConfig
    main_input_name = "pixel_values"
    _can_record_outputs = {"hidden_states": TimmWrapperModel, "attentions": TimmWrapperModel}

    def get_input_embeddings(self):
        raise NotImplementedError("Can't get input embeddings from timm wrapper model")

    @check_model_inputs
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, EdgeTamVisionEncoderOutput]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Forward through backbone
        backbone_output = self.backbone(pixel_values)
        intermediate_hidden_states = backbone_output.last_hidden_state
        intermediate_hidden_states = [hidden_state.permute(0, 2, 3, 1) for hidden_state in intermediate_hidden_states]

        fpn_hidden_states, fpn_position_encoding = self.neck(intermediate_hidden_states)
        # Select last `num_feature_levels` feature levels from FPN and reverse order to get features from high to low resolution
        fpn_hidden_states = fpn_hidden_states[-self.num_feature_levels :][::-1]
        fpn_position_encoding = fpn_position_encoding[-self.num_feature_levels :][::-1]

        return EdgeTamVisionEncoderOutput(
            last_hidden_state=intermediate_hidden_states[-1],
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )


class EdgeTamModel(Sam2Model):
    _keys_to_ignore_on_load_unexpected = [
        r"^memory_.*",
        r"^mask_downsample.*",
        r"spatial_perceiver.*",
        r"^object_pointer_proj.*",
        r"^temporal_positional_encoding_projection_layer.*",
        "no_memory_positional_encoding",
        "no_object_pointer",
        "occlusion_spatial_embedding_parameter",
    ]

    def get_input_embeddings(self):
        raise NotImplementedError("Can't get input embeddings from timm wrapper model")


__all__ = [
    "EdgeTamModel",
    "EdgeTamVisionModel",
    "EdgeTamPreTrainedModel",
    "EdgeTamConfig",
    "EdgeTamVisionConfig",
    "EdgeTamPromptEncoderConfig",
    "EdgeTamMaskDecoderConfig",
]
