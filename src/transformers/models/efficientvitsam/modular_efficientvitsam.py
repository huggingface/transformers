# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Modular EfficientViT-SAM configuration classes.
"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from huggingface_hub.dataclasses import strict
from ..sam.configuration_sam import SamPromptEncoderConfig, SamMaskDecoderConfig, SamConfig

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamPromptEncoderConfig(SamPromptEncoderConfig):
    pass


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamMaskDecoderConfig(SamMaskDecoderConfig):
    pass


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EfficientViTSamVisionModel`]. It is used to
    instantiate an EfficientViT-SAM vision encoder according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        width_list (`list[int]`, *optional*, defaults to `[32, 64, 128, 256, 512]`):
            Channel dimensions for each stage of the backbone.
        depth_list (`list[int]`, *optional*, defaults to `[1, 1, 1, 6, 6]`):
            Number of blocks for each stage of the backbone.
        block_list (`list[str]`, *optional*):
            Type of block to use in each stage of a large backbone (e.g. `["res", "fmb", "fmb", "fmb", "att@3", "att@3"]`).
        expand_list (`list[float]`, *optional*):
            Expand ratios for blocks in each stage of a large backbone.
        fewer_norm_list (`list[bool]`, *optional*):
            Whether to use fewer normalization layers in each stage of a large backbone.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        qkv_dim (`int`, *optional*, defaults to 32):
            Query/Key/Value dimension in the attention layer.
        norm (`str`, *optional*, defaults to `"bn2d"`):
            Type of normalization layer to use.
        act_func (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        fid_list (`list[str]`, *optional*, defaults to `["stage4", "stage3", "stage2"]`):
            Stages from which to aggregate features in the Neck.
        in_channel_list (`list[int]`, *optional*, defaults to `[512, 256, 128]`):
            Channel widths of features aggregated by the Neck.
        head_width (`int`, *optional*, defaults to 256):
            Projection dimension of Neck inputs.
        head_depth (`int`, *optional*, defaults to 8):
            Depth of intermediate layers in the Neck.
        expand_ratio (`float`, *optional*, defaults to 1.0):
            Expansion ratio in the Neck's FusedMBConv blocks.
        middle_op (`str`, *optional*, defaults to `"fmb"`):
            Middle operator for the Neck blocks.
        out_dim (`int`, *optional*, defaults to 256):
            Output feature dimension of the Neck.
        image_size (`int`, *optional*, defaults to 512):
            Image size expected by the model.
    """

    model_type = "efficientvitsam_vision_model"

    def __init__(
        self,
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        block_list=None,
        expand_list=None,
        fewer_norm_list=None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1.0,
        middle_op="fmb",
        out_dim=256,
        image_size=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width_list = width_list
        self.depth_list = depth_list
        self.block_list = block_list
        self.expand_list = expand_list
        self.fewer_norm_list = fewer_norm_list
        self.in_channels = in_channels
        self.qkv_dim = qkv_dim
        self.norm = norm
        self.act_func = act_func
        self.fid_list = fid_list
        self.in_channel_list = in_channel_list
        self.head_width = head_width
        self.head_depth = head_depth
        self.expand_ratio = expand_ratio
        self.middle_op = middle_op
        self.out_dim = out_dim
        self.image_size = image_size


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamConfig(SamConfig):
    r"""
    [`EfficientViTSamConfig`] is the configuration class to store the configuration of a [`EfficientViTSamModel`]. It is
    used to instantiate an EfficientViT-SAM model according to the specified arguments, defining the vision encoder,
    prompt encoder and mask decoder configs.

    Configuration objects inherit from [`SamConfig`] and can be used to control the model outputs. Read the
    documentation from [`SamConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `EfficientViTSamVisionConfig`], *optional*):
            Dictionary of configuration options or an `EfficientViTSamVisionConfig` object used to initialize the
            vision encoder.
        prompt_encoder_config (Union[`dict`, `EfficientViTSamPromptEncoderConfig`], *optional*):
            Dictionary of configuration options or a `EfficientViTSamPromptEncoderConfig` object used to initialize the prompt
            encoder.
        mask_decoder_config (Union[`dict`, `EfficientViTSamMaskDecoderConfig`], *optional*):
            Dictionary of configuration options or a `EfficientViTSamMaskDecoderConfig` object used to initialize the mask
            decoder.
    """

    model_type = "efficientvitsam"
    sub_configs = {
        "prompt_encoder_config": EfficientViTSamPromptEncoderConfig,
        "mask_decoder_config": EfficientViTSamMaskDecoderConfig,
        "vision_config": EfficientViTSamVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = EfficientViTSamVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = EfficientViTSamVisionConfig()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = EfficientViTSamPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = EfficientViTSamPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = EfficientViTSamMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = EfficientViTSamMaskDecoderConfig()

        PreTrainedConfig.__post_init__(self)


__all__ = [
    "EfficientViTSamPromptEncoderConfig",
    "EfficientViTSamMaskDecoderConfig",
    "EfficientViTSamVisionConfig",
    "EfficientViTSamConfig",
]
