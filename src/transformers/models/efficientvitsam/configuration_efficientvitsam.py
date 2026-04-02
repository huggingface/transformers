# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..sam.configuration_sam import SamMaskDecoderConfig, SamPromptEncoderConfig


_VARIANT_DEFAULTS = {
    "l0": {
        "image_size": 512,
        "width_list": (32, 64, 128, 256, 512),
        "depth_list": (1, 1, 1, 4, 4),
        "block_list": ("res", "fmb", "fmb", "mb", "att"),
        "expand_list": (1.0, 4.0, 4.0, 4.0, 6.0),
        "fewer_norm_list": (False, False, False, True, True),
        "neck_hidden_sizes": (512, 256, 128),
        "neck_feature_names": ("stage4", "stage3", "stage2"),
        "head_depth": 4,
        "head_expand_ratio": 1.0,
        "head_middle_op": "fmb",
    },
    "l1": {
        "image_size": 512,
        "width_list": (32, 64, 128, 256, 512),
        "depth_list": (1, 1, 1, 6, 6),
        "block_list": ("res", "fmb", "fmb", "mb", "att"),
        "expand_list": (1.0, 4.0, 4.0, 4.0, 6.0),
        "fewer_norm_list": (False, False, False, True, True),
        "neck_hidden_sizes": (512, 256, 128),
        "neck_feature_names": ("stage4", "stage3", "stage2"),
        "head_depth": 8,
        "head_expand_ratio": 1.0,
        "head_middle_op": "fmb",
    },
    "l2": {
        "image_size": 512,
        "width_list": (32, 64, 128, 256, 512),
        "depth_list": (1, 2, 2, 8, 8),
        "block_list": ("res", "fmb", "fmb", "mb", "att"),
        "expand_list": (1.0, 4.0, 4.0, 4.0, 6.0),
        "fewer_norm_list": (False, False, False, True, True),
        "neck_hidden_sizes": (512, 256, 128),
        "neck_feature_names": ("stage4", "stage3", "stage2"),
        "head_depth": 12,
        "head_expand_ratio": 1.0,
        "head_middle_op": "fmb",
    },
    "xl0": {
        "image_size": 1024,
        "width_list": (32, 64, 128, 256, 512, 1024),
        "depth_list": (0, 1, 1, 2, 3, 3),
        "block_list": ("res", "fmb", "fmb", "fmb", "att@3", "att@3"),
        "expand_list": (1.0, 4.0, 4.0, 4.0, 4.0, 6.0),
        "fewer_norm_list": (False, False, False, False, True, True),
        "neck_hidden_sizes": (1024, 512, 256),
        "neck_feature_names": ("stage5", "stage4", "stage3"),
        "head_depth": 6,
        "head_expand_ratio": 4.0,
        "head_middle_op": "fmb",
    },
    "xl1": {
        "image_size": 1024,
        "width_list": (32, 64, 128, 256, 512, 1024),
        "depth_list": (1, 2, 2, 4, 6, 6),
        "block_list": ("res", "fmb", "fmb", "fmb", "att@3", "att@3"),
        "expand_list": (1.0, 4.0, 4.0, 4.0, 4.0, 6.0),
        "fewer_norm_list": (False, False, False, False, True, True),
        "neck_hidden_sizes": (1024, 512, 256),
        "neck_feature_names": ("stage5", "stage4", "stage3"),
        "head_depth": 12,
        "head_expand_ratio": 4.0,
        "head_middle_op": "fmb",
    },
}


@strict
class EfficientvitsamPromptEncoderConfig(SamPromptEncoderConfig):
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16


@strict
class EfficientvitsamMaskDecoderConfig(SamMaskDecoderConfig):
    pass


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam")
@strict
class EfficientvitsamVisionConfig(PreTrainedConfig):
    base_config_key = "vision_config"
    model_type = "efficientvitsam_vision_model"

    variant: str = "l0"
    num_channels: int = 3
    image_size: int = 512
    prompt_image_size: int = 1024
    output_channels: int = 256
    width_list: tuple[int, ...] | list[int] | None = None
    depth_list: tuple[int, ...] | list[int] | None = None
    block_list: tuple[str, ...] | list[str] | None = None
    expand_list: tuple[float, ...] | list[float] | None = None
    fewer_norm_list: tuple[bool, ...] | list[bool] | None = None
    qkv_dim: int = 32
    num_pos_feats: int = 128
    scale: float = 1.0
    norm: str = "bn2d"
    act_func: str = "gelu"
    neck_hidden_sizes: tuple[int, ...] | list[int] | None = None
    neck_feature_names: tuple[str, ...] | list[str] | None = None
    head_width: int = 256
    head_depth: int = 4
    head_expand_ratio: float = 1.0
    head_middle_op: str = "fmb"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        variant = self.variant.lower()
        if variant not in _VARIANT_DEFAULTS:
            raise ValueError(f"Unknown EfficientViT-SAM variant: {self.variant}")

        for key in [
            "width_list",
            "depth_list",
            "block_list",
            "expand_list",
            "fewer_norm_list",
            "neck_hidden_sizes",
            "neck_feature_names",
        ]:
            value = getattr(self, key)
            if isinstance(value, list):
                setattr(self, key, tuple(value))

        defaults = _VARIANT_DEFAULTS[variant]
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

        if not self.neck_hidden_sizes or not self.neck_feature_names:
            raise ValueError("`neck_hidden_sizes` and `neck_feature_names` must be defined.")

        if len(self.neck_hidden_sizes) != len(self.neck_feature_names):
            raise ValueError("`neck_hidden_sizes` and `neck_feature_names` must have the same length.")

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam")
@strict
class EfficientvitsamConfig(PreTrainedConfig):
    model_type = "efficientvitsam"
    sub_configs = {
        "prompt_encoder_config": EfficientvitsamPromptEncoderConfig,
        "mask_decoder_config": EfficientvitsamMaskDecoderConfig,
        "vision_config": EfficientvitsamVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = EfficientvitsamVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = EfficientvitsamVisionConfig()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = EfficientvitsamPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = EfficientvitsamPromptEncoderConfig(
                image_size=self.vision_config.prompt_image_size
            )

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = EfficientvitsamMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = EfficientvitsamMaskDecoderConfig()

        self.prompt_encoder_config.image_size = self.vision_config.prompt_image_size
        super().__post_init__(**kwargs)


__all__ = [
    "EfficientvitsamConfig",
    "EfficientvitsamMaskDecoderConfig",
    "EfficientvitsamPromptEncoderConfig",
    "EfficientvitsamVisionConfig",
]
