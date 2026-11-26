# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""LLaVA-OneVision-2 model configuration."""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct")
@strict
class LlavaOnevision2VisionConfig(PreTrainedConfig):
    r"""
    spatial_merge_size (`int`, *optional*, defaults to 2):
        Number of patches merged along each spatial dimension.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output dimension of the patch merger, equal to the language model hidden size.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period of the three-dimensional vision rotary embeddings.
    frame_windows_size (`int`, *optional*, defaults to 4):
        Maximum number of consecutive frames attending jointly within one grid entry.
    layer_norm_type (`str`, *optional*, defaults to `"layer_norm"`):
        Normalization used before the encoder and inside its blocks.
    use_patch_position_encoding (`bool`, *optional*, defaults to `False`):
        Whether the merger adds learned absolute spatial position embeddings.
    patch_position_encoding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of the optional merger position embeddings.
    """

    model_type = "llava_onevision2_vision"
    base_config_key = "vision_config"
    attribute_map = {"num_heads": "num_attention_heads"}

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    out_hidden_size: int = 4096
    frame_windows_size: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    layer_norm_type: Literal["layer_norm", "rms_norm"] = "layer_norm"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    use_patch_position_encoding: bool = False
    patch_position_encoding_type: Literal["absolute"] = "absolute"
    max_position_embeddings: int = 8192

    def __post_init__(self, **kwargs):
        if kwargs.get("model_type") == "onevision_encoder":
            kwargs["model_type"] = type(self).model_type
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.temporal_patch_size != 1:
            raise ValueError("OV2 requires temporal_patch_size=1")
        if self.hidden_size % self.num_attention_heads or (self.hidden_size // self.num_attention_heads) % 32:
            raise ValueError("Vision head dimension must be divisible by 32 for the 4:6:6 rotary split")


@auto_docstring(checkpoint="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct")
@strict
class LlavaOnevision2Config(PreTrainedConfig):
    model_type = "llava_onevision2"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(config_class=LlavaOnevision2VisionConfig),
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen3",
            init_kwargs={
                "num_hidden_layers": 36,
                "intermediate_size": 12288,
                "num_key_value_heads": 8,
                "max_position_embeddings": 262144,
                "rope_parameters": {"rope_type": "default", "rope_theta": 8000000},
            },
        ),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def validate_architecture(self):
        super().validate_architecture()
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("vision_config.out_hidden_size must match text_config.hidden_size")


__all__ = ["LlavaOnevision2Config", "LlavaOnevision2VisionConfig"]
