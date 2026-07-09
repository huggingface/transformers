# Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.
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
"""SAM3 model configuration"""

from huggingface_hub.dataclasses import strict

from transformers import CLIPTextConfig

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3ViTConfig(PreTrainedConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base frequency for RoPE.
    window_size (`int`, *optional*, defaults to 24):
        Window size for windowed attention.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
        Indexes of layers with global attention.
    pretrain_image_size (`int`, *optional*, defaults to 336):
        Pretrained model image size for position embedding initialization.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for hidden states.
    """

    base_config_key = "backbone_config"
    model_type = "sam3_vit_model"

    hidden_size: int = 1024
    intermediate_size: int = 4736
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 1008
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    rope_theta: float = 10000.0
    window_size: int = 24
    global_attn_indexes: list[int] | None = None
    layer_scale_init_value: float | None = None
    pretrain_image_size: int | list[int] | tuple[int, int] = 336
    hidden_dropout: float | int = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.global_attn_indexes is None:
            self.global_attn_indexes = [7, 15, 23, 31]


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3VisionConfig(PreTrainedConfig):
    r"""
    fpn_hidden_size (`int`, *optional*, defaults to 256):
        The hidden dimension of the FPN.
    backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[288, 288], [144, 144], [72, 72]]`):
        The spatial sizes (height, width) of the feature maps from the backbone at different scales.
    scale_factors (`list[float]`, *optional*, defaults to `[4.0, 2.0, 1.0, 0.5]`):
        Scale factors for FPN multi-scale features. List of scaling factors for each FPN level.
    """

    base_config_key = "vision_config"
    model_type = "sam3_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    backbone_config: dict | PreTrainedConfig | None = None
    fpn_hidden_size: int = 256
    backbone_feature_sizes: list | None = None
    scale_factors: list[float] | None = None
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.scale_factors = [4.0, 2.0, 1.0, 0.5] if self.scale_factors is None else self.scale_factors
        if self.backbone_feature_sizes is None:
            self.backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam3_vit_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = CONFIG_MAPPING["sam3_vit_model"]()

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the vision encoder."""
        return self.backbone_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to backbone."""
        self.backbone_config.image_size = value


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3GeometryEncoderConfig(PreTrainedConfig):
    r"""
    roi_size (`int`, *optional*, defaults to 7):
        ROI size for box pooling operations.
    """

    model_type = "sam3_geometry_encoder"

    hidden_size: int = 256
    num_layers: int = 3
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float | int = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6
    roi_size: int = 7
    initializer_range: float = 0.02


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3DETREncoderConfig(PreTrainedConfig):
    r"""
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for hidden states.
    """

    model_type = "sam3_detr_encoder"

    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float | int = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3DETRDecoderConfig(PreTrainedConfig):
    r"""
    num_queries (`int`, *optional*, defaults to 200):
        Number of object queries.
    """

    model_type = "sam3_detr_decoder"

    hidden_size: int = 256
    num_layers: int = 6
    num_queries: int = 200
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float | int = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3MaskDecoderConfig(PreTrainedConfig):
    r"""
    num_upsampling_stages (`int`, *optional*, defaults to 3):
        Number of upsampling stages in the pixel decoder (FPN).
    """

    model_type = "sam3_mask_decoder"

    hidden_size: int = 256
    num_upsampling_stages: int = 3
    layer_norm_eps: float = 1e-6
    dropout: float | int = 0.0
    num_attention_heads: int = 8
    initializer_range: float = 0.02


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3Config(PreTrainedConfig):
    r"""
    geometry_encoder_config (`dict` or `Sam3GeometryEncoderConfig`, *optional*):
        Configuration for the geometry encoder.
    detr_encoder_config (`dict` or `Sam3DETREncoderConfig`, *optional*):
        Configuration for the DETR encoder.
    detr_decoder_config (`dict` or `Sam3DETRDecoderConfig`, *optional*):
        Configuration for the DETR decoder.
    mask_decoder_config (`dict` or `Sam3MaskDecoderConfig`, *optional*):
        Configuration for the mask decoder.

    Example:
    ```python
    >>> from transformers import Sam3Config, Sam3Model

    >>> # Initializing a SAM3 configuration
    >>> configuration = Sam3Config()

    >>> # Initializing a model from the configuration
    >>> model = Sam3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "sam3"
    is_composition = True
    sub_configs = {
        "vision_config": Sam3VisionConfig,
        "text_config": CLIPTextConfig,
        "geometry_encoder_config": Sam3GeometryEncoderConfig,
        "detr_encoder_config": Sam3DETREncoderConfig,
        "detr_decoder_config": Sam3DETRDecoderConfig,
        "mask_decoder_config": Sam3MaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    geometry_encoder_config: dict | PreTrainedConfig | None = None
    detr_encoder_config: dict | PreTrainedConfig | None = None
    detr_decoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Sam3VisionConfig()
        if isinstance(self.vision_config, dict):
            self.vision_config = Sam3VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = CLIPTextConfig(
                **{
                    "vocab_size": 49408,
                    "hidden_size": 1024,
                    "intermediate_size": 4096,  # hidden_size * mlp_ratio (1024 * 4)
                    "projection_dim": 512,  # CLIP's internal projection dimension
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "max_position_embeddings": 32,
                    "hidden_act": "gelu",
                }
            )
        if isinstance(self.text_config, dict):
            self.text_config = CLIPTextConfig(**self.text_config)

        if self.geometry_encoder_config is None:
            self.geometry_encoder_config = Sam3GeometryEncoderConfig()
        if isinstance(self.geometry_encoder_config, dict):
            self.geometry_encoder_config = Sam3GeometryEncoderConfig(**self.geometry_encoder_config)

        if self.detr_encoder_config is None:
            self.detr_encoder_config = Sam3DETREncoderConfig()
        if isinstance(self.detr_encoder_config, dict):
            self.detr_encoder_config = Sam3DETREncoderConfig(**self.detr_encoder_config)

        if self.detr_decoder_config is None:
            self.detr_decoder_config = Sam3DETRDecoderConfig()
        if isinstance(self.detr_decoder_config, dict):
            self.detr_decoder_config = Sam3DETRDecoderConfig(**self.detr_decoder_config)

        if self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3MaskDecoderConfig()
        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3MaskDecoderConfig(**self.mask_decoder_config)

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the SAM3 model."""
        return self.vision_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to vision config."""
        self.vision_config.image_size = value


__all__ = [
    "Sam3Config",
    "Sam3ViTConfig",
    "Sam3VisionConfig",
    "Sam3GeometryEncoderConfig",
    "Sam3DETREncoderConfig",
    "Sam3DETRDecoderConfig",
    "Sam3MaskDecoderConfig",
]
