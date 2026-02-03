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

from transformers import CLIPTextConfig

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class Sam3ViTConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Vision Encoder (ViT backbone).

    Instantiating a configuration defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers.
        intermediate_size (`int`, *optional*, defaults to 4736):
            Dimensionality of the feedforward (MLP) layers.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        image_size (`int`, *optional*, defaults to 1008):
            Expected input image size.
        patch_size (`int`, *optional*, defaults to 14):
            Size of image patches.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for RoPE.
        window_size (`int`, *optional*, defaults to 24):
            Window size for windowed attention.
        global_attn_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
            Indexes of layers with global attention.
        layer_scale_init_value (`float`, *optional*):
            Initial value for layer scale. None means no layer scale.
        pretrain_image_size (`int`, *optional*, defaults to 336):
            Pretrained model image size for position embedding initialization.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    base_config_key = "backbone_config"
    model_type = "sam3_vit_model"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4736,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        image_size=1008,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        rope_theta=10000.0,
        window_size=24,
        global_attn_indexes=None,
        layer_scale_init_value=None,
        pretrain_image_size=336,
        hidden_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if global_attn_indexes is None:
            global_attn_indexes = [7, 15, 23, 31]

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.layer_scale_init_value = layer_scale_init_value
        self.pretrain_image_size = pretrain_image_size
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range


class Sam3VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam3VisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `Sam3ViTConfig()`):
            Configuration for the vision backbone. This is used to instantiate the backbone using
            `AutoModel.from_config`.
        fpn_hidden_size (`int`, *optional*, defaults to 256):
            The hidden dimension of the FPN.
        backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[288, 288], [144, 144], [72, 72]]`):
            The spatial sizes (height, width) of the feature maps from the backbone at different scales.
        scale_factors (`list[float]`, *optional*, defaults to `[4.0, 2.0, 1.0, 0.5]`):
            Scale factors for FPN multi-scale features. List of scaling factors for each FPN level.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "vision_config"
    model_type = "sam3_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    def __init__(
        self,
        backbone_config=None,
        fpn_hidden_size=256,
        backbone_feature_sizes=None,
        scale_factors=None,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        scale_factors = [4.0, 2.0, 1.0, 0.5] if scale_factors is None else scale_factors
        if backbone_feature_sizes is None:
            backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]

        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = backbone_config.get("model_type", "sam3_vit_model")
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif backbone_config is None:
            backbone_config = CONFIG_MAPPING["sam3_vit_model"]()

        self.backbone_config = backbone_config

        # Neck
        self.fpn_hidden_size = fpn_hidden_size
        self.scale_factors = scale_factors
        self.backbone_feature_sizes = backbone_feature_sizes

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the vision encoder."""
        return self.backbone_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to backbone."""
        self.backbone_config.image_size = value


class Sam3GeometryEncoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Geometry Encoder.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        num_layers (`int`, *optional*, defaults to 3):
            Number of transformer encoder layers for processing geometry prompts.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the geometry encoder.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the feedforward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        roi_size (`int`, *optional*, defaults to 7):
            ROI size for box pooling operations.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_geometry_encoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        hidden_act="relu",
        hidden_dropout=0.0,
        layer_norm_eps=1e-6,
        roi_size=7,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.roi_size = roi_size
        self.initializer_range = initializer_range


class Sam3DETREncoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 DETR Encoder (vision-text fusion encoder).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        num_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the feedforward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_detr_encoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        hidden_act="relu",
        hidden_dropout=0.0,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class Sam3DETRDecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 DETR Decoder (object query decoder).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the decoder layers.
        num_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        num_queries (`int`, *optional*, defaults to 200):
            Number of object queries.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the feedforward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_detr_decoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=6,
        num_queries=200,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        hidden_act="relu",
        hidden_dropout=0.0,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class Sam3MaskDecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Mask Decoder (pixel-level mask prediction).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the mask decoder.
        num_upsampling_stages (`int`, *optional*, defaults to 3):
            Number of upsampling stages in the pixel decoder (FPN).
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for prompt cross-attention.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for prompt cross-attention.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_mask_decoder"

    def __init__(
        self,
        hidden_size=256,
        num_upsampling_stages=3,
        layer_norm_eps=1e-6,
        dropout=0.0,
        num_attention_heads=8,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_upsampling_stages = num_upsampling_stages
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range


class Sam3Config(PreTrainedConfig):
    r"""
    Configuration class to store the configuration of a [`Sam3Model`].

    Instantiating a configuration defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    This is the main configuration class that combines all sub-configurations for the SAM3 model.

    Args:
        vision_config (`dict` or `Sam3VisionConfig`, *optional*):
            Configuration for the vision encoder.
        text_config (`dict` or `Sam3TextConfig`, *optional*):
            Configuration for the text encoder.
        geometry_encoder_config (`dict` or `Sam3GeometryEncoderConfig`, *optional*):
            Configuration for the geometry encoder.
        detr_encoder_config (`dict` or `Sam3DETREncoderConfig`, *optional*):
            Configuration for the DETR encoder.
        detr_decoder_config (`dict` or `Sam3DETRDecoderConfig`, *optional*):
            Configuration for the DETR decoder.
        mask_decoder_config (`dict` or `Sam3MaskDecoderConfig`, *optional*):
            Configuration for the mask decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.

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

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        geometry_encoder_config=None,
        detr_encoder_config=None,
        detr_decoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        # Vision config
        if vision_config is None:
            vision_config = {}
        if isinstance(vision_config, dict):
            self.vision_config = Sam3VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Text config (CLIPTextModelWithProjection defaults)
        if text_config is None:
            text_config = {
                "vocab_size": 49408,
                "hidden_size": 1024,
                "intermediate_size": 4096,  # hidden_size * mlp_ratio (1024 * 4)
                "projection_dim": 512,  # CLIP's internal projection dimension
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "max_position_embeddings": 32,
                "hidden_act": "gelu",
            }
        if isinstance(text_config, dict):
            self.text_config = CLIPTextConfig(**text_config)
        else:
            self.text_config = text_config

        # Geometry encoder config
        if geometry_encoder_config is None:
            geometry_encoder_config = {}
        if isinstance(geometry_encoder_config, dict):
            self.geometry_encoder_config = Sam3GeometryEncoderConfig(**geometry_encoder_config)
        else:
            self.geometry_encoder_config = geometry_encoder_config

        # DETR encoder config
        if detr_encoder_config is None:
            detr_encoder_config = {}
        if isinstance(detr_encoder_config, dict):
            self.detr_encoder_config = Sam3DETREncoderConfig(**detr_encoder_config)
        else:
            self.detr_encoder_config = detr_encoder_config

        # DETR decoder config
        if detr_decoder_config is None:
            detr_decoder_config = {}
        if isinstance(detr_decoder_config, dict):
            self.detr_decoder_config = Sam3DETRDecoderConfig(**detr_decoder_config)
        else:
            self.detr_decoder_config = detr_decoder_config

        # Mask decoder config
        if mask_decoder_config is None:
            mask_decoder_config = {}
        if isinstance(mask_decoder_config, dict):
            self.mask_decoder_config = Sam3MaskDecoderConfig(**mask_decoder_config)
        else:
            self.mask_decoder_config = mask_decoder_config

        self.initializer_range = initializer_range
        super().__init__(**kwargs)

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
