# coding=utf-8
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


class Sam3ViTConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Vision Encoder (ViT backbone).

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers.
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
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to QKV projections.
        mlp_ratio (`float`, *optional*, defaults to 4.625):
            Ratio of mlp hidden dim to embedding dim.
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embeddings.
        tile_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to tile absolute position embeddings instead of interpolation.
        use_rope (`bool`, *optional*, defaults to `True`):
            Whether to use RoPE (Rotary Position Embeddings).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for RoPE.
        use_interp_rope (`bool`, *optional*, defaults to `True`):
            Whether to interpolate RoPE frequencies.
        window_size (`int`, *optional*, defaults to 24):
            Window size for windowed attention.
        global_attn_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
            Indexes of layers with global attention.
        use_rel_pos (`bool`, *optional*, defaults to `False`):
            Whether to use relative position embeddings (only for specific blocks).
        rel_pos_zero_init (`bool`, *optional*, defaults to `True`):
            Whether to zero-initialize relative position embeddings.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        layer_scale_init_value (`float`, *optional*):
            Initial value for layer scale. None means no layer scale.
        pretrain_img_size (`int`, *optional*, defaults to 336):
            Pretrained model image size for position embedding initialization.
        pretrain_use_cls_token (`bool`, *optional*, defaults to `True`):
            Whether pretrained model used cls token.
        retain_cls_token (`bool`, *optional*, defaults to `False`):
            Whether to retain cls token in the output.
        ln_pre (`bool`, *optional*, defaults to `True`):
            Whether to apply layer norm before transformer blocks.
        ln_post (`bool`, *optional*, defaults to `False`):
            Whether to apply layer norm after transformer blocks.
        output_channels (`int`, *optional*, defaults to 256):
            Output dimensionality after the neck.
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
        qkv_bias=True,
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
        self.qkv_bias = qkv_bias
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
    defaults will yield a similar configuration to that of SAM 3 architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*):
            Configuration for the vision backbone. This is used to instantiate the backbone using
            `AutoModel.from_config`.
        backbone_channel_list (`List[int]`, *optional*, defaults to `[768, 384, 192, 96]`):
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
    model_type = "sam3_vision_model"
    sub_configs = {
        "backbone_config": Sam3ViTConfig,
    }

    def __init__(
        self,
        backbone_config=None,
        backbone_channel_list=None,
        backbone_feature_sizes=None,
        fpn_hidden_size=256,
        fpn_kernel_size=2,
        fpn_stride=2,
        num_feature_levels=3,
        scale_factors=None,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        backbone_channel_list = [1024, 512, 256, 128] if backbone_channel_list is None else backbone_channel_list
        backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if backbone_feature_sizes is None else backbone_feature_sizes
        )
        scale_factors = [4.0, 2.0, 1.0, 0.5] if scale_factors is None else scale_factors

        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = backbone_config.get("model_type", "sam3_vit_model")
            backbone_config = Sam3ViTConfig(**backbone_config)
        elif isinstance(backbone_config, Sam3ViTConfig):
            pass
        elif backbone_config is None:
            backbone_config = Sam3ViTConfig()

        self.backbone_config = backbone_config

        # Neck
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_kernel_size = fpn_kernel_size
        self.fpn_stride = fpn_stride
        self.num_feature_levels = num_feature_levels
        self.scale_factors = scale_factors

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


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
        activation_function (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
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
        activation_function (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
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
        activation_function (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for layer normalization.
        box_rpb_mode (`str`, *optional*, defaults to `"log"`):
            Mode for box relative position bias ("log" or "linear").
        use_presence_token (`bool`, *optional*, defaults to `True`):
            Whether to use presence token for object detection.
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
        box_rpb_mode="log",
        use_presence_token=True,
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
        self.box_rpb_mode = box_rpb_mode
        self.use_presence_token = use_presence_token
        self.initializer_range = initializer_range


class Sam3MaskDecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Mask Decoder (pixel-level mask prediction).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the mask decoder.
        num_upsampling_stages (`int`, *optional*, defaults to 3):
            Number of upsampling stages in the pixel decoder (FPN).
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
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

    This is the main configuration class that combines all sub-configurations for the SAM3 model.

    Args:
        vision_config (`dict` or `Sam3VisionConfig`, *optional*):
            Configuration for the vision encoder.
        text_config (`dict` or `Sam3TextConfig`, *optional*):
            Configuration for the text encoder.
        geometry_encoder_config (`dict` or `Sam3GeometryEncoderConfig`, *optional*):
            Configuration for the geometry encoder.
        transformer_config (`dict` or `Sam3TransformerConfig`, *optional*):
            Configuration for the transformer.
        segmentation_config (`dict` or `Sam3SegmentationConfig`, *optional*):
            Configuration for the segmentation head.
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


__all__ = [
    "Sam3Config",
    "Sam3ViTConfig",
    "Sam3VisionConfig",
    "Sam3GeometryEncoderConfig",
    "Sam3DETREncoderConfig",
    "Sam3DETRDecoderConfig",
    "Sam3MaskDecoderConfig",
]
