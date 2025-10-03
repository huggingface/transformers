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

from ...configuration_utils import PretrainedConfig


class Sam3VisionConfig(PretrainedConfig):
    r"""
    Configuration class for SAM3 Vision Encoder (ViT backbone).

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        image_size (`int`, *optional*, defaults to 1024):
            Expected input image size.
        patch_size (`int`, *optional*, defaults to 16):
            Size of image patches.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to QKV projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of mlp hidden dim to embedding dim.
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embeddings.
        use_rope (`bool`, *optional*, defaults to `True`):
            Whether to use RoPE (Rotary Position Embeddings).
        window_size (`int`, *optional*, defaults to 24):
            Window size for windowed attention.
        global_attn_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
            Indexes of layers with global attention.
        output_channels (`int`, *optional*, defaults to 256):
            Output dimensionality after the neck.
    """

    model_type = "sam3_vision"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        image_size=1008,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        qkv_bias=True,
        mlp_ratio=4.625,
        use_abs_pos=True,
        use_rope=True,
        window_size=24,
        global_attn_indexes=[7, 15, 23, 31],
        output_channels=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.use_abs_pos = use_abs_pos
        self.use_rope = use_rope
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.output_channels = output_channels


class Sam3TextConfig(PretrainedConfig):
    r"""
    Configuration class for SAM3 Text Encoder.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the text encoder.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (feed-forward) layer.
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            Maximum sequence length.
        output_dim (`int`, *optional*, defaults to 256):
            Output dimensionality for projection to common space.
    """

    model_type = "sam3_text"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        max_position_embeddings=77,
        output_dim=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.output_dim = output_dim


class Sam3GeometryEncoderConfig(PretrainedConfig):
    r"""
    Configuration class for SAM3 Geometry Encoder (encodes points, boxes, masks).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the geometry embeddings.
        num_layers (`int`, *optional*, defaults to 3):
            Number of transformer layers in geometry encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of feed-forward layer.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
    """

    model_type = "sam3_geometry_encoder"

    def __init__(
        self,
        hidden_size=256,
        num_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout


class Sam3TransformerConfig(PretrainedConfig):
    r"""
    Configuration class for SAM3 Transformer (encoder/decoder with fusion).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the transformer.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of feed-forward layer.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        num_queries (`int`, *optional*, defaults to 200):
            Number of object queries in decoder.
        num_feature_levels (`int`, *optional*, defaults to 1):
            Number of feature levels from the backbone.
    """

    model_type = "sam3_transformer"

    def __init__(
        self,
        hidden_size=256,
        encoder_layers=6,
        decoder_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        num_queries=200,
        num_feature_levels=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels


class Sam3SegmentationConfig(PretrainedConfig):
    r"""
    Configuration class for SAM3 Segmentation Head.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the segmentation head.
        num_upsampling_stages (`int`, *optional*, defaults to 3):
            Number of upsampling stages in pixel decoder.
        interpolation_mode (`str`, *optional*, defaults to `"nearest"`):
            Interpolation mode for upsampling.
    """

    model_type = "sam3_segmentation"

    def __init__(
        self,
        hidden_size=256,
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode


class Sam3Config(PretrainedConfig):
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

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        geometry_encoder_config=None,
        transformer_config=None,
        segmentation_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}
        if geometry_encoder_config is None:
            geometry_encoder_config = {}
        if transformer_config is None:
            transformer_config = {}
        if segmentation_config is None:
            segmentation_config = {}

        if isinstance(vision_config, Sam3VisionConfig):
            vision_config = vision_config.to_dict()
        if isinstance(text_config, Sam3TextConfig):
            text_config = text_config.to_dict()
        if isinstance(geometry_encoder_config, Sam3GeometryEncoderConfig):
            geometry_encoder_config = geometry_encoder_config.to_dict()
        if isinstance(transformer_config, Sam3TransformerConfig):
            transformer_config = transformer_config.to_dict()
        if isinstance(segmentation_config, Sam3SegmentationConfig):
            segmentation_config = segmentation_config.to_dict()

        self.vision_config = Sam3VisionConfig(**vision_config)
        self.text_config = Sam3TextConfig(**text_config)
        self.geometry_encoder_config = Sam3GeometryEncoderConfig(**geometry_encoder_config)
        self.transformer_config = Sam3TransformerConfig(**transformer_config)
        self.segmentation_config = Sam3SegmentationConfig(**segmentation_config)
        self.initializer_range = initializer_range


__all__ = [
    "Sam3Config",
    "Sam3VisionConfig",
    "Sam3TextConfig",
    "Sam3GeometryEncoderConfig",
    "Sam3TransformerConfig",
    "Sam3SegmentationConfig",
]
