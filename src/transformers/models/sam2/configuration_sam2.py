# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""SAM2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class Sam2HieraDetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2HieraDetModel`]. It is used to instantiate
    a HieraDet model as defined in the original sam2 repo according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 96):
            The hidden dimension of the image encoder.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the image.
        image_size (`list[int]`, *optional*, defaults to `[1024, 1024]`):
            The size of the image.
        patch_kernel_size (`list[int]`, *optional*, defaults to `[7, 7]`):
            The kernel size of the patch.
        patch_stride (`list[int]`, *optional*, defaults to `[4, 4]`):
            The stride of the patch.
        patch_padding (`list[int]`, *optional*, defaults to `[3, 3]`):
            The padding of the patch.
        query_stride (`list[int]`, *optional*, defaults to `[2, 2]`):
            The downsample stride between stages.
        window_positional_embedding_background_size (`list[int]`, *optional*, defaults to `[7, 7]`):
            The window size per stage when not using global attention.
        num_query_pool_stages (`int`, *optional*, defaults to 3):
            The number of query pool stages.
        blocks_per_stage (`list[int]`, *optional*, defaults to `[1, 2, 7, 2]`):
            The number of blocks per stage.
        embed_dim_per_stage (`list[int]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The embedding dimension per stage.
        num_attention_heads_per_stage (`list[int]`, *optional*, defaults to `[1, 2, 4, 8]`):
            The number of attention heads per stage.
        window_size_per_stage (`list[int]`, *optional*, defaults to `[8, 4, 14, 7]`):
            The window size per stage.
        global_attention_blocks (`list[int]`, *optional*, defaults to `[5, 7, 9]`):
            The blocks where global attention is used.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the MLP hidden dimension to the embedding dimension.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "backbone_config"
    model_type = "sam2_hiera_det_model"

    def __init__(
        self,
        hidden_size=96,
        num_attention_heads=1,
        num_channels=3,
        image_size=None,
        patch_kernel_size=None,
        patch_stride=None,
        patch_padding=None,
        query_stride=None,
        window_positional_embedding_background_size=None,
        num_query_pool_stages=3,
        blocks_per_stage=None,
        embed_dim_per_stage=None,
        num_attention_heads_per_stage=None,
        window_size_per_stage=None,
        global_attention_blocks=None,
        mlp_ratio=4.0,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        image_size = image_size if image_size is not None else [1024, 1024]
        patch_kernel_size = patch_kernel_size if patch_kernel_size is not None else [7, 7]
        patch_stride = patch_stride if patch_stride is not None else [4, 4]
        patch_padding = patch_padding if patch_padding is not None else [3, 3]
        query_stride = query_stride if query_stride is not None else [2, 2]
        window_positional_embedding_background_size = (
            window_positional_embedding_background_size
            if window_positional_embedding_background_size is not None
            else [7, 7]
        )
        blocks_per_stage = blocks_per_stage if blocks_per_stage is not None else [1, 2, 7, 2]
        embed_dim_per_stage = embed_dim_per_stage if embed_dim_per_stage is not None else [96, 192, 384, 768]
        num_attention_heads_per_stage = (
            num_attention_heads_per_stage if num_attention_heads_per_stage is not None else [1, 2, 4, 8]
        )
        window_size_per_stage = window_size_per_stage if window_size_per_stage is not None else [8, 4, 14, 7]
        global_attention_blocks = global_attention_blocks if global_attention_blocks is not None else [5, 7, 9]

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.query_stride = query_stride
        self.window_positional_embedding_background_size = window_positional_embedding_background_size
        self.num_query_pool_stages = num_query_pool_stages
        self.blocks_per_stage = blocks_per_stage
        self.embed_dim_per_stage = embed_dim_per_stage
        self.num_attention_heads_per_stage = num_attention_heads_per_stage
        self.window_size_per_stage = window_size_per_stage
        self.global_attention_blocks = global_attention_blocks
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class Sam2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2VisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PretrainedConfig"]`, *optional*):
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
    model_type = "sam2_vision_model"
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

        backbone_channel_list = [768, 384, 192, 96] if backbone_channel_list is None else backbone_channel_list
        backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if backbone_feature_sizes is None else backbone_feature_sizes
        )
        fpn_top_down_levels = [2, 3] if fpn_top_down_levels is None else fpn_top_down_levels

        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = backbone_config.get("model_type", "sam2_hiera_det_model")
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif isinstance(backbone_config, Sam2HieraDetConfig):
            backbone_config = backbone_config
        elif backbone_config is None:
            backbone_config = Sam2HieraDetConfig()

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


class Sam2PromptEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2PromptEncoder`]. The [`Sam2PromptEncoder`]
    module is used to encode the input 2D points and bounding boxes.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        scale (`float`, *optional*, defaults to 1):
            The scale factor for the prompt encoder.
    """

    base_config_key = "prompt_encoder_config"

    def __init__(
        self,
        hidden_size=256,
        image_size=1024,
        patch_size=16,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        scale=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale


class Sam2MaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MaskDecoder`]. It is used to instantiate a SAM2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the SAM2 mask decoder.
        mlp_dim (`int`, *optional*, defaults to 2048):
            The dimension of the MLP in the two-way transformer.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            The number of hidden layers in the two-way transformer.
        num_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads in the two-way transformer.
        attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsample rate for the attention layers.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of multimask outputs.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The depth of the IoU head.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the IoU head.
        dynamic_multimask_via_stability (`bool`, *optional*, defaults to `True`):
            Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (`float`, *optional*, defaults to 0.05):
            The stability delta for the dynamic multimask.
        dynamic_multimask_stability_thresh (`float`, *optional*, defaults to 0.98):
            The stability threshold for the dynamic multimask.

    """

    base_config_key = "mask_decoder_config"

    def __init__(
        self,
        hidden_size=256,
        hidden_act="gelu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.hidden_act = hidden_act
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # TwoWayTransformer configuration
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate


class Sam2Config(PretrainedConfig):
    r"""
    [`Sam2Config`] is the configuration class to store the configuration of a [`Sam2Model`]. It is used to instantiate a
    SAM2 model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `Sam2VisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2VisionConfig`].
        prompt_encoder_config (Union[`dict`, `Sam2PromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2PromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `Sam2MaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MaskDecoderConfig`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for parameter initialization.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam2VisionConfig,
    ...     Sam2PromptEncoderConfig,
    ...     Sam2MaskDecoderConfig,
    ...     Sam2Model,
    ... )

    >>> # Initializing a Sam2Config with `"facebook/sam2.1_hiera_tiny"` style configuration
    >>> configuration = Sam2config()

    >>> # Initializing a Sam2Model (with random weights) from the `"facebook/sam2.1_hiera_tiny"` style configuration
    >>> model = Sam2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam2Config from a Sam2VisionConfig, Sam2PromptEncoderConfig, and Sam2MaskDecoderConfig

    >>> # Initializing SAM2 vision encoder, memory attention, and memory encoder configurations
    >>> vision_config = Sam2VisionConfig()
    >>> prompt_encoder_config = Sam2PromptEncoderConfig()
    >>> mask_decoder_config = Sam2MaskDecoderConfig()

    >>> config = Sam2Config(vision_config, prompt_encoder_config, mask_decoder_config)
    ```"""

    model_type = "sam2"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam2PromptEncoderConfig,
        "mask_decoder_config": Sam2MaskDecoderConfig,
    }

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vision_config = vision_config if vision_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "sam2_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            vision_config = vision_config
        if isinstance(prompt_encoder_config, Sam2PromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam2MaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        self.vision_config = vision_config
        self.prompt_encoder_config = Sam2PromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam2MaskDecoderConfig(**mask_decoder_config)

        self.initializer_range = initializer_range


__all__ = [
    "Sam2Config",
    "Sam2HieraDetConfig",
    "Sam2VisionConfig",
    "Sam2PromptEncoderConfig",
    "Sam2MaskDecoderConfig",
]
