# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


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


class Sam2MemoryAttentionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MemoryAttention`]. It is used to instantiate a SAM 2
    memory attention module according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        num_layers (`int`, *optional*, defaults to 4):
            The number of layers in the memory attention module.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the memory attention module.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            The dimension of the feedforward network in the memory attention module.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the memory attention module.
        rope_theta (`float`, *optional*, defaults to 10000):
            The Rope theta parameter.
        rope_feat_sizes (`Tuple[int, int]`, *optional*, defaults to `[32, 32]`):
            The feature sizes for the Rope positional encoding.
        rope_embedding_dim (`int`, *optional*, defaults to 256):
            The dimension of the Rope positional encoding.
        rope_num_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the Rope positional encoding.
        rope_downsample_rate (`int`, *optional*, defaults to 1):
            The downsample rate for the Rope positional encoding.
        rope_dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the Rope positional encoding.
        apply_pe_at_self_attn (`bool`, *optional*, defaults to `False`):
            Whether to apply positional encoding at the self-attention of the memory attention module.
        apply_pe_at_cross_attn_keys (`bool`, *optional*, defaults to `True`):
            Whether to apply positional encoding at the keys of the cross-attention of the memory attention module.
        apply_pe_at_cross_attn_queries (`bool`, *optional*, defaults to `False`):
            Whether to apply positional encoding at the queries of the cross-attention of the memory attention module.

    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        hidden_act="relu",
        dim_feedforward=2048,
        dropout=0.1,
        rope_theta=10000,
        rope_feat_sizes=[32, 32],
        rope_embedding_dim=256,
        rope_num_heads=1,
        rope_downsample_rate=1,
        rope_dropout=0.1,
        apply_pe_at_self_attn=False,
        apply_pe_at_cross_attn_keys=True,
        apply_pe_at_cross_attn_queries=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_act = hidden_act
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes
        self.rope_embedding_dim = rope_embedding_dim
        self.rope_num_heads = rope_num_heads
        self.rope_downsample_rate = rope_downsample_rate
        self.rope_dropout = rope_dropout
        self.apply_pe_at_self_attn = apply_pe_at_self_attn
        self.apply_pe_at_cross_attn_keys = apply_pe_at_cross_attn_keys
        self.apply_pe_at_cross_attn_queries = apply_pe_at_cross_attn_queries


class Sam2MemoryEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MemoryEncoder`]. It is used to instantiate a SAM 2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        output_channels (`int`, *optional*, defaults to 64):
            The number of output channels for the mask downsampler.
        mask_downsampler_embed_dim (`int`, *optional*, defaults to 256):
            The dimension of the mask downsampler embedding.
        mask_downsampler_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the mask downsampler.
        mask_downsampler_stride (`int`, *optional*, defaults to 2):
            The stride for the mask downsampler.
        mask_downsampler_padding (`int`, *optional*, defaults to 1):
            The padding for the mask downsampler.
        mask_downsampler_total_stride (`int`, *optional*, defaults to 16):
            The total stride for the mask downsampler.
        mask_downsampler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the mask downsampler.
        memory_fuser_num_layers (`int`, *optional*, defaults to 2):
            The number of layers in the memory fuser.
        memory_fuser_embed_dim (`int`, *optional*, defaults to 256):
            The dimension of the memory fuser embedding.
        memory_fuser_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size for the memory fuser.
        memory_fuser_padding (`int`, *optional*, defaults to 3):
            The padding for the memory fuser.
        memory_fuser_layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            The initial value for the layer scale in the memory fuser.
        memory_fuser_use_depthwise_conv (`bool`, *optional*, defaults to `True`):
            Whether to use a depthwise convolution for the memory fuser.
        memory_fuser_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the memory fuser.

    """

    def __init__(
        self,
        hidden_size=256,
        output_channels=64,
        mask_downsampler_embed_dim=256,
        mask_downsampler_kernel_size=3,
        mask_downsampler_stride=2,
        mask_downsampler_padding=1,
        mask_downsampler_total_stride=16,
        mask_downsampler_hidden_act="gelu",
        memory_fuser_num_layers=2,
        memory_fuser_embed_dim=256,
        memory_fuser_kernel_size=7,
        memory_fuser_padding=3,
        memory_fuser_layer_scale_init_value=1e-6,
        memory_fuser_use_depthwise_conv=True,
        memory_fuser_hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            mask_downsampler_stride
            ** int(math.log2(mask_downsampler_total_stride) // math.log2(mask_downsampler_stride))
            == mask_downsampler_total_stride
        )

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.mask_downsampler_embed_dim = mask_downsampler_embed_dim
        self.mask_downsampler_kernel_size = mask_downsampler_kernel_size
        self.mask_downsampler_stride = mask_downsampler_stride
        self.mask_downsampler_padding = mask_downsampler_padding
        self.mask_downsampler_total_stride = mask_downsampler_total_stride
        self.mask_downsampler_hidden_act = mask_downsampler_hidden_act
        self.memory_fuser_num_layers = memory_fuser_num_layers
        self.memory_fuser_embed_dim = memory_fuser_embed_dim
        self.memory_fuser_kernel_size = memory_fuser_kernel_size
        self.memory_fuser_padding = memory_fuser_padding
        self.memory_fuser_layer_scale_init_value = memory_fuser_layer_scale_init_value
        self.memory_fuser_use_depthwise_conv = memory_fuser_use_depthwise_conv
        self.memory_fuser_hidden_act = memory_fuser_hidden_act


class Sam2MaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MaskDecoder`]. It is used to instantiate a SAM 2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of multimask outputs.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the SAM mask decoder.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The depth of the IoU head.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the IoU head.
        iou_prediction_use_sigmoid (`bool`, *optional*, defaults to `True`):
            Whether to use a sigmoid function for the IoU prediction.
        dynamic_multimask_via_stability (`bool`, *optional*, defaults to `True`):
            Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (`float`, *optional*, defaults to 0.05):
            The stability delta for the dynamic multimask.
        dynamic_multimask_stability_thresh (`float`, *optional*, defaults to 0.98):
            The stability threshold for the dynamic multimask.
        use_multimask_token_for_object_pointer (`bool`, *optional*, defaults to `True`):
            Whether to use the multimask token for the object pointer.
        feed_forward_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the feed-forward network.
        two_way_transformer_depth (`int`, *optional*, defaults to 2):
            The depth of the two-way transformer.
        two_way_transformer_embedding_dim (`int`, *optional*, defaults to 256):
            The embedding dimension of the two-way transformer.
        two_way_transformer_num_heads (`int`, *optional*, defaults to 8):
            The number of attention heads in the two-way transformer.
        two_way_transformer_mlp_dim (`int`, *optional*, defaults to 2048):
            The dimension of the feed-forward network in the two-way transformer.
        two_way_transformer_activation (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the two-way transformer.
        two_way_transformer_attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsample rate of the attention in the two-way transformer.

    """

    def __init__(
        self,
        hidden_size=256,
        num_multimask_outputs=3,
        hidden_act="gelu",
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        iou_prediction_use_sigmoid=True,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        use_multimask_token_for_object_pointer=True,
        feed_forward_hidden_act="relu",
        two_way_transformer_depth=2,
        two_way_transformer_embedding_dim=256,
        two_way_transformer_num_heads=8,
        two_way_transformer_mlp_dim=2048,
        two_way_transformer_activation="relu",
        two_way_transformer_attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert hidden_size == two_way_transformer_embedding_dim

        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.hidden_act = hidden_act
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self.use_multimask_token_for_object_pointer = use_multimask_token_for_object_pointer
        self.feed_forward_hidden_act = feed_forward_hidden_act

        # TwoWayTransformer configuration
        self.two_way_transformer_depth = two_way_transformer_depth
        self.two_way_transformer_embedding_dim = two_way_transformer_embedding_dim
        self.two_way_transformer_num_heads = two_way_transformer_num_heads
        self.two_way_transformer_mlp_dim = two_way_transformer_mlp_dim
        self.two_way_transformer_activation = two_way_transformer_activation
        self.two_way_transformer_attention_downsample_rate = two_way_transformer_attention_downsample_rate


class Sam2ImageEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2ImageEncoder`]. It is used to instantiate a SAM
    image encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM 2 Hiera-B+
    [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 96):
            The hidden dimension of the image encoder.
        num_heads (`int`, *optional*, defaults to 1):
            Initial number of attention heads.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the image.
        image_size (`int`, *optional*, defaults to 1024):
            The size of the image.
        patch_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size of the patch.
        patch_stride (`int`, *optional*, defaults to 4):
            The stride of the patch.
        patch_padding (`int`, *optional*, defaults to 3):
            The padding of the patch.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The stochastic depth rate.
        q_pool (`int`, *optional*, defaults to 3):
            The number of q_pool stages.
        q_stride (`Tuple[int, int]`, *optional*, defaults to `(2, 2)`):
            The downsample stride between stages.
        stages (`Tuple[int, ...]`, *optional*, defaults to `(1, 2, 7, 2)`):
            The number of blocks per stage.
        dim_mul (`float`, *optional*, defaults to 2.0):
            The dimension multiplier factor at stage shift.
        head_mul (`float`, *optional*, defaults to 2.0):
            The head multiplier factor at stage shift.
        window_positional_embedding_background_size (`Tuple[int, int]`, *optional*, defaults to `(7, 7)`):
            The window size per stage when not using global attention.
        window_spec (`Tuple[int, ...]`, *optional*, defaults to `(8, 4, 14, 7)`):
            The window specifications for each stage.
        global_attention_blocks (`Tuple[int, ...]`, *optional*, defaults to `(5, 7, 9)`):
            The blocks where global attention is used.
        backbone_channel_list (`List[int]`, *optional*, defaults to `[768, 384, 192, 96]`):
            The list of channel dimensions for the backbone.
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
        fpn_interpolation_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation model for the FPN.
        fuse_type (`str`, *optional*, defaults to `"sum"`):
            The type of fusion to use in the neck.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.

    """

    def __init__(
        self,
        hidden_size=96,
        num_heads=1,
        num_channels=3,
        image_size=1024,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=(2, 2),
        stages=(1, 2, 7, 2),
        dim_mul=2.0,
        head_mul=2.0,
        window_positional_embedding_background_size=(7, 7),
        window_spec=(8, 4, 14, 7),
        global_attention_blocks=(5, 7, 9),
        backbone_channel_list=[768, 384, 192, 96],
        backbone_feature_sizes=[[256, 256], [128, 128], [64, 64]],
        fpn_hidden_size=256,
        fpn_kernel_size=1,
        fpn_stride=1,
        fpn_padding=0,
        fpn_top_down_levels=[2, 3],
        fpn_interpolation_mode="nearest",
        num_feature_levels=3,
        fuse_type="sum",
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(stages) == len(window_spec) == len(backbone_channel_list)
        assert fuse_type in ["sum", "average"]

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_positional_embedding_background_size = window_positional_embedding_background_size
        self.window_spec = window_spec
        self.global_attention_blocks = global_attention_blocks

        # Neck
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_kernel_size = fpn_kernel_size
        self.fpn_stride = fpn_stride
        self.fpn_padding = fpn_padding
        self.fpn_top_down_levels = fpn_top_down_levels
        self.fpn_interpolation_mode = fpn_interpolation_mode
        self.fuse_type = fuse_type
        self.num_feature_levels = num_feature_levels

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps


class Sam2Config(PretrainedConfig):
    r"""
    [`Sam2Config`] is the configuration class to store the configuration of a [`Sam2Model`]. It is used to instantiate a
    SAM2 model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2 Hiera-B+
    [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_encoder_config (Union[`dict`, `Sam2ImageEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2ImageEncoderConfig`].
        prompt_encoder_config (Union[`dict`, `Sam2PromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2PromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `Sam2MaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MaskDecoderConfig`].
        memory_attention_config (Union[`dict`, `Sam2MemoryAttentionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MemoryAttentionConfig`].
        memory_encoder_config (Union[`dict`, `Sam2MemoryEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MemoryEncoderConfig`].

        initializer_range (`float`, *optional*, defaults to 0.02): std for parameter initialization
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam2ImageEncoderConfig,
    ...     Sam2PromptEncoderConfig,
    ...     Sam2MaskDecoderConfig,
    ...     Sam2MemoryAttentionConfig,
    ...     Sam2MemoryEncoderConfig,
    ...     Sam2Model,
    ... )

    >>> # Initializing a Sam2Config with `"facebook/hiera-base-plus"` style configuration
    >>> configuration = Sam2config()

    >>> # Initializing a Sam2Model (with random weights) from the `"facebook/sam-vit-huge"` style configuration
    >>> model = Sam2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam2Config from a Sam2ImageEncoderConfig, Sam2MemoryAttentionConfig, and Sam2MemoryEncoderConfig

    >>> # Initializing SAM2 image encoder, memory attention, and memory encoder configurations
    >>> image_encoder_config = Sam2ImageEncoderConfig()
    >>> prompt_encoder_config = Sam2PromptEncoderConfig()
    >>> mask_decoder_config = Sam2MaskDecoderConfig()
    >>> memory_attention_config = Sam2MemoryAttentionConfig()
    >>> memory_encoder_config = Sam2MemoryEncoderConfig()

    >>> config = Sam2Config(image_encoder_config, prompt_encoder_config, mask_decoder_config, memory_attention_config, memory_encoder_config)
    ```"""

    model_type = "sam2"

    def __init__(
        self,
        image_encoder_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        memory_attention_config=None,
        memory_encoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        image_encoder_config = image_encoder_config if image_encoder_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}
        memory_attention_config = memory_attention_config if memory_attention_config is not None else {}
        memory_encoder_config = memory_encoder_config if memory_encoder_config is not None else {}

        if isinstance(image_encoder_config, Sam2ImageEncoderConfig):
            image_encoder_config = image_encoder_config.to_dict()
        if isinstance(prompt_encoder_config, Sam2PromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam2MaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()
        if isinstance(memory_attention_config, Sam2MemoryAttentionConfig):
            memory_attention_config = memory_attention_config.to_dict()
        if isinstance(memory_encoder_config, Sam2MemoryEncoderConfig):
            memory_encoder_config = memory_encoder_config.to_dict()

        self.image_encoder_config = Sam2ImageEncoderConfig(**image_encoder_config)
        self.prompt_encoder_config = Sam2PromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam2MaskDecoderConfig(**mask_decoder_config)
        self.memory_attention_config = Sam2MemoryAttentionConfig(**memory_attention_config)
        self.memory_encoder_config = Sam2MemoryEncoderConfig(**memory_encoder_config)

        self.initializer_range = initializer_range
        self.num_maskmem = 7  # default 1 input frame + 6 previous frames
        self.image_size = 1024
        self.backbone_stride = 16  # stride of the image backbone output
        self.sigmoid_scale_for_mem_enc = 20.0  # scale factor for mask sigmoid prob
        self.sigmoid_bias_for_mem_enc = -10.0  # bias factor for mask sigmoid prob
        # During evaluation whether to binarize the sigmoid mask logits on interacted frames with clicks
        self.binarize_mask_from_pts_for_mem_enc = True
        self.use_mask_input_as_output_without_sam = True  # on frames with mask input whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        self.max_cond_frames_in_attn = -1
        self.enable_occlusion_spatial_embedding = True
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        self.multimask_output_in_sam = True
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both meaning that only the first click gives multimask output; also note that a box counts as two points)
        self.multimask_min_pt_num = 0
        self.multimask_max_pt_num = 1
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        self.multimask_output_for_tracking = True
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_object_pointers_in_encoder=True and multimask_output_for_tracking=True
        self.use_multimask_token_for_object_pointer = True
        # whether to use sigmoid to restrict ious prediction to [0-1]
        self.iou_prediction_use_sigmoid = True
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1 the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames plus the last frame.
        self.memory_temporal_stride_for_eval = 1
        # if `add_all_frames_to_correct_as_cond` is True we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False we conditioning frame list to only use those initial conditioning frames
        self.add_all_frames_to_correct_as_cond = False
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        self.non_overlap_masks_for_mem_enc = False
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        self.use_object_pointers_in_encoder = True
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_object_pointers_in_encoder=True`)
        self.max_object_pointers_in_encoder = 16
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_object_pointers_in_encoder=True`)
        self.enable_temporal_pos_encoding_for_object_pointers = True
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_object_pointers_in_encoder=True` and `enable_temporal_pos_encoding_for_object_pointers=True`)
        self.project_temporal_pos_encoding_in_object_pointers = True
        self.preserve_temporal_direction_in_object_pointers = True

        # Video inference specific parameters
        self.fill_hole_area = 8  # area threshold for filling holes in masks
        self.non_overlap_masks = False  # whether to apply non-overlapping constraints on output masks


__all__ = [
    "Sam2Config",
    "Sam2ImageEncoderConfig",
    "Sam2PromptEncoderConfig",
    "Sam2MaskDecoderConfig",
    "Sam2MemoryAttentionConfig",
    "Sam2MemoryEncoderConfig",
]
