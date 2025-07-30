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

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from transformers.models.sam2.configuration_sam2 import (
    Sam2MaskDecoderConfig,
    Sam2PromptEncoderConfig,
)
from transformers.models.sam2.modeling_sam2 import (
    Sam2Attention,
    Sam2FeedForward,
    Sam2LayerNorm,
    Sam2MemoryAttention,
    Sam2MemoryEncoder,
    Sam2MemoryFuserCXBlock,
    Sam2Model,
    Sam2PreTrainedModel,
    Sam2RoPEAttention,
    Sam2TwoWayAttentionBlock,
    Sam2VideoInferenceSession,
    Sam2VideoModel,
    Sam2VisionEncoderOutput,
    Sam2VisionModel,
    Sam2VisionRotaryEmbedding,
    eager_attention_forward,
    get_1d_sine_pe,
    rotate_half,
    window_partition,
)
from transformers.utils.generic import OutputRecorder, TransformersKwargs, check_model_inputs

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    auto_docstring,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel


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
        fpn_interpolation_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation model for the FPN.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels from the FPN to use.
        fuse_type (`str`, *optional*, defaults to `"sum"`):
            The type of fusion to use in the neck.
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
        backbone_channel_list=[384, 192, 96, 48],
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
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = (
                backbone_config["model_type"] if "model_type" in backbone_config else "hiera"
            )
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif isinstance(backbone_config, AutoConfig):
            backbone_config = backbone_config
        elif backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(
                "timm/repvit_m1.dist_in1k",
                model_args={"in_chans": 3, "features_only": True, "out_indices": (0, 1, 2, 3)},
            )

        self.backbone_config = backbone_config

        assert fuse_type in ["sum", "average"]
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
        self.initializer_range = initializer_range


class EdgeTamPromptEncoderConfig(Sam2PromptEncoderConfig):
    pass


class EdgeTamMaskDecoderConfig(Sam2MaskDecoderConfig):
    pass


class EdgeTamConfig(PretrainedConfig):
    r"""
    [`EdgeTamConfig`] is the configuration class to store the configuration of a [`EdgeTamModel`]. It is used to instantiate a
    EDGETAM model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2.1 Hiera-tiny
    [facebook/EdgeTAM](https://huggingface.co/facebook/EdgeTAM) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `EdgeTamVisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`EdgeTamVisionConfig`].
        prompt_encoder_config (Union[`dict`, `EdgeTamPromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`EdgeTamPromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `EdgeTamMaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`EdgeTamMaskDecoderConfig`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for parameter initialization.
        num_maskmem (`int`, *optional*, defaults to 7):
            The number of memory slots for the mask memory.
        image_size (`int`, *optional*, defaults to 1024):
            The size of the input images.
        sigmoid_scale_for_mem_enc (`float`, *optional*, defaults to 20.0):
            Scale factor for the sigmoid function in the memory encoder.
        sigmoid_bias_for_mem_enc (`float`, *optional*, defaults to -10.0):
            Bias for the sigmoid function in the memory encoder.
        binarize_mask_from_pts_for_mem_enc (`bool`, *optional*, defaults to `True`):
            Whether to binarize the mask from points for the memory encoder.
        enable_occlusion_spatial_embedding (`bool`, *optional*, defaults to `True`):
            Whether to enable spatial embedding for occlusions.
        multimask_output_in_sam (`bool`, *optional*, defaults to `True`):
            Whether to output multiple masks from the SAM head.
        multimask_min_pt_num (`int`, *optional*, defaults to 0):
            The minimum number of points to trigger multimask output.
        multimask_max_pt_num (`int`, *optional*, defaults to 1):
            The maximum number of points to trigger multimask output.
        multimask_output_for_tracking (`bool`, *optional*, defaults to `True`):
            Whether to use multimask output for tracking.
        non_overlap_masks_for_mem_enc (`bool`, *optional*, defaults to `False`):
            Whether to enforce non-overlapping masks for the memory encoder.
        max_object_pointers_in_encoder (`int`, *optional*, defaults to 16):
            The maximum number of object pointers in the encoder.
        enable_temporal_pos_encoding_for_object_pointers (`bool`, *optional*, defaults to `True`):
            Whether to enable temporal positional encoding for object pointers.
        project_temporal_pos_encoding_in_object_pointers (`bool`, *optional*, defaults to `True`):
            Whether to project temporal positional encoding in object pointers.
        preserve_temporal_direction_in_object_pointers (`bool`, *optional*, defaults to `True`):
            Whether to preserve temporal direction in object pointers.
        memory_attention_hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the memory attention hidden states.
        memory_attention_num_layers (`int`, *optional*, defaults to 4):
            The number of layers in the memory attention module.
        memory_attention_num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the memory attention.
        memory_attention_downsample_rate (`int`, *optional*, defaults to 1):
            The downsample rate for the attention layers.
        memory_attention_feed_forward_hidden_size (`int`, *optional*, defaults to 2048):
            The dimension of the feedforward network in the memory attention module.
        memory_attention_feed_forward_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the feedforward network in the memory attention module.
        memory_attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the memory attention module.
        memory_attention_rope_theta (`float`, *optional*, defaults to 10000):
            The Rope theta parameter.
        memory_attention_rope_feat_sizes (`Tuple[int, int]`, *optional*, defaults to `[64, 64]`):
            The feature sizes for the Rope positional encoding.
        memory_attention_rope_dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the Rope positional encoding.
        memory_attention_apply_pe_at_self_attn (`bool`, *optional*, defaults to `False`):
            Whether to apply positional encoding at the self-attention of the memory attention module.
        memory_attention_apply_pe_at_cross_attn_keys (`bool`, *optional*, defaults to `True`):
            Whether to apply positional encoding at the keys of the cross-attention of the memory attention module.
        memory_attention_apply_pe_at_cross_attn_queries (`bool`, *optional*, defaults to `False`):
            Whether to apply positional encoding at the queries of the cross-attention of the memory attention module.
        memory_encoder_hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the memory encoder hidden states.
        memory_encoder_output_channels (`int`, *optional*, defaults to 64):
            The number of output channels for the memory encoder.
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
        fill_hole_area (`int`, *optional*, defaults to 8):
            The maximum area of holes to fill in the masks.
        non_overlap_masks (`bool`, *optional*, defaults to `False`):
            Whether to enforce non-overlapping masks.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     EdgeTamVisionConfig,
    ...     EdgeTamPromptEncoderConfig,
    ...     EdgeTamMaskDecoderConfig,
    ...     EdgeTamModel,
    ... )

    >>> # Initializing a EdgeTamConfig with `"facebook/edgetam.1_hiera_tiny"` style configuration
    >>> configuration = EdgeTamconfig()

    >>> # Initializing a EdgeTamModel (with random weights) from the `"facebook/edgetam.1_hiera_tiny"` style configuration
    >>> model = EdgeTamModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a EdgeTamConfig from a EdgeTamVisionConfig, EdgeTamPromptEncoderConfig, and EdgeTamMaskDecoderConfig

    >>> # Initializing EDGETAM vision encoder, memory attention, and memory encoder configurations
    >>> vision_config = EdgeTamVisionConfig()
    >>> prompt_encoder_config = EdgeTamPromptEncoderConfig()
    >>> mask_decoder_config = EdgeTamMaskDecoderConfig()

    >>> config = EdgeTamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
    ```"""

    model_type = "edgetam"
    sub_configs = {
        "vision_config": EdgeTamVisionConfig,
        "prompt_encoder_config": EdgeTamPromptEncoderConfig,
        "mask_decoder_config": EdgeTamMaskDecoderConfig,
    }

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        binarize_mask_from_pts_for_mem_enc=True,
        enable_occlusion_spatial_embedding=True,
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        non_overlap_masks_for_mem_enc=False,
        max_object_pointers_in_encoder=16,
        enable_temporal_pos_encoding_for_object_pointers=True,
        project_temporal_pos_encoding_in_object_pointers=True,
        preserve_temporal_direction_in_object_pointers=True,
        # memory attention
        memory_attention_hidden_size=256,
        memory_attention_num_layers=2,
        memory_attention_num_attention_heads=1,
        memory_attention_downsample_rate=1,
        memory_attention_feed_forward_hidden_size=2048,
        memory_attention_feed_forward_hidden_act="relu",
        memory_attention_dropout=0.1,
        memory_attention_rope_theta=10000,
        memory_attention_rope_feat_sizes=[64, 64],
        memory_attention_rope_q_sizes=[64, 64],
        memory_attention_rope_k_sizes=[16, 16],
        memory_attention_rope_dropout=0.1,
        memory_attention_apply_pe_at_self_attn=False,
        memory_attention_apply_pe_at_cross_attn_keys=True,
        memory_attention_apply_pe_at_cross_attn_queries=False,
        # spatial perceiver
        num_latents=256,
        num_latents_2d=256,
        dim=64,
        dim_head=64,
        heads=1,
        depth=2,
        use_self_attn=True,
        hidden_dropout_p=0.0,
        attention_dropout_p=0.0,
        concat_kv_latents=False,
        pos_enc_at_key_value=True,
        ff_mult=4,
        # memory encoder
        memory_encoder_hidden_size=256,
        memory_encoder_output_channels=64,
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
        # post-processing parameters
        fill_hole_area=8,
        non_overlap_masks=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vision_config = vision_config if vision_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}

        if isinstance(vision_config, EdgeTamVisionConfig):
            vision_config = vision_config.to_dict()
        if isinstance(prompt_encoder_config, EdgeTamPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, EdgeTamMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        self.vision_config = EdgeTamVisionConfig(**vision_config)
        self.prompt_encoder_config = EdgeTamPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = EdgeTamMaskDecoderConfig(**mask_decoder_config)

        self.initializer_range = initializer_range
        self.num_maskmem = num_maskmem  # default 1 input frame + 6 previous frames
        self.image_size = image_size
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc  # scale factor for mask sigmoid prob
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc  # bias factor for mask sigmoid prob
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.enable_occlusion_spatial_embedding = enable_occlusion_spatial_embedding
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.max_object_pointers_in_encoder = max_object_pointers_in_encoder
        self.enable_temporal_pos_encoding_for_object_pointers = enable_temporal_pos_encoding_for_object_pointers
        self.project_temporal_pos_encoding_in_object_pointers = project_temporal_pos_encoding_in_object_pointers
        self.preserve_temporal_direction_in_object_pointers = preserve_temporal_direction_in_object_pointers

        # memory attention
        self.memory_attention_hidden_size = memory_attention_hidden_size
        self.memory_attention_num_layers = memory_attention_num_layers
        self.memory_attention_num_attention_heads = memory_attention_num_attention_heads
        self.memory_attention_downsample_rate = memory_attention_downsample_rate
        self.memory_attention_feed_forward_hidden_size = memory_attention_feed_forward_hidden_size
        self.memory_attention_feed_forward_hidden_act = memory_attention_feed_forward_hidden_act
        self.memory_attention_dropout = memory_attention_dropout
        self.memory_attention_rope_theta = memory_attention_rope_theta
        self.memory_attention_rope_feat_sizes = memory_attention_rope_feat_sizes
        self.memory_attention_rope_q_sizes = memory_attention_rope_q_sizes
        self.memory_attention_rope_k_sizes = memory_attention_rope_k_sizes
        self.memory_attention_rope_dropout = memory_attention_rope_dropout
        self.memory_attention_apply_pe_at_self_attn = memory_attention_apply_pe_at_self_attn
        self.memory_attention_apply_pe_at_cross_attn_keys = memory_attention_apply_pe_at_cross_attn_keys
        self.memory_attention_apply_pe_at_cross_attn_queries = memory_attention_apply_pe_at_cross_attn_queries

        # spatial perceiver
        self.num_latents = num_latents
        self.num_latents_2d = num_latents_2d
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.depth = depth
        self.use_self_attn = use_self_attn
        self.hidden_dropout_p = hidden_dropout_p
        self.attention_dropout_p = attention_dropout_p
        self.concat_kv_latents = concat_kv_latents
        self.pos_enc_at_key_value = pos_enc_at_key_value
        self.ff_mult = ff_mult

        # memory encoder
        self.memory_encoder_hidden_size = memory_encoder_hidden_size
        self.memory_encoder_output_channels = memory_encoder_output_channels
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

        # post-processing parameters
        self.fill_hole_area = fill_hole_area  # area threshold for filling holes in masks
        self.non_overlap_masks = non_overlap_masks  # whether to apply non-overlapping constraints on output masks


class EdgeTamHieraDetModel:
    pass


class EdgeTamLayerNorm(Sam2LayerNorm):
    pass


class EdgeTamMemoryFuserCXBlock(Sam2MemoryFuserCXBlock):
    pass


class EdgeTamVisionEncoderOutput(Sam2VisionEncoderOutput):
    pass


class EdgeTamVisionRotaryEmbedding(Sam2VisionRotaryEmbedding):
    pass


class EdgeTamAttention(Sam2Attention):
    pass


class EdgeTamRoPEAttention(Sam2RoPEAttention):
    pass


class EdgeTamTwoWayAttentionBlock(Sam2TwoWayAttentionBlock):
    pass


class EdgeTamMemoryEncoder(Sam2MemoryEncoder):
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
        elif isinstance(module, EdgeTamVideoModel):
            if module.no_memory_positional_encoding is not None:
                module.no_memory_positional_encoding.data.zero_()
            if module.memory_temporal_positional_encoding is not None:
                module.memory_temporal_positional_encoding.data.zero_()
            if module.no_object_pointer is not None:
                module.no_object_pointer.data.zero_()
            if module.occlusion_spatial_embedding_parameter is not None:
                module.occlusion_spatial_embedding_parameter.data.zero_()
        if isinstance(module, EdgeTamMemoryFuserCXBlock):
            if module.scale is not None:
                module.scale.data.zero_()


@auto_docstring(
    custom_intro="""
    The vision model from Sam without any head or projection on top.
    """
)
class EdgeTamVisionModel(Sam2VisionModel):
    config_class = EdgeTamVisionConfig
    main_input_name = "pixel_values"
    _can_record_outputs = {"hidden_states": AutoModel, "attentions": AutoModel}

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


def apply_rotary_pos_emb_2d_v2(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    repeat_freqs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors for vision models.
    Follows the standard transformers library pattern.

    Args:
        q: Query tensor of shape (..., seq_len, head_dim)
        k: Key tensor of shape (..., seq_len, head_dim)
        cos: Cosine position embedding of shape (seq_len, head_dim)
        sin: Sine position embedding of shape (seq_len, head_dim)
        repeat_freqs_k: Whether to repeat frequencies for keys (for cross-attention)

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim)
    cos = torch.flatten(torch.cat((cos.unsqueeze(-1), cos.unsqueeze(-1)), dim=-1), -2)
    sin = torch.flatten(torch.cat((sin.unsqueeze(-1), sin.unsqueeze(-1)), dim=-1), -2)
    batch_size, num_heads, num_tokens, channels_per_head = x.shape
    if num_tokens == cos.shape[-2]:
        x_rope = x
        x_no_rope = None
    else:
        rope_tokens = cos.shape[-2]
        no_rope_tokens = num_tokens // repeat_freqs - rope_tokens
        x = x.view(batch_size, num_heads, repeat_freqs, num_tokens // repeat_freqs, channels_per_head)
        x_rope = x[..., no_rope_tokens:, :].reshape(batch_size, num_heads, -1, channels_per_head)
        x_no_rope = x[..., :no_rope_tokens, :].reshape(batch_size, num_heads, -1, channels_per_head)

    if repeat_freqs > 1:
        cos = cos.repeat(1, 1, repeat_freqs, 1)
        sin = sin.repeat(1, 1, repeat_freqs, 1)
    x_embed = (x_rope * cos) + (rotate_half(x_rope) * sin)
    if x_no_rope is not None:
        x_embed = x_embed.view(batch_size, num_heads, repeat_freqs, -1, channels_per_head)
        x_no_rope = x_no_rope.view(batch_size, num_heads, repeat_freqs, -1, channels_per_head)
        x_embed = torch.cat((x_no_rope, x_embed), dim=3).view(batch_size, num_heads, num_tokens, channels_per_head)
    return x_embed.type_as(x)


class EdgeTamModel(Sam2Model):
    pass


class EdgeTamVideoInferenceSession(Sam2VideoInferenceSession):
    pass


class EdgeTamRoPEAttentionV2(EdgeTamAttention):
    """Attention with rotary position encoding."""

    def __init__(self, *args, dropout=0.0, rope_theta=10000.0, q_sizes=(64, 64), k_sizes=(16, 16), **kwargs):
        super().__init__(*args, **kwargs)

        head_dim = self.internal_dim // self.num_attention_heads
        self.rotary_emb_q = EdgeTamVisionRotaryEmbedding(
            dim=head_dim, end_x=q_sizes[0], end_y=q_sizes[1], theta=rope_theta
        )
        self.rotary_emb_k = EdgeTamVisionRotaryEmbedding(
            dim=head_dim, end_x=k_sizes[0], end_y=k_sizes[1], theta=rope_theta
        )
        self.q_sizes = q_sizes
        self.k_sizes = k_sizes
        self.dropout_p = dropout

        # Cache for position embeddings
        self._cached_cos_q = None
        self._cached_sin_q = None
        self._cached_cos_k = None
        self._cached_sin_k = None
        self._cached_feat_sizes_q = None
        self._cached_feat_sizes_k = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_k_exclude_rope: int = 0,
        rope_k_repeat: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # Determine feature map size - assume square for simplicity and infer from sequence length
        seq_len_q = query.shape[-2]
        width_q = height_q = int(math.sqrt(seq_len_q))
        current_feat_sizes_q = (width_q, height_q)
        seq_len_k = key.shape[-2]
        width_k = height_k = int(math.sqrt(seq_len_k))
        current_feat_sizes_k = (width_k, height_k)
        # Generate or use cached position embeddings
        if (
            self._cached_cos_q is None
            or self._cached_sin_q is None
            or self._cached_feat_sizes_q != current_feat_sizes_q
        ):
            cos_q, sin_q = self.rotary_emb_q(current_feat_sizes_q)
            self._cached_cos_q = cos_q
            self._cached_sin_q = sin_q
            self._cached_feat_sizes_q = current_feat_sizes_q
        else:
            cos_q = self._cached_cos_q
            sin_q = self._cached_sin_q
        if (
            self._cached_cos_k is None
            or self._cached_sin_k is None
            or self._cached_feat_sizes_k != current_feat_sizes_k
        ):
            cos_k, sin_k = self.rotary_emb_k(current_feat_sizes_k)
            self._cached_cos_k = cos_k
            self._cached_sin_k = sin_k
            self._cached_feat_sizes_k = current_feat_sizes_k
        else:
            cos_k = self._cached_cos_k
            sin_k = self._cached_sin_k

        query = apply_rotary_pos_emb_2d_v2(query, cos_q, sin_q, repeat_freqs=1)
        num_k_rope = key.shape[-2] - num_k_exclude_rope
        key[:, :, :num_k_rope] = apply_rotary_pos_emb_2d_v2(
            key[:, :, :num_k_rope], cos_k, sin_k, repeat_freqs=rope_k_repeat
        )
        scale = query.shape[-1] ** -0.5

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0 if not self.training else self.dropout_p,
            scaling=scale,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = self._recombine_heads(attn_output, point_batch_size)
        attn_output = self.out_proj(attn_output)
        return attn_output


class EdgeTamMemoryAttentionLayer(nn.Module):
    def __init__(self, config: EdgeTamConfig):
        super().__init__()
        hidden_size = config.memory_attention_hidden_size
        self.self_attn = EdgeTamRoPEAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=config.memory_attention_num_attention_heads,
            downsample_rate=config.memory_attention_downsample_rate,
            rope_theta=config.memory_attention_rope_theta,
            feat_sizes=config.memory_attention_rope_feat_sizes,
            dropout=config.memory_attention_rope_dropout,
        )
        self.cross_attn_image = EdgeTamRoPEAttentionV2(
            config,
            hidden_size=hidden_size,
            num_attention_heads=config.memory_attention_num_attention_heads,
            downsample_rate=config.memory_attention_downsample_rate,
            rope_theta=config.memory_attention_rope_theta,
            dropout=config.memory_attention_rope_dropout,
            q_sizes=config.memory_attention_rope_q_sizes,
            k_sizes=config.memory_attention_rope_k_sizes,
            kv_in_dim=64,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_size, config.memory_attention_feed_forward_hidden_size)
        self.dropout = nn.Dropout(config.memory_attention_dropout)
        self.linear2 = nn.Linear(config.memory_attention_feed_forward_hidden_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(config.memory_attention_dropout)
        self.dropout2 = nn.Dropout(config.memory_attention_dropout)
        self.dropout3 = nn.Dropout(config.memory_attention_dropout)

        self.activation = ACT2FN[config.memory_attention_feed_forward_hidden_act]

        # Where to add pos enc
        self.apply_pe_at_self_attn = config.memory_attention_apply_pe_at_self_attn
        self.apply_pe_at_cross_attn_queries = config.memory_attention_apply_pe_at_cross_attn_queries
        self.apply_pe_at_cross_attn_keys = config.memory_attention_apply_pe_at_cross_attn_keys

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Optional[Tensor] = None,
        key_point_embedding: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
        rope_k_repeat: int = 0,
    ) -> torch.Tensor:
        # Self-Attention
        query = self.layer_norm1(queries)
        if self.apply_pe_at_self_attn:
            query = self.self_attn(query=query + query_point_embedding, key=query + query_point_embedding, value=query)
        else:
            query = self.self_attn(query=query, key=query, value=query)
        queries = queries + self.dropout1(query)

        # Cross-Attention
        query = self.layer_norm2(queries)
        query = self.cross_attn_image(
            query=query + query_point_embedding if self.apply_pe_at_cross_attn_queries else query,
            key=keys + key_point_embedding if self.apply_pe_at_cross_attn_keys else keys,
            value=keys,
            num_k_exclude_rope=num_k_exclude_rope,
            rope_k_repeat=rope_k_repeat,
        )
        queries = queries + self.dropout2(query)
        # MLP
        query = self.layer_norm3(queries)
        query = self.linear2(self.dropout(self.activation(self.linear1(query))))
        queries = queries + self.dropout3(query)
        return queries


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class EdgeTamPerceiverAttention(nn.Module):
    def __init__(self, config, dim, dim_head=64, heads=8, dropout_p=0.05, concat_kv_latents=True):
        super().__init__()
        self.config = config
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm_x = nn.LayerNorm(dim)
        self.layer_norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout_p = dropout_p
        self.concat_kv_latents = concat_kv_latents
        self.is_causal = False

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_tokens, n_heads, c_per_head = x.shape
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, latents, x, pos=None, **kwargs):
        latents = self.layer_norm_latents(latents)
        x = self.layer_norm_x(x)

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if self.concat_kv_latents:
            kv_input = torch.cat((x, latents), dim=-2)
        else:
            kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        if pos is not None:
            if self.concat_kv_latents:
                raise ValueError("Position encoding is not supported when concat_kv_latents is True")
            pos = self._separate_heads(pos, self.heads)
            k, v = k + pos, v + pos

        scale = q.shape[-1] ** -0.5
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=None,
            dropout=0.0 if not self.training else self.dropout_p,
            scaling=scale,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = self._recombine_heads(attn_output)
        return self.to_out(attn_output)


class EdgeTamPerceiverSelfAttention(nn.Module):
    def __init__(self, config, dim, dim_head=64, heads=8, dropout_p=0.05):
        super().__init__()
        self.config = config
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout_p = dropout_p
        self.is_causal = False

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_tokens, n_heads, c_per_head = x.shape
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, x, **kwargs):
        x = self.layer_norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        scale = q.shape[-1] ** -0.5
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=None,
            dropout=0.0 if not self.training else self.dropout_p,
            scaling=scale,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = self._recombine_heads(attn_output)
        return self.to_out(attn_output)


class EdgeTamPerceiverEncoderLayer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        hidden_dropout_p=0.0,
        attention_dropout_p=0.0,
        concat_kv_latents=False,
        use_self_attn=False,
    ):
        super().__init__()
        self.attn = EdgeTamPerceiverAttention(
            config,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout_p=attention_dropout_p,
            concat_kv_latents=concat_kv_latents,
        )
        self.ff = FeedForward(dim=dim, mult=ff_mult)
        self.dropout = nn.Dropout(hidden_dropout_p)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = EdgeTamPerceiverSelfAttention(
                config,
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout_p=attention_dropout_p,
            )
            self.self_ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, latents, x, pos=None):
        latents = self.attn(latents, x, pos) + latents
        latents = self.dropout(latents)
        latents = self.ff(latents) + latents
        if self.use_self_attn:
            latents = self.self_attn(latents) + latents
            latents = self.self_ff(latents) + latents
        return latents


class EdgeTamPositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class EdgeTamPerceiverResampler(nn.Module):
    def __init__(self, config: EdgeTamConfig):
        super().__init__()
        self.num_latents = config.num_latents
        self.num_latents_2d = config.num_latents_2d

        if self.num_latents > 0:
            self.latents = nn.Parameter(torch.randn(self.num_latents, config.dim))
        if self.num_latents_2d > 0:
            self.latents_2d = nn.Parameter(torch.randn(self.num_latents_2d, config.dim))
        self.position_encoding = EdgeTamPositionEmbeddingSine(config.dim)

        self.layers = nn.ModuleList([])
        for _ in range(config.depth):
            self.layers.append(
                EdgeTamPerceiverEncoderLayer(
                    config,
                    dim=config.dim,
                    dim_head=config.dim_head,
                    heads=config.heads,
                    ff_mult=config.ff_mult,
                    hidden_dropout_p=config.hidden_dropout_p,
                    attention_dropout_p=config.attention_dropout_p,
                    concat_kv_latents=config.concat_kv_latents,
                    use_self_attn=config.use_self_attn,
                )
            )

        self.layer_norm = nn.LayerNorm(config.dim)
        self.pos_enc_at_key_value = config.pos_enc_at_key_value

    def forward(self, x, pos=None):
        out_latents = []
        out_pos = []
        if self.num_latents > 0:
            latents_1d, pos_1d = self.forward_1d(x, pos)
            out_latents.append(latents_1d)
            out_pos.append(pos_1d)
        if self.num_latents_2d > 0:
            latents_2d, pos_2d = self.forward_2d(x)
            out_latents.append(latents_2d)
            out_pos.append(pos_2d)

        latents = torch.concat(out_latents, dim=1)
        if pos is not None:
            pos = torch.concat(out_pos, dim=1)

        return latents, pos

    def forward_1d(self, x, pos):
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)

        if not self.pos_enc_at_key_value:
            _pos = None
        if pos is not None:
            _pos = pos.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            _pos = None

        for layer in self.layers:
            latents = layer(latents, x, _pos)

        if pos is not None:
            pos = torch.zeros_like(latents)

        latents = self.layer_norm(latents)
        return latents, pos

    def forward_2d(self, x):
        B, C, H, W = x.shape

        latents_2d = self.latents_2d.unsqueeze(0).expand(B, -1, -1).view(-1, 1, C)

        num_window = int(math.sqrt(self.num_latents_2d))
        window_size = H // num_window
        x = x.permute(0, 2, 3, 1)

        x, _ = window_partition(x, window_size)
        x = x.flatten(1, 2)

        for layer in self.layers:
            latents_2d = layer(latents_2d, x)

        latents_2d = latents_2d.view(B, num_window, num_window, C).permute(0, 3, 1, 2)

        pos_2d = self.position_encoding(latents_2d).to(dtype=x.dtype)
        pos_2d = pos_2d.permute(0, 2, 3, 1).flatten(1, 2)

        latents_2d = latents_2d.permute(0, 2, 3, 1).flatten(1, 2)

        latents_2d = self.layer_norm(latents_2d)

        return latents_2d, pos_2d


class EdgeTamMemoryAttention(Sam2MemoryAttention):
    def forward(
        self,
        current_vision_features: torch.Tensor,
        memory: torch.Tensor,
        current_vision_position_embeddings: Optional[Tensor] = None,
        memory_posision_embeddings: Optional[Tensor] = None,
        num_object_pointer_tokens: int = 0,
        num_spatial_memory_tokens: int = -1,
    ):
        """
        Args:
            current_vision_features (`torch.FloatTensor`):
                The current vision features used for self-attention.
            memory (`torch.FloatTensor`):
                The memory features used for cross-attention.
            current_vision_position_embeddings (`torch.FloatTensor`, *optional*):
                The position embeddings for the current vision features.
            memory_posision_embeddings (`torch.FloatTensor`, *optional*):
                The position embeddings for the memory features.
            num_object_pointer_tokens (`int`, *optional*, defaults to 0):
                The number of object pointer tokens.
        """
        if isinstance(current_vision_features, list) and isinstance(current_vision_position_embeddings, list):
            current_vision_features, current_vision_position_embeddings = (
                current_vision_features[0],
                current_vision_position_embeddings[0],
            )

        output = current_vision_features
        if current_vision_position_embeddings is not None:
            output = output + 0.1 * current_vision_position_embeddings

        # Convert to batch first
        output = output.transpose(0, 1)
        current_vision_position_embeddings = current_vision_position_embeddings.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_posision_embeddings = memory_posision_embeddings.transpose(0, 1)

        for layer in self.layers:
            output = layer(
                queries=output.unsqueeze(1) if output.ndim == 3 else output,
                keys=memory.unsqueeze(1),
                query_point_embedding=current_vision_position_embeddings.unsqueeze(1),
                key_point_embedding=memory_posision_embeddings.unsqueeze(1),
                num_k_exclude_rope=num_object_pointer_tokens,
                rope_k_repeat=num_spatial_memory_tokens,
            )

        normed_output = self.layer_norm(output)

        # Convert back to seq first
        normed_output = normed_output.transpose(0, 1)
        current_vision_position_embeddings = current_vision_position_embeddings.transpose(0, 1)

        return normed_output


@auto_docstring
class EdgeTamVideoModel(Sam2VideoModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]
    # need to be ignored, as it's a buffer and will not be correctly detected as tied weight
    _keys_to_ignore_on_load_missing = ["prompt_encoder.shared_embedding.positional_embedding"]
    _keys_to_ignore_on_load_unexpected = []
    _can_record_outputs = {"mask_decoder_attentions": OutputRecorder(EdgeTamTwoWayAttentionBlock, index=2)}

    def __init__(self, config: EdgeTamConfig):
        super().__init__(config)
        # For video sequence inference
        self.memory_attention = EdgeTamMemoryAttention(config)
        self.memory_encoder = EdgeTamMemoryEncoder(config)
        self.spatial_perceiver = EdgeTamPerceiverResampler(config)
        self.no_memory_positional_encoding = torch.nn.Parameter(
            torch.zeros(1, 1, config.vision_config.fpn_hidden_size)
        )
        self.mem_dim = config.memory_encoder_output_channels
        self.num_maskmem = config.num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.memory_temporal_positional_encoding = torch.nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )

        # prompt encoder part
        self.project_temporal_pos_encoding_in_object_pointers = (
            config.project_temporal_pos_encoding_in_object_pointers
        )  # compatibility with EdgeTam

        self.no_object_pointer = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        # A conv layer to downsample the mask prompt to stride 4 (the same stride as
        # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
        # so that it can be fed into the SAM mask decoder to generate a pointer.
        self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        # a feedforward layer on SAM output tokens to turn them into object pointers
        self.object_pointer_proj = EdgeTamFeedForward(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)

        if self.project_temporal_pos_encoding_in_object_pointers:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.temporal_positional_encoding_projection_layer = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.temporal_positional_encoding_projection_layer = torch.nn.Identity()

        self.occlusion_spatial_embedding_parameter = None  # compatibility with EdgeTam
        if config.enable_occlusion_spatial_embedding:
            self.occlusion_spatial_embedding_parameter = torch.nn.Parameter(torch.zeros(1, self.mem_dim))

        # Video Inference specific parameters
        self.sigmoid_scale_for_mem_enc = config.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = config.sigmoid_bias_for_mem_enc
        # Additional configuration for video tracking
        self.non_overlap_masks = config.non_overlap_masks
        self.fill_hole_area = config.fill_hole_area
        self.multimask_output_in_sam = config.multimask_output_in_sam
        self.multimask_min_pt_num = config.multimask_min_pt_num
        self.multimask_max_pt_num = config.multimask_max_pt_num
        self.non_overlap_masks_for_mem_enc = config.non_overlap_masks_for_mem_enc
        self.max_object_pointers_in_encoder = config.max_object_pointers_in_encoder
        # Compatibility with EDGETAM
        self.enable_temporal_pos_encoding_for_object_pointers = config.enable_temporal_pos_encoding_for_object_pointers
        self.binarize_mask_from_pts_for_mem_enc = config.binarize_mask_from_pts_for_mem_enc
        # Compatibility with EDGETAM
        self.preserve_temporal_direction_in_object_pointers = config.preserve_temporal_direction_in_object_pointers
        self.multimask_output_for_tracking = config.multimask_output_for_tracking

        self.post_init()

    def _prepare_memory_conditioned_features(
        self,
        inference_session: EdgeTamVideoInferenceSession,
        frame_idx: int,
        obj_idx: int,
        is_initial_conditioning_frame: bool,
        current_vision_features: list[torch.Tensor],
        current_vision_positional_embeddings: list[torch.Tensor],
        num_total_frames: int,
        track_in_reverse_time: bool = False,
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Fuse current frame's visual features with memory from previous frames for enhanced object tracking.

        This method conditions the current frame's visual features on temporal memory from previous frames,
        enabling consistent object tracking across video sequences. For initial conditioning frames, it uses
        no-memory embeddings. For subsequent frames, it retrieves and integrates memory features from both
        conditioning frames (user interactions) and non-conditioning frames (tracked results) via cross-attention.

        Args:
            inference_session (`EdgeTamVideoInferenceSession`):
                The video inference session object.
            frame_idx (`int`):
                Index of the current frame being processed.
            obj_idx (`int`):
                Index of the object being processed.
            is_initial_conditioning_frame (`bool`):
                Whether this is an initial conditioning frame with user inputs (True) or a subsequent
                tracking frame (False).
            current_vision_features (`list[torch.Tensor]`):
                List of vision feature tensors for the current frame, with the last element being the
                highest-level features of shape `(seq_len, batch_size, channels)`.
            current_vision_positional_embeddings (`list[torch.Tensor]`):
                List of positional embedding tensors corresponding to the vision features.
            num_total_frames (`int`):
                Total number of frames in the video sequence.
            track_in_reverse_time (`bool`, *optional*, defaults to `False`):
                Whether tracking is performed in reverse temporal order.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether this is streaming inference mode.

        Returns:
            `torch.Tensor`: Memory-conditioned feature tensor of shape `(batch_size, channels, height, width)`
                suitable for input to the SAM decoder.
        """
        # Get dimensions from the highest-level (lowest-resolution) feature map
        batch_size = current_vision_features[-1].size(1)
        num_channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]
        device = current_vision_features[-1].device

        # If memory is disabled (e.g., for single image SAM), return current features directly.
        if self.num_maskmem == 0:
            # Permute (SeqLen, Batch, Channels) -> (Batch, Channels, SeqLen) then view as (Batch, Channels, Height, Width)
            # Assuming SeqLen = Height * Width for the last feature map
            current_feature_map = (
                current_vision_features[-1].permute(1, 2, 0).view(batch_size, num_channels, height, width)
            )
            return current_feature_map

        num_object_pointer_tokens = 0
        temporal_position_sign_multiplier = -1 if track_in_reverse_time else 1

        # Step 1: Condition the visual features of the current frame on previous memories
        if not is_initial_conditioning_frame:
            # Retrieve memories encoded from previous frames
            memories_to_concatenate = []
            memory_positional_embeddings_to_concatenate = []

            # Ensure there are conditioning frame outputs to process
            conditioning_outputs = inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
            if not conditioning_outputs:
                raise ValueError(
                    "maskmem_features in conditioning outputs cannot be empty when not is_initial_conditioning_frame"
                )

            # Select a maximum number of temporally closest conditioning frames for cross-attention
            # Store (temporal_position, output_data) tuples
            temporal_positions_and_previous_outputs = [(0, out) for out in conditioning_outputs.values()]

            # Add non-conditioning memory frames (up to self.num_maskmem - 1)
            # These are typically frames tracked by the model without direct user input.
            # Frames are selected with a stride, prioritizing the most recent ones.
            for temporal_pos_offset in range(1, self.num_maskmem):
                # relative_temporal_offset: how many frames before (or after if reversing) the current frame
                relative_temporal_offset = self.num_maskmem - temporal_pos_offset
                previous_frame_idx = -1  # Initialize with an invalid index

                if relative_temporal_offset == 1:
                    # For the immediately preceding/succeeding frame, always take it regardless of stride
                    if not track_in_reverse_time:
                        previous_frame_idx = frame_idx - relative_temporal_offset
                    else:
                        previous_frame_idx = frame_idx + relative_temporal_offset
                else:
                    # For other memory frames, select based on stride
                    if not track_in_reverse_time:
                        # Find the nearest frame among every stride-th frame before the current one (excluding current-1)
                        base_idx = frame_idx - 2
                        previous_frame_idx = base_idx - (relative_temporal_offset - 2)
                    else:
                        base_idx = frame_idx + 2
                        previous_frame_idx = base_idx + (relative_temporal_offset - 2)

                # check if the output is already stored without using get_output to avoid unnecessary memory transfers between CPU and GPU
                output_data = inference_session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].get(
                    previous_frame_idx, None
                )

                temporal_positions_and_previous_outputs.append((temporal_pos_offset, output_data))

            for temporal_pos_offset, prev_output_data in temporal_positions_and_previous_outputs:
                if prev_output_data is None:
                    continue  # Skip if no output data for this temporal position (e.g., padding frames)

                # Load memory features (potentially from CPU to GPU)
                # Features are flattened: (Batch, Channels, H, W) -> (H*W, Batch, Channels)
                memory_features = prev_output_data["maskmem_features"].to(device, non_blocking=True)
                memories_to_concatenate.append(memory_features.permute(1, 0, 2))

                # Spatial positional encoding (potentially from CPU to GPU)
                spatial_memory_pos_embed = prev_output_data["maskmem_pos_enc"][-1].to(device, non_blocking=True)
                spatial_memory_pos_embed = spatial_memory_pos_embed.squeeze(1).permute(1, 0, 2)
                # Add temporal positional encoding
                # self.memory_temporal_positional_encoding shape: (NumMaskMem, 1, 1, MemDim)
                temporal_encoding_index = self.num_maskmem - temporal_pos_offset - 1
                combined_memory_pos_embed = (
                    spatial_memory_pos_embed + self.memory_temporal_positional_encoding[temporal_encoding_index]
                )
                memory_positional_embeddings_to_concatenate.append(combined_memory_pos_embed)

            num_spatial_memory_tokens = len(memories_to_concatenate)

            # Construct the list of past object pointers to be used in attention
            if streaming:
                max_object_pointers_to_use = self.max_object_pointers_in_encoder
            else:
                max_object_pointers_to_use = min(num_total_frames, self.max_object_pointers_in_encoder)
            temporal_diff_and_pointers = []

            # Add object pointers from selected conditioning frames
            # Optionally, only include pointers from past frames during evaluation
            eligible_conditioning_outputs = conditioning_outputs
            if not self.training:
                eligible_conditioning_outputs = {
                    t: out
                    for t, out in conditioning_outputs.items()
                    if (t >= frame_idx if track_in_reverse_time else t <= frame_idx)
                }

            for t_idx, out_data in eligible_conditioning_outputs.items():
                temporal_difference = (frame_idx - t_idx) * temporal_position_sign_multiplier
                if not self.preserve_temporal_direction_in_object_pointers:
                    temporal_difference = abs(temporal_difference)
                temporal_diff_and_pointers.append((temporal_difference, out_data["object_pointer"]))

            # Add object pointers from non-conditioning frames (up to max_object_pointers_to_use - 1)
            for t_diff_offset in range(1, max_object_pointers_to_use):
                ref_frame_idx = frame_idx + t_diff_offset if track_in_reverse_time else frame_idx - t_diff_offset
                if ref_frame_idx < 0 or (
                    not streaming and num_total_frames is not None and ref_frame_idx >= num_total_frames
                ):
                    break  # Stop if frame index is out of bounds

                # check if the output is already stored without using get_output to avoid unnecessary memory transfers between CPU and GPU
                out_data = inference_session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].get(
                    ref_frame_idx, None
                )
                if out_data is not None:
                    temporal_diff_and_pointers.append((t_diff_offset, out_data["object_pointer"]))

            if temporal_diff_and_pointers:
                temporal_differences, object_pointers_list = zip(*temporal_diff_and_pointers)
                # Stack object pointers: List of (Batch, Channels) -> (SeqLen_ptr, Batch, Channels)
                object_pointers = torch.stack(object_pointers_list, dim=0)

                if self.enable_temporal_pos_encoding_for_object_pointers:
                    max_temporal_diff = float(max_object_pointers_to_use - 1)
                    # Determine dimensionality for temporal positional encoding of pointers
                    pointer_tpos_dim = (
                        num_channels if self.project_temporal_pos_encoding_in_object_pointers else self.mem_dim
                    )

                    # Normalize temporal differences before sine PE calculation
                    normalized_temporal_diffs = (
                        torch.tensor(temporal_differences, device=device, dtype=torch.float32) / max_temporal_diff
                    )
                    sine_pe = get_1d_sine_pe(normalized_temporal_diffs, dim=pointer_tpos_dim).to(object_pointers.dtype)
                    projected_sine_pe = self.temporal_positional_encoding_projection_layer(sine_pe)
                    object_pointers_pos_embed = projected_sine_pe.unsqueeze(1).expand(-1, batch_size, self.mem_dim)
                else:
                    object_pointers_pos_embed = object_pointers.new_zeros(
                        len(temporal_differences), batch_size, self.mem_dim, dtype=object_pointers.dtype
                    )

                if self.mem_dim < num_channels:
                    # If memory dimension is smaller, reshape/split pointers and repeat positional encoding
                    num_splits = num_channels // self.mem_dim
                    object_pointers = object_pointers.reshape(-1, batch_size, num_splits, self.mem_dim)
                    object_pointers = object_pointers.permute(0, 2, 1, 3).flatten(
                        0, 1
                    )  # (SeqLen_ptr*num_splits, Batch, MemDim)
                    object_pointers_pos_embed = object_pointers_pos_embed.repeat_interleave(num_splits, dim=0)

                memories_to_concatenate.append(object_pointers)
                memory_positional_embeddings_to_concatenate.append(object_pointers_pos_embed)
                num_object_pointer_tokens = object_pointers.shape[0]
        else:
            # For initial conditioning frames, no prior memory is used directly in this block.
            # The model might handle this with a special token or mechanism.
            # If configured, directly add a learnable "no memory" embedding.
            # current_vision_features[-1] has shape (SeqLen, Batch, Channels)
            conditioned_feature_map_flat = current_vision_features[-1] + self.no_memory_embedding
            # Reshape to (Batch, Channels, Height, Width)
            conditioned_feature_map = conditioned_feature_map_flat.permute(1, 2, 0).view(
                batch_size, num_channels, height, width
            )
            return conditioned_feature_map

        # Step 2: Concatenate all retrieved memories and their positional embeddings.
        combined_memory = torch.cat(memories_to_concatenate, dim=0)
        combined_memory_positional_embeddings = torch.cat(memory_positional_embeddings_to_concatenate, dim=0)

        # Step 3: Forward through the memory attention mechanism.
        conditioned_feature_map_flat = self.memory_attention(
            current_vision_features=current_vision_features,  # Pass the list as expected
            current_vision_position_embeddings=current_vision_positional_embeddings,
            memory=combined_memory,
            memory_posision_embeddings=combined_memory_positional_embeddings,  # Corrected typo from API
            num_object_pointer_tokens=num_object_pointer_tokens,
            num_spatial_memory_tokens=num_spatial_memory_tokens,
        )

        # Reshape from (Batch, H*W, Channels) to (Batch, Channels, Height, Width)
        conditioned_feature_map = (
            conditioned_feature_map_flat.squeeze(1).permute(0, 2, 1).view(batch_size, num_channels, height, width)
        )
        return conditioned_feature_map

    def _encode_new_memory(
        self,
        current_vision_feats: list[torch.Tensor],
        pred_masks_high_res: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_pts: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode the current image and its prediction into a memory feature."""
        batch_size = current_vision_feats[-1].size(1)  # batch size on this frame
        channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(batch_size, channels, height, width)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).to(pred_masks_high_res.dtype)
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_features, maskmem_pos_enc = self.memory_encoder(
            pix_feat,
            mask_for_mem,
            skip_mask_sigmoid=True,  # sigmoid already applied
        )
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.occlusion_spatial_embedding_parameter is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None]) * self.occlusion_spatial_embedding_parameter[
                ..., None, None
            ].expand(*maskmem_features.shape)

        maskmem_features, maskmem_pos_enc[0] = self.spatial_perceiver(maskmem_features, maskmem_pos_enc[0])

        return maskmem_features, maskmem_pos_enc


__all__ = [
    "EdgeTamModel",
    "EdgeTamVideoModel",
    "EdgeTamVisionModel",
    "EdgeTamVideoInferenceSession",
    "EdgeTamPreTrainedModel",
    "EdgeTamConfig",
    "EdgeTamVisionConfig",
    "EdgeTamPromptEncoderConfig",
    "EdgeTamMaskDecoderConfig",
]
