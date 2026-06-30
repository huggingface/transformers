# Copyright 2025 the HuggingFace Team. All rights reserved.
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


import torch
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..sam2_video.configuration_sam2_video import Sam2VideoMaskDecoderConfig, Sam2VideoPromptEncoderConfig
from ..sam2_video.modeling_sam2_video import (
    Sam2VideoAttention,
    Sam2VideoFeedForward,
    Sam2VideoImageSegmentationOutput,
    Sam2VideoInferenceCache,
    Sam2VideoInferenceSession,
    Sam2VideoLayerNorm,
    Sam2VideoMaskDecoder,
    Sam2VideoMaskDownSampler,
    Sam2VideoMaskDownSamplerLayer,
    Sam2VideoMaskEmbedding,
    Sam2VideoMemoryAttention,
    Sam2VideoMemoryAttentionLayer,
    Sam2VideoMemoryEncoder,
    Sam2VideoMemoryFuser,
    Sam2VideoMemoryFuserCXBlock,
    Sam2VideoModel,
    Sam2VideoPositionalEmbedding,
    Sam2VideoPositionEmbeddingSine,
    Sam2VideoPreTrainedModel,
    Sam2VideoPromptEncoder,
    Sam2VideoRoPEAttention,
    Sam2VideoSegmentationOutput,
    Sam2VideoTwoWayAttentionBlock,
    Sam2VideoTwoWayTransformer,
    Sam2VideoVisionEncoderOutput,
    Sam2VideoVisionRotaryEmbedding,
)
from ..sam2_video.processing_sam2_video import Sam2VideoProcessor


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerVideoPromptEncoderConfig(Sam2VideoPromptEncoderConfig):
    r"""
    mask_input_channels (`int`, *optional*, defaults to 16):
        The number of channels to be fed to the `MaskDecoder` module.
    num_point_embeddings (`int`, *optional*, defaults to 4):
        The number of point embeddings to be used.
    scale (`float`, *optional*, defaults to 1):
        The scale factor for the prompt encoder.
    """

    base_config_key = "prompt_encoder_config"

    image_size: int | list[int] | tuple[int, int] = 1008
    patch_size: int | list[int] | tuple[int, int] = 14


class Sam3TrackerVideoProcessor(Sam2VideoProcessor):
    pass


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerVideoMaskDecoderConfig(Sam2VideoMaskDecoderConfig):
    pass


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerVideoConfig(PreTrainedConfig):
    r"""
    prompt_encoder_config (Union[`dict`, `Sam3TrackerVideoPromptEncoderConfig`], *optional*):
        Dictionary of configuration options used to initialize [`Sam3TrackerVideoPromptEncoderConfig`].
    mask_decoder_config (Union[`dict`, `Sam3TrackerVideoMaskDecoderConfig`], *optional*):
        Dictionary of configuration options used to initialize [`Sam3TrackerVideoMaskDecoderConfig`].
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation for parameter initialization.
    num_maskmem (`int`, *optional*, defaults to 7):
        The number of memory slots for the mask memory.
    sigmoid_scale_for_mem_enc (`float`, *optional*, defaults to 20.0):
        Scale factor for the sigmoid function in the memory encoder.
    sigmoid_bias_for_mem_enc (`float`, *optional*, defaults to -10.0):
        Bias for the sigmoid function in the memory encoder.
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
    max_object_pointers_in_encoder (`int`, *optional*, defaults to 16):
        The maximum number of object pointers in the encoder.
    max_cond_frame_num (`int`, *optional*, defaults to 4):
        Maximum number of conditioning frames to use in memory attention.
    enable_temporal_pos_encoding_for_object_pointers (`bool`, *optional*, defaults to `True`):
        Whether to enable temporal positional encoding for object pointers.
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
    memory_attention_rope_feat_sizes (`list[int]`, *optional*, defaults to `[72, 72]`):
        The feature sizes for the Rope positional encoding.
    memory_attention_rope_dropout (`float`, *optional*, defaults to 0.1):
        The dropout rate for the Rope positional encoding.
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
        The dimension of the embedding layer in the memory fuser.
    memory_fuser_intermediate_dim (`int`, *optional*, defaults to 1024):
        The dimension of the intermediate layer in the memory fuser.
    memory_fuser_kernel_size (`int`, *optional*, defaults to 7):
        The kernel size for the memory fuser.
    memory_fuser_padding (`int`, *optional*, defaults to 3):
        The padding for the memory fuser.
    memory_fuser_layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
        The initial value for the layer scale in the memory fuser.
    memory_fuser_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function in the memory fuser.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam3VisionConfig,
    ...     Sam3TrackerVideoPromptEncoderConfig,
    ...     Sam3TrackerVideoMaskDecoderConfig,
    ...     Sam3TrackerVideoModel,
    ... )

    >>> # Initializing a Sam3TrackerVideoConfig with `"facebook/sam3"` style configuration
    >>> configuration = Sam3TrackerVideoConfig()

    >>> # Initializing a Sam3TrackerVideoModel (with random weights) from the `"facebook/sam3"` style configuration
    >>> model = Sam3TrackerVideoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam3TrackerVideoConfig from a Sam3TrackerVideoVisionConfig, Sam3TrackerVideoPromptEncoderConfig, and Sam3TrackerVideoMaskDecoderConfig

    >>> # Initializing SAM3 tracker video vision encoder, memory attention, and memory encoder configurations
    >>> vision_config = Sam3TrackerVideoVisionConfig()
    >>> prompt_encoder_config = Sam3TrackerVideoPromptEncoderConfig()
    >>> mask_decoder_config = Sam3TrackerVideoMaskDecoderConfig()

    >>> config = Sam3TrackerVideoConfig(vision_config, prompt_encoder_config, mask_decoder_config)
    ```"""

    model_type = "sam3_tracker_video"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam3TrackerVideoPromptEncoderConfig,
        "mask_decoder_config": Sam3TrackerVideoMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    num_maskmem: int = 7
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    enable_occlusion_spatial_embedding: bool = True
    multimask_output_in_sam: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1
    multimask_output_for_tracking: bool = True
    max_object_pointers_in_encoder: int = 16
    max_cond_frame_num: int = 4
    enable_temporal_pos_encoding_for_object_pointers: bool = True
    memory_attention_hidden_size: int = 256
    memory_attention_num_layers: int = 4
    memory_attention_num_attention_heads: int = 1
    memory_attention_downsample_rate: int = 1
    memory_attention_feed_forward_hidden_size: int = 2048
    memory_attention_feed_forward_hidden_act: str = "relu"
    memory_attention_dropout: float | int = 0.1
    memory_attention_rope_theta: int = 10000
    memory_attention_rope_feat_sizes: list | None = None
    memory_attention_rope_dropout: float | int = 0.1
    memory_encoder_hidden_size: int = 256
    memory_encoder_output_channels: int = 64
    mask_downsampler_embed_dim: int = 256
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_stride: int = 2
    mask_downsampler_padding: int = 1
    mask_downsampler_total_stride: int = 16
    mask_downsampler_hidden_act: str = "gelu"
    memory_fuser_num_layers: int = 2
    memory_fuser_embed_dim: int = 256
    memory_fuser_intermediate_dim: int = 1024
    memory_fuser_kernel_size: int = 7
    memory_fuser_padding: int = 3
    memory_fuser_layer_scale_init_value: float = 1e-6
    memory_fuser_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        self.memory_attention_rope_feat_sizes = (
            [72, 72] if self.memory_attention_rope_feat_sizes is None else self.memory_attention_rope_feat_sizes
        )

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_vision_model"](
                backbone_feature_sizes=[[288, 288], [144, 144], [72, 72]]
            )

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam3TrackerVideoPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam3TrackerVideoPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3TrackerVideoMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3TrackerVideoMaskDecoderConfig()

        self.image_size = kwargs.pop("image_size", 1008)
        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the tracker video model."""
        return self.vision_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to sub-configs. Calculates feature sizes based on patch_size."""
        self.prompt_encoder_config.image_size = value
        self.vision_config.image_size = value

        patch_size = self.vision_config.backbone_config.patch_size
        self.vision_config.backbone_feature_sizes = [
            [4 * value // patch_size, 4 * value // patch_size],
            [2 * value // patch_size, 2 * value // patch_size],
            [value // patch_size, value // patch_size],
        ]
        self.memory_attention_rope_feat_sizes = [
            value // patch_size,
            value // patch_size,
        ]

        # keep the image_size in the __dict__ to save the value in the config file (backward compatibility)
        self.__dict__["image_size"] = value


class Sam3TrackerVideoInferenceCache(Sam2VideoInferenceCache):
    pass


class Sam3TrackerVideoInferenceSession(Sam2VideoInferenceSession):
    pass


class Sam3TrackerVideoLayerNorm(Sam2VideoLayerNorm):
    pass


class Sam3TrackerVideoPositionEmbeddingSine(Sam2VideoPositionEmbeddingSine):
    pass


class Sam3TrackerVideoAttention(Sam2VideoAttention):
    pass


class Sam3TrackerVideoTwoWayAttentionBlock(Sam2VideoTwoWayAttentionBlock):
    pass


class Sam3TrackerVideoFeedForward(Sam2VideoFeedForward):
    pass


class Sam3TrackerVideoImageSegmentationOutput(Sam2VideoImageSegmentationOutput):
    pass


class Sam3TrackerVideoSegmentationOutput(Sam2VideoSegmentationOutput):
    pass


class Sam3TrackerVideoPreTrainedModel(Sam2VideoPreTrainedModel):
    base_model_prefix = "tracker_model"


class Sam3TrackerVideoVisionRotaryEmbedding(Sam2VideoVisionRotaryEmbedding):
    pass


class Sam3TrackerVideoRoPEAttention(Sam2VideoRoPEAttention):
    pass


class Sam3TrackerVideoMemoryAttentionLayer(Sam2VideoMemoryAttentionLayer):
    pass


class Sam3TrackerVideoMemoryAttention(Sam2VideoMemoryAttention):
    pass


class Sam3TrackerVideoMemoryFuserCXBlock(Sam2VideoMemoryFuserCXBlock):
    pass


class Sam3TrackerVideoMemoryFuser(Sam2VideoMemoryFuser):
    pass


class Sam3TrackerVideoMaskDownSamplerLayer(Sam2VideoMaskDownSamplerLayer):
    pass


class Sam3TrackerVideoMaskDownSampler(Sam2VideoMaskDownSampler):
    pass


class Sam3TrackerVideoMemoryEncoder(Sam2VideoMemoryEncoder):
    pass


class Sam3TrackerVideoVisionEncoderOutput(Sam2VideoVisionEncoderOutput):
    pass


class Sam3TrackerVideoPositionalEmbedding(Sam2VideoPositionalEmbedding):
    pass


class Sam3TrackerVideoMaskEmbedding(Sam2VideoMaskEmbedding):
    pass


class Sam3TrackerVideoPromptEncoder(Sam2VideoPromptEncoder):
    pass


class Sam3TrackerVideoTwoWayTransformer(Sam2VideoTwoWayTransformer):
    pass


class Sam3TrackerVideoMaskDecoder(Sam2VideoMaskDecoder):
    pass


class Sam3TrackerVideoModel(Sam2VideoModel):
    _keys_to_ignore_on_load_unexpected = [r"^detector_model."]

    def __init__(self, config: Sam3TrackerVideoConfig, remove_vision_encoder: bool = False):
        r"""
        remove_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to remove the vision encoder. If True, the vision encoder will be set to None.
        """
        # loading from a sam3_video config
        if hasattr(config, "tracker_config") and config.tracker_config is not None:
            tracker_config = config.tracker_config
            if isinstance(tracker_config, dict):
                tracker_config = Sam3TrackerVideoConfig(**tracker_config)
            config = tracker_config
        Sam3TrackerVideoPreTrainedModel.__init__(config)
        self.shared_image_embedding = Sam3TrackerVideoPositionalEmbedding(config.prompt_encoder_config)
        self.vision_encoder = AutoModel.from_config(config.vision_config) if not remove_vision_encoder else None
        self.prompt_encoder = Sam3TrackerVideoPromptEncoder(config.prompt_encoder_config)
        # The module using it is not a PreTrainedModel subclass so we need this
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.mask_decoder = Sam3TrackerVideoMaskDecoder(config.mask_decoder_config)

        self.backbone_feature_sizes = config.vision_config.backbone_feature_sizes
        # a single token to indicate no memory embedding from previous frames
        self.hidden_dim = config.vision_config.fpn_hidden_size
        self.no_memory_embedding = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.config = config
        # For video sequence inference
        self.image_size = config.image_size
        self.memory_attention = Sam3TrackerVideoMemoryAttention(config)
        self.memory_encoder = Sam3TrackerVideoMemoryEncoder(config)
        self.no_memory_positional_encoding = torch.nn.Parameter(
            torch.zeros(1, 1, config.vision_config.fpn_hidden_size)
        )
        self.mem_dim = config.memory_encoder_output_channels
        self.num_maskmem = config.num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.memory_temporal_positional_encoding = torch.nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )

        self.no_object_pointer = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        # A conv layer to downsample the mask prompt to stride 4 (the same stride as
        # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
        # so that it can be fed into the SAM mask decoder to generate a pointer.
        self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        # a feedforward layer on SAM output tokens to turn them into object pointers
        self.object_pointer_proj = Sam3TrackerVideoFeedForward(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)

        if self.config.enable_temporal_pos_encoding_for_object_pointers:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.temporal_positional_encoding_projection_layer = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.temporal_positional_encoding_projection_layer = torch.nn.Identity()

        self.occlusion_spatial_embedding_parameter = None  # compatibility with Sam2
        if config.enable_occlusion_spatial_embedding:
            self.occlusion_spatial_embedding_parameter = torch.nn.Parameter(torch.zeros(1, self.mem_dim))

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3TrackerVideoVisionEncoderOutput:
        r"""
        pixel_values (`torch.FloatTensor`):
            Input pixel values of shape `(batch_size, num_channels, height, width)`.
        """
        vision_outputs: Sam3TrackerVideoVisionEncoderOutput = self.vision_encoder(
            pixel_values, return_dict=True, **kwargs
        )

        feature_maps = vision_outputs.fpn_hidden_states
        feature_maps_position_embeddings = vision_outputs.fpn_position_encoding

        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        feature_maps = list(feature_maps[:-1])
        feature_maps[0] = self.mask_decoder.conv_s0(feature_maps[0])
        feature_maps[1] = self.mask_decoder.conv_s1(feature_maps[1])

        # flatten NxCxHxW to HWxNxC
        feature_maps = [feature_map.flatten(2).permute(2, 0, 1) for feature_map in feature_maps]
        feature_maps_position_embeddings = [
            feature_map_position_embedding.flatten(2).permute(2, 0, 1)
            for feature_map_position_embedding in feature_maps_position_embeddings[:-1]
        ]
        vision_outputs.fpn_hidden_states = feature_maps
        vision_outputs.fpn_position_encoding = feature_maps_position_embeddings

        return vision_outputs


__all__ = [
    "Sam3TrackerVideoMaskDecoderConfig",
    "Sam3TrackerVideoPromptEncoderConfig",
    "Sam3TrackerVideoConfig",
    "Sam3TrackerVideoModel",
    "Sam3TrackerVideoInferenceSession",
    "Sam3TrackerVideoPreTrainedModel",
    "Sam3TrackerVideoProcessor",
]
