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
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from ...utils.generic import OutputRecorder, TransformersKwargs
from ...video_utils import VideoInput
from ..auto import CONFIG_MAPPING, AutoConfig
from ..sam2.configuration_sam2 import (
    Sam2MaskDecoderConfig,
    Sam2PromptEncoderConfig,
)
from ..sam2.modeling_sam2 import (
    Sam2FeedForward,
    Sam2ImageSegmentationOutput,
    Sam2LayerNorm,
    Sam2Model,
    Sam2SinePositionEmbedding,
    Sam2TwoWayAttentionBlock,
    eager_attention_forward,
)
from ..sam2.processing_sam2 import Sam2Processor


logger = logging.get_logger(__name__)


class Sam2VideoPromptEncoderConfig(Sam2PromptEncoderConfig):
    pass


class Sam2VideoMaskDecoderConfig(Sam2MaskDecoderConfig):
    pass


class Sam2VideoConfig(PretrainedConfig):
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
        num_maskmem (`int`, *optional*, defaults to 7):
            The number of memory slots for the mask memory.
        image_size (`int`, *optional*, defaults to 1024):
            The size of the input images.
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
        memory_attention_rope_feat_sizes (`list[int]`, *optional*, defaults to `[64, 64]`):
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

    model_type = "sam2_video"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam2VideoPromptEncoderConfig,
        "mask_decoder_config": Sam2VideoMaskDecoderConfig,
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
        enable_occlusion_spatial_embedding=True,
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        max_object_pointers_in_encoder=16,
        enable_temporal_pos_encoding_for_object_pointers=True,
        # memory attention
        memory_attention_hidden_size=256,
        memory_attention_num_layers=4,
        memory_attention_num_attention_heads=1,
        memory_attention_downsample_rate=1,
        memory_attention_feed_forward_hidden_size=2048,
        memory_attention_feed_forward_hidden_act="relu",
        memory_attention_dropout=0.1,
        memory_attention_rope_theta=10000,
        memory_attention_rope_feat_sizes=None,
        memory_attention_rope_dropout=0.1,
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
        memory_fuser_intermediate_dim=1024,
        memory_fuser_kernel_size=7,
        memory_fuser_padding=3,
        memory_fuser_layer_scale_init_value=1e-6,
        memory_fuser_hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        vision_config = vision_config if vision_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}
        memory_attention_rope_feat_sizes = (
            [64, 64] if memory_attention_rope_feat_sizes is None else memory_attention_rope_feat_sizes
        )

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "sam2_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        if isinstance(prompt_encoder_config, Sam2VideoPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, Sam2VideoMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        self.vision_config = vision_config
        self.prompt_encoder_config = Sam2VideoPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = Sam2VideoMaskDecoderConfig(**mask_decoder_config)

        self.initializer_range = initializer_range
        self.num_maskmem = num_maskmem  # default 1 input frame + 6 previous frames
        self.image_size = image_size
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.max_object_pointers_in_encoder = max_object_pointers_in_encoder
        # The next 4 are True for sam2.1 and False for sam2
        self.enable_occlusion_spatial_embedding = enable_occlusion_spatial_embedding
        self.enable_temporal_pos_encoding_for_object_pointers = enable_temporal_pos_encoding_for_object_pointers

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
        self.memory_attention_rope_dropout = memory_attention_rope_dropout

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
        self.memory_fuser_intermediate_dim = memory_fuser_intermediate_dim
        self.memory_fuser_kernel_size = memory_fuser_kernel_size
        self.memory_fuser_padding = memory_fuser_padding
        self.memory_fuser_layer_scale_init_value = memory_fuser_layer_scale_init_value
        self.memory_fuser_hidden_act = memory_fuser_hidden_act


class Sam2VideoInferenceCache:
    """Cache for vision features and model constants."""

    def __init__(
        self,
        inference_device: Union[torch.device, str] = "cpu",
        inference_state_device: Union[torch.device, str] = "cpu",
        max_vision_features_cache_size: int = 1,
    ):
        self.inference_device = inference_device
        self.inference_state_device = inference_state_device
        self.max_vision_features_cache_size = max_vision_features_cache_size

        self._vision_features = {}

    def cache_vision_features(self, frame_idx: int, features: dict):
        """Cache vision features with automatic device management."""
        cached = {}
        if len(self._vision_features) >= self.max_vision_features_cache_size:
            # remove the oldest frame
            self._vision_features.pop(min(self._vision_features.keys()))

        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                cached[key] = value.to(self.inference_state_device, non_blocking=True)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                cached[key] = [v.to(self.inference_state_device, non_blocking=True) for v in value]
            else:
                cached[key] = value
        self._vision_features[frame_idx] = cached

    def get_vision_features(self, frame_idx: int) -> Optional[dict]:
        """Get cached vision features, automatically moved to inference device."""
        if frame_idx not in self._vision_features:
            return None

        cached = self._vision_features[frame_idx]
        moved = {}
        for key, value in cached.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.inference_device, non_blocking=True)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                moved[key] = [v.to(self.inference_device, non_blocking=True) for v in value]
            else:
                moved[key] = value
        return moved

    def clear_all(self):
        """Clear all cached data."""
        self._vision_features.clear()


class Sam2VideoInferenceSession:
    r"""
    Manages video inference session parameters, state and cache.

    Args:
        video (`torch.FloatTensor`, *optional*):
            The video to process. No need to provide when streaming.
        video_height (`int`, *optional*):
            The height of the video.
        video_width (`int`, *optional*):
            The width of the video.
        inference_device (`torch.device`, *optional*, defaults to `"cpu"`):
            The device to use for inference.
        inference_state_device (`torch.device`, *optional*, defaults to `"cpu"`):
            The device to store the inference state on.
        video_storage_device (`torch.device`, *optional*, defaults to `"cpu"`):
            The device to store the video on.
        dtype (`torch.dtype`, *optional*, defaults to `"float32"`):
            The dtype to use for the video.
        max_vision_features_cache_size (`int`, *optional*, defaults to 1):
            The maximum number of vision features to cache.
    """

    def __init__(
        self,
        video: Optional[torch.FloatTensor] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        inference_device: Union[torch.device, str] = "cpu",
        inference_state_device: Union[torch.device, str] = "cpu",
        video_storage_device: Union[torch.device, str] = "cpu",
        dtype: Union[torch.dtype, str] = "float32",
        max_vision_features_cache_size: int = 1,
    ):
        # store as a dictionary to avoid double memory allocation with torch.cat when adding new frames
        self.processed_frames = (
            dict(enumerate(video.to(video_storage_device, dtype=dtype))) if video is not None else None
        )
        self.video_height = video_height
        self.video_width = video_width

        self.inference_device = inference_device
        self.inference_state_device = inference_state_device
        self.video_storage_device = video_storage_device
        self.dtype = dtype
        self.max_vision_features_cache_size = max_vision_features_cache_size

        # Cache for computed features
        self.cache = Sam2VideoInferenceCache(
            inference_device=self.inference_device,
            inference_state_device=self.inference_state_device,
            max_vision_features_cache_size=self.max_vision_features_cache_size,
        )

        # Persistent object tracking state
        self._obj_id_to_idx = OrderedDict()
        self._obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        # Persistent user inputs
        self.point_inputs_per_obj = {}
        self.mask_inputs_per_obj = {}

        # Persistent model outputs/history
        self.output_dict_per_obj = {}
        self.frames_tracked_per_obj = {}

        # Session state flags
        self.obj_with_new_inputs = []

    @property
    def num_frames(self) -> Optional[int]:
        return len(self.processed_frames) if self.processed_frames is not None else None

    # Object management
    def obj_id_to_idx(self, obj_id: int) -> int:
        """Map object ID to index, creating new entry if needed."""
        obj_idx = self._obj_id_to_idx.get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        obj_idx = len(self._obj_id_to_idx)
        self._obj_id_to_idx[obj_id] = obj_idx
        self._obj_idx_to_id[obj_idx] = obj_id
        self.obj_ids = list(self._obj_id_to_idx)

        self.point_inputs_per_obj[obj_idx] = {}
        self.mask_inputs_per_obj[obj_idx] = {}
        self.output_dict_per_obj[obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        self.frames_tracked_per_obj[obj_idx] = {}

        return obj_idx

    # Video Inference specific functions
    def obj_idx_to_id(self, obj_idx: int) -> int:
        """Map model-side object index to client-side object id."""
        return self._obj_idx_to_id[obj_idx]

    def get_obj_num(self) -> int:
        """Get the total number of unique object ids received so far in this session."""
        return len(self._obj_idx_to_id)

    # Input management with device handling
    def add_point_inputs(self, obj_idx: int, frame_idx: int, inputs: dict):
        """Add point inputs with automatic device placement."""
        device_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                device_inputs[key] = value.to(self.inference_device, non_blocking=True)
            else:
                device_inputs[key] = value
        self.point_inputs_per_obj[obj_idx][frame_idx] = device_inputs

    def remove_point_inputs(self, obj_idx: int, frame_idx: int):
        """Remove point inputs."""
        self.point_inputs_per_obj[obj_idx].pop(frame_idx, None)

    def add_mask_inputs(self, obj_idx: int, frame_idx: int, inputs: torch.Tensor):
        """Add mask inputs with automatic device placement."""
        self.mask_inputs_per_obj[obj_idx][frame_idx] = inputs.to(
            self.inference_device, dtype=self.dtype, non_blocking=True
        )

    def remove_mask_inputs(self, obj_idx: int, frame_idx: int):
        """Remove mask inputs."""
        self.mask_inputs_per_obj[obj_idx].pop(frame_idx, None)

    # Output management with smart device placement
    def store_output(
        self,
        obj_idx: int,
        frame_idx: int,
        output_key: Optional[str] = None,
        output_value: Optional[Union[torch.Tensor, dict]] = None,
        is_conditioning_frame: bool = True,
    ):
        """
        Store output with smart device management.
        If output_key is None, the output is stored as a dictionary.

        Args:
            obj_idx (int): The index of the object.
            frame_idx (int): The index of the frame.
            output_key (Optional[str]): The key of the output. If None, the output is stored as a dictionary.
            output_value (Optional[Union[torch.Tensor, dict]]): The value of the output.
            is_conditioning_frame (bool): Whether the output is for a conditioning frame.
        """
        storage_key = "cond_frame_outputs" if is_conditioning_frame else "non_cond_frame_outputs"

        if output_key is None and isinstance(output_value, dict):
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx] = {}
            for key, value in output_value.items():
                self.store_output(obj_idx, frame_idx, key, value, is_conditioning_frame)
            return

        # Device placement: small tensors stay on inference device, large ones go to inference state device
        if output_key in ["object_pointer", "object_score_logits"]:  # Small tensors
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx][output_key] = output_value
        elif isinstance(output_value, torch.Tensor):  # Large tensors like masks, features
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx][output_key] = output_value.to(
                self.inference_state_device, non_blocking=True
            )
        else:
            self.output_dict_per_obj[obj_idx][storage_key][frame_idx][output_key] = output_value

    def get_output(
        self,
        obj_idx: int,
        frame_idx: int,
        output_key: str,
        is_conditioning_frame: bool = True,
    ):
        """
        Get output with smart device management.

        Args:
            obj_idx (int): The index of the object.
            frame_idx (int): The index of the frame.
            output_key (str): The key of the output.
            is_conditioning_frame (bool): Whether the output is for a conditioning frame.
        """
        storage_key = "cond_frame_outputs" if is_conditioning_frame else "non_cond_frame_outputs"
        out = self.output_dict_per_obj[obj_idx][storage_key].get(frame_idx, None)
        # move to inference device if needed
        if out is None:
            return None
        value = out[output_key]
        if isinstance(value, torch.Tensor):
            value = value.to(self.inference_device, non_blocking=True)
        return value

    # Video frame management
    def add_new_frame(self, pixel_values: torch.Tensor, frame_idx: Optional[int] = None) -> int:
        """Add new frame with automatic device placement."""
        pixel_values = pixel_values.to(self.video_storage_device, dtype=self.dtype, non_blocking=True)
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.squeeze(0)

        if frame_idx is None:
            frame_idx = len(self.processed_frames) if self.processed_frames is not None else 0

        if self.processed_frames is None:
            self.processed_frames = {frame_idx: pixel_values}
        else:
            self.processed_frames[frame_idx] = pixel_values

        return frame_idx

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        """Get frame from video."""
        return self.processed_frames[frame_idx].to(self.inference_device, non_blocking=True)

    def reset_tracking_data(self):
        """Reset tracking data but keep cache."""
        self._obj_id_to_idx.clear()
        self._obj_idx_to_id.clear()
        self.obj_ids.clear()
        self.point_inputs_per_obj.clear()
        self.mask_inputs_per_obj.clear()
        self.output_dict_per_obj.clear()
        self.frames_tracked_per_obj.clear()
        self.obj_with_new_inputs = []
        # Note: cache and video data are preserved

    def reset_inference_session(self):
        """Reset tracking data and cache."""
        self._obj_id_to_idx.clear()
        self._obj_idx_to_id.clear()
        self.obj_ids.clear()
        self.point_inputs_per_obj.clear()
        self.mask_inputs_per_obj.clear()
        self.output_dict_per_obj.clear()
        self.frames_tracked_per_obj.clear()
        self.obj_with_new_inputs = []
        self.cache.clear_all()


class Sam2VideoProcessor(Sam2Processor):
    r"""
    Constructs a SAM2 processor which wraps a SAM2 image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`Sam2VideoProcessor`] offers all the functionalities of [`Sam2ImageProcessorFast`] and [`Sam2VideoProcessor`]. See the docstring of
    [`~Sam2ImageProcessorFast.__call__`] and [`~Sam2VideoProcessor.__call__`] for more information.

    Args:
        image_processor (`Sam2ImageProcessorFast`):
            An instance of [`Sam2ImageProcessorFast`].
        video_processor (`Sam2VideoVideoProcessor`):
            An instance of [`Sam2VideoVideoProcessor`].
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        point_pad_value (`int`, *optional*, defaults to -10):
            The value used for padding input points.
    """

    attributes = ["image_processor", "video_processor"]
    image_processor_class = "Sam2ImageProcessorFast"
    video_processor_class = "Sam2VideoVideoProcessor"

    def __init__(
        self, image_processor, video_processor, target_size: Optional[int] = None, point_pad_value: int = -10, **kwargs
    ):
        ProcessorMixin.__init__(self, image_processor, video_processor, **kwargs)
        self.point_pad_value = point_pad_value
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    def init_video_session(
        self,
        video: Optional[VideoInput] = None,
        inference_device: Union[str, "torch.device"] = "cpu",
        inference_state_device: Union[str, "torch.device"] = None,
        processing_device: Union[str, "torch.device"] = None,
        video_storage_device: Union[str, "torch.device"] = None,
        max_vision_features_cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a video session for inference.
        If a video is provided (async inference), the video will be processed and stored on the `video_storage_device`.

        Args:
            video (`VideoInput`, *optional*):
                The video to process. No need to provide when streaming.
            inference_device (`str` or `torch.device`, *optional*, defaults to "cpu"):
                The device to use for inference.
            inference_state_device (`str` or `torch.device`, *optional*):
                The device to store the inference state on.
            processing_device (`str` or `torch.device`, *optional*):
                The device to use for video processing.
            video_storage_device (`str` or `torch.device`, *optional*):
                The device to store the processed video frames on.
            max_vision_features_cache_size (`int`, *optional*, defaults to 1):
                The maximum number of vision features to cache.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The torch dtype to use for the whole session.
        """
        video_storage_device = video_storage_device if video_storage_device is not None else inference_device
        inference_state_device = inference_state_device if inference_state_device is not None else inference_device
        processing_device = processing_device if processing_device is not None else inference_device
        pixel_values_video = None
        video_height = None
        video_width = None
        if video is not None:
            processed_video = self.video_processor(videos=video, device=processing_device, return_tensors="pt")
            pixel_values_video = processed_video.pixel_values_videos[0]
            video_height = processed_video.original_sizes[0][0]
            video_width = processed_video.original_sizes[0][1]
        inference_session = Sam2VideoInferenceSession(
            video=pixel_values_video,
            video_height=video_height,
            video_width=video_width,
            inference_device=inference_device,
            video_storage_device=video_storage_device,
            inference_state_device=inference_state_device,
            dtype=dtype,
            max_vision_features_cache_size=max_vision_features_cache_size,
        )
        return inference_session

    def add_inputs_to_inference_session(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: Union[list[int], int],
        input_points: Optional[Union[list[list[list[list[float]]]], torch.Tensor]] = None,
        input_labels: Optional[Union[list[list[list[int]]], torch.Tensor]] = None,
        input_boxes: Optional[Union[list[list[list[float]]], torch.Tensor]] = None,
        input_masks: Optional[Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]]] = None,
        original_size: Optional[tuple[int, int]] = None,
        clear_old_inputs: bool = True,
    ) -> Sam2VideoInferenceSession:
        """
        Process new points, boxes, or masks for a video frame and add them to the inference session.

        Args:
            inference_session (`Sam2VideoInferenceSession`):
                The inference session for the video.
            frame_idx (`int`):
                The index of the frame to process.
            obj_ids (`list[int]` or `int`):
                The object ID(s) to associate with the points or box.
                These can be any integers and can be reused later on to specify an object.
            input_points (`list[list[list[list[float]]]]`, `torch.Tensor`, *optional*):
                The points to add to the frame.
            input_labels (`list[list[list[int]]]`, `torch.Tensor`, *optional*):
                The labels for the points.
            input_boxes (`list[list[list[float]]]`, `torch.Tensor`, *optional*):
                The bounding boxes to add to the frame.
            input_masks (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, or `list[torch.Tensor]`, *optional*):
                The mask(s) to add to the frame.
            original_size (`tuple[int, int]`, *optional*):
                The original size of the video. Provide when streaming.
            clear_old_inputs (`bool`, *optional*, defaults to `True`):
                Whether to clear old inputs for the object.
        """

        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]

        # Validate inputs
        if (input_points is not None) != (input_labels is not None):
            raise ValueError("points and labels must be provided together")
        if input_points is None and input_boxes is None and input_masks is None:
            raise ValueError("at least one of points, boxes, or masks must be provided as input")
        if input_masks is not None and (input_points is not None or input_boxes is not None):
            raise ValueError("masks cannot be provided together with points or boxes")

        if input_masks is not None:
            return self.process_new_mask_for_video_frame(inference_session, frame_idx, obj_ids, input_masks)
        else:
            return self.process_new_points_or_boxes_for_video_frame(
                inference_session,
                frame_idx,
                obj_ids,
                input_points,
                input_labels,
                input_boxes,
                original_size,
                clear_old_inputs,
            )

    def process_new_points_or_boxes_for_video_frame(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_points: Optional[Union[list[list[list[list[float]]]], torch.Tensor]] = None,
        input_labels: Optional[Union[list[list[list[int]]], torch.Tensor]] = None,
        input_boxes: Optional[Union[list[list[list[float]]], torch.Tensor]] = None,
        original_size: Optional[tuple[int, int]] = None,
        clear_old_inputs: bool = True,
    ) -> Sam2VideoInferenceSession:
        """
        Process new points or boxes for a video frame and add them to the inference session.

        Args:
            inference_session (`Sam2VideoInferenceSession`):
                The inference session for the video.
            frame_idx (`int`):
                The index of the frame to process.
            obj_ids (`list[int]`):
                The object ID(s) to associate with the points or box.
                These can be any integers and can be reused later on to specify an object.
            input_points (`list[list[list[list[float]]]]`, `torch.Tensor`, *optional*):
                The points to add to the frame.
            input_labels (`list[list[list[int]]]`, `torch.Tensor`, *optional*):
                The labels for the points.
            input_boxes (`list[list[list[float]]]`, `torch.Tensor`, *optional*):
                The bounding boxes to add to the frame.
            original_size (`tuple[int, int]`, *optional*):
                The original size of the video. Provide when streaming.
            clear_old_inputs (`bool`, *optional*, defaults to `True`):
                Whether to clear old inputs for the object.
        """
        if original_size is not None:
            inference_session.video_height = original_size[0]
            inference_session.video_width = original_size[1]
        elif inference_session.video_height is None or inference_session.video_width is None:
            raise ValueError("original_size must be provided when adding points or boxes on a first streamed frame")

        original_sizes = [[inference_session.video_height, inference_session.video_width]]

        encoded_inputs = self(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            return_tensors="pt",
        )
        input_points = encoded_inputs.get("input_points", None)
        input_labels = encoded_inputs.get("input_labels", None)
        input_boxes = encoded_inputs.get("input_boxes", None)

        if input_points is not None:
            if input_points.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of points ({input_points.shape[1]})"
                )
        else:
            input_points = torch.zeros(1, len(obj_ids), 0, 2, dtype=torch.float32)
        if input_labels is not None:
            if input_labels.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of labels ({input_labels.shape[1]})"
                )
        else:
            input_labels = torch.zeros(1, len(obj_ids), 0, dtype=torch.int32)
        if input_boxes is not None:
            if input_boxes.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of boxes ({input_boxes.shape[1]})"
                )

        if input_boxes is not None:
            if not clear_old_inputs:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            box_coords = input_boxes.reshape(1, -1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32).repeat(1, box_coords.shape[1], 1)
            input_points = torch.cat([box_coords, input_points], dim=2)
            input_labels = torch.cat([box_labels, input_labels], dim=2)

        for obj_id, idx in zip(obj_ids, range(len(obj_ids))):
            obj_idx = inference_session.obj_id_to_idx(obj_id)
            input_points_for_obj = input_points[:, idx, :, :].unsqueeze(1)
            input_labels_for_obj = input_labels[:, idx, :].unsqueeze(1)
            # Handle existing points
            if not clear_old_inputs:
                existing_points = inference_session.point_inputs_per_obj[obj_idx].get(frame_idx, None)
                if existing_points is not None:
                    # Concatenate with existing points
                    input_points_for_obj = torch.cat(
                        [existing_points["point_coords"].to(input_points_for_obj.device), input_points_for_obj], dim=2
                    )
                    input_labels_for_obj = torch.cat(
                        [existing_points["point_labels"].to(input_labels_for_obj.device), input_labels_for_obj], dim=2
                    )
            point_inputs = {
                "point_coords": input_points_for_obj,
                "point_labels": input_labels_for_obj,
            }

            inference_session.add_point_inputs(obj_idx, frame_idx, point_inputs)
            inference_session.remove_mask_inputs(obj_idx, frame_idx)  # Clear any mask inputs

        inference_session.obj_with_new_inputs = obj_ids

    def process_new_mask_for_video_frame(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_masks: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]],
    ):
        """
        Add new mask to a frame and add them to the inference session.

        Args:
            inference_session (`Sam2VideoInferenceSession`):
                The inference session for the video.
            frame_idx (`int`):
                The index of the frame to process.
            obj_ids (`list[int]`):
                The object ID(s) to associate with the mask.
                These can be any integers and can be reused later on to specify an object.
            input_masks (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, or `list[torch.Tensor]`):
                The mask(s) to add to the frame.
        """
        if not isinstance(input_masks, list):
            input_masks = [input_masks]
        if len(input_masks) != len(obj_ids):
            raise ValueError(
                f"Number of object ids ({len(obj_ids)}) does not match number of masks ({len(input_masks)})"
            )

        for obj_id, mask in zip(obj_ids, input_masks):
            obj_idx = inference_session.obj_id_to_idx(obj_id)

            device = inference_session.inference_device

            # Process mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            nb_dim = mask.dim()
            if nb_dim > 4 or nb_dim < 2:
                raise ValueError(f"Mask has an unsupported number of dimensions: {nb_dim}")
            for i in range(4 - nb_dim):
                mask = mask.unsqueeze(0)

            mask_H, mask_W = mask.shape[-2:]
            mask_inputs_orig = mask.to(device)
            mask_inputs_orig = mask_inputs_orig.float().to(device)

            # Resize mask if needed
            if mask_H != self.target_size or mask_W != self.target_size:
                mask_inputs = torch.nn.functional.interpolate(
                    mask_inputs_orig,
                    size=(self.target_size, self.target_size),
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
                mask_inputs = (mask_inputs >= 0.5).float()
            else:
                mask_inputs = mask_inputs_orig

            inference_session.add_mask_inputs(obj_idx, frame_idx, mask_inputs)
            inference_session.remove_point_inputs(obj_idx, frame_idx)  # Clear any point inputs

        inference_session.obj_with_new_inputs = obj_ids


class Sam2VideoLayerNorm(Sam2LayerNorm):
    pass


class Sam2VideoPositionEmbeddingSine(Sam2SinePositionEmbedding):
    pass


class Sam2VideoTwoWayAttentionBlock(Sam2TwoWayAttentionBlock):
    pass


class Sam2VideoFeedForward(Sam2FeedForward):
    pass


class Sam2VideoImageSegmentationOutput(Sam2ImageSegmentationOutput):
    r"""
    iou_scores (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_masks)`):
        The Intersection over Union (IoU) scores of the predicted masks.
    pred_masks (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_masks, height, width)`):
        The predicted low-resolution masks. This is an alias for `low_res_masks`. These masks need to be post-processed
        by the processor to be brought to the original image size.
    object_score_logits (`torch.FloatTensor` of shape `(batch_size, point_batch_size, 1)`):
        Logits for the object score, indicating if an object is present.
    image_embeddings (`tuple(torch.FloatTensor)`):
        The features from the FPN, which are used by the mask decoder. This is a tuple of `torch.FloatTensor` where each
        tensor has shape `(batch_size, channels, height, width)`.
    vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, height, width, hidden_size)`.
        Hidden-states of the vision model at the output of each stage.
    vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights of the vision model.
    mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights of the mask decoder.
    high_res_masks (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_masks, image_size, image_size)`, *optional*):
        The predicted masks, upscaled to the original image size. Only used for Sam2VideoModel.
    object_pointer (`torch.FloatTensor` of shape `(batch_size, point_batch_size, hidden_size)`, *optional*):
        A tensor representing the object pointer, used for tracking in videos. Only used for Sam2VideoModel.
    """

    high_res_masks: Optional[torch.FloatTensor] = None
    object_pointer: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(custom_intro="Base class for the Sam2 model's output.")
class Sam2VideoSegmentationOutput(ModelOutput):
    r"""
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
        The predicted masks stored at the model's resolution.
    frame_idx (`int`):
        The frame index of the video.
    """

    pred_masks: Optional[torch.FloatTensor] = None
    frame_idx: Optional[int] = None


@auto_docstring
class Sam2VideoPreTrainedModel(PreTrainedModel):
    config_class = Sam2VideoConfig
    base_model_prefix = "sam2_video"
    main_input_name = "pixel_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_attention_backend = True

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
        elif isinstance(module, (nn.LayerNorm, Sam2VideoLayerNorm)):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Sam2VideoModel):
            if module.no_memory_positional_encoding is not None:
                module.no_memory_positional_encoding.data.zero_()
            if module.memory_temporal_positional_encoding is not None:
                module.memory_temporal_positional_encoding.data.zero_()
            if module.no_object_pointer is not None:
                module.no_object_pointer.data.zero_()
            if module.occlusion_spatial_embedding_parameter is not None:
                module.occlusion_spatial_embedding_parameter.data.zero_()
        if isinstance(module, Sam2VideoMemoryFuserCXBlock):
            if module.scale is not None:
                module.scale.data.zero_()


class Sam2VideoVisionRotaryEmbedding(nn.Module):
    """
    Vision Rotary Position Embedding for SAM2, following transformers library standards.
    Supports 2D (axial) rotary embeddings for spatial dimensions.
    """

    def __init__(self, config: Sam2VideoConfig):
        super().__init__()
        dim = config.memory_attention_hidden_size // (
            config.memory_attention_downsample_rate * config.memory_attention_num_attention_heads
        )
        # Ensure even dimension for proper axial splitting
        if dim % 4 != 0:
            raise ValueError("Dimension must be divisible by 4 for axial RoPE")
        end_x, end_y = config.memory_attention_rope_feat_sizes
        freqs = 1.0 / (config.memory_attention_rope_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

        # Generate 2D position indices for axial rotary embedding
        flattened_indices = torch.arange(end_x * end_y, dtype=torch.long)
        x_positions = flattened_indices % end_x
        y_positions = torch.div(flattened_indices, end_x, rounding_mode="floor")
        freqs_x = torch.outer(x_positions, freqs).float()
        freqs_y = torch.outer(y_positions, freqs).float()
        inv_freq = torch.cat([freqs_x, freqs_y], dim=-1)
        inv_freq = inv_freq.repeat_interleave(2, dim=-1)
        # directly register the cos and sin embeddings as we have a fixed feature shape
        self.register_buffer("rope_embeddings_cos", inv_freq.cos(), persistent=False)
        self.register_buffer("rope_embeddings_sin", inv_freq.sin(), persistent=False)

    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        # As the feature map size is fixed, we can just return the pre-computed embeddings.
        return self.rope_embeddings_cos, self.rope_embeddings_sin


def rotate_pairwise(x):
    """
    pairwise rotation of the hidden dims of the input. Differerent from Llama Half-Tensor Rotation.

    This is an optimized version of the following more explicit implementation:
    ```python
    x_rotated = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    x_rotated[..., ::2] = -x[..., 1::2]
    x_rotated[..., 1::2] = x[..., ::2]
    return x_rotated
    ```
    """
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


# TODO: This leads to ~1e-07 max diff and ~1e-09 avg diff for q_embed and k_embed from the original implementation, most likely due to the use of complex tensors in the original implementation.
def apply_rotary_pos_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_k_exclude_rope: int = 0,
    repeat_freqs_k: bool = False,
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
    k_rot, k_pass = k[..., : k.shape[-2] - num_k_exclude_rope, :], k[..., k.shape[-2] - num_k_exclude_rope :, :]
    q_embed = q.float()  # force upscale to float32 as in the original implementation
    q_embed = (q_embed * cos) + (rotate_pairwise(q_embed) * sin)
    if k_rot.shape[-2] == 0:
        # Handle case where keys might be empty due to dropout
        return q_embed.type_as(q), torch.cat([k_rot, k_pass], dim=-2)

    # Handle key tensor - may need to repeat frequencies if different sequence length
    if repeat_freqs_k and k_rot.shape[-2] != q.shape[-2]:
        # Repeat cos/sin to match key sequence length
        repeat_factor = k_rot.shape[-2] // q.shape[-2]
        cos_k = cos.repeat(1, 1, repeat_factor, 1)
        sin_k = sin.repeat(1, 1, repeat_factor, 1)
    else:
        cos_k = cos
        sin_k = sin

    # Apply rotary embedding to keys
    k_embed = k_rot.float()  # force upscale to float32 as in the original implementation
    k_embed = (k_embed * cos_k) + (rotate_pairwise(k_embed) * sin_k)
    # Concatenate back to full shape
    k_embed = torch.cat([k_embed.type_as(k), k_pass], dim=-2)
    return q_embed.type_as(q), k_embed


class Sam2VideoRoPEAttention(nn.Module):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        config: Sam2VideoConfig,
        kv_in_dim: Optional[int] = None,
        rope_k_repeat=False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.memory_attention_hidden_size
        self.internal_dim = self.hidden_size // config.memory_attention_downsample_rate
        self.num_attention_heads = config.memory_attention_num_attention_heads
        self.head_dim = self.internal_dim // config.memory_attention_num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else self.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, self.hidden_size)

        self.rope_k_repeat = rope_k_repeat
        self.dropout_p = config.memory_attention_rope_dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor:
        # Input projections
        batch_size, point_batch_size = query.shape[:2]
        new_shape = (batch_size * point_batch_size, -1, self.num_attention_heads, self.head_dim)

        query = self.q_proj(query).view(*new_shape).transpose(1, 2)
        key = self.k_proj(key).view(*new_shape).transpose(1, 2)
        value = self.v_proj(value).view(*new_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # Apply rotary position encoding, excluding some keys if specified
        query, key = apply_rotary_pos_emb_2d(
            query, key, cos, sin, repeat_freqs_k=self.rope_k_repeat, num_k_exclude_rope=num_k_exclude_rope
        )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0 if not self.training else self.dropout_p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(
            batch_size, point_batch_size, -1, self.num_attention_heads * self.head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Sam2VideoMemoryAttentionLayer(nn.Module):
    def __init__(self, config: Sam2VideoConfig):
        super().__init__()
        hidden_size = config.memory_attention_hidden_size
        self.self_attn = Sam2VideoRoPEAttention(config)
        self.cross_attn_image = Sam2VideoRoPEAttention(config, kv_in_dim=64, rope_k_repeat=True)

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

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        key_point_embedding: Tensor,
        rope_position_embeddings: tuple[Tensor, Tensor],
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        # Self-Attention
        query = self.layer_norm1(queries)
        query, _ = self.self_attn(query=query, key=query, value=query, position_embeddings=rope_position_embeddings)
        queries = queries + self.dropout1(query)

        # Cross-Attention
        query = self.layer_norm2(queries)
        query, _ = self.cross_attn_image(
            query=query,
            key=keys + key_point_embedding,
            value=keys,
            position_embeddings=rope_position_embeddings,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        queries = queries + self.dropout2(query)
        # MLP
        query = self.layer_norm3(queries)
        query = self.linear2(self.dropout(self.activation(self.linear1(query))))
        queries = queries + self.dropout3(query)
        return queries


class Sam2VideoMemoryAttention(nn.Module):
    def __init__(self, config: Sam2VideoConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Sam2VideoMemoryAttentionLayer(config) for _ in range(config.memory_attention_num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.memory_attention_hidden_size)
        self.rotary_emb = Sam2VideoVisionRotaryEmbedding(config=config)

    def forward(
        self,
        current_vision_features: torch.Tensor,
        memory: torch.Tensor,
        current_vision_position_embeddings: Optional[Tensor] = None,
        memory_posision_embeddings: Optional[Tensor] = None,
        num_object_pointer_tokens: int = 0,
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
        output = current_vision_features
        if current_vision_position_embeddings is not None:
            output = output + 0.1 * current_vision_position_embeddings

        # Convert to batch first
        output = output.transpose(0, 1)
        memory = memory.transpose(0, 1).unsqueeze(1)
        memory_posision_embeddings = memory_posision_embeddings.transpose(0, 1).unsqueeze(1)
        rope_position_embeddings = self.rotary_emb()
        for layer in self.layers:
            output = layer(
                queries=output.unsqueeze(1) if output.ndim == 3 else output,
                keys=memory,
                key_point_embedding=memory_posision_embeddings,
                rope_position_embeddings=rope_position_embeddings,
                num_k_exclude_rope=num_object_pointer_tokens,
            )

        normed_output = self.layer_norm(output)

        # Convert back to seq first
        normed_output = normed_output.transpose(0, 1)

        return normed_output


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class Sam2VideoMemoryFuserCXBlock(GradientCheckpointingLayer):
    def __init__(self, config: Sam2VideoConfig):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            config.memory_fuser_embed_dim,
            config.memory_fuser_embed_dim,
            kernel_size=config.memory_fuser_kernel_size,
            padding=config.memory_fuser_padding,
            groups=config.memory_fuser_embed_dim,
        )  # depthwise conv
        self.layer_norm = Sam2VideoLayerNorm(config.memory_fuser_embed_dim, eps=1e-6, data_format="channels_first")
        self.activation = ACT2FN[config.memory_fuser_hidden_act]
        self.pointwise_conv1 = nn.Linear(
            config.memory_fuser_embed_dim, config.memory_fuser_intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.pointwise_conv2 = nn.Linear(config.memory_fuser_intermediate_dim, config.memory_fuser_embed_dim)
        self.scale = nn.Parameter(
            config.memory_fuser_layer_scale_init_value * torch.ones(config.memory_fuser_embed_dim),
            requires_grad=True,
        )

    def forward(self, hidden_states):
        input = hidden_states
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.scale * hidden_states
        hidden_states = hidden_states.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        hidden_states = input + hidden_states
        return hidden_states


class Sam2VideoMemoryFuser(nn.Module):
    def __init__(self, config: Sam2VideoConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Sam2VideoMemoryFuserCXBlock(config) for _ in range(config.memory_fuser_num_layers)]
        )

    def forward(self, hidden_states):
        # normally hidden_states: (N, C, H, W)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Sam2VideoMaskDownSamplerLayer(nn.Module):
    def __init__(self, config: Sam2VideoConfig, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.mask_downsampler_kernel_size,
            stride=config.mask_downsampler_stride,
            padding=config.mask_downsampler_padding,
        )
        self.layer_norm = Sam2VideoLayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.activation = ACT2FN[config.mask_downsampler_hidden_act]

    def forward(self, x):
        return self.activation(self.layer_norm(self.conv(x)))


class Sam2VideoMaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(self, config: Sam2VideoConfig):
        super().__init__()

        num_layers = int(math.log2(config.mask_downsampler_total_stride) // math.log2(config.mask_downsampler_stride))

        self.layers = nn.ModuleList()
        self.activation = ACT2FN[config.mask_downsampler_hidden_act]
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (config.mask_downsampler_stride**2)
            self.layers.append(Sam2VideoMaskDownSamplerLayer(config, mask_in_chans, mask_out_chans))
            mask_in_chans = mask_out_chans

        self.final_conv = nn.Conv2d(mask_out_chans, config.mask_downsampler_embed_dim, kernel_size=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return x


class Sam2VideoMemoryEncoder(nn.Module):
    def __init__(self, config: Sam2VideoConfig):
        super().__init__()

        hidden_size = config.memory_encoder_hidden_size
        output_channels = config.memory_encoder_output_channels
        self.mask_downsampler = Sam2VideoMaskDownSampler(config)
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.memory_fuser = Sam2VideoMemoryFuser(config)
        self.position_encoding = Sam2VideoPositionEmbeddingSine(num_pos_feats=output_channels // 2, normalize=True)
        self.projection = nn.Conv2d(hidden_size, output_channels, kernel_size=1)

    def forward(
        self,
        vision_features: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        masks = self.mask_downsampler(masks)
        ## Fuse pixel_features and downsampled masks

        vision_features = self.feature_projection(vision_features)
        vision_features = vision_features + masks
        vision_features = self.memory_fuser(vision_features)
        vision_features = self.projection(vision_features)

        vision_pos_enc = self.position_encoding(vision_features.shape, vision_features.device, vision_features.dtype)

        return vision_features, vision_pos_enc


# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


@auto_docstring
class Sam2VideoModel(Sam2Model):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]
    # need to be ignored, as it's a buffer and will not be correctly detected as tied weight
    _keys_to_ignore_on_load_missing = ["prompt_encoder.shared_embedding.positional_embedding"]
    _keys_to_ignore_on_load_unexpected = []
    _can_record_outputs = {"mask_decoder_attentions": OutputRecorder(Sam2VideoTwoWayAttentionBlock, index=2)}

    def __init__(self, config: Sam2VideoConfig):
        super().__init__(config)
        self.config = config
        # For video sequence inference
        self.image_size = config.image_size
        self.memory_attention = Sam2VideoMemoryAttention(config)
        self.memory_encoder = Sam2VideoMemoryEncoder(config)
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
        self.object_pointer_proj = Sam2VideoFeedForward(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)

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

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    def _prepare_vision_features(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Prepare vision features for a frame."""

        # Check if features are cached
        if cached_features := inference_session.cache.get_vision_features(frame_idx):
            vision_feats = cached_features["vision_feats"]
            vision_pos_embeds = cached_features["vision_pos_embeds"]
        else:
            # Compute features using image encoder
            image_batch = inference_session.get_frame(frame_idx).unsqueeze(0)  # Add batch dimension
            vision_feats, vision_pos_embeds, _, _ = self.get_image_features(image_batch)
            # Cache features
            inference_session.cache.cache_vision_features(
                frame_idx, {"vision_feats": vision_feats, "vision_pos_embeds": vision_pos_embeds}
            )

        # Expand to batch size if needed
        if batch_size > 1:
            vision_feats = vision_feats.expand(batch_size, -1, -1, -1)
            vision_pos_embeds = [pe.expand(batch_size, -1, -1, -1) for pe in vision_pos_embeds]

        return vision_feats, vision_pos_embeds

    def _single_frame_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam2VideoImageSegmentationOutput:
        """
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box
        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decoder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        """
        if not ((pixel_values is None) ^ (image_embeddings is None)):
            raise ValueError("Exactly one of pixel_values or image_embeddings must be provided.")
        if input_points is not None and input_boxes is not None:
            if input_points.shape[1] != input_boxes.shape[1]:
                raise ValueError(
                    f"You should provide as many bounding boxes as input points per box. Got {input_points.shape[1]} and {input_boxes.shape[1]}."
                )
        elif input_points is not None:
            num_objects = input_points.shape[1]
        elif input_boxes is not None:
            num_objects = input_boxes.shape[1]
        elif input_masks is not None:
            num_objects = input_masks.shape[1]
        else:
            num_objects = 1

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings[-1].shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            feature_maps, _, vision_hidden_states, vision_attentions = self.get_image_features(
                pixel_values,
                **kwargs,
            )

            # add no memory embedding to the last feature map
            feature_maps[-1] = feature_maps[-1] + self.no_memory_embedding

            # reshape feature maps to the same shape as the backbone feature sizes
            image_embeddings = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(feature_maps, self.backbone_feature_sizes)
            ]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is None and input_boxes is None:
            # If no points are provide, pad with an empty point (with label -1)
            input_points = torch.zeros(
                batch_size, 1, 1, 2, dtype=image_embeddings[-1].dtype, device=image_embeddings[-1].device
            )
            input_labels = -torch.ones(batch_size, 1, 1, dtype=torch.int32, device=image_embeddings[-1].device)

        if input_masks is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            if input_masks.shape[-2:] != self.prompt_encoder.mask_input_size:
                input_masks = F.interpolate(
                    input_masks.float(),
                    size=self.prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                ).to(input_masks.dtype)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        low_res_multimasks, iou_scores, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:-1],
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            **kwargs,
        )

        is_obj_appearing = object_score_logits > 0
        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
        # consistent with the actual mask prediction
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        high_res_multimasks = (
            F.interpolate(
                low_res_multimasks.squeeze(1).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            .unsqueeze(1)
            .to(low_res_multimasks.dtype)
        )
        sam_output_token = sam_output_tokens[:, :, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(iou_scores, dim=-1)
            batch_inds = torch.arange(batch_size, device=high_res_multimasks.device)
            object_batch_inds = torch.arange(num_objects, device=high_res_multimasks.device)
            low_res_masks = low_res_multimasks[batch_inds, object_batch_inds, best_iou_inds]
            high_res_masks = high_res_multimasks[batch_inds, object_batch_inds, best_iou_inds]
            if sam_output_tokens.size(2) > 1:
                sam_output_token = sam_output_tokens[batch_inds, object_batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks[:, :, 0], high_res_multimasks[:, :, 0]

        # Extract object pointer from the SAM output token (with occlusion handling)
        object_pointer = self.object_pointer_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.to(object_pointer.dtype)

        object_pointer = lambda_is_obj_appearing * object_pointer
        object_pointer = object_pointer + (1 - lambda_is_obj_appearing) * self.no_object_pointer

        return Sam2VideoImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=low_res_masks,
            high_res_masks=high_res_masks,
            object_pointer=object_pointer,
            object_score_logits=object_score_logits,
            image_embeddings=image_embeddings,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
        )

    def _use_mask_as_output(
        self,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        mask_inputs: torch.Tensor,
    ) -> Sam2VideoImageSegmentationOutput:
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in forward above).
        """
        # Use -10/+20 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.to(backbone_features[0].dtype)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks.float(),
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        ).to(backbone_features[0].dtype)
        # a dummy IoU prediction of all 1's under mask input
        iou_scores = mask_inputs.new_ones(mask_inputs.size(0), 1).to(backbone_features[0].dtype)
        # produce an object pointer using the SAM decoder from the mask input
        object_pointer = self._single_frame_forward(
            input_masks=self.mask_downsample(mask_inputs_float.to(backbone_features[0].dtype)),
            image_embeddings=high_res_features + [backbone_features],
        ).object_pointer
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.to(backbone_features[0].dtype)
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        object_pointer = lambda_is_obj_appearing * object_pointer
        object_pointer = object_pointer + (1 - lambda_is_obj_appearing) * self.no_object_pointer
        return Sam2VideoImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=low_res_masks,
            high_res_masks=high_res_masks,
            object_pointer=object_pointer,
            object_score_logits=object_score_logits,
            image_embeddings=high_res_features + [backbone_features],
        )

    def _gather_memory_frame_outputs(
        self,
        inference_session: Sam2VideoInferenceSession,
        obj_idx: int,
        frame_idx: int,
        track_in_reverse_time: bool = False,
    ) -> list[tuple[int, dict]]:
        """
        Get memory frames from conditioning and non-conditioning outputs.

        Returns:
            List of (relative_temporal_offset, output_data) tuples.
        """
        temporal_positions_and_previous_outputs = []

        # Add conditioning frame outputs (no limit here, as is the case in the original checkpoints)
        conditioning_outputs = inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
        if not conditioning_outputs:
            raise ValueError(
                "maskmem_features in conditioning outputs cannot be empty when not is_initial_conditioning_frame"
            )

        # Store (temporal_position, output_data) tuples
        temporal_positions_and_previous_outputs = [(0, out) for out in conditioning_outputs.values()]

        # Add non-conditioning memory frames (up to self.num_maskmem - 1)
        # These are typically frames tracked by the model without direct user input.
        # Frames are selected with a stride, prioritizing the most recent ones. Here we only support stride = 1 for simplicity.
        for relative_temporal_offset in range(self.num_maskmem - 1, 0, -1):
            # relative_temporal_offset: how many frames before (or after if reversing) the current frame
            if not track_in_reverse_time:
                previous_frame_idx = frame_idx - relative_temporal_offset
            else:
                previous_frame_idx = frame_idx + relative_temporal_offset

            # check if the output is already stored without using get_output to avoid unnecessary memory transfers between CPU and GPU
            output_data = inference_session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].get(
                previous_frame_idx, None
            )

            temporal_positions_and_previous_outputs.append((relative_temporal_offset, output_data))

        return temporal_positions_and_previous_outputs

    def _build_memory_attention_inputs(
        self,
        temporal_positions_and_previous_outputs: list[tuple[int, dict]],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Concatenate memory features and positional embeddings from previous frames.

        Returns:
            Tuple of (memories_to_concatenate, memory_positional_embeddings_to_concatenate).
        """
        memories_to_concatenate = []
        memory_positional_embeddings_to_concatenate = []

        for relative_temporal_offset, prev_output_data in temporal_positions_and_previous_outputs:
            if prev_output_data is None:
                continue  # Skip if no output data for this temporal position (e.g., padding frames)

            # Load memory features (potentially from CPU to GPU)
            # Features are flattened: (Batch, Channels, H, W) -> (H*W, Batch, Channels)
            memory_features = prev_output_data["maskmem_features"].to(device, non_blocking=True)
            memories_to_concatenate.append(memory_features)

            # Spatial positional encoding (potentially from CPU to GPU)
            spatial_memory_pos_embed = prev_output_data["maskmem_pos_enc"].to(device, non_blocking=True)

            # Add temporal positional encoding
            # self.memory_temporal_positional_encoding shape: (NumMaskMem, 1, 1, MemDim)
            combined_memory_pos_embed = (
                spatial_memory_pos_embed + self.memory_temporal_positional_encoding[relative_temporal_offset - 1]
            )
            memory_positional_embeddings_to_concatenate.append(combined_memory_pos_embed)

        return memories_to_concatenate, memory_positional_embeddings_to_concatenate

    def _get_object_pointers(
        self,
        inference_session: Sam2VideoInferenceSession,
        obj_idx: int,
        frame_idx: int,
        num_total_frames: int,
        device: torch.device,
        track_in_reverse_time: bool = False,
        streaming: bool = False,
    ) -> tuple[list[int], list[torch.Tensor], int]:
        """
        Get object pointers and their positional embeddings from past frames.

        Returns:
            Tuple of (temporal_offsets, pointer_tokens, max_object_pointers_to_use).
        """
        temporal_position_sign_multiplier = -1 if track_in_reverse_time else 1

        # Determine max object pointers to use
        if streaming:
            max_object_pointers_to_use = self.config.max_object_pointers_in_encoder
        else:
            max_object_pointers_to_use = min(num_total_frames, self.config.max_object_pointers_in_encoder)

        temporal_offsets: list[int] = []
        pointer_tokens: list[torch.Tensor] = []

        # Add object pointers from selected conditioning frames
        # Optionally, only include pointers from past frames during evaluation
        conditioning_outputs = inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
        eligible_conditioning_outputs = conditioning_outputs
        if not self.training:
            eligible_conditioning_outputs = {
                temporal_idx: out
                for temporal_idx, out in conditioning_outputs.items()
                if (temporal_idx >= frame_idx if track_in_reverse_time else temporal_idx <= frame_idx)
            }

        for temporal_idx, out_data in eligible_conditioning_outputs.items():
            temporal_difference = (frame_idx - temporal_idx) * temporal_position_sign_multiplier
            temporal_offsets.append(temporal_difference)
            pointer_tokens.append(out_data["object_pointer"].to(device))

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
                temporal_offsets.append(t_diff_offset)
                pointer_tokens.append(out_data["object_pointer"].to(device))

        return temporal_offsets, pointer_tokens, max_object_pointers_to_use

    def _process_object_pointers(
        self,
        temporal_offsets: list[int],
        pointer_tokens: list[torch.Tensor],
        max_object_pointers_to_use: int,
        batch_size: int,
        num_channels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process object pointers and compute their positional embeddings.

        Returns:
            Tuple of (object_pointers, object_pointers_pos_embed).
        """
        if not pointer_tokens:
            return None, None

        # Stack object pointers: List of (Batch, Channels) -> (SeqLen_ptr, Batch, Channels)
        object_pointers = torch.stack(pointer_tokens, dim=0)

        if self.config.enable_temporal_pos_encoding_for_object_pointers:
            max_temporal_diff = float(max_object_pointers_to_use - 1)
            # Determine dimensionality for temporal positional encoding of pointers
            pointer_tpos_dim = num_channels

            # Normalize temporal differences before sine PE calculation
            normalized_temporal_diffs = (
                torch.tensor(temporal_offsets, device=device, dtype=torch.float32) / max_temporal_diff
            )
            sine_pe = get_1d_sine_pe(normalized_temporal_diffs, dim=pointer_tpos_dim).to(object_pointers.dtype)
            projected_sine_pe = self.temporal_positional_encoding_projection_layer(sine_pe)
            object_pointers_pos_embed = projected_sine_pe.unsqueeze(1).expand(-1, batch_size, self.mem_dim)
        else:
            object_pointers_pos_embed = object_pointers.new_zeros(
                len(temporal_offsets), batch_size, self.mem_dim, dtype=object_pointers.dtype
            )

        if self.mem_dim < num_channels:
            # If memory dimension is smaller, reshape/split pointers and repeat positional encoding
            num_splits = num_channels // self.mem_dim
            object_pointers = object_pointers.reshape(-1, batch_size, num_splits, self.mem_dim)
            object_pointers = object_pointers.permute(0, 2, 1, 3).flatten(
                0, 1
            )  # (SeqLen_ptr*num_splits, Batch, MemDim)
            object_pointers_pos_embed = object_pointers_pos_embed.repeat_interleave(num_splits, dim=0)

        return object_pointers, object_pointers_pos_embed

    def _prepare_memory_conditioned_features(
        self,
        inference_session: Sam2VideoInferenceSession,
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
            inference_session (`Sam2VideoInferenceSession`):
                The video inference session object.
            frame_idx (`int`):
                Index of the current frame being processed.
            obj_idx (`int`):
                Index of the object being processed.
            is_initial_conditioning_frame (`bool`):
                Whether this is an initial conditioning frame with user inputs (True) or a subsequent
                tracking frame (False).
            current_vision_features (`torch.Tensor`):
                Highest-level vision features of shape `(seq_len, batch_size, channels)`.
            current_vision_positional_embeddings (`torch.Tensor`):
                Positional embedding tensors corresponding to the highest-level vision features.
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
        batch_size = current_vision_features.size(1)
        num_channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]
        device = current_vision_features.device

        # If memory is disabled (e.g., for single image SAM), return current features directly.
        if self.num_maskmem == 0:
            # Permute (SeqLen, Batch, Channels) -> (Batch, Channels, SeqLen) then view as (Batch, Channels, Height, Width)
            # Assuming SeqLen = Height * Width for the last feature map
            current_feature_map = current_vision_features.permute(1, 2, 0).view(
                batch_size, num_channels, height, width
            )
            return current_feature_map

        # Step 1: Handle initial conditioning frames
        if is_initial_conditioning_frame:
            # For initial conditioning frames, no prior memory is used directly in this block.
            # If configured, directly add a learnable "no memory" embedding.
            # current_vision_features has shape (SeqLen, Batch, Channels)
            conditioned_feature_map_flat = current_vision_features + self.no_memory_embedding
            # Reshape to (Batch, Channels, Height, Width)
            conditioned_feature_map = conditioned_feature_map_flat.permute(1, 2, 0).view(
                batch_size, num_channels, height, width
            )
            return conditioned_feature_map

        # Step 2: Get memory frames and concatenate their features
        temporal_positions_and_previous_outputs = self._gather_memory_frame_outputs(
            inference_session, obj_idx, frame_idx, track_in_reverse_time
        )

        memories_to_concatenate, memory_positional_embeddings_to_concatenate = self._build_memory_attention_inputs(
            temporal_positions_and_previous_outputs, device
        )

        # Step 3: Get and process object pointers
        temporal_offsets, pointer_tokens, max_object_pointers_to_use = self._get_object_pointers(
            inference_session, obj_idx, frame_idx, num_total_frames, device, track_in_reverse_time, streaming
        )

        num_object_pointer_tokens = 0
        if pointer_tokens:
            object_pointers, object_pointers_pos_embed = self._process_object_pointers(
                temporal_offsets, pointer_tokens, max_object_pointers_to_use, batch_size, num_channels, device
            )

            if object_pointers is not None:
                memories_to_concatenate.append(object_pointers)
                memory_positional_embeddings_to_concatenate.append(object_pointers_pos_embed)
                num_object_pointer_tokens = object_pointers.shape[0]

        # Step 4: Concatenate all retrieved memories and their positional embeddings
        combined_memory = torch.cat(memories_to_concatenate, dim=0)
        combined_memory_positional_embeddings = torch.cat(memory_positional_embeddings_to_concatenate, dim=0)

        # Step 5: Forward through the memory attention mechanism
        conditioned_feature_map_flat = self.memory_attention(
            current_vision_features=current_vision_features,
            current_vision_position_embeddings=current_vision_positional_embeddings,
            memory=combined_memory,
            memory_posision_embeddings=combined_memory_positional_embeddings,  # Corrected typo from API
            num_object_pointer_tokens=num_object_pointer_tokens,
        )

        # Reshape from (Batch, H*W, Channels) to (Batch, Channels, Height, Width)
        conditioned_feature_map = (
            conditioned_feature_map_flat.squeeze(1).permute(0, 2, 1).view(batch_size, num_channels, height, width)
        )
        return conditioned_feature_map

    def _use_multimask(self, is_init_cond_frame: bool, point_inputs: Optional[dict]) -> bool:
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(2)
        multimask_output = (
            self.config.multimask_output_in_sam
            and (is_init_cond_frame or self.config.multimask_output_for_tracking)
            and (self.config.multimask_min_pt_num <= num_pts <= self.config.multimask_max_pt_num)
        )
        return multimask_output

    def _run_single_frame_inference(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: Optional[torch.Tensor],
        mask_inputs: Optional[torch.Tensor],
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: Optional[torch.Tensor] = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """
        Perform a single tracking step for video object segmentation.

        Args:
            inference_session (`Sam2VideoInferenceSession`):
                The video inference session object.
            frame_idx (`int`):
                Index of the current frame.
            obj_idx (`int`):
                Index of the current object.
            batch_size (`int`):
                Batch size of the current frame.
            is_init_cond_frame (`bool`):
                Whether this is an initial conditioning frame with user inputs.
            point_inputs (`dict`, *optional*):
                Point prompt inputs for the current frame.
            mask_inputs (`torch.Tensor`, *optional*):
                Mask prompt inputs for the current frame.
            reverse (`bool`, *optional*, defaults to `False`):
                Whether to track in reverse time order.
            run_mem_encoder (`bool`, *optional*, defaults to `True`):
                Whether to run the memory encoder on predicted masks.
            prev_sam_mask_logits (`torch.Tensor`, *optional*):
                Previously predicted SAM mask logits that can be fed with new clicks.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether this is streaming inference.

        Returns:
            `dict`: Dictionary containing the tracking results for the current frame, including:
                - pred_masks: Predicted low-resolution masks.
                - object_pointer: Object pointer for memory.
                - object_score_logits: Object score logits (inference only).
                - maskmem_features: Memory features for future frames.
                - maskmem_pos_enc: Memory positional encodings.
        """
        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds = self._prepare_vision_features(
            inference_session, frame_idx, batch_size
        )
        # point and mask should not appear as input simultaneously on the same frame
        if point_inputs is not None and mask_inputs is not None:
            raise ValueError(
                "point_inputs and mask_inputs should not appear as input simultaneously on the same frame"
            )
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], self.backbone_feature_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None:
            # We directly output the mask input (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *self.backbone_feature_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_idx=obj_idx,
                is_initial_conditioning_frame=is_init_cond_frame,
                current_vision_features=current_vision_feats[-1],
                current_vision_positional_embeddings=current_vision_pos_embeds[-1],
                num_total_frames=inference_session.num_frames,
                track_in_reverse_time=reverse,
                streaming=streaming,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._single_frame_forward(
                pixel_values=None,  # Vision features already computed
                input_points=point_inputs["point_coords"] if point_inputs is not None else None,
                input_labels=point_inputs["point_labels"] if point_inputs is not None else None,
                input_masks=mask_inputs,
                image_embeddings=high_res_features + [pix_feat],
                multimask_output=multimask_output,
            )

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (which will be used to condition vision features in future frames)
        maskmem_features = None
        maskmem_pos_enc = None
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats[-1],
                pred_masks_high_res=sam_outputs.high_res_masks,
                object_score_logits=sam_outputs.object_score_logits,
                is_mask_from_pts=(point_inputs is not None or mask_inputs is not None),
            )

        current_out = {
            "pred_masks": sam_outputs.pred_masks,
            "object_pointer": sam_outputs.object_pointer,
            "maskmem_features": maskmem_features if maskmem_features is not None else None,
            "maskmem_pos_enc": maskmem_pos_enc,
        }
        if not self.training:
            current_out["object_score_logits"] = sam_outputs.object_score_logits

        return current_out

    def _encode_new_memory(
        self,
        current_vision_feats: torch.Tensor,
        pred_masks_high_res: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_pts: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode the current image and its prediction into a memory feature."""
        batch_size = current_vision_feats.size(1)  # batch size on this frame
        channels = self.hidden_dim
        height, width = self.backbone_feature_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats.permute(1, 2, 0).view(batch_size, channels, height, width)
        if is_mask_from_pts and not self.training:
            # binarize the mask logits
            mask_for_mem = (pred_masks_high_res > 0).to(pred_masks_high_res.dtype)
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        mask_for_mem = mask_for_mem * self.config.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.config.sigmoid_bias_for_mem_enc

        maskmem_features, maskmem_pos_enc = self.memory_encoder(
            pix_feat,
            mask_for_mem,
        )
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.occlusion_spatial_embedding_parameter is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None]) * self.occlusion_spatial_embedding_parameter[
                ..., None, None
            ].expand(*maskmem_features.shape)

        # convert to bfloat16 to save memory, and for consistency with the original implementation
        maskmem_features = maskmem_features.to(torch.bfloat16).flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc.to(pred_masks_high_res.dtype).flatten(2).permute(2, 0, 1)

        return maskmem_features, maskmem_pos_enc

    @torch.inference_mode()
    @auto_docstring(custom_intro="Propagate the objects through a streamed video frame.")
    def forward(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: Optional[int] = None,
        frame: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Sam2VideoSegmentationOutput:
        r"""
        inference_session (`Sam2VideoInferenceSession`):
            The video inference session object.
        frame_idx (`int`, *optional*):
            The index of the frame on which to run inference. No need to provide when inferring
            on a new streamed frame.
        frame (`torch.Tensor`, *optional*):
            The frame to process. Provide when streaming.
        reverse (`bool`, *optional*, defaults to `False`):
            Whether to propagate in reverse.
        """
        if frame is not None:
            frame_idx = inference_session.add_new_frame(frame, frame_idx)

        if frame is not None and inference_session.get_obj_num() == 0:
            raise ValueError("No objects are provided for tracking; please add inputs first.")

        num_objects = inference_session.get_obj_num()
        pred_masks_per_obj = [None] * num_objects
        # Note: We avoid batched inference here because per-object inputs (clicks/masks)
        # can differ across objects.
        for obj_idx in range(num_objects):
            obj_id = inference_session.obj_idx_to_id(obj_idx)
            has_new_inputs = obj_id in inference_session.obj_with_new_inputs
            has_cond_output = frame_idx in inference_session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
            # If this object has no new inputs and this frame already has a
            # conditioning output, reuse the cached masks instead of recomputing.
            if (not has_new_inputs) and has_cond_output:
                pred_masks = inference_session.get_output(obj_idx, frame_idx, "pred_masks", is_conditioning_frame=True)
                is_init_cond_frame = True
            else:
                # Defaults when there are no new inputs
                is_init_cond_frame = False
                point_inputs = None
                mask_inputs = None

                if has_new_inputs:
                    is_init_cond_frame = frame_idx not in inference_session.frames_tracked_per_obj[obj_idx]
                    if is_init_cond_frame:
                        reverse = False
                    point_inputs = inference_session.point_inputs_per_obj[obj_idx].get(frame_idx, None)
                    mask_inputs = inference_session.mask_inputs_per_obj[obj_idx].get(frame_idx, None)
                    if point_inputs is not None or mask_inputs is not None:
                        inference_session.obj_with_new_inputs.remove(obj_id)

                current_out = self._run_single_frame_inference(
                    inference_session=inference_session,
                    obj_idx=obj_idx,
                    frame_idx=frame_idx,
                    batch_size=1,  # run on the slice of a single object
                    is_init_cond_frame=is_init_cond_frame,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    reverse=reverse,
                    run_mem_encoder=True,
                    streaming=frame is not None,
                )
                inference_session.store_output(
                    obj_idx, frame_idx, output_value=current_out, is_conditioning_frame=is_init_cond_frame
                )
                pred_masks = current_out["pred_masks"]

            pred_masks_per_obj[obj_idx] = pred_masks
            if not is_init_cond_frame:
                # only for tracked frames, not for initial conditioning frames
                inference_session.frames_tracked_per_obj[obj_idx][frame_idx] = {"reverse": reverse}

        # Resize the output mask to the original video resolution (we directly use
        # the mask scores on GPU for output to avoid any CPU conversion in between)
        if len(pred_masks_per_obj) > 1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]

        return Sam2VideoSegmentationOutput(pred_masks=all_pred_masks, frame_idx=frame_idx)

    @torch.inference_mode()
    @auto_docstring(
        custom_intro="""
        Propagate the objects through the video frames. Used when initializing an inference session with a whole video.
        Yields Sam2VideoSegmentationOutput for each frame.
        """
    )
    def propagate_in_video_iterator(
        self,
        inference_session: Sam2VideoInferenceSession,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ) -> Iterator[Sam2VideoSegmentationOutput]:
        r"""
        inference_session (`Sam2VideoInferenceSession`):
            The video inference session object.
        start_frame_idx (`int`, *optional*):
            The starting frame index for propagation.
            Need to be provided if `forward` hasn't been called on new inputs yet.
            If not provided, the starting frame index will be the earliest frame with input points.
        max_frame_num_to_track (`int`, *optional*):
            The maximum number of frames to track.
        reverse (`bool`, *optional*, defaults to `False`):
            Whether to propagate in reverse.
        """
        num_frames = inference_session.num_frames

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            frames_with_inputs = [
                frame_idx
                for obj_output_dict in inference_session.output_dict_per_obj.values()
                for frame_idx in obj_output_dict["cond_frame_outputs"]
            ]
            if not frames_with_inputs:
                raise ValueError(
                    "Cannot determine the starting frame index; please specify it manually, or run inference on a frame with inputs first."
                )
            start_frame_idx = min(frames_with_inputs)
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            sam2_video_output = self(inference_session, frame_idx=frame_idx, reverse=reverse)
            yield sam2_video_output


__all__ = [
    "Sam2VideoModel",
    "Sam2VideoInferenceSession",
    "Sam2VideoPreTrainedModel",
    "Sam2VideoMaskDecoderConfig",
    "Sam2VideoPromptEncoderConfig",
    "Sam2VideoProcessor",
    "Sam2VideoConfig",
]
