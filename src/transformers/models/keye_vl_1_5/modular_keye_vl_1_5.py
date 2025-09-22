# coding=utf-8
# Copyright 2025 The Kwai Keye Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


import os
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List, Tuple, Dict

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLTextModel,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from ..qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipMLP,
    SiglipVisionModelOutput,
)


from ...activations import GELUActivation
from ...cache_utils import Cache, StaticCache, DynamicCache, SlidingWindowCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...image_processing_utils import BatchFeature, BaseImageProcessor
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    is_valid_image,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    sdpa_attention_forward,
)
from ...processing_utils import Unpack, ProcessorMixin, ProcessingKwargs, VideosKwargs
from ...utils import (
    TensorType,
    TransformersKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    auto_docstring,
    can_return_tuple,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import requires
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING
from ...video_utils import VideoInput, VideoMetadata, group_videos_by_shape, reorder_videos, make_batched_videos
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


logger = logging.get_logger(__name__)

try:
    from keye_vl_utils import BicubicVideoProcessor
except:
    BicubicVideoProcessor = None

if BicubicVideoProcessor is not None:
    bicubic = BicubicVideoProcessor()


def eager_attention_forward(
        module: nn.Module,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def make_batched_images(
        images: Union[List[List[ImageInput]], List[ImageInput], ImageInput]
) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


class KeyeVL1_5VideosProcessorKwargs(VideosKwargs, total=False):

    fps: Optional[Union[List[float], float]]
    """
    the fps of the video.
    """
    width: Optional[Union[List[int], int]]
    """
    the width to resize in for slow frames.
    """
    height: Optional[Union[List[int], int]]
    """
    the height to resize in for slow frames.
    """
    fast_width: Optional[Union[List[int], int]]
    """
    the width to resize in for fast frames.
    """
    fast_height: Optional[Union[List[int], int]]
    """
    the height to resize in for fast frames.
    """
    timestamps: Optional[Union[List[torch.Tensor], torch.Tensor]]
    """
    used to mark the timestamp of each frame, the quantity is equal to the number of frames.
    """
    frame_types: Optional[Union[List[torch.Tensor], torch.Tensor]]
    """
    used to mark whether each frame is of type slow or fast, where 0 for alow and 1 for fast.
    """


class KeyeVL1_5ProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: KeyeVL1_5VideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "videos_kwargs": {"fps": 2.0},
    }


def select_slow_fast_frames(
        frames: torch.FloatTensor,
        frame_types: torch.LongTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Selects frames from a tensor based on a mask list.

    Args:
        frames (torch.FloatTensor): A tensor of shape (nframes, channel, height, width).
        frame_types (torch.LongTensor): A int tensor of shape (nframes, ), 1 for fast frames and 0 for slow frames.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: A tuple containing two tensors:
            - slow_frames: Frames which the type is 0.
            - fast_frames: Frames where the type is 1.
    """
    nframes, _, _, _ = frames.shape
    if frame_types.shape[-1] != nframes:
        raise ValueError("Length of mask must be equal to the number of frames.")

    mask = (frame_types == 0)

    slow_frames = frames[mask]
    fast_frames = frames[~mask]

    return slow_frames, fast_frames


def split_thw(thw: torch.LongTensor) -> torch.LongTensor:
    """
    Split grid_thw in t dimension, the result tensor should like [[1, h, w],...]
    Example:

    ```python
    >>> thw = torch.LongTensor([[2, 10, 12], [3, 4, 6]])
    >>> split_thw(thw) -> torch.LongTensor([[1, 10, 12], [1, 10, 12], [1, 4, 6], [1, 4, 6], [1, 4, 6]])
    """
    if thw.dim() == 1:
        thw = thw.unsqueeze(0)

    clone = thw.clone()
    clone[:, 0] = 1
    return torch.repeat_interleave(clone, thw[:, 0], dim=0)


def merge_thw(
        thw: Union[torch.LongTensor, List[torch.LongTensor]],
        num_frames: Optional[List[int]] = None,
) -> torch.LongTensor:
    """
    Merge same grid_thw in t dimension, if num_frames is provided, may split adjacent identical values.
    Example:

    ```python
    >>> thw = torch.LongTensor([[1, 3, 4], [1, 3, 4], [1, 3, 4], [1, 23, 10], [1, 89, 18], [1, 9, 10], [1, 9, 10]])
    >>> num_frames = None
    >>> merge_thw(thw, num_frames) -> torch.LongTensor([3, 3, 4], [1, 23, 10], [1, 89, 18], [2, 9, 10]])

    >>> thw = torch.LongTensor([[1, 3, 4], [1, 3, 4], [1, 3, 4], [1, 23, 10], [1, 89, 18], [1, 9, 10], [1, 9, 10]])
    >>> num_frames = [2, 1, 3, 1]
    >>> merge_thw(thw, num_frames) -> torch.LongTensor([2, 3, 4], [1, 3, 4], [1, 23, 10], [1, 89, 18], [1, 9, 10], [1, 9, 10]])

    Args:
        thw (torch.LongTensor, List[torch.LongTensor]): A tensor of shape (N, 3) or a list of (3, ) tensor with lengths N, the value of t dimension is always 1.
        num_frames (List[int]): Indicates the number of frames in each video.

    Returns:
        torch.LongTensor: A tensor merged adjacent identical values.
    """

    if isinstance(thw, list):
        thw = torch.stack(thw, dim=0)

    assert thw.dim() == 2, thw.shape
    assert torch.all(thw[:, 0] == 1), thw

    if num_frames is None:
        mask = (thw[:-1] != thw[1:]).any(1)
        mask = F.pad(mask, (1, 0), value=True)

        indices = torch.where(mask)[0]
        append = torch.LongTensor([mask.shape[0]])
        count = torch.diff(indices, append=append).unsqueeze(-1)
        return torch.concat([count, thw[indices][:, 1:]], dim=1)

    assert thw.shape[0] == sum(num_frames), (thw.shape, num_frames)

    return torch.concat(
        [
            merge_thw(part)
            for part in thw.split(num_frames, dim=0)
        ],
        dim=0
    )


def repeat_kv(
        hidden_states: torch.FloatTensor,
        n_rep: int
) -> torch.FloatTensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class KeyeVL1_5VisionConfig(SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`KeyeVL1_5VisionModel`]. It is used to instantiate a
    KeyeVL1_5 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Dimensionality of spatial merging for visual features. Visual patches are merged in HxW dimensions using a window of this size (e.g., 2 means merging 2x2 adjacent patches into one) to reduce feature resolution and computational cost.
        tokens_per_second (`int`, *optional*, defaults to 2):
            Number of tokens used to represent one second of temporal data (e.g., video frames). Used to calculate temporal position embeddings for video inputs, defining the granularity of time-based positional encoding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal distribution used to initialize weight parameters (e.g., linear layers, embedding layers) in the model. Controls the initial spread of weights to stabilize training.


    Example:

    ```python
    >>> from transformers import KeyeVL1_5VisionConfig, KeyeVL1_5VisionModel

    >>> # Initializing a KeyeVL1_5VisionConfig with Kwai-Keye/Keye-VL-1_5-8B style configuration
    >>> configuration = KeyeVL1_5VisionConfig()

    >>> # Initializing a KeyeVL1_5VisionModel (with random weights) from the Kwai-Keye/Keye-VL-1_5-8B style configuration
    >>> model = KeyeVL1_5VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "keye_vl_1_5_vision_model"
    base_config_key = "vision_config"

    def __init__(
            self,
            hidden_size: int = 768,
            intermediate_size: int = 3072,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            num_channels: int = 3,
            image_size: int = 224,
            patch_size: int = 14,
            hidden_act: Union[str, Callable] = "gelu_pytorch_tanh",
            layer_norm_eps: float = 1e-6,
            attention_dropout: float = 0.0,
            spatial_merge_size: int = 2,
            tokens_per_second: int = 2,
            initializer_range: float = 0.02,
            **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.spatial_merge_size = spatial_merge_size
        self.tokens_per_second = tokens_per_second
        self.initializer_range = initializer_range


class KeyeVL1_5TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`KeyeVL1_5TextModel`]. It is used to instantiate a
    KeyeVL1_5 model according to the specified arguments, defining the model architecture.
    e.g. [Kwai-Keye/Keye-1_5-VL-8B](https://huggingface.co/Kwai-Keye/Keye-VL-1_5-8B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the KeyeVL1_5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`KeyeVL1_5Model`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 29568):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        image_token_id (`int`, *optional*):
            Token index used as placeholder for image embeddings.
        video_token_id (`int`, *optional*):
            Token index used as placeholder for video embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias term in the attention mechanism layers.

    ```python
    >>> from transformers import KeyeVL1_5TextModel, KeyeVL1_5Config

    >>> # Initializing a KeyeVL1_5 style configuration
    >>> configuration = KeyeVL1_5Config()

    >>> # Initializing a model from the KeyeVL1_5 configuration
    >>> model = KeyeVL1_5TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "keye_vl_1_5_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size: int = 152064,
            hidden_size: int = 8192,
            intermediate_size: int = 29568,
            num_hidden_layers: int = 80,
            num_attention_heads: int = 64,
            num_key_value_heads: int = 8,
            hidden_act: Union[str, Callable] = "silu",
            max_position_embeddings: int = 32768,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-05,
            use_cache: bool = True,
            tie_word_embeddings: bool = False,
            rope_theta: float = 1000000.0,
            use_sliding_window: bool = False,
            sliding_window: int = 4096,
            layer_types: Optional[List[str]] = None,
            attention_dropout: float = 0.0,
            rope_scaling: Optional[Dict[str, Any]] = None,
            image_token_id: int = None,
            video_token_id: int = None,
            attention_bias: bool = False,
            **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        if layer_types is None:
            layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]

        self.layer_types = layer_types
        layer_type_validation(self.layer_types)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.attention_bias = attention_bias


class KeyeVL1_5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`KeyeVL1_5Model`]. It is used to instantiate a
    KeyeVL-1.5 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `KeyeVL1_5TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `KeyeVL1_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.

    ```python
    >>> from transformers import KeyeVL1_5ForConditionalGeneration, KeyeVL1_5Config

    >>> # Initializing a KeyeVL1_5 style configuration
    >>> configuration = KeyeVL1_5Config()

    >>> # Initializing a model from the KeyeVL-1.5-8B style configuration
    >>> model = KeyeVL1_5ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "KeyeVL1_5"
    sub_configs = {"vision_config": KeyeVL1_5VisionConfig, "text_config": KeyeVL1_5TextConfig}

    def __init__(
            self,
            text_config=None,
            vision_config=None,
            image_token_id=151655,
            video_token_id=151656,
            **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
            self.sliding_window = text_config.get("sliding_window", None)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)
            self.sliding_window = kwargs.get("sliding_window", None)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


class KeyeVL1_5ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Keye-VL-1.5 image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
            self,
            do_resize: bool = True,
            resample: PILImageResampling = PILImageResampling.BILINEAR,
            do_rescale: bool = True,
            rescale_factor: Union[int, float] = 1 / 255,
            do_normalize: bool = True,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            do_convert_rgb: bool = True,
            min_pixels: int = 56 * 56,
            max_pixels: int = 28 * 28 * 1280,
            patch_size: int = 14,
            temporal_patch_size: int = 1,
            merge_size: int = 2,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb
        assert self.temporal_patch_size == 1, "temporal_patch_size != 1 is not supported yet."

    def _preprocess(
            self,
            images: Union[ImageInput, VideoInput],
            do_resize: Optional[bool] = None,
            size: Optional[Dict[str, int]] = None,
            resample: Optional[PILImageResampling] = None,
            do_rescale: Optional[bool] = None,
            rescale_factor: Optional[float] = None,
            do_normalize: Optional[bool] = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            do_convert_rgb: Optional[bool] = None,
            data_format: ChannelDimension = ChannelDimension.FIRST,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[int, int, int]]:
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`, `VideoInput`):
                Image/Video or batch of images/videos to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = list()
        for image in images:
            if do_resize:
                if size is not None and "height" in size.keys():
                    resized_height, resized_width = size["height"], size["width"]
                else:
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=self.patch_size * self.merge_size,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        init_patches = patches
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel, self.patch_size, self.patch_size
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
            self,
            images: Optional[ImageInput] = None,
            videos: Optional[VideoInput] = None,
            do_resize: Optional[bool] = None,
            size: Optional[Dict[str, int]] = None,
            resample: Optional[PILImageResampling] = None,
            do_rescale: Optional[bool] = None,
            rescale_factor: Optional[float] = None,
            do_normalize: Optional[bool] = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            do_convert_rgb: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            data_format: ChannelDimension = ChannelDimension.FIRST,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if images is not None:
            pixel_values, vision_grid_thws = list(), list()
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size = size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        if videos is not None:
            pixel_values, vision_grid_thws = list(), list()
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    size = size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values_videos": pixel_values, "video_grid_thw": vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)


@add_start_docstrings(
    "Constructs a fast Keye-VL-1.5 image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        min_frames (`int`, *optional*, defaults to 4):
            The minimum number of frames that can be sampled.
        max_frames (`int`, *optional*, defaults to 768):
            The maximum number of frames that can be sampled.
    """,
)
class KeyeVL1_5Processor(ProcessorMixin):
    r"""
    [`KeyeVL1_5Processor`] offers all the functionalities of [`KeyeVL1_5ImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~KeyeVL1_5Processor.__call__`] and [`~KeyeVL1_5Processor.decode`] for more information.
    Args:
        image_processor ([`KeyeVL1_5ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template","image_std", "min_pixels", "image_mean", "merge_size", "image_processor_type",
        "temporal_patch_size", "patch_size", "max_pixels"
    ]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
            self,
            image_processor: Optional[KeyeVL1_5ImageProcessor] = None,
            tokenizer: Optional[Qwen2TokenizerFast] = None,
            chat_template: Optional[str] = None,
            **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.frame_token = getattr(tokenizer, "frame_token", "<|frame|>")
        self.fast_start = getattr(tokenizer, "fast_start", "<|fast_start|>")
        self.fast_end = getattr(tokenizer, "fast_end", "<|fast_end|>")

        self.merge_size = getattr(image_processor, "merge_size", 2)
        self.patch_size = getattr(image_processor, "patch_size", 14)
        self.min_pixels = getattr(image_processor, "min_pixels", 28 * 28 * 4)
        self.max_pixels = getattr(image_processor, "max_pixels", 28 * 28 * 1280)
        self.scale = 255 if not hasattr(image_processor, "rescale_factor") else int(round(1.0 / image_processor.rescale_factor))
        self.image_mean = getattr(image_processor, "image_mean", OPENAI_CLIP_MEAN)
        self.image_std = getattr(image_processor, "image_std", OPENAI_CLIP_STD)

        if not isinstance(self.image_mean, (list, tuple)):
            self.image_mean = [self.image_mean] * 3
        if not isinstance(self.image_std, (list, tuple)):
            self.image_std = [self.image_std] * 3
        self.factor = self.merge_size * self.patch_size

        self.enable_fusion_op = bool(int(os.environ.get("ENABLE_FUSION_PROCESSOR_OP", 1))) and \
                                (bicubic is not None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            images: Optional[ImageInput] = None,
            videos: Optional[VideoInput] = None,
            **kwargs: Unpack[KeyeVL1_5ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        KeyeVL1_5ImageProcessor's [`~KeyeVL1_5ImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- Tensor of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- Tensor of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- Tensor of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **num_frames** -- Tensor of number of frames for each videos in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            KeyeVL1_5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
            image_inputs['pixel_values'] = image_inputs['pixel_values']
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = dict()
            image_grid_thw = None

        num_frames = list()
        if videos is not None:
            batch_slow_frames = list()
            batch_fast_frames = list()

            videos_kwargs = output_kwargs["videos_kwargs"]
            num_videos = len(videos)
            batch_frame_types = videos_kwargs.get("frame_types", [None] * num_videos)
            batch_timestamps = videos_kwargs.get("timestamps", [None] * num_videos)
            batch_width = videos_kwargs.get("width", [None] * num_videos)
            batch_height = videos_kwargs.get("height", [None] * num_videos)
            batch_fast_width = videos_kwargs.get("fast_width", [None] * num_videos)
            batch_fast_height = videos_kwargs.get("fast_height", [None] * num_videos)

            for index, frames in enumerate(videos):
                if isinstance(frames, np.ndarray):
                    frames = torch.from_numpy(frames)
                nframes, channel, ori_height, ori_width = frames.shape
                num_frames.append(nframes)
                assert nframes > 0, "No frames in video"
                if batch_frame_types[index] is None:
                    # default to all slow frames
                    batch_frame_types[index] = torch.zeros((nframes, ), dtype=torch.long)
                frame_types = batch_frame_types[index]
                if not self.enable_fusion_op:
                    slow_frames, fast_frames = select_slow_fast_frames(frames, frame_types)
                    has_fast_frames = fast_frames.shape[0] > 0
                    # resize slow frames
                    resized_width = batch_width[index]
                    resized_height = batch_height[index]
                    if resized_width is not None and resized_height is not None:
                        slow_frames = nn.functional.interpolate(
                            slow_frames,
                            [resized_height, resized_width],
                            mode="bilinear",
                            antialias=True,
                        ).float()
                        do_resize = False
                    else:
                        slow_frames = slow_frames.float()
                        do_resize = True

                    slow_video_inputs = self.image_processor(
                        images=None, videos=[slow_frames], **output_kwargs["images_kwargs"], do_resize=do_resize)
                    slow_video_grid_thw = slow_video_inputs["video_grid_thw"]
                    batch_slow_frames.append(slow_video_inputs)

                    if has_fast_frames:
                        # TODO: shrink fast_frames
                        fast_resized_width = batch_fast_width[index]
                        fast_resized_height = batch_fast_height[index]
                        if fast_resized_width is not None and fast_resized_height is not None:
                            fast_frames = nn.functional.interpolate(
                                fast_frames,
                                [fast_resized_height, fast_resized_width],
                                mode="bilinear",
                                antialias=True,
                            ).float()
                            do_fast_resize = False
                        else:
                            fast_frames = fast_frames.float()
                            do_fast_resize = True

                        fast_video_inputs = self.image_processor(
                            images=None, videos=[fast_frames], **output_kwargs["images_kwargs"], do_resize=do_fast_resize)
                        fast_video_grid_thw = fast_video_inputs["video_grid_thw"]
                        batch_fast_frames.append(fast_video_inputs)
                else:
                    slow_indices = (frame_types == 0).nonzero().flatten().tolist()
                    fast_indices = (frame_types == 1).nonzero().flatten().tolist()
                    has_fast_frames = len(fast_indices) > 0
                    resized_width = batch_width[index] or 0
                    resized_height = batch_height[index] or 0
                    fast_width = batch_fast_width[index] or 0
                    fast_height = batch_fast_height[index] or 0

                    slow_inputs = bicubic.interp(
                        frames,
                        nframes,
                        slow_indices,
                        ori_height,
                        ori_width,
                        resized_height,
                        resized_width,
                        patch=self.patch_size,
                        factor=self.factor,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                        scale=self.scale,
                        image_mean=self.image_mean,
                        image_std=self.image_std,
                    )
                    batch_slow_frames.append(slow_inputs)

                    if has_fast_frames:
                        fast_inputs = bicubic.interp(
                            frames,
                            nframes,
                            fast_indices,
                            ori_height,
                            ori_width,
                            fast_height,
                            fast_width,
                            patch=self.patch_size,
                            factor=self.factor,
                            min_pixels=self.min_pixels,
                            max_pixels=self.max_pixels,
                            scale=self.scale,
                            image_mean=self.image_mean,
                            image_std=self.image_std,
                        )
                        batch_fast_frames.append(fast_inputs)

            assert len(batch_slow_frames) > 0, "Slow frames should not be empty."
            slow_pixel_values_videos_list = [
                video["pixel_values_videos"] for video in batch_slow_frames if video is not None]
            slow_video_grid_thw_list = [
                video["video_grid_thw"] for video in batch_slow_frames if video is not None]

            slow_pixel_values_videos = torch.concat(slow_pixel_values_videos_list, dim=0)
            slow_video_grid_thw = torch.concat(slow_video_grid_thw_list, dim=0)

            if has_fast_frames:
                fast_pixel_values_videos_list = [
                    video["pixel_values_videos"] for video in batch_fast_frames \
                    if video is not None]
                fast_video_grid_thw_list = [
                    video["video_grid_thw"] for video in batch_fast_frames \
                    if video is not None]

                fast_pixel_values_videos = \
                    torch.concat(fast_pixel_values_videos_list, dim=0)
                fast_video_grid_thw = \
                    torch.concat(fast_video_grid_thw_list, dim=0)
            else:
                fast_video_grid_thw = None
        else:
            slow_video_grid_thw = None
            fast_video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    image_place_holder_tempale = "<|placeholder|>" * (
                            image_grid_thw[index].prod() // self.image_processor.merge_size ** 2)
                    text[i] = text[i].replace(
                        self.image_token,
                        image_place_holder_tempale,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        pixel_values_videos = list()
        video_grid_thw = list()
        videos_inputs = dict()
        if slow_video_grid_thw is not None:
            slow_video_grid_thw = split_thw(slow_video_grid_thw)
            if fast_video_grid_thw is not None:
                fast_video_grid_thw = split_thw(fast_video_grid_thw)
            index = 0
            slow_index = 0
            fast_index = 0
            slow_pixels_index = 0
            fast_pixels_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    video_place_holder_tempale = ""

                    for j in range(batch_frame_types[index].shape[-1]):
                        if batch_timestamps[index] is not None: # If has timestamps
                            video_place_holder_tempale += self.frame_token + format(batch_timestamps[index][j], ".1f")
                        else:
                            video_place_holder_tempale += self.frame_token

                        # Current frame is slow frame
                        if batch_frame_types[index][j] == 0:
                            num_patches = int(slow_video_grid_thw[slow_index].prod())
                            video_place_holder_tempale += "<|placeholder|>" * (
                                    num_patches // self.image_processor.merge_size ** 2)
                            pixel_values_videos.append(
                                slow_pixel_values_videos[slow_pixels_index:slow_pixels_index + num_patches])
                            slow_pixels_index = slow_pixels_index + num_patches
                            video_grid_thw.append(slow_video_grid_thw[slow_index])
                            slow_index += 1

                        # Current frame is fast frame
                        elif batch_frame_types[index][j] == 1:
                            num_patches = int(fast_video_grid_thw[fast_index].prod())
                            video_place_holder_tempale += self.fast_start + "<|placeholder|>" * (
                                    num_patches // self.image_processor.merge_size ** 2) + \
                                                          self.fast_end
                            pixel_values_videos.append(
                                fast_pixel_values_videos[fast_pixels_index:fast_pixels_index + num_patches])
                            fast_pixels_index = fast_pixels_index + num_patches
                            video_grid_thw.append(fast_video_grid_thw[fast_index])
                            fast_index += 1
                    text[i] = text[i].replace(
                        self.video_token,
                        video_place_holder_tempale,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

            videos_inputs["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)
            videos_inputs["video_grid_thw"] = merge_thw(video_grid_thw, num_frames)
            videos_inputs["num_frames"] = torch.LongTensor(num_frames)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
            self,
            generated_outputs: Union[torch.LongTensor, NDArray[np.long]],
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = False,
            **kwargs
    ) -> List[str]:
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length, )`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            Clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


class KeyeVL1_5VisionEmbeddings(nn.Module):
    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_positions = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def interpolate_pos_encoding(self, height: int, width: int, dim: int) -> torch.FloatTensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on dynamic resolution
        images.
        """
        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        sqrt_num_positions = torch_int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            position_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            pixel_values: (batch, length, channel, height, width)  batch * length = sum(t * h * w for all images in batch)
            image_grid_thw: torch.LongTensor  (nimages, 3), nimages represents number of images, 3 represents (t, h, w)
        Returns:
            (batch, length, dim)
        """
        if pixel_values.dim() == 6:
            assert pixel_values.shape[0] == 1
            pixel_values = pixel_values.squeeze(0)

        if pixel_values.dim() != 5:
            raise NotImplementedError(f"Expected 5-D input (batch, length, channel, height, width), got {pixel_values.shape}")

        batch, length, channel, height, width = pixel_values.shape
        assert image_grid_thw is not None

        tokens_per_img = torch.prod(image_grid_thw, dim=1).tolist()
        total_tokens = sum(tokens_per_img)
        assert total_tokens == batch * length, f"token mismatch: {total_tokens} vs {length}"

        embeddings = pixel_values.view(batch * length, channel, height, width)

        target_dtype = self.patch_embedding.weight.dtype
        embeddings = embeddings.to(dtype=target_dtype)
        embeddings = self.patch_embedding(embeddings)  # (batch * length, dim, gh=1, gw=1)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (batch * length, 1, dim)

        dim = embeddings.size(-1)

        token_embed_list = torch.split(embeddings.view(-1, dim), tokens_per_img, dim=0)

        hw_pos_dict = dict()
        outs = list()
        for img_embeds, (t, h, w) in zip(token_embed_list, image_grid_thw.tolist()):
            img_embeds = img_embeds.view(t, h * w, -1)  # (t, h * w, dim)
            if (h, w) not in hw_pos_dict:
                pos = self.interpolate_pos_encoding(h, w, dim)  # (1, h * w, dim)
                hw_pos_dict[(h, w)] = pos
            else:
                pos = hw_pos_dict[(h, w)]
            img_embeds = img_embeds + pos.expand(t, -1, -1)
            outs.append(img_embeds.view(-1, dim))  # (t * h * w, dim)

        embeddings = torch.cat(outs, dim=0)  # (batch * length, dim)
        embeddings = embeddings.view(batch, length, -1)
        return embeddings


def apply_rotary_pos_emb_flashatt(
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class KeyeVL1_5VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: bool = False,
            cu_seqlens: Optional[torch.IntTensor] = None,
            rope_emb: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """Input shape: Batch x Time x Channel"""

        use_flash_attn = self.config._attn_implementation == "flash_attention_2"
        if self.config._attn_implementation == "flash_attention_2" and cu_seqlens is None:
            raise ValueError(
                f"cu_seqlens must be not None when _attn_implementation is setting to `flash_attention_2`."
            )

        if attention_mask is not None:
            raise ValueError(
                f"Attention_mask is not None for vision model is not supported yet."
            )

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)

        cos, sin = rope_emb

        if use_flash_attn:
            queries, keys = apply_rotary_pos_emb_flashatt(queries, keys, cos, sin)
        else:
            queries, keys = apply_rotary_pos_emb_vision(queries, keys, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if use_flash_attn:

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

            attn_output = flash_attn_varlen_func(
                queries.squeeze(0),
                keys.squeeze(0),
                values.squeeze(0),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=self.is_causal,
                softmax_scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_output = attn_output.flatten(-2).unsqueeze(0)
        else:
            queries = queries.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [
                torch.split(tensor, lengths, dim=2) for tensor in (queries, keys, values)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=self.dropout if self.training else 0.0,
                    is_causal=self.is_causal,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
            attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class KeyeVL1_5VisionMLP(SiglipMLP):
    pass


class KeyeVL1_5VisionBlock(GradientCheckpointingLayer):
    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = KeyeVL1_5VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = KeyeVL1_5VisionMLP(config)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            output_attentions: bool = False,
            cu_seqlens: Optional[torch.IntTensor] = None,
            rope_emb: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            rope_emb=rope_emb,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs


class KeyeVL1_5VisionEncoderLayer(nn.Module):
    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = KeyeVL1_5VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = KeyeVL1_5VisionMLP(config)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            output_attentions: bool = False,
            cu_seqlens: Optional[torch.IntTensor] = None,
            rope_emb: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            rope_emb=rope_emb,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs


class KeyeVL1_5VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`KeyeVL1_5VisionBlock`].

    Args:
        config: KeyeVL1_5VisionConfig
    """

    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.layers = nn.ModuleList([KeyeVL1_5VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.rotary_pos_emb = KeyeVL1_5VisionRotaryEmbedding(head_dim // 2, getattr(config, "rope_theta", 10000.0))
        self.gradient_checkpointing = False

    def forward(
            self,
            inputs_embeds: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cu_seqlens: Optional[torch.IntTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            height_position_ids: Optional[torch.LongTensor] = None,
            width_position_ids: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        device = inputs_embeds.device
        hidden_states = inputs_embeds
        attention_mask = attention_mask.to(inputs_embeds.dtype) if attention_mask is not None else None

        pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
        max_grid_size = pids.max() + 1
        rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
        rope_emb = rope_emb_max_grid[pids].flatten(1)
        rope_emb = rope_emb.repeat(1, 2)
        rope_emb = (rope_emb.cos(), rope_emb.sin())

        if hidden_states.shape[0] != 1:
            batch, length, dim = hidden_states.shape
            hidden_states = hidden_states.reshape(1, batch * length, dim)

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    cu_seqlens,
                    rope_emb,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens,
                    rope_emb=rope_emb,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class KeyeVL1_5VisionModelOutput(SiglipVisionModelOutput):
    pass


KEYE_VL_1_5_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class KeyeVL1_5VisionTransformer(nn.Module):
    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = KeyeVL1_5VisionEmbeddings(config)
        self.encoder = KeyeVL1_5VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(KEYE_VL_1_5_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=KeyeVL1_5VisionConfig)
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            height_position_ids: Optional[torch.LongTensor] = None,
            width_position_ids: Optional[torch.LongTensor] = None,
            cu_seqlens: Optional[torch.IntTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:
            keye_vl_1_5_vision_model_output (`KeyeVL1_5VisionModelOutput` see class KeyeVL1_5VisionModelOutput for details.)
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        hidden_states = self.embeddings(
            pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_state = last_hidden_state.squeeze(0)
        assert hidden_state.shape[0] == cu_seqlens[-1].item()

        sample_hidden_state_list = list(torch.split(hidden_state, lengths, dim=0))

        return KeyeVL1_5VisionModelOutput(
            last_hidden_state=sample_hidden_state_list,
            image_embeds=None,
            hidden_states=None,
            attentions=encoder_outputs.attentions,
        )


class KeyeVL1_5RMSNorm(Qwen2RMSNorm):
    pass


@auto_docstring
class KeyeVL1_5PreTrainedModel(Qwen2_5_VLPreTrainedModel):
    config_class: KeyeVL1_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KeyeVL1_5DecoderLayer", "KeyeVL1_5VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0. is the standard default value accross the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.0)

        if isinstance(module, KeyeVL1_5VisionEmbeddings):
            nn.init.normal_(module.position_embedding.weight, std=std)
            module.patch_embedding.weight.data.normal_(mean=0.0, std=std)
            if module.patch_embedding.bias is not None:
                module.patch_embedding.bias.data.zero_()

        elif isinstance(module, (KeyeVL1_5RMSNorm, )):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, KeyeVL1_5VisionAttention):
            nn.init.normal_(module.q_proj.weight, std=std)
            nn.init.normal_(module.k_proj.weight, std=std)
            nn.init.normal_(module.v_proj.weight, std=std)
            nn.init.normal_(module.out_proj.weight, std=std)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, KeyeVL1_5Attention):
            nn.init.normal_(module.q_proj.weight, std=std)
            nn.init.normal_(module.k_proj.weight, std=std)
            nn.init.normal_(module.v_proj.weight, std=std)
            nn.init.normal_(module.o_proj.weight, std=std)
            if self.config.attention_bias:
                nn.init.zeros_(module.q_proj.bias)
                nn.init.zeros_(module.k_proj.bias)
                nn.init.zeros_(module.v_proj.bias)
                nn.init.zeros_(module.o_proj.bias)
            module.q_norm.weight.data.fill_(1.0)
            module.k_norm.weight.data.fill_(1.0)

        elif isinstance(module, KeyeVL1_5VisionMLP):
            nn.init.normal_(module.fc1.weight, std=std)
            nn.init.normal_(module.fc2.weight, std=std)
            nn.init.zeros_(module.fc1.bias)
            nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, KeyeVL1_5MLP):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring
class KeyeVL1_5VisionModel(KeyeVL1_5PreTrainedModel):
    config_class = KeyeVL1_5VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: KeyeVL1_5VisionConfig):
        super().__init__(config)

        self.vision_model = KeyeVL1_5VisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(KEYE_VL_1_5_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=KeyeVL1_5VisionConfig)
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            cu_seqlens: Optional[torch.IntTensor] = None,
            height_position_ids: Optional[torch.LongTensor] = None,
            width_position_ids: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:
            keye_vl_1_5_vision_model_output (`KeyeVL1_5VisionModelOutput` see class KeyeVL1_5VisionModelOutput for details.)
        ```"""

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
        )


class KeyeVL1_5VisionRotaryEmbedding(nn.Module):

    def __init__(
            self,
            dim: int,
            theta: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self) -> None:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
            self,
            seqlen: int
    ) -> torch.FloatTensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for KeyeVL1_5 causal language model (or autoregressive) outputs.
    """
)
class KeyeVL1_5CausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    pass


class KeyeVL1_5ModelOutputWithPast(Qwen2_5_VLModelOutputWithPast):
    pass


class KeyeVL1_5RotaryEmbedding(nn.Module):
    def __init__(self, config: KeyeVL1_5TextConfig, device=None) -> None:
        super().__init__()
        self.rope_kwargs = dict()

        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device) -> None:
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, KeyeVL1_5 has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rope_init(self) -> None:
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device=None, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


class KeyeVL1_5Attention(nn.Module):
    def __init__(self, config: KeyeVL1_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.attention_bias = config.attention_bias

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=self.attention_bias
        )
        self.q_norm = KeyeVL1_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = KeyeVL1_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

        self.rotary_emb = KeyeVL1_5RotaryEmbedding(config=config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,  # necessary, but kept here for BC
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        bsz, q_len, _ = hidden_states.size()

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = self.q_norm(queries.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        keys = self.k_norm(keys.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        values = values.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        queries, keys = apply_multimodal_rotary_pos_emb(
            queries, keys, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            keys, values = past_key_values.update(keys, values, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        attn_weights = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : keys.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in KeyeVL-1.5 float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if queries.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, values)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class KeyeVL1_5FlashAttention2(KeyeVL1_5Attention):
    """
    KeyeVL1_5 flash attention module, following KeyeVL1_5 attention module. This module inherits from `KeyeVL1_5Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,  # necessary, but kept here for BC
            cu_seqlens: Optional[torch.IntTensor] = None,
            sliding_window = -1,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        bsz, q_len, _ = hidden_states.size()
        q= self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        queries = self.q_norm(q)
        keys = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim))
        values = self.v_proj(hidden_states)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        queries, keys = apply_multimodal_rotary_pos_emb(
            queries, keys, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            keys, values = past_key_values.update(keys, values, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = queries.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            queries = queries.to(target_dtype)
            keys = keys.to(target_dtype)
            values = values.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if (
                sliding_window == -1
                and self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = -1

        if cu_seqlens is not None:
            # Sample packing with FA2
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            cu_seqlens = cu_seqlens.to(torch.int32)
            # remove batch_dim first: q.squeeze(0)
            attn_output = flash_attn_varlen_func(
                queries.squeeze(0),
                keys.squeeze(0),
                values.squeeze(0),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=dropout_rate,
                window_size=(sliding_window, sliding_window),
                causal=self.is_causal
            )
        else:
            attn_output = _flash_attention_forward(
                queries,
                keys,
                values,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                sliding_window=sliding_window,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class KeyeVL1_5SdpaAttention(KeyeVL1_5Attention):
    """
    KeyeVL-1.5 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `KeyeVL1_5Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,  # necessary, but kept here for BC
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "KeyeVL1_5Model is using KeyeVL1_5SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        queries = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim))
        keys = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim))
        values = self.v_proj(hidden_states)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        queries, keys = apply_multimodal_rotary_pos_emb(
            queries, keys, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            keys, values = past_key_values.update(keys, values, self.layer_idx, cache_kwargs)

        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : keys.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if queries.device.type == "cuda" and attention_mask is not None:
            queries = queries.contiguous()
            keys = keys.contiguous()
            values = values.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_values


class KeyeVL1_5MLP(Qwen2MLP):
    pass


KEYE_VL_1_5_ATTENTION_CLASSES = {
    "eager": KeyeVL1_5Attention,
    "flash_attention_2": KeyeVL1_5FlashAttention2,
    "sdpa": KeyeVL1_5SdpaAttention,
}


class KeyeVL1_5DecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: KeyeVL1_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = KEYE_VL_1_5_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = KeyeVL1_5MLP(config)
        self.input_layernorm = KeyeVL1_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KeyeVL1_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,  # necessary, but kept here for BC
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


@auto_docstring
class KeyeVL1_5TextModel(KeyeVL1_5PreTrainedModel):
    def __init__(self, config: KeyeVL1_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [KeyeVL1_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = KeyeVL1_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = KeyeVL1_5RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
            self,
            attention_mask: torch.FloatTensor,
            input_tensor: torch.FloatTensor,
            cache_position: torch.LongTensor,
            past_key_values: Cache,
            output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of KeyeVL1_5. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
                self.config._attn_implementation == "sdpa"
                and not (using_static_cache or using_sliding_window_cache)
                and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    sliding_window=self.config.sliding_window,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type in ["cuda", "xpu"]
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.FloatTensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            cache_position: torch.LongTensor,
            batch_size: int,
            config: KeyeVL1_5Config,
            past_key_values: Cache,
            device: torch.device = None,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`KeyeVL1_5Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            device = device or cache_position.device
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                            cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KeyeVL1_5Projector(nn.Module):
    def __init__(self, config: KeyeVL1_5Config):
        super().__init__()
        spatial_merge_size = config.vision_config.spatial_merge_size
        hidden_size = self.vision_config.hidden_size * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        
        self.merge_kernel_size = (spatial_merge_size, spatial_merge_size)

        self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(hidden_size, self.text_config.hidden_size, bias=True)

    def forward(
            self,
            image_features: List[torch.FloatTensor],
            image_grid_thw: torch.LongTensor,
    ) -> torch.FloatTensor:
        h_kernel, w_kernel = self.merge_kernel_size
        processed_features = list()
        for image_feature, (temporal, height, width) in zip(image_features, image_grid_thw.tolist()):
            image_feature = (
                image_feature.view(temporal, height // h_kernel, h_kernel, width // w_kernel, w_kernel, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 2)  # temporal * height * width
                .flatten(1, 3)  # h_kernel * w_kernel * dim
            )

            image_feature = self.pre_norm(image_feature)
            hidden_states = self.linear_1(image_feature)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed_features.append(hidden_states)

        processed_features = torch.concat(processed_features, dim=0)
        return processed_features


@auto_docstring
class KeyeVL1_5Model(KeyeVL1_5PreTrainedModel, Qwen2_5_VLModel):
    config: KeyeVL1_5Config
    base_model_prefix = ""
    _no_split_modules = ["KeyeVL1_5DecoderLayer", "KeyeVL1_5VisionBlock"]

    def __init__(self, config: KeyeVL1_5Config):
        super().__init__(config)
        self.visual = KeyeVL1_5VisionModel._from_config(config.vision_config)
        self.language_model = KeyeVL1_5TextModel._from_config(config.text_config)
        self.mm_projector = KeyeVL1_5Projector(config)

        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_rope_index(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4]
                vision height position_ids: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
                vision width position_ids: [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size, 1)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        mrope_position_deltas = list()
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            device = input_ids.device
            batch, length = input_ids.shape
            vision_token = torch.LongTensor([image_token_id, video_token_id]).to(device)

            batch_is_vision_token = (input_ids.unsqueeze(-1) == vision_token).any(-1)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            attention_mask = attention_mask.to(device)

            assert (~(batch_is_vision_token) | attention_mask.bool()).all(), \
                "Attention mask is False for vision token is not supported yet."

            batch_is_vision_token = batch_is_vision_token.unbind(0)
            attention_mask = attention_mask.unbind(0)

            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            if image_grid_thw is not None:
                llm_image_grid = torch.div(image_grid_thw, spatial_merge_size, rounding_mode="floor")
                llm_image_grid[:, 0] = image_grid_thw[:, 0]
                llm_image_grid = llm_image_grid.tolist()

            if video_grid_thw is not None:
                llm_video_grid = torch.div(video_grid_thw, spatial_merge_size, rounding_mode="floor")
                llm_video_grid[:, 0] = video_grid_thw[:, 0]
                llm_video_grid = llm_video_grid.tolist()

            def _get_text_pos_ids(mask, beg, fsh, pos_ids_list):
                text_length = mask[beg: fsh].sum().long().item()
                start_pos = pos_ids_list[-1].max() + 1 if pos_ids_list else 0
                text_pos_ids = torch.arange(start_pos, start_pos + text_length, device=device).expand(3, -1)
                return text_pos_ids

            image_index, video_index = 0, 0

            pos_ids_dict = dict()
            for i, inp_ids in enumerate(input_ids):
                vision_token_indices = (batch_is_vision_token[i] == True).nonzero().flatten().tolist()
                attn_mask = attention_mask[i]

                start = 0
                indices_iter = 0
                pos_ids = list()
                while indices_iter < len(vision_token_indices):
                    end = vision_token_indices[indices_iter]

                    if end > start:
                        pos_ids.append(_get_text_pos_ids(attn_mask, start, end, pos_ids))

                    if inp_ids[end] == image_token_id:
                        t, h, w = llm_image_grid[image_index]
                        image_index += 1
                    else:
                        t, h, w = llm_video_grid[video_index]
                        video_index += 1

                    start_pos = pos_ids[-1].max() + 1 if pos_ids else 0

                    if (t, h, w) in pos_ids_dict:
                        pos_ids.append(pos_ids_dict[(t, h, w)] + start_pos)
                    else:
                        # Here t is all 1, so the temporal position id is all 0.
                        thw_pos_ids = torch.stack(
                            torch.meshgrid(
                                torch.arange(t, device=device),
                                torch.arange(h, device=device),
                                torch.arange(w, device=device),
                                indexing="ij",
                            ),
                            dim=0
                        ).flatten(1)
                        pos_ids.append(thw_pos_ids + start_pos)
                        pos_ids_dict[(t, h, w)] = thw_pos_ids

                    indices_iter += t * h * w
                    start = end + t * h * w

                if start < length:
                    pos_ids.append(_get_text_pos_ids(attn_mask, start, length, pos_ids))

                mrope_position_deltas.append(pos_ids[-1].max() + 1 - length)

                position_ids[..., i, attn_mask == 1] = torch.concat(pos_ids, dim=1)

            mrope_position_deltas = torch.tensor(mrope_position_deltas).reshape(-1, 1).to(device)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
            self,
            pixel_values_videos: torch.FloatTensor,
            video_grid_thw: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        device = pixel_values_videos.device
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        pixel_values_videos = pixel_values_videos.unsqueeze(0)
        video_grid_thw = split_thw(video_grid_thw.squeeze(0)).to(device)

        assert torch.all(video_grid_thw[:, 0] == 1)

        total_patches = video_grid_thw.prod(dim=1)
        width = torch.repeat_interleave(video_grid_thw[:, 2], total_patches)
        cu_seqlens = total_patches.cumsum(0)
        arange = torch.arange(cu_seqlens[-1], dtype=torch.long, device=device)
        video_position_ids = arange - torch.repeat_interleave(cu_seqlens.to(device) - total_patches, total_patches)

        width_position_ids = torch.remainder(video_position_ids, width)
        height_position_ids = torch.div(video_position_ids, width, rounding_mode="floor")
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32, device=device)
        width_position_ids = width_position_ids.to(device)
        height_position_ids = height_position_ids.to(device)

        vision_outputs = self.visual(
            pixel_values=pixel_values_videos,
            image_grid_thw=video_grid_thw,
            position_ids=video_position_ids,
            cu_seqlens=cu_seqlens,
            width_position_ids=width_position_ids,
            height_position_ids=height_position_ids,
        )

        video_embeds = vision_outputs.last_hidden_state
        return video_embeds

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            image_grid_thw: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        device = pixel_values.device
        pixel_values = pixel_values.type(self.visual.dtype)
        pixel_values = pixel_values.unsqueeze(0)
        assert torch.all(image_grid_thw[:, 0] == 1)
        image_grid_thw = image_grid_thw.to(device)

        total_patches = image_grid_thw.prod(dim=1)
        width = torch.repeat_interleave(image_grid_thw[:, 2], total_patches)
        cu_seqlens = total_patches.cumsum(0)

        arange = torch.arange(cu_seqlens[-1], dtype=torch.long, device=device)
        image_position_ids = arange - torch.repeat_interleave(cu_seqlens.to(device) - total_patches, total_patches)

        width_position_ids = torch.remainder(image_position_ids, width)
        height_position_ids = torch.div(image_position_ids, width, rounding_mode="floor")
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32, device=device)
        width_position_ids = width_position_ids.to(device)
        height_position_ids = height_position_ids.to(device)

        vision_outputs = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=image_position_ids,
            cu_seqlens=cu_seqlens,
            width_position_ids=width_position_ids,
            height_position_ids=height_position_ids,
        )

        image_embeds = vision_outputs.last_hidden_state
        return image_embeds

    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_frames: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, KeyeVL1_5ModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        num_frames (`torch.LongTensor` of shape `(num_videos, )`, *optional*):
            The number of frames for each video.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            device = pixel_values.device
            pixel_values = pixel_values.type(self.visual.dtype)
            pixel_values = pixel_values.unsqueeze(0)
            assert torch.all(image_grid_thw[:, 0] == 1)
            image_grid_thw = image_grid_thw.to(device)

            total_patches = image_grid_thw.prod(dim=1)
            width = torch.repeat_interleave(image_grid_thw[:, 2], total_patches)
            cu_seqlens = total_patches.cumsum(0)

            arange = torch.arange(cu_seqlens[-1], dtype=torch.long, device=device)
            image_position_ids = arange - torch.repeat_interleave(cu_seqlens.to(device) - total_patches, total_patches)

            width_position_ids = torch.remainder(image_position_ids, width)
            height_position_ids = torch.div(image_position_ids, width, rounding_mode="floor")
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32, device=device)
            width_position_ids = width_position_ids.to(device)
            height_position_ids = height_position_ids.to(device)

            vision_outputs = self.visual(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=image_position_ids,
                cu_seqlens=cu_seqlens,
                width_position_ids=width_position_ids,
                height_position_ids=height_position_ids,
            )

            image_embeds = vision_outputs.last_hidden_state

            image_embeds = self.mm_projector(image_embeds, image_grid_thw)

            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            n_image_tokens = image_mask.sum().item()

            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(device)

            image_embeds = image_embeds.to(device, inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            device = pixel_values_videos.device
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            pixel_values_videos = pixel_values_videos.unsqueeze(0)
            video_grid_thw = split_thw(video_grid_thw.squeeze(0)).to(device)

            assert torch.all(video_grid_thw[:, 0] == 1)

            total_patches = video_grid_thw.prod(dim=1)
            width = torch.repeat_interleave(video_grid_thw[:, 2], total_patches)
            cu_seqlens = total_patches.cumsum(0)
            arange = torch.arange(cu_seqlens[-1], dtype=torch.long, device=device)
            video_position_ids = arange - torch.repeat_interleave(cu_seqlens.to(device) - total_patches, total_patches)

            width_position_ids = torch.remainder(video_position_ids, width)
            height_position_ids = torch.div(video_position_ids, width, rounding_mode="floor")
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32, device=device)
            width_position_ids = width_position_ids.to(device)
            height_position_ids = height_position_ids.to(device)

            vision_outputs = self.visual(
                pixel_values=pixel_values_videos,
                image_grid_thw=video_grid_thw,
                position_ids=video_position_ids,
                cu_seqlens=cu_seqlens,
                width_position_ids=width_position_ids,
                height_position_ids=height_position_ids,
            )

            video_embeds = vision_outputs.last_hidden_state
            video_embeds = self.mm_projector(video_embeds, video_grid_thw)
            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            n_video_tokens = video_mask.sum().item()

            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(device)

            video_embeds = video_embeds.to(device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas

            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        return KeyeVL1_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        ) if return_dict else (outputs + (self.rope_deltas, ))


class KeyeVL1_5ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        "^mlp_AR": "model.mm_projector",
        r"^model(?!\.(language_model|visual|mm_projector))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    config_class = KeyeVL1_5Config
    _no_split_modules = ["KeyeVL1_5DecoderLayer", "KeyeVL1_5VisionEncoderLayer"]

    @can_return_tuple
    @replace_return_docstrings(output_type=KeyeVL1_5CausalLMOutputWithPast, config_class="KeyeVL1_5Config")
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_frames: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple, KeyeVL1_5CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            keye_vl_1_5_causal_lm_output_with_past (`KeyeVL1_5CausalLMOutputWithPast` see class KeyeVL1_5CausalLMOutputWithPast for details.)

        Example:

        ```python
        >>> !pip install --upgrade keye-vl-utils==1.5.2 -i https://pypi.org/simple
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, KeyeVL1_5ForConditionalGeneration
        >>> from keye_vl_utils import process_vision_info

        >>> model = KeyeVL1_5ForConditionalGeneration.from_pretrained("Kwai-Keye/Keye-VL-1_5-8B").cuda()
        >>> processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-1_5-8B", trust_remote_code=True)
        >>> tokenizer = processor.tokenizer
        >>> url = "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg"

        >>> messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": url},
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        >>> # Since we support the slow-fast architecture and keye-vl-utils has additional return parameters,
        >>> # we did not adopt the combined form of:
        >>> # inputs = processor.apply_chat_template( messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt" )

        >>> text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        >>> image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
        >>> inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs
            ).to(model.device)

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=51200)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "system\nYou are a helpful assistant.\nuser\nWhat is shown in this image?\nassistant\n<analysis>"
        "This question is straightforward and asks for a description of what is shown in the image. "
        "Therefore, /no_think mode is more appropriate.</analysis>The image shows a white cat lying on a pink and "
        "purple patterned surface. The cat appears to be a breed with a flat face, possibly a Persian or a similar "
        "breed. It has large, round eyes and is looking directly at the camera. One of its front paws is raised, "
        "giving the impression that it might be waving or reaching out. The background is neutral, likely a wall "
        "or a piece of furniture, which helps to focus attention on the cat. The overall scene conveys a sense of cuteness and playfulness."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_frames=num_frames,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return KeyeVL1_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Cache] = None,
            attention_mask: torch.FloatTensor = None,
            inputs_embeds: torch.FloatTensor = None,
            cache_position: torch.LongTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            pixel_values: torch.FloatTensor = None,
            pixel_values_videos: torch.FloatTensor = None,
            image_grid_thw: torch.LongTensor = None,
            video_grid_thw: torch.LongTensor = None,
            **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # KeyeVL-1.5 position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
            self,
            input_ids: Optional[torch.LongTensor],
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size)`)
            video_nums (`torch.LongTensor` of shape `(batch_size)`)
        """
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        if inputs_embeds is not None:
            vision_start_mask = (
                    inputs_embeds
                    == self.get_input_embeddings()(
                torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            )[..., 0]
            image_mask = (
                    inputs_embeds
                    == self.get_input_embeddings()(
                torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        # Because our video data does not immediately follow a `vision_start_token_id`
        # with a `video_token`, there are `frame_token` and `timestamps-infomation` in
        # between. So we determine that it is of the video type by checking that what
        # follows the `vision_start_token_id `` is not an image_token.
        video_nums = torch.sum(vision_first_mask & ~image_mask, dim=1)
        return image_nums, video_nums

    def _expand_inputs_for_generation(
            self,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            input_ids: Optional[torch.LongTensor] = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand: Dict[str, Any]) -> Dict[str, Any]:
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds")
            )
            image_nums = image_nums.tolist()
            video_nums = video_nums.tolist()

            def _repeat_interleave_samples(x: torch.Tensor, lengths: List[int], repeat_times: int) -> torch.Tensor:
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            merge_size = self.config.vision_config.spatial_merge_size ** 2

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, image_nums)
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() // merge_size for sample in samples]

                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=image_nums, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, video_nums)
                    lengths = [torch.prod(sample, dim=1).sum() // merge_size for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=video_nums, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand: Dict[str, Any]) -> Dict[str, Any]:
            for key in dict_to_expand:
                if (
                        key != "cache_position"
                        and dict_to_expand[key] is not None
                        and isinstance(dict_to_expand[key], torch.Tensor)
                        and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None or model_kwargs.get("inputs_embeds") is not None:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "KeyeVL1_5ForConditionalGeneration",
    "KeyeVL1_5VisionModel",
    "KeyeVL1_5Model",
    "KeyeVL1_5Config",
    "KeyeVL1_5PreTrainedModel",
    "KeyeVL1_5Processor",
    "KeyeVL1_5TextConfig",
    "KeyeVL1_5TextModel",
    "KeyeVL1_5ImageProcessor",
]
