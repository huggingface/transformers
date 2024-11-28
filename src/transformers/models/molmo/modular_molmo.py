# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import einops
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    normalize,
    pad,
    resize,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ..clip.modeling_clip import (
    CLIPVisionTransformer,
)
from ..llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaForConditionalGeneration
from ..qwen2.configuration_qwen2 import Qwen2Config
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2SdpaAttention,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipFlashAttention2,
    SiglipMLP,
    SiglipSdpaAttention,
    SiglipVisionModel,
)


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class MolmoVisionConfig(SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoVisionModel`]. It is used to instantiate a
    `MolmoVisionModel` according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Molmo
    [allenai/Molmo-7B-D-0924-hf](https://huggingface.co/allenai/Molmo-7B-D-0924-hf) architecture.

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
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    Example:

    ```python
    >>> from transformers import MolmoVisionConfig, MolmoVisionModel

    >>> # Initializing a MolmoVisionConfig with allenai/Molmo-7B-D-0924-hf style configuration
    >>> configuration = MolmoVisionConfig()

    >>> # Initializing a MolmoVisionModel (with random weights) from the allenai/Molmo-7B-D-0924-hf style configuration
    >>> model = MolmoVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        num_hidden_layers=23,
        num_image_positions=577,
        image_size=576,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_positions = num_image_positions
        self.hidden_act = hidden_act


class MolmoPoolingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoAdapterModel`]. It is used to instantiate an
    `MolmoAdapterModel` according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Molmo-7B-D.

    e.g. [allenai/Molmo-7B-D-0924-hf](https://huggingface.co/allenai/Molmo-7B-D-0924-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the pooler attention layer.
        text_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the text encoder layers.
        text_intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the text Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer pooler.
        head_dim (`int`, *optional*, defaults to 64):
            The poolinng attention head dimension.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        pooling_height (`int`, *optional*, defaults to 2):
            The height of image features requred for pooling operation.
        pooling_width (`int`, *optional*, defaults to 2):
            The width of image features requred for pooling operation.
        pad_embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of a padding tensor which is multiplied with the image mask.
        image_num_patches (`int`, *optional*, defaults to 24):
            Number of patches each image feature has after the vision tower.
        image_feature_dropout (`float`, *optional*, defaults to 0.9):
            The dropout ratio for the image features after vision tower.
        image_pooling_type (`str`, *optional*, defaults to `"attention_meanq"`):
            Type of pooling to apply on image features. Can be one of ["attention", "attention_meanq", "attention_2wide", "attention_v2", "stack"] or `None`
        image_padding_embed (`str`, *optional*, defaults to `"pad_and_partial_pad"`):
            Type of padding to apply of image masks. Can be one of ["pad_embed", "regress", "pad_and_partial_pad]
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import MolmoAdapterModel, MolmoPoolingConfig

    >>> # Initializing a Molmo-pooling config
    >>> pooling_config = MolmoPoolingConfig()

    >>> # Initializing a adapter model from the allenai/Molmo-7B-D-0924-hf style configuration
    >>> model = MolmoAdapterModel(pooling_config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        hidden_size=2048,
        num_attention_heads=16,
        head_dim=64,
        attention_dropout=0.0,
        initializer_range=0.02,
        pooling_height=2,
        pooling_width=2,
        pad_embed_dim=2048,
        image_num_patches=24,
        image_feature_dropout=0.0,
        text_intermediate_size=37888,
        text_hidden_size=3584,
        image_pooling_type="attention_meanq",
        image_padding_embed="pad_and_partial_pad",
        projector_hidden_act="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.pooling_height = pooling_height
        self.pooling_width = pooling_width
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.pad_embed_dim = pad_embed_dim
        self.image_num_patches = image_num_patches
        self.image_feature_dropout = image_feature_dropout
        self.text_intermediate_size = text_intermediate_size
        self.text_hidden_size = text_hidden_size
        self.image_pooling_type = image_pooling_type
        self.image_padding_embed = image_padding_embed
        self.projector_hidden_act = projector_hidden_act


class MolmoTextConfig(Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`MolmoModel`]. It is used to instantiate a
    Molmo model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Molmo-7B-beta [Qwen/Molmo-7B-beta](https://huggingface.co/Qwen/Molmo-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Molmo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MolmoTextModel`]
        additional_vocab_size  (`int`, *optional*, defaults to 128):
            Number of additional tokens added to the vocabulary size of the Molmo model.
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 37888):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        head_dim (`int`, *optional*, defaults to 128):
            The poolinng attention head dimension.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"swiglu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
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
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import MolmoTextModel, MolmoTextConfig

    >>> # Initializing a Molmo style configuration
    >>> configuration = MolmoTextConfig()

    >>> # Initializing a model from the Molmo-7B style configuration
    >>> model = MolmoTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        hidden_size=3584,
        num_key_value_heads=4,
        num_attention_heads=28,
        num_hidden_layers=28,
        head_dim=128,
        vocab_size=152064,
        additional_vocab_size=128,
        intermediate_size=37888,
        hidden_act="swiglu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.additional_vocab_size = additional_vocab_size
        self.head_dim = head_dim
        super().__init__(**kwargs)


class MolmoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoForConditionalGeneration`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Molmo-7B-D.

    e.g. [allenai/Molmo-7B-D-0924-hf](https://huggingface.co/allenai/Molmo-7B-D-0924-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `MolmoVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MolmoTextConfig`):
            The config object or dictionary of the text backbone.
        pooling_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MolmoPoolingConfig`):
            The config object or dictionary of the adapter backbone.
        image_token_index (`int`, *optional*, defaults to 152069):
            The image token index to encode the image prompt.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layers (`List[int]`, *optional*, defaults to (-2, -9)):
            The indices of the layers to select the vision feature.

    Example:

    ```python
    >>> from transformers import MolmoForConditionalGeneration, MolmoConfig, MolmoVisionConfig, MolmoTextConfig, MolmoPoolingConfig

    >>> # Initializing a Molmo-vision config
    >>> vision_config = MolmoVisionConfig()

    >>> # Initializing a Molmo-text config
    >>> text_config = MolmoTextConfig()

    >>> # Initializing a Molmo-pooling config
    >>> pooling_config = MolmoPoolingConfig()

    >>> # Initializing a Molmo allenai/Molmo-7B-D-0924-hf style configuration
    >>> configuration = MolmoConfig.from_text_vision_configs(vision_config, text_config, pooling_config)

    >>> # Initializing a model from the allenai/Molmo-7B-D-0924-hf style configuration
    >>> model = MolmoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmo"
    sub_configs = {
        "text_config": MolmoTextConfig,
        "vision_config": MolmoVisionConfig,
        "pooling_config": MolmoPoolingConfig,
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        pooling_config=None,
        image_token_index=152069,
        image_seq_length=576,
        initializer_range=0.02,
        vision_feature_select_strategy="default",
        vision_feature_layers=(-2, -9),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layers = vision_feature_layers
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the MolmoVisionConfig with default values.")
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. initializing the MolmoTextConfig with default values.")
        if pooling_config is None:
            pooling_config = {}
            logger.info("pooling_config is None. initializing the MolmoPoolingConfig with default values.")
        self.vision_config = MolmoVisionConfig(**vision_config)
        self.text_config = MolmoTextConfig(**text_config)
        self.pooling_config = MolmoPoolingConfig(**pooling_config)
        self.initializer_range = initializer_range

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: MolmoTextConfig,
        vision_config: MolmoVisionConfig,
        pooling_config: MolmoPoolingConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MolmoConfig`] (or a derived class) from molmo text model configuration, molmo vision model
        configuration and molmo pooling module conffiguration.

        Returns:
            [`MolmoConfig`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            pooling_config=pooling_config.to_dict(),
            **kwargs,
        )


class MolmoCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


# swiglu activation
class MolmoSwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return nn.functional.silu(gate) * x


# text modules inherited from Qwen2
class MolmoMLP(SiglipMLP):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = MolmoSwiGLU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=False)


# We have different attention classes for the txt and the image components, they need to be propagated back correctly
class MolmoTextAttention(Qwen2Attention):
    pass


class MolmoTextSdpaAttention(MolmoTextAttention, Qwen2SdpaAttention):
    pass


class MolmoTextFlashAttention2(MolmoTextAttention, Qwen2FlashAttention2):
    pass


MOLMO_TEXT_ATTENTION_CLASSES = {
    "eager": MolmoTextAttention,
    "sdpa": MolmoTextSdpaAttention,
    "flash_attention_2": MolmoTextFlashAttention2,
}


class MolmoDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.mlp = MolmoMLP(config)
        self.self_attn = MOLMO_TEXT_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)


MOLMO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MolmoConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Molmo Model outputting raw hidden-states without any specific head on top.",
    MOLMO_START_DOCSTRING,
)
class MolmoPreTrainedModel(PreTrainedModel):
    config_class = MolmoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MolmoDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)


class MolmoTextModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size + config.additional_vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [MolmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.post_init()


# TODO the name matching here is error-inducing as MolmoForCausalLM isn't a standalone generative model
class MolmoForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MolmoTextModel(config)
        self.post_init()


# New Molmo multimodal projection and image pooling


class MolmoMultiModalProjector(nn.Module):
    def __init__(self, config: MolmoPoolingConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.hidden_size // 2,
            config.text_intermediate_size // 2,
            bias=False,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_3 = nn.Linear(
            config.hidden_size // 2,
            config.text_intermediate_size // 2,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_intermediate_size // 2,
            config.text_hidden_size,
            bias=False,
        )

    def forward(self, image_features):
        hidden_states = self.act(self.linear_1(image_features)) * self.linear_3(image_features)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# Molmo image components inherited from SiglipVision
# We have different attention classes for the txt and the image components, they need to be propagated back correctly


class MolmoVisionAttention(SiglipAttention):
    pass


class MolmoVisionSdpaAttention(MolmoVisionAttention, SiglipSdpaAttention):
    pass


class MolmoVisionFlashAttention2(MolmoVisionAttention, SiglipFlashAttention2):
    pass


MOLMO_VISION_ATTENTION_CLASSES = {
    "eager": MolmoVisionAttention,
    "sdpa": MolmoVisionSdpaAttention,
    "flash_attention_2": MolmoVisionFlashAttention2,
}


class MolmoVisionEmbeddings(nn.Module):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Linear(
            self.patch_size**2 * 3,
            self.embed_dim,
            bias=False,
        )

        self.position_embedding = nn.Embedding(config.num_image_positions, config.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(config.num_image_positions).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, patches, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size):
            raise ValueError(f"Input image size ({height}) doesn't match model" f" ({self.image_size}).")
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        class_embeds = self.class_embedding.expand(batch_size, patches, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=2)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids).unsqueeze(1)
        return embeddings.flatten(0, 1)  # NOTE: DON'T FLATTEN MORE TO MATCH ORIG IMPL


class MolmoVisionMLP(SiglipMLP):
    pass


class MolmoVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.self_attn = MOLMO_VISION_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = MolmoVisionMLP(config)


class MolmoVisionEncoder(SiglipEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MolmoVisionEncoderLayer`].

    Args:
        config: MolmoConfig
    """

    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([MolmoVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class MolmoVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.embeddings = MolmoVisionEmbeddings(config)
        embed_dim = config.hidden_size
        self.encoder = MolmoVisionEncoder(config)  # necessary because of renaming issue in modular
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        del self.post_layernorm

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MolmoVisionConfig)
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MolmoPoolingAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim // 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()
        seq_len = key_value_hidden_states.shape[1]
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = (
            self.k_proj(key_value_hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            self.v_proj(key_value_hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = (
            query_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(*proj_shape)
        )
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights_reshaped


class MolmoPoolingSdpaAttention(MolmoPoolingAttention):
    """
    SDPA attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MolmoPoolingAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Molmo is using MolmoPoolingSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
                "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
                'be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                key_value_hidden_states=key_value_hidden_states,
                output_attentions=output_attentions,
            )

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(key_value_hidden_states)
        value_states = self.v_proj(key_value_hidden_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


class MolmoPoolingFlashAttention2(MolmoPoolingAttention):
    """
    MolmoPoolingAttention flash attention module. This module inherits from `MolmoPoolingAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output_attentions = False

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(key_value_hidden_states)
        value_states = self.v_proj(key_value_hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim)

        dropout_rate = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        input_dtype = query_states.dtype
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

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            None,
            q_len,
            dropout=dropout_rate,
            is_causal=False,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


MOLMO_POOLING_ATTENTION_CLASSES = {
    "eager": MolmoPoolingAttention,
    "sdpa": MolmoPoolingSdpaAttention,
    "flash_attention_2": MolmoPoolingFlashAttention2,
}


@add_start_docstrings(
    """The adapter model from MOLMO that takes in image hidden states from vision tower.""",
    MOLMO_START_DOCSTRING,
)
class MolmoAdapterModel(MolmoPreTrainedModel):
    config_class = MolmoPoolingConfig
    main_input_name = "image_features"

    def __init__(self, config: MolmoPoolingConfig):
        super().__init__(config)

        attention_class = MOLMO_POOLING_ATTENTION_CLASSES[config._attn_implementation]
        if config.image_pooling_type in {"attention", "attention_meanq"}:
            self.image_pooling_2d = attention_class(config)
        elif config.image_pooling_type == "attention_2wide":
            self.image_pooling_2d = attention_class(config)
        elif config.image_pooling_type == "attention_v2":
            self.image_pooling_2d = attention_class(config)
        elif config.image_pooling_type in [None, "stack"]:
            self.image_pooling_2d = None
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.pooling_config.image_pooling_type}")

        if config.image_padding_embed is not None:
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(torch.zeros((config.pad_embed_dim,)))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(torch.zeros((2, config.pad_embed_dim)))
            else:
                raise ValueError(config.image_padding_embed)

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)
        self.multi_modal_projector = MolmoMultiModalProjector(config)

    def forward(self, image_features, image_masks) -> torch.FloatTensor:
        batch_size, patches = image_features.shape[:2]
        if self.config.image_padding_embed is not None:
            image_padding_embed = self.config.image_padding_embed
            if image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(
                    torch.maximum(image_masks, torch.zeros_like(image_masks)), -1
                )
            elif image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(
                    dtype=image_features.dtype
                )
                all_pad = all_pad.to(dtype=image_features.dtype)
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(image_padding_embed)

        image_features = self.image_feature_dropout(image_features)
        num_patches = self.config.image_num_patches
        image_features = image_features.reshape(
            (batch_size, patches) + (num_patches, num_patches) + (-1,),
        )

        if num_patches % self.config.pooling_height == 1:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            )

        # image pooling
        image_features = einops.rearrange(
            image_features,
            "b n (h dh) (w dw) c -> (b n h w) (dh dw) c",
            dh=self.config.pooling_height,
            dw=self.config.pooling_width,
        )

        if self.config.image_pooling_type == "attention_meanq":
            queries = image_features.mean(-2, keepdim=True)
            image_features = self.image_pooling_2d(queries, image_features)[0]
        elif self.config.image_pooling_type not in {None, "stack"}:
            queries = image_features[:, :1, :]
            image_features = self.image_pooling_2d(queries, image_features)[0]

        # Round up in case we need to pad the image features for pooling
        h = (num_patches + self.config.pooling_height - 1) // self.config.pooling_height
        w = (num_patches + self.config.pooling_width - 1) // self.config.pooling_width

        image_features = image_features.reshape(batch_size, patches, h * w, -1)
        image_features = self.multi_modal_projector(image_features)
        return image_features


class MolmoVisionModel(SiglipVisionModel):
    config_class = MolmoVisionConfig  # needed because renames

    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.vision_model = MolmoVisionTransformer(config)


class MolmoForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        self.adapter = MolmoAdapterModel._from_config(config.pooling_config)

        self.language_model = MolmoForCausalLM._from_config(config.text_config)
        self.vision_tower = MolmoVisionModel._from_config(config.vision_config)
        self.post_init()

        del self.multi_modal_projector

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_masks,
        vision_feature_layers: List,
        vision_feature_select_strategy: str,
    ):
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        batch_size, patches, height, width = pixel_values.shape

        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        features = []
        image_features = image_outputs.hidden_states
        for layer in vision_feature_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        image_features = image_features.view(batch_size, patches, -1, image_features.shape[-1])
        if vision_feature_select_strategy == "default":
            image_features = image_features[:, :, 1:, :]

        image_features = self.adapter(image_features, image_masks)

        return image_features

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_masks=None,
        image_token_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layers: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, MolmoCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MolmoForConditionalGeneration

        >>> model = MolmoForConditionalGeneration.from_pretrained("molmo-hf/molmo-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("molmo-hf/molmo-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None and image_token_indices is not None:
            batch_size, num_crops, height, width = pixel_values.shape
            seq_len = inputs_embeds.size(1)
            hidden_size = inputs_embeds.size(2)
            valid_crops = pixel_values.abs().sum(dim=[2, 3]) > 0

            pixel_values_flat = pixel_values.view(-1, height, width)
            image_masks_flat = image_masks.view(-1, image_masks.size(-1))
            image_token_indices_flat = image_token_indices.view(-1, image_token_indices.size(-1))

            valid_crops_flat = valid_crops.view(-1)

            all_pixel_values = pixel_values_flat[valid_crops_flat]
            all_image_masks = image_masks_flat[valid_crops_flat]
            all_image_token_indices = image_token_indices_flat[valid_crops_flat]

            batch_indices = (
                torch.arange(batch_size, device=pixel_values.device).unsqueeze(1).expand(-1, num_crops).reshape(-1)
            )
            valid_batch_indices = batch_indices[valid_crops_flat]
            # now all valid crops together
            image_features = self.get_image_features(
                pixel_values=all_pixel_values.unsqueeze(1),
                image_masks=all_image_masks.unsqueeze(1),
                vision_feature_layers=vision_feature_layers,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )  # this returns [total_valid_crops, num_image_tokens, hidden_size]

            image_features_flat = image_features.view(-1, hidden_size)
            image_token_indices_flat = all_image_token_indices.view(-1)

            valid_indices_mask = image_token_indices_flat != -100
            image_token_indices_flat[valid_indices_mask] += 1  # adjustment, TODO is this still needed

            valid_batch_indices_expanded = (
                valid_batch_indices.unsqueeze(1).expand(-1, all_image_token_indices.size(-1)).reshape(-1)
            )

            valid_positions = image_token_indices_flat >= 0
            valid_indices = image_token_indices_flat[valid_positions].long()
            valid_features = image_features_flat[valid_positions]
            valid_batch_indices = valid_batch_indices_expanded[valid_positions].long()

            flat_indices = valid_batch_indices * seq_len + valid_indices
            inputs_embeds_flat = inputs_embeds.view(-1, hidden_size)

            inputs_embeds_flat.index_add_(0, flat_indices, valid_features.to(inputs_embeds_flat.device))
            inputs_embeds = inputs_embeds_flat.view(batch_size, seq_len, hidden_size)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MolmoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_masks=None,
        image_token_indices=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_indices"] = image_token_indices
            model_inputs["image_masks"] = image_masks

        return model_inputs


### IMAGE PROCESSING CODE


def make_batched_images(images) -> List[List[ImageInput]]:
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

    raise ValueError(f"Could not make batched video from {images}")


def get_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
) -> tuple:
    original_height, original_width = get_image_size(image)

    scale_y = size["height"] / original_height
    scale_x = size["width"] / original_width
    scale = min(scale_x, scale_y)

    # Compute new dimensions
    new_height = round(original_height * scale)
    new_width = round(original_width * scale)
    return {"height": new_height, "width": new_width}


def pad_to_bounding_box(
    image: np.ndarray, offset_height: int, offset_width: int, target_height: int, target_width: int, value: int = 0
) -> np.ndarray:
    """
    Pad the input image to the target height and width using the transformers `pad` function.

    Args:
        image: The input image to be padded.
        offset_height: The number of pixels to add to the top of the image.
        offset_width: The number of pixels to add to the left of the image.
        target_height: The target height of the padded image.
        target_width: The target width of the padded image.
        value: The constant value used for padding (default is 0).

    Returns:
        A padded image of size (target_height, target_width).
    """
    height, width = image.shape[:2]
    after_padding_height = target_height - offset_height - height
    after_padding_width = target_width - offset_width - width
    return np.pad(
        image,
        [
            (offset_height, after_padding_height),
            (offset_width, after_padding_width),
            (0, 0),  # don't pad on the channel dim
        ],
        mode="constant",
        constant_values=value,
    )


class MolmoImageProcessor(BaseImageProcessor):
    """
    Image processor for the Molmo model.

    This processor handles resizing, padding, grid shape, and patch extraction from images,
    converting them into inputs suitable for the Molmo model.
    """

    model_input_names = ["pixel_values", "input_ids", "image_input_idx", "image_masks"]

    def __init__(
        self,
        max_num_crops: int = 12,
        overlap_margins: Tuple[int, int] = [4, 4],
        size: Dict[str, int] = None,
        tokens_per_image_width: int = 12,
        tokens_per_image_height: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_pad: Optional[bool] = True,
        padding_value: float = 1.0,
        padding_mode: str = "constant",
        do_split_into_crops: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        image_patch_token: str = "<im_patch>",
        image_column_token: str = "<im_col>",
        image_start_token: str = "<im_start>",
        image_end_token: str = "<im_end>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 336, "width": 336}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_pad = do_pad
        self.padding_value = padding_value
        self.padding_mode = padding_mode
        self.do_split_into_crops = do_split_into_crops
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.max_num_crops = max_num_crops
        self.overlap_margins = overlap_margins
        self.tokens_per_image_width = tokens_per_image_width
        self.tokens_per_image_height = tokens_per_image_height
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.image_patch_token = image_patch_token
        self.image_column_token = image_column_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
            "do_pad",
            "do_split_into_crops",
            "padding_mode",
            "padding_value",
        ]

        # TODO move these to configuration once processing is done.
        self.tokens_per_image = tokens_per_image_height * tokens_per_image_width
        self.patches_per_image_width = size["width"] // image_patch_size
        self.patches_per_image_height = size["height"] // image_patch_size
        self.total_margin_pixels = image_patch_size * (overlap_margins[1] + overlap_margins[0])
        self.crop_patches = self.size["width"] // self.image_patch_size  # patches per crop dim
        self.crop_window_patches = self.crop_patches - (
            self.overlap_margins[1] + self.overlap_margins[0]
        )  # usable patches
        self.crop_window_size = self.crop_window_patches * self.image_patch_size
        self.crop_size = size["width"]

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def pad(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if "height" not in size or "width" not in size:
            raise ValueError("Size must contain 'height' and 'width'.")
        new_size = get_resize_output_image_size(image, size)
        padding_height = size["height"] - new_size["height"]
        padding_width = size["width"] - new_size["width"]
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        mask_padding = [
            [padding_top, size["height"] - new_size["height"] - padding_top],
            [padding_left, size["width"] - new_size["width"] - padding_left],
        ]
        if input_data_format == ChannelDimension.FIRST:
            image_to_pad = image[0, :, :]
        elif input_data_format == ChannelDimension.LAST:
            image_to_pad = image[:, :, 0]
        else:
            raise ValueError(f"Invalid channel dimension format: {input_data_format}")

        image_mask = np.pad(np.ones_like(image_to_pad, dtype=bool), mask_padding)

        return padded_image, image_mask

    def find_best_crop_grid_for_image_size(self, image: ImageInput):
        """
        Decide how best to divide an image of size {"width": width, "height": height}]
        in up to max_num_crops of size crop_size
        """
        original_size = np.array(
            [image.shape[0] - self.total_margin_pixels, image.shape[1] - self.total_margin_pixels], dtype=np.float32
        )
        crop_grid = [(i, j) for i in range(1, self.max_num_crops + 1) for j in range(1, (self.max_num_crops // i) + 1)]

        # sort so argmin and argmax favour smaller crop_grid in the event of a tie
        crop_grid.sort(key=lambda x: (x[0] * x[1], x[0]))
        candidate_crop_grid = np.array(crop_grid, dtype=np.int32)  # [n_resolutions, 2]
        candidate_resolutions = candidate_crop_grid * self.crop_window_size  # [n_resolutions, 2]

        required_scale_step = candidate_resolutions.astype(np.float32) / original_size
        required_scale = np.min(required_scale_step, axis=-1, keepdims=True)  # [n_resolutions, 1]

        if np.all(required_scale < 1):
            # min downscaling
            selected_index = np.argmax(required_scale)
        else:
            # same with upscaling
            required_scale = np.where(required_scale < 1.0, np.inf, required_scale)
            selected_index = np.argmin(required_scale)

        return candidate_crop_grid[selected_index]

    def reshape_into_patches(self, global_image, input_data_format):
        if input_data_format == ChannelDimension.FIRST:
            global_image = np.transpose(global_image, (1, 2, 0))
        channels = global_image.shape[-1]

        global_image = global_image.reshape(
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
            channels,
        )
        global_image = global_image.transpose(0, 2, 1, 3, 4)
        global_image = global_image.reshape(
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size * channels,
        )
        return global_image

    def split_image_into_crops(
        self,
        image: np.ndarray,
        image_mask: np.ndarray,
        crop_grid: Tuple[int, int],
        input_data_format,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the image into crops (patches), while keeping track of the patch ordering and generating masks for each crop.

        Args:
            image: The resized and padded image as a NumPy array.
            image_mask: The mask corresponding to the image, indicating valid pixels.
            crop_grid: Tuple (num_rows, num_cols) representing how the image is divided into crops (crop grid).
            crop_stride: The step size or stride used to move between crops.
            patch_grid_height: The number of patches along the height of the image grid.
            patch_grid_width: The number of patches along the width of the image grid.

        Returns:
            crops: Array of image patches/crops.
            patch_ordering: Array representing the ordering of patches within the original image.
            cropped_masks: Array of masks corresponding to the image crops.
        """
        if input_data_format == ChannelDimension.FIRST:
            image = np.transpose(image, (1, 2, 0))
        crops = []
        cropped_masks = []
        patch_orderings = []

        # Check if patch grid size matches expected dimensions
        if ((self.patches_per_image_height + 1) // 2 != self.tokens_per_image_height) or (
            (self.patches_per_image_width + 1) // 2 != self.tokens_per_image_width
        ):
            raise ValueError("Number of patches per crop does not fit number of tokens per image dimension.")

        patch_index = 0  # Track the index for patch ordering
        for row in range(crop_grid[0]):  # Loop over rows of crops
            crop_y_start = row * self.crop_window_size

            # calculate crop height, accounting for margins (there are overlaps, remember)
            current_crop_height = self.patches_per_image_height - (self.overlap_margins[1] + self.overlap_margins[0])
            if row == 0:  # add left margin for the first row
                current_crop_height += self.overlap_margins[0]
            if row == (crop_grid[0] - 1):  # add right margin for the last row
                current_crop_height += self.overlap_margins[1]

            crop_y_offset = self.overlap_margins[0] // 2 if row > 0 else 0
            for column in range(crop_grid[1]):  # Loop over columns of crops
                crop_x_start = column * self.crop_window_size

                # Calculate crop width, accounting for margins
                current_crop_width = self.patches_per_image_width - (self.overlap_margins[1] + self.overlap_margins[0])
                if column == 0:  # add left margin for the first column
                    current_crop_width += self.overlap_margins[0]
                if column == (crop_grid[1] - 1):  # add right margin for the last column
                    current_crop_width += self.overlap_margins[1]

                pooled_width = (current_crop_width + 1) // 2
                pooled_height = (current_crop_height + 1) // 2

                # Correct padding based on margins and offsets
                crop_x_offset = self.overlap_margins[0] // 2 if column > 0 else 0

                # Track patch ordering: generate an array representing the order of patches (overlaps (on crops))
                reshaped_image = np.reshape(
                    np.arange(patch_index, patch_index + pooled_height * pooled_width, dtype=np.int32),
                    (pooled_height, pooled_width, 1),
                )
                patch_orderings.append(
                    pad_to_bounding_box(
                        reshaped_image,
                        offset_height=crop_y_offset,
                        offset_width=crop_x_offset,
                        target_height=self.tokens_per_image_height,
                        target_width=self.tokens_per_image_width,
                        value=-1,
                    )[:, :, 0]
                )

                # Extract the image crop
                crops.append(
                    image[crop_y_start : crop_y_start + self.crop_size, crop_x_start : crop_x_start + self.crop_size]
                )

                # Extract the corresponding mask for the crop
                cropped_masks.append(
                    image_mask[
                        crop_y_start : crop_y_start + self.crop_size, crop_x_start : crop_x_start + self.crop_size
                    ]
                )
                # Update the patch index for ordering (there are several patches in a crop)
                patch_index += pooled_height * pooled_width
        # Stack the crops, patch orderings, and masks into arrays
        crops = np.stack(crops)
        patch_orderings = np.stack(patch_orderings)
        cropped_masks = np.stack(cropped_masks)
        # rearrange patches
        leading_crops_dim, channels = crops.shape[0], crops.shape[-1]
        crops = crops.reshape(
            leading_crops_dim,
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
            channels,
        )
        crops = crops.transpose(0, 1, 3, 2, 4, 5)
        crops = crops.reshape(
            leading_crops_dim,
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size * channels,
        )
        leading_mask_dim = cropped_masks.shape[0]
        cropped_masks = cropped_masks.reshape(
            leading_mask_dim,
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
        )
        cropped_masks = cropped_masks.transpose(0, 1, 3, 2, 4)
        cropped_masks = cropped_masks.reshape(
            leading_mask_dim,
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size,
        )

        cropped_masks = cropped_masks.astype(np.float32).mean(axis=-1)
        cropped_masks = np.pad(cropped_masks, [[0, 1], [0, 0]], constant_values=-1)
        patch_orderings = np.reshape(patch_orderings, [-1])
        return crops, patch_orderings, cropped_masks

    def transpose_patch_orderings(self, crop_grid, patch_orderings):
        patch_ordering_left_right = np.reshape(
            patch_orderings, [crop_grid[0], crop_grid[1], self.tokens_per_image_height, self.tokens_per_image_width]
        )
        patch_ordering_left_right = np.transpose(patch_ordering_left_right, [0, 2, 1, 3])
        patch_ordering_left_right = np.reshape(patch_ordering_left_right, [-1])

        # The transpose will mess up which patches are masked, project the
        # new order into sparse structure of `patch_ordering` to fix this
        patch_orderings[patch_orderings >= 0] = patch_ordering_left_right[patch_ordering_left_right >= 0]
        return patch_orderings

    def _prepare_crop_grids(self, data):
        """
        Prepares crop_grids by stacking them into a batch dimension.
        """
        crop_grids = data["crop_grids"]  # List of arrays with shape (2,)
        data["crop_grids"] = np.stack(crop_grids, axis=0)  # Shape: (batch_size, 2)

    def _pad_patch_orderings(self, data):
        """
        Pads patch_orderings to have the same length across the batch.
        """
        patch_orderings = data["patch_orderings"]  # List of arrays with shape (length_i,)
        batch_size = len(patch_orderings)
        max_length = max(ordering.shape[0] for ordering in patch_orderings)

        # use a fill value that doesn't interfere with valid data (e.g., -2)
        fill_value = -2
        batched_patch_orderings = np.full(
            (batch_size, max_length), fill_value=fill_value, dtype=patch_orderings[0].dtype
        )

        patch_orderings_mask = np.zeros((batch_size, max_length), dtype=bool)

        for idx, ordering in enumerate(patch_orderings):
            length = ordering.shape[0]
            batched_patch_orderings[idx, :length] = ordering
            patch_orderings_mask[idx, :length] = True

        # Update the data dictionary
        data["patch_orderings"] = batched_patch_orderings  # Shape: (batch_size, max_length)

    def _pad_for_batching(
        self,
        data: Dict,
    ):
        """
        Pads crops obtained with the largest amount of crops in the batch. Will penalize queries with high
        number of crops. Pads as well the patch orderings and so on.
        """
        crops = data["pixel_values"]
        max_num_crops = max(image.shape[0] for image in crops)
        batch_size = len(crops)
        crop_shape = crops[0].shape[1:]

        batched_crops = np.zeros((batch_size, max_num_crops) + crop_shape, dtype=crops[0].dtype)
        crop_masks = np.zeros((batch_size, max_num_crops), dtype=np.bool_)
        for idx, image in enumerate(crops):
            num_crops = image.shape[0]
            batched_crops[idx, :num_crops, ...] = image
            crop_masks[idx, :num_crops] = True

        data["pixel_values"] = batched_crops

        # pad image_masks with -1
        image_masks = data["image_masks"]
        mask_shape = image_masks[0].shape[1:]
        batched_image_masks = np.full(
            (batch_size, max_num_crops) + mask_shape, fill_value=-1, dtype=image_masks[0].dtype
        )
        for idx, mask in enumerate(image_masks):
            num_crops = mask.shape[0]
            batched_image_masks[idx, :num_crops, ...] = mask

        data["image_masks"] = batched_image_masks
        self._pad_patch_orderings(data)

        self._prepare_crop_grids(data)
        return data

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_pad: Optional[bool] = None,
        do_split_into_crops: Optional[bool] = None,
        padding_value: Optional[float] = None,
        padding_mode: Optional[str] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = OPENAI_CLIP_MEAN,
        image_std: Optional[Union[float, List[float]]] = OPENAI_CLIP_STD,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images for the Molmo model.

        Args:
            images (ImageInput): Image or batch of images to preprocess.
            image_patch_token_id (int): Token ID for image patches.
            image_col_token_id (int): Token ID for image columns.
            image_start_token_id (int): Token ID for the start of an image.
            image_end_token_id (int): Token ID for the end of an image.

        Returns:
            BatchFeature: A dictionary containing processed image patches, tokens, indices, and masks.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_split_into_crops = do_split_into_crops if do_split_into_crops is not None else self.do_split_into_crops
        padding_value = padding_value if padding_value is not None else self.padding_value
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        images = make_batched_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

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

        all_images = []
        all_crop_grids = []
        all_cropped_masks = []
        all_patch_orderings = []
        for image in images:
            # 1. First, for a given image, figure out the best crop grid for the input image.
            # We need to keep track of a few values here.
            crop_grid = self.find_best_crop_grid_for_image_size(image)
            # 2. Then, resize and pad, figure out number of crops (large ones) and patches (small ones)
            if do_resize:
                # we resize both the global image to the wanted size, as well as the crops.
                global_image_size = get_resize_output_image_size(image, size)
                global_image = self.resize(
                    image=image, size=global_image_size, resample=resample, input_data_format=input_data_format
                )
                new_crop_size = {}
                new_crop_size["height"] = crop_grid[0] * self.crop_window_size + self.total_margin_pixels
                new_crop_size["width"] = crop_grid[1] * self.crop_window_size + self.total_margin_pixels
                crop_output_size = get_resize_output_image_size(
                    image,
                    size=new_crop_size,
                )

                image = self.resize(
                    image=image, size=crop_output_size, resample=resample, input_data_format=input_data_format
                )
            # TODO do_pad and do_split_into_crops should not be optional. Removing them will break the processing.
            if do_pad:
                # 2.1 after padding, we also get the image mask
                image, image_mask = self.pad(
                    image=image, size=new_crop_size, input_data_format=input_data_format, constant_values=0
                )
                # 2.2 (from original code) the image mask padding is increased by 1 dim
                global_image, _ = self.pad(
                    image=global_image, size=size, input_data_format=input_data_format, constant_values=0
                )
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                global_image = self.rescale(
                    image=global_image, scale=rescale_factor, input_data_format=input_data_format
                )
            if do_normalize:
                image = normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                global_image = normalize(
                    image=global_image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            # 3. Then split the padded and rescaled image into crops. Don't touch the global image.
            if do_split_into_crops:
                crops, patch_orderings, cropped_masks = self.split_image_into_crops(
                    image=image, image_mask=image_mask, crop_grid=crop_grid, input_data_format=input_data_format
                )
                # 4. Reorder patches left-to-right instead of crop-by-crop.
                patch_orderings = self.transpose_patch_orderings(crop_grid, patch_orderings)
            global_image = self.reshape_into_patches(global_image, input_data_format=input_data_format)
            # 5. Concatenate patches and the global image
            crops = np.concatenate([np.expand_dims(global_image, 0), crops], 0)

            # 6. Global image goes first, so the order of patches in previous crops gets increased
            # by an amount corresponding to the number of tokens per image
            patch_orderings = np.where(patch_orderings >= 0, patch_orderings + self.tokens_per_image, -1)
            patch_orderings = np.concatenate([np.arange(0, self.tokens_per_image), patch_orderings], 0)
            # 7. Add an extra dim for the image mask padding

            all_images.append(crops)
            all_crop_grids.append(crop_grid)
            all_cropped_masks.append(cropped_masks)
            all_patch_orderings.append(patch_orderings)
        data = {
            "pixel_values": all_images,
            "crop_grids": all_crop_grids,
            "patch_orderings": all_patch_orderings,
            "image_masks": all_cropped_masks,
        }
        if do_pad:
            data = self._pad_for_batching(data)
        return BatchFeature(data=data, tensor_type=return_tensors)


### PROCESSING CODE


class MolmoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "images_kwargs": {
            "max_crops": 12,
            "overlap_margins": (4, 4),
            "tokens_per_image_width": 12,
            "tokens_per_image_height": 12,
            "image_patch_size": 14,
            "image_padding_mask": True,
        },
        "text_kwargs": {
            "padding": False,
        },
    }


class MolmoProcessor(ProcessorMixin):
    r"""
    Constructs a Molmo processor which wraps a Molmo image processor and a Molmo tokenizer into a single processor.

    [`MolmoProcessor`] offers all the functionalities of [`MolmoImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~MolmoProcessor.__call__`] and [`~MolmoProcessor.decode`] for more information.

    Args:
        image_processor ([`MolmoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.im_patch_token = tokenizer.im_patch_token
        self.im_col_token = tokenizer.im_col_token
        self.bos_token = tokenizer.bos_token or tokenizer.eos_token

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[MolmoProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MolmoImageProcessor's [`~MolmoImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            MolmoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        # TODO should be vectorizable
        if image_inputs.get("pixel_values") is not None and image_inputs.get("crop_grids") is not None:
            for crop_grid, patch_ordering in zip(image_inputs.pop("crop_grids"), image_inputs.pop("patch_orderings")):
                overlap_margins = self.image_processor.overlap_margins
                crop_window_patches = self.image_processor.crop_window_patches

                full_height = crop_grid[0] * crop_window_patches + (overlap_margins[1] + overlap_margins[0])
                full_width = crop_grid[1] * crop_window_patches + (overlap_margins[1] + overlap_margins[0])
                tokens_per_row = np.full(
                    ((full_width + 1) // 2,),
                    self.im_patch_token,
                )
                tokens_per_row = np.concatenate([tokens_per_row, [self.im_col_token]], 0)

                crop_tokens = np.tile(tokens_per_row, [(full_height + 1) // 2])
                crop_tokens = [[self.boi_token], crop_tokens, [self.eoi_token]]

                # for the global image

                global_tokens_per_row = np.full(
                    (self.image_processor.tokens_per_image_width,),
                    self.im_patch_token,
                )
                global_tokens_per_row = np.concatenate([global_tokens_per_row, [self.im_col_token]], 0)
                extra_tokens = np.tile(global_tokens_per_row, [self.image_processor.tokens_per_image_height])
                all_image_tokens = [
                    [self.boi_token],
                    extra_tokens,
                    [self.eoi_token],
                ] + crop_tokens
                all_image_tokens = np.concatenate(all_image_tokens, 0)

                # then build the image token indices with the patch ordering baked in

                image_token_mask = np.nonzero(all_image_tokens == self.im_patch_token)[0].astype(np.int32)
                number_of_tokens = image_token_mask.shape[0]
                patch_ordering = np.reshape(patch_ordering, [-1])
                valid = patch_ordering >= 0
                number_of_valid_patches = valid.sum()

                sorted_patch_ixs = np.zeros([number_of_tokens], np.int32)
                sorted_patch_ixs[patch_ordering[valid]] = np.arange(number_of_valid_patches, dtype=np.int32)

                # Project the inverted mapping into same sparse structure
                sorted_patch_ixs_ex = np.full(np.shape(patch_ordering), -1)
                sorted_patch_ixs_ex[valid] = sorted_patch_ixs

                # Do the gather and then re-masked outputs that were masked in `sorted_patch_ixs`
                valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
                image_token_mask = image_token_mask[sorted_patch_ixs_ex * valid]
                image_token_mask = image_token_mask * valid - 100 * (1 - valid)
                image_token_mask = np.reshape(
                    image_token_mask,
                    [-1, self.image_processor.tokens_per_image_width * self.image_processor.tokens_per_image_height],
                )
                image_inputs.setdefault("image_token_indices", []).append(image_token_mask)

                # Replace the image token with the expanded image token sequence
                prompt_strings = []
                for sample in text:
                    sample = sample.replace(self.image_token, "".join(all_image_tokens))
                    prompt_strings.append(sample)
        text_inputs = self.tokenizer(
            [f"{self.bos_token}{prompt}" for prompt in prompt_strings], **output_kwargs["text_kwargs"]
        )
        # there is no bos token in Qwen tokenizer
        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=output_kwargs["common_kwargs"]["return_tensors"]
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = [
    "MolmoConfig",
    "MolmoImageProcessor",
    "MolmoProcessor",
    "MolmoVisionConfig",
    "MolmoVisionEmbeddings",
    "MolmoVisionModel",
    "MolmoTextAttention",
    "MolmoVisionAttention",
    "MolmoPoolingAttention",
    "MolmoAdapterModel",
    "MolmoTextModel",
    "MolmoPreTrainedModel",
    "MolmoForCausalLM",
    "MolmoForConditionalGeneration",
]
