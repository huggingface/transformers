# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.generic import TransformersKwargs, check_model_inputs
from ..phi3.configuration_phi3 import Phi3Config
from ..phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3ForCausalLM,
    Phi3Model,
    Phi3PreTrainedModel,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
    SiglipPreTrainedModel,
    SiglipVisionEmbeddings,
    default_flax_embed_init,
    lecun_normal_,
)


logger = logging.get_logger(__name__)


class Phi4MultimodalVisionConfig(SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi4MultimodalVisionModel`]. It is used to instantiate a
    Phi4Multimodal vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of
    [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        crop_size (`int`, *optional*, defaults to 448):
            Crop size for the input images.
        image_token_id (`int`, *optional*, defaults to 200010):
            The image token id.
        feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer of the encoder from which to extract image features.

    Example:

    ```python
    >>> from transformers import Phi4MultimodalVisionConfig

    >>> # Initializing a Phi4MultimodalVisionConfig with microsoft/Phi-4-multimodal-instruct style configuration
    >>> configuration = Phi4MultimodalVisionConfig()
    ```"""

    model_type = "phi4_multimodal_vision"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        crop_size: int = 448,
        image_token_id: int = 200010,
        feature_layer: int = -2,
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
        self.crop_size = crop_size
        self.image_token_id = image_token_id
        self.feature_layer = feature_layer


class Phi4MultimodalAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi4MultimodalAudioModel`]. It is used to instantiate a
    Phi4Multimodal audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of
    [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_blocks (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        activation (`str`, *optional*, defaults to `"swish"`):
            The non-linear activation function in the MLPs.
        chunk_size (`int`, *optional*, defaults to -1):
            The chunk size to create the masks.
        left_chunk (`int`, *optional*, defaults to 18):
            The left chunk to create the masks.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio.
        ext_pw_out_channel (`int`, *optional*, defaults to 1024):
            Number of out channels in the point-wise conv modules.
        depthwise_seperable_out_channel (`int`, *optional*, defaults to 1024):
            Number of out channels in the depth-wise separable conv modules.
        depthwise_multiplier (`int`, *optional*, defaults to 1):
            Input size multiplier for the depth-wise separable conv modules.
        kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the depth-wise separable conv modules.
        conv_activation (`str`, *optional*, defaults to `"swish"`):
            The non-linear activation function in the conv modules.
        input_size (`int`, *optional*, defaults to 80):
            Input size for the audio model.
        conv_glu_type (`str`, *optional*, defaults to `"swish"`):
            The non-linear activation function in the point-wise conv modules.
        time_reduction (`int`, *optional*, defaults to 8):
            Time reduction (subsampling factor).
        bias_max_distance (`int`, *optional*, defaults to 1000):
            Max distance for the relative attention bias module.
        bias_symmetric (`bool`, *optional*, defaults to `False`):
            Whether the relative attention bias should be symmetric or not.
        nemo_activation (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the nemo conv modules.
        nemo_conv_channels (`int`, *optional*, defaults to 1024):
            Number of channels in the nemo conv modules.
        downsample_rate (`int`, *optional*, defaults to 1):
            Downsample rate for the audio feature extractor.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        audio_token_id (`int`, *optional*, defaults to 200011):
            The audio token id.
        feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer of the encoder from which to extract audio features.

    Example:

    ```python
    >>> from transformers import Phi4MultimodalAudioConfig

    >>> # Initializing a Phi4MultimodalAudioConfig with microsoft/Phi-4-multimodal-instruct style configuration
    >>> configuration = Phi4MultimodalAudioConfig()
    ```"""

    model_type = "phi4_multimodal_audio"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 1536,
        num_blocks: int = 24,
        num_attention_heads: int = 16,
        activation: str = "swish",
        chunk_size: int = -1,
        left_chunk: int = 18,
        dropout_rate: float = 0.0,
        ext_pw_out_channel: int = 1024,
        depthwise_seperable_out_channel: int = 1024,
        depthwise_multiplier: int = 1,
        kernel_size: int = 3,
        conv_activation: str = "swish",
        input_size: int = 80,
        conv_glu_type: str = "swish",
        time_reduction: int = 8,
        bias_max_distance: int = 1000,
        bias_symmetric: bool = False,
        nemo_activation: str = "relu",
        nemo_conv_channels: int = 1024,
        downsample_rate: int = 1,
        initializer_range: float = 0.02,
        audio_token_id: int = 200011,
        feature_layer: int = -2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.ext_pw_out_channel = ext_pw_out_channel
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.depthwise_multiplier = depthwise_multiplier
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.input_size = input_size
        self.conv_glu_type = conv_glu_type
        self.time_reduction = time_reduction
        self.bias_max_distance = bias_max_distance
        self.bias_symmetric = bias_symmetric
        self.nemo_activation = nemo_activation
        self.nemo_conv_channels = nemo_conv_channels
        self.downsample_rate = downsample_rate
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range
        self.feature_layer = feature_layer

        if time_reduction % 2 != 0:
            raise ValueError("`time_reduction` should be a multiple of 2!")
        length = input_size
        for _ in range(int(math.log(time_reduction, 2))):
            length = math.floor((length - 1) / 2 + 1)
        self.nemo_final_size = length


class Phi4MultimodalConfig(Phi3Config):
    r"""
    This is the configuration class to store the configuration of a [`Phi4MultimodalModel`]. It is used to instantiate a
    Phi4Multimodal model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 200064):
            Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Phi3Model`].
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            Dropout probability for mlp outputs.
        embd_pdrop (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value used for the RMSNorm.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            The scaling strategy for the RoPE embeddings. If `None`, no scaling is applied. If a dictionary, it must
            contain the following keys: `type`, `short_factor` and `long_factor`. The `type` must be `longrope` and
            the `short_factor` and `long_factor` must be lists of numbers with the same length as the hidden size
            divided by the number of attention heads divided by 2.
        partial_rotary_factor (`float`, *optional*, defaults to `1.0`):
            Percentage of the query and keys which will have rotary embedding. Must be between 0.0 and 1.0.
        bos_token_id (`int`, *optional*, defaults to 199999):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int` or `list[int]`, *optional*, defaults to `[199999, 200020]`):
            The id of the "end-of-sequence" token.
        pad_token_id (`int`, *optional*, defaults to 199999):
            The id of the padding token.
        original_max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model was trained with. This is used to determine the size of the
            original RoPE embeddings when using long scaling.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If `None`, no sliding window is applied.
        vision_config (`Phi4MultimodalVisionConfig` or `dict`, *optional*):
            The vision config for the underlying image embedding model. If not provided, will default to the configuration
            used to instantiate a model similar in architecture as
            [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).
        audio_config (`Phi4MultimodalAudioConfig` or `dict`, *optional*):
            The audio config for the underlying audio embedding model. If not provided, will default to the configuration
            used to instantiate a model similar in architecture as
            [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).

    Example:

    ```python
    >>> from transformers import Phi4MultimodalModel, Phi4MultimodalConfig

    >>> # Initializing a Phi4Multimodal style configuration
    >>> configuration = Phi4MultimodalConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

    >>> # Initializing a model from the configuration
    >>> model = Phi4MultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    sub_configs = {"audio_config": Phi4MultimodalAudioConfig, "vision_config": Phi4MultimodalVisionConfig}

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1,
        bos_token_id=199999,
        eos_token_id=[199999, 200020],
        pad_token_id=199999,
        original_max_position_embeddings=4096,
        sliding_window=None,
        vision_config=None,
        audio_config=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            original_max_position_embeddings=original_max_position_embeddings,
            sliding_window=sliding_window,
            **kwargs,
        )

        if isinstance(vision_config, dict):
            vision_config = Phi4MultimodalVisionConfig(**vision_config)
        elif vision_config is None:
            Phi4MultimodalVisionConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Phi4MultimodalAudioConfig(**audio_config)
        elif vision_config is None:
            audio_config = Phi4MultimodalAudioConfig()
        self.audio_config = audio_config


class Phi4MultimodalVisionMLP(SiglipMLP):
    pass


def simple_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Phi4MultimodalVisionAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Phi4MultimodalVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.self_attn = Phi4MultimodalVisionAttention(config)
        self.mlp = Phi4MultimodalVisionMLP(config)


class Phi4MultimodalVisionEncoder(SiglipEncoder):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Phi4MultimodalVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class Phi4MultimodalVisionPreTrainedModel(SiglipPreTrainedModel):
    config: Phi4MultimodalVisionConfig
    base_model_prefix = "phi4_vision"
    supports_gradient_checkpointing = True

    _no_split_modules = ["Phi4MultimodalVisionEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Phi4MultimodalVisionEmbeddings):
            width = (
                self.config.hidden_size
                if isinstance(self.config, Phi4MultimodalVisionConfig)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Phi4MultimodalVisionAttention):
            nn.init.normal_(module.q_proj.weight)
            nn.init.normal_(module.k_proj.weight)
            nn.init.normal_(module.v_proj.weight)
            nn.init.normal_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, Phi4MultimodalVisionMLP):
            nn.init.normal_(module.fc1.weight)
            nn.init.normal_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, Phi4MultimodalVisionMultiheadAttentionPoolingHead):
            nn.init.normal_(module.probe.data)
            nn.init.normal_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Phi4MultimodalVisionEmbeddings(SiglipVisionEmbeddings, nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // self.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(self.num_patches_per_side**2, config.hidden_size)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full((batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Phi4MultimodalVisionMultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.mlp = Phi4MultimodalVisionMLP(config)

    def forward(self, hidden_state, attention_mask):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(
            query=probe, key=hidden_state, value=hidden_state, key_padding_mask=~attention_mask
        )[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Phi4MultimodalVisionModel(Phi4MultimodalVisionPreTrainedModel):
    config: Phi4MultimodalVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Phi4MultimodalVisionEmbeddings(config)
        self.encoder = Phi4MultimodalVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = Phi4MultimodalVisionMultiheadAttentionPoolingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = (
                _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
                if not self.config._attn_implementation == "flash_attention_2"
                else patch_attention_mask
            )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(
            hidden_state=last_hidden_state,
            attention_mask=patch_attention_mask,
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Phi4MultimodalImageEmbedding(nn.Module):
    """Image embedding."""

    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.vision_config.feature_layer
        self.crop_size = config.vision_config.crop_size
        self.image_dim_out = config.vision_config.hidden_size

        n_patches = config.vision_config.image_size // config.vision_config.patch_size
        if n_patches % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            n_patches += 1
        self.num_img_tokens = (n_patches // 2) ** 2

        self.drop = nn.Dropout(config.embd_pdrop)
        self.img_processor = Phi4MultimodalVisionModel._from_config(config.vision_config)
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.img_projection_up = nn.Linear(self.image_dim_out, config.hidden_size)
        self.img_projection_down = nn.Linear(config.hidden_size, config.hidden_size)
        self.global_img_feature_extensor = nn.Parameter(torch.zeros([1, 1, self.image_dim_out]))
        self.sub_img_feature_extensor = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out]))

    def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None) -> torch.FloatTensor:
        img_processor_output = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask, output_hidden_states=True
        )
        img_feature = img_processor_output.hidden_states[self.layer_idx]

        patch_feature = img_feature
        # reshape to 2D tensor
        width = int(math.sqrt(patch_feature.size(1)))
        patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
        # convert to NCHW
        patch_feature = patch_feature.permute(0, 3, 1, 2)
        if getattr(self, "img_processor_padding", None) is not None:
            patch_feature = self.img_processor_padding(patch_feature)
        patch_feature = self.image_token_compression(patch_feature)
        # convert to NHWC
        patch_feature = patch_feature.permute(0, 2, 3, 1)
        patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
        return patch_feature

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_pixel_values: torch.FloatTensor,
        image_sizes: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        image_pixel_values = image_pixel_values.to(self.img_processor.embeddings.patch_embedding.weight.dtype)

        target_device = self.img_projection_up.bias.device
        target_dtype = self.img_projection_up.bias.dtype

        batch_size = image_pixel_values.shape[0]

        img_features = self.get_img_features(
            image_pixel_values.flatten(0, 1),
            attention_mask=image_attention_mask.flatten(0, 1).to(dtype=bool, device=target_device),
        )
        base_feat_size = int(np.sqrt(img_features.shape[1]))
        img_features = img_features.view(batch_size, -1, base_feat_size**2, self.image_dim_out)
        image_sizes = image_sizes.view(-1, 2)

        output_imgs = []
        for idx in range(batch_size):
            height, width = image_sizes[idx]
            height_ratio = height // self.crop_size
            width_ratio = width // self.crop_size
            area_ratio = height_ratio * width_ratio

            global_img = img_features[idx, :1]
            global_img = global_img.reshape(1, base_feat_size, base_feat_size, self.image_dim_out).contiguous()
            temporary_extensor = self.sub_img_feature_extensor.repeat(1, base_feat_size, 1, 1)
            global_img = torch.cat([global_img, temporary_extensor], dim=2).reshape(1, -1, self.image_dim_out)

            sub_img = img_features[idx, 1:]
            sub_img = sub_img[:area_ratio]
            sub_img = (
                sub_img.reshape(height_ratio, width_ratio, base_feat_size, base_feat_size, self.image_dim_out)
                .transpose(1, 2)
                .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size, self.image_dim_out)
                .contiguous()
            )

            if image_attention_mask is not None:
                reshaped_image_attention_mask = (
                    image_attention_mask[idx, 1 : area_ratio + 1, 0::2, 0::2]
                    .reshape(height_ratio, width_ratio, base_feat_size, base_feat_size)
                    .transpose(1, 2)
                    .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size)
                )
                useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temporary_extensor = self.sub_img_feature_extensor.repeat(1, useful_height, 1, 1)
            else:
                temporary_extensor = self.sub_img_feature_extensor.repeat(1, height_ratio * base_feat_size, 1, 1)

            sub_img = torch.cat([sub_img, temporary_extensor], dim=2).reshape(1, -1, self.image_dim_out)

            # Merge global and sub
            output_imgs.append(torch.cat([sub_img, self.global_img_feature_extensor, global_img], dim=1))

        img_set_tensor = []
        for output_img in output_imgs:
            output_img = output_img.to(device=target_device, dtype=target_dtype)
            img_feature_proj = self.img_projection_up(output_img)
            img_feature_proj = nn.functional.gelu(img_feature_proj)
            img_feature_proj = self.img_projection_down(img_feature_proj)
            img_set_tensor.append(img_feature_proj)

        merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
        merged_img_set_tensor = merged_img_set_tensor.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        with torch.no_grad():
            positions_tuple = torch.nonzero(input_ids == self.config.vision_config.image_token_id, as_tuple=True)

        # Temporarily disable autocast to avoid issue on bf16 tensors
        # Ref: https://github.com/pytorch/pytorch/issues/132715
        with torch.autocast(device_type=inputs_embeds.device.type, enabled=False):
            image_embeds = inputs_embeds.index_put(
                indices=positions_tuple, values=merged_img_set_tensor, accumulate=False
            )

        image_embeds = self.drop(image_embeds)

        return image_embeds


########################################################## AUDIO #############################################


class Phi4MultimodalAudioMLP(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.activation]
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        up_states = self.gate_up_proj(hidden_states)
        up_states, gate = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_fn(gate)
        up_states = self.dropout(up_states)
        hidden_states = self.down_proj(up_states)
        out = self.dropout(hidden_states)

        return out


class Phi4MultimodalAudioAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.dropout_rate
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Phi4MultimodalAudioDepthWiseSeperableConv1d(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig, padding: int = 0):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size * config.depthwise_multiplier,
            config.kernel_size,
            1,
            padding=padding,
            groups=config.hidden_size,
        )
        self.pw_conv = nn.Conv1d(
            config.hidden_size * config.depthwise_multiplier, config.depthwise_seperable_out_channel, 1, 1, 0
        )

    def forward(self, hidden_states):
        return self.pw_conv(self.dw_conv(hidden_states))


class Phi4MultimodalAudioGluPointWiseConv(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.ext_pw_out_channel

        self.ext_pw_conv_1d = nn.Conv1d(config.hidden_size, config.ext_pw_out_channel * 2, kernel_size=1, stride=1)
        self.glu_act = ACT2FN[config.conv_glu_type]
        self.b1 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))
        self.b2 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))

    def forward(self, hidden_states):
        # we assume the input always has the #channel (#dim) in the last dimension of the
        # tensor, so need to switch the dimension first for 1D-Conv case
        hidden_states = hidden_states.permute([0, 2, 1])
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        out = hidden_states[:, 0 : self.output_dim, :] + self.b1
        out = out * self.glu_act(hidden_states[:, self.output_dim : self.output_dim * 2, :] + self.b2)
        return out.permute([0, 2, 1])


class Phi4MultimodalAudioConvModule(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.kernel_size = config.kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.glu = Phi4MultimodalAudioGluPointWiseConv(config)
        self.dw_sep_conv_1d = Phi4MultimodalAudioDepthWiseSeperableConv1d(config, padding=config.kernel_size - 1)
        self.act = ACT2FN[config.conv_activation]
        self.ext_pw_conv_1d = nn.Conv1d(config.hidden_size, config.ext_pw_out_channel, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.glu(self.layer_norm(hidden_states))
        hidden_states = self.dw_sep_conv_1d(hidden_states.permute([0, 2, 1]))

        if self.kernel_size > 1:
            hidden_states = hidden_states[:, :, : -(self.kernel_size - 1)]

        hidden_states = self.act(hidden_states)
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        out = self.dropout(hidden_states.permute([0, 2, 1]))
        return out


class Phi4MultimodalAudioConformerEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()

        self.feed_forward_in = Phi4MultimodalAudioMLP(config)
        self.self_attn = Phi4MultimodalAudioAttention(config)
        self.conv = Phi4MultimodalAudioConvModule(config)
        self.feed_forward_out = Phi4MultimodalAudioMLP(config)
        self.layer_norm_att = nn.LayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        residual = hidden_states + 0.5 * self.feed_forward_in(hidden_states)
        hidden_states = self.layer_norm_att(residual)

        hidden_states = residual + self.self_attn(hidden_states, attention_mask)
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.feed_forward_out(hidden_states)

        out = self.layer_norm(hidden_states)

        return out


class Phi4MultimodalAudioNemoConvSubsampling(torch.nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.subsampling_factor = config.time_reduction
        self.sampling_num = int(math.log(self.subsampling_factor, 2))
        self.act_fn = ACT2FN[config.nemo_activation]
        conv_channels = config.nemo_conv_channels

        layers = [
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            self.act_fn,
        ]
        for _ in range(self.sampling_num - 1):
            layers.extend(
                [
                    nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1, groups=conv_channels),
                    nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1, padding=0, groups=1),
                    self.act_fn,
                ]
            )

        # Aggregate the layers
        self.conv = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(conv_channels * config.nemo_final_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]):
        # Unsqueeze Channel Axis
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = self.conv(hidden_states)

        # Flatten Channel and Frequency Axes
        b, _, t, _ = hidden_states.size()
        hidden_states = self.out(hidden_states.transpose(1, 2).reshape(b, t, -1))

        if mask is None:
            return hidden_states, None

        max_audio_length = hidden_states.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        arange_ = torch.arange(0, max_audio_length, device=hidden_states.device)
        pad_mask = arange_.expand(padding_length.size(0), -1) < padding_length.unsqueeze(1)
        return hidden_states, pad_mask.unsqueeze(1)


class Phi4MultimodalAudioRelativeAttentionBias(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()

        self.max_distance = config.bias_max_distance
        self.symmetric = config.bias_symmetric
        self.num_buckets = self.max_distance
        if not config.bias_symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, config.num_attention_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        max_pos = x.size(1)
        context_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[:, None]
        memory_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX export
        relative_position = relative_position.masked_fill(relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        # mapping from relative position to index in the bias parameter
        bias_idx = relative_position
        bias_idx = bias_idx.abs() if self.symmetric else bias_idx + self.num_buckets // 2

        att_bias = self.bias_values(bias_idx)
        att_bias = att_bias.permute(2, 0, 1).unsqueeze(0)

        return att_bias


class Phi4MultimodalAudioMeanVarianceNormLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.register_buffer("global_mean", torch.zeros(config.input_size))
        self.register_buffer("global_invstd", torch.ones(config.input_size))

    def forward(self, x):
        return (x - self.global_mean) * self.global_invstd


@auto_docstring
class Phi4MultimodalAudioPreTrainedModel(PreTrainedModel):
    config: Phi4MultimodalAudioConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi4MultimodalAudioConformerEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Phi4MultimodalAudioGluPointWiseConv):
            module.b1.data.zero_()
            module.b2.data.zero_()


class Phi4MultimodalAudioModel(Phi4MultimodalAudioPreTrainedModel):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__(config)
        self.config = config

        self.encoder_embedding = Phi4MultimodalAudioMeanVarianceNormLayer(config)
        self.embed = Phi4MultimodalAudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4MultimodalAudioRelativeAttentionBias(config)
        self.encoders = nn.ModuleList(
            [Phi4MultimodalAudioConformerEncoderLayer(config) for _ in range(config.num_blocks)]
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size)
        # avoid randomness when run evaluation or decoding
        if self.training and np.random.rand() > 0.5:
            # Either first or last chunk is not complete.
            # If only the last one is not complete, EOS is not effective
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)

        enc_streaming_mask = (
            adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def forward_embeddings(self, hidden_states, masks):
        """Forwarding the inputs through the top embedding layers"""
        seq_len = math.ceil(hidden_states.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"The sequence length after time reduction is invalid: {seq_len}. Your input feature is too short."
            )

        batch_size = hidden_states.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.config.chunk_size, self.config.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(hidden_states.device)

        hidden_states, masks = self.embed(hidden_states, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        return hidden_states, hs_mask, masks

    def calculate_hs_mask(self, hidden_states, device, mask):
        max_audio_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        enc_streaming_mask = self._streaming_mask(
            max_audio_length, batch_size, self.config.chunk_size, self.config.left_chunk
        )
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]):
        hidden_states = self.encoder_embedding(hidden_states)
        hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)

        unfolded = False
        bs, seq_len, _ = hidden_states.shape
        max_seq_len = 500  # maximum position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                hidden_states_pad = F.pad(hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0)
                hidden_states = hidden_states_pad.to(hidden_states.device)

            hidden_states = unfold_tensor(hidden_states, max_seq_len)
            masks_unfold = None
            if mask is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = mask.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            hs_mask = self.calculate_hs_mask(
                hidden_states, hidden_states.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        relative_attention_bias = self.relative_attention_bias_layer(hidden_states)
        attention_mask = hs_mask.unsqueeze(1) + relative_attention_bias

        for layer in self.encoders:
            hidden_states = layer(hidden_states, attention_mask)

        if unfolded:
            embed_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(bs, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states


def unfold_tensor(tensor, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        tensor: N, T, D
    """
    _, _, D = tensor.shape
    tensor = tensor.transpose(-1, -2)
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    tensor = F.unfold(tensor[..., None, :], kernel_size=(1, max_seq_len), stride=(1, max_seq_len))

    new_bsz, _, slen = tensor.shape
    tensor = tensor.view(new_bsz, -1, max_seq_len, slen)
    tensor = tensor.permute(0, 3, 2, 1)
    tensor = tensor.view(-1, max_seq_len, D).contiguous()
    return tensor


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class Phi4MultimodalAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.audio_config.feature_layer

        self.drop = nn.Dropout(config.embd_pdrop)
        self.encoder = Phi4MultimodalAudioModel._from_config(config.audio_config)
        self.up_proj_for_speech = nn.Linear(
            config.audio_config.hidden_size * config.audio_config.downsample_rate, config.hidden_size
        )
        self.down_proj_for_speech = nn.Linear(config.hidden_size, config.hidden_size)
        self.up_proj_for_vision_speech = nn.Linear(
            config.audio_config.hidden_size * config.audio_config.downsample_rate, config.hidden_size
        )
        self.down_proj_for_vision_speech = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        audio_input_features: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
    ) -> torch.FloatTensor:
        with torch.no_grad():
            positions_tuple = torch.nonzero(input_ids == self.config.audio_config.audio_token_id, as_tuple=True)

        up_proj = self.up_proj_for_speech if audio_projection_mode == "speech" else self.up_proj_for_vision_speech
        down_proj = (
            self.down_proj_for_speech if audio_projection_mode == "speech" else self.down_proj_for_vision_speech
        )

        target_device = up_proj.bias.device
        target_dtype = up_proj.bias.dtype

        audio_input_features = audio_input_features.to(device=target_device, dtype=target_dtype)

        audio_encoder_hidden_states = self.encoder(audio_input_features, audio_attention_mask)
        audio_encoder_hidden_states = up_proj(audio_encoder_hidden_states)
        audio_encoder_hidden_states = nn.functional.gelu(audio_encoder_hidden_states)
        audio_embeds = down_proj(audio_encoder_hidden_states)

        merged_audio_embeds = torch.cat(
            [audio_embeds[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], dim=0
        )
        merged_audio_embeds = merged_audio_embeds.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Temporarily disable autocast to avoid issue on bf16 tensors
        # Ref: https://github.com/pytorch/pytorch/issues/132715
        with torch.autocast(device_type=inputs_embeds.device.type, enabled=False):
            audio_embeds = inputs_embeds.index_put(
                indices=positions_tuple, values=merged_audio_embeds, accumulate=False
            )

        audio_embeds = self.drop(audio_embeds)

        return audio_embeds


#################################################### TEXT ####################################################


class Phi4MultimodalRMSNorm(Phi3RMSNorm):
    pass


class Phi4MultimodalDecoderLayer(Phi3DecoderLayer):
    pass


class Phi4MultimodalFeatureEmbedding(nn.Module):
    """Image-audio embedding."""

    def __init__(self, config: Phi4MultimodalConfig) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = config.vision_config.image_token_id
        self.audio_token_id = config.audio_config.audio_token_id
        self.image_embed = Phi4MultimodalImageEmbedding(config)
        self.audio_embed = Phi4MultimodalAudioEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_pixel_values: Optional[torch.FloatTensor] = None,
        audio_input_features: Optional[torch.FloatTensor] = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
    ) -> torch.FloatTensor:
        with torch.no_grad():
            image_position_mask = (input_ids == self.config.vision_config.image_token_id).unsqueeze(-1)
            non_image_position_mask = ~image_position_mask

        image_embeds = None
        audio_embeds = None
        if image_pixel_values is not None and (input_ids == self.image_token_id).any():
            image_embeds = self.image_embed(
                input_ids,
                inputs_embeds,
                image_pixel_values=image_pixel_values,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
            )
        if audio_input_features is not None and (input_ids == self.audio_token_id).any():
            audio_projection_mode = "vision" if image_pixel_values is not None else "speech"
            audio_embeds = self.audio_embed(
                input_ids,
                inputs_embeds,
                audio_input_features=audio_input_features,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
            )

        # merge image and audio
        if image_embeds is not None and audio_embeds is not None:
            inputs_embeds = image_embeds * image_position_mask + audio_embeds * non_image_position_mask
        elif image_embeds is not None:
            inputs_embeds = image_embeds
        elif audio_embeds is not None:
            inputs_embeds = audio_embeds

        return inputs_embeds


class Phi4MultimodalRotaryEmbedding(Phi3RotaryEmbedding):
    pass


class Phi4MultimodalPreTrainedModel(Phi3PreTrainedModel):
    def _init_weights(self, module):
        Phi3PreTrainedModel._init_weights(module)
        if isinstance(module, Phi4MultimodalImageEmbedding):
            module.global_img_feature_extensor.data.zero_()
            module.sub_img_feature_extensor.data.zero_()


class Phi4MultimodalModel(Phi3Model, nn.Module):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)

        self.embed_tokens_extend = Phi4MultimodalFeatureEmbedding(config)

        self.layers = nn.ModuleList(
            [Phi4MultimodalDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi4MultimodalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        audio_input_features: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        r"""
        image_pixel_values (`torch.FloatTensor`, *optional*):
            If the input contains images, these correspond to the pixel values after transformations (as returned by
            the Processor)
        image_sizes (`torch.LongTensor`, *optional*):
            If the input contains images, these correspond to size of each image.
        image_attention_mask (`torch.LongTensor`, *optional*):
            Attention mask for the images.
        audio_input_features (`torch.FloatTensor`, *optional*):
            If the input contains audio samples, these correspond to the values after transformation (as returned by
            the Processor).
        audio_embed_sizes (`torch.Tensor`, *optional*):
            Size of the audio inputs.
        audio_attention_mask (`torch.Tensor, *optional*):
            Attention mask for the audio inputs.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.embed_tokens_extend(
                input_ids,
                inputs_embeds,
                image_pixel_values=image_pixel_values,
                audio_input_features=audio_input_features,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Phi4MultimodalForCausalLM(Phi3ForCausalLM, nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi4MultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        audio_input_features: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        image_pixel_values (`torch.FloatTensor`, *optional*):
            If the input contains images, these correspond to the pixel values after transformations (as returned by
            the Processor)
        image_sizes (`torch.LongTensor`, *optional*):
            If the input contains images, these correspond to size of each image.
        image_attention_mask (`torch.LongTensor`, *optional*):
            Attention mask for the images.
        audio_input_features (`torch.FloatTensor`, *optional*):
            If the input contains audio samples, these correspond to the values after transformation (as returned by
            the Processor).
        audio_embed_sizes (`torch.Tensor`, *optional*):
            Size of the audio inputs.
        audio_attention_mask (`torch.Tensor, *optional*):
            Attention mask for the audio inputs.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:
        ```python
        >>> from transformers import AutoTokenizer, Phi4MultimodalForCausalLM
        >>> model = Phi4MultimodalForCausalLM.from_pretrained("TBA")
        >>> tokenizer = AutoTokenizer.from_pretrained("TBA")
        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_pixel_values=image_pixel_values,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            audio_input_features=audio_input_features,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_pixel_values=None,
        image_sizes=None,
        image_attention_mask=None,
        audio_input_features=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=0,
        **kwargs,
    ):
        # Overwritten -- this model may need to switch between short and long rope, invalidating the cache in the
        # process

        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_pixel_values=image_pixel_values,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            audio_input_features=audio_input_features,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return model_inputs


__all__ = [
    "Phi4MultimodalAudioPreTrainedModel",
    "Phi4MultimodalAudioModel",
    "Phi4MultimodalVisionPreTrainedModel",
    "Phi4MultimodalVisionModel",
    "Phi4MultimodalPreTrainedModel",
    "Phi4MultimodalModel",
    "Phi4MultimodalForCausalLM",
    "Phi4MultimodalVisionConfig",
    "Phi4MultimodalAudioConfig",
    "Phi4MultimodalConfig",
]
