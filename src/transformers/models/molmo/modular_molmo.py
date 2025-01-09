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

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    is_flash_attn_2_available,
    logging,
)
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from ..cohere.configuration_cohere import CohereConfig
from ..cohere.modeling_cohere import (
    CohereAttention,
    CohereModel,
    CoherePreTrainedModel,
)
from ..llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaForConditionalGeneration
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "MolmoConfig"


class MolmoVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoVisionModel`]. It is used to instantiate a
    `MolmoVisionModel` according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Molmo
    [allenai/Molmo-7B-D-0924-hf](https://huggingface.co/allenai/Molmo-7B-D-0924-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 23):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 576):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
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
        num_image_positions (`int`, *optional*, defaults to 577):
            The number of image tokens per crop.
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

    model_type = "molmo_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=23,
        num_attention_heads=16,
        image_size=576,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        num_image_positions=577,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_image_positions = num_image_positions


class MolmoPoolingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoAdapterModel`]. It is used to instantiate an
    `MolmoAdapterModel` according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Molmo-7B-D.

    e.g. [allenai/Molmo-7B-D-0924-hf](https://huggingface.co/allenai/Molmo-7B-D-0924-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the pooler attention layer.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer pooler.
        head_dim (`int`, *optional*, defaults to 64):
            The poolinng attention head dimension.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pooling_height (`int`, *optional*, defaults to 2):
            The height of image features requred for pooling operation.
        pooling_width (`int`, *optional*, defaults to 2):
            The width of image features requred for pooling operation.
        pad_embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of a padding tensor which is multiplied with the image mask.
        image_num_patches (`int`, *optional*, defaults to 24):
            Number of patches each image feature has after the vision tower.
        image_feature_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the image features after vision tower.
        text_intermediate_size (`int`, *optional*, defaults to 37888):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the text Transformer encoder.
        text_hidden_size (`int`, *optional*, defaults to 3584):
            Dimensionality of the text encoder layers.
        image_pooling_type (`str`, *optional*, defaults to `"attention_meanq"`):
            Type of pooling to apply on image features. Can be one of ["attention", "attention_meanq", "attention_2wide", "attention_v2", "stack"] or `None`
        image_padding_embed (`str`, *optional*, defaults to `"pad_and_partial_pad"`):
            Type of padding to apply of image masks. Can be one of ["pad_embed", "regress", "pad_and_partial_pad]
        projector_hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used by the multimodal projector.

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


class MolmoTextConfig(CohereConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoModel`]. It is used to instantiate a
    Molmo model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Molmo-7B-beta [Qwen/Molmo-7B-beta](https://huggingface.co/Qwen/Molmo-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        head_dim (`int`, *optional*, defaults to 128):
            The poolinng attention head dimension.
        vocab_size (`int`, *optional*, defaults to 152192):
            Vocabulary size of the Molmo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MolmoTextModel`]
        intermediate_size (`int`, *optional*, defaults to 37888):
            Dimension of the MLP representations.
        hidden_act (`str` or `function`, *optional*, defaults to `"swiglu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
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
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_qk_norm (`bool), *optional*, defaults to `False`):
            Whther to apply layer norm to keys and queries in attention module.
        use_postnorm (`bool), *optional*, defaults to `True`):
            Whther to apply pre or post layer normalization in each decoder layer.

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
        vocab_size=152192,
        intermediate_size=37888,
        hidden_act="swiglu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        sliding_window=4096,
        attention_dropout=0.0,
        attention_bias=False,
        use_qk_norm=False,
        use_postnorm=True,
        **kwargs,
    ):
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.use_postnorm = use_postnorm
        self.sliding_window = sliding_window
        super().__init__(**kwargs)
        del self.logit_scale


class MolmoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoForConditionalGeneration`]. It is used to instantiate an
    Momlmo model according to the specified arguments, defining the model architecture. Instantiating a configuration
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
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layers (`List[int]`, *optional*, defaults to `(-2, -9)`):
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
        initializer_range=0.02,
        vision_feature_select_strategy="default",
        vision_feature_layers=(-2, -9),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
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
class MolmoTextMLP(CLIPMLP):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = MolmoSwiGLU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=False)


class MolmoTextRotaryEmbedding(Qwen2RotaryEmbedding):
    pass  # cohere has special RoPE so we need to get qwen2


# cohere has special RoPE so we need to copy to not dispatch all dependencies of attn class
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MolmoTextLayerNorm(Qwen2RMSNorm):
    pass


class MolmoTextAttention(CohereAttention):
    def __init__(self, config: MolmoTextConfig, layer_idx: Optional[int] = None):
        self.hidden_size = config.hidden_size
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


class MolmoTextDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = MolmoTextLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = MolmoTextLayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class MolmoTextPrenormDecoderLayer(MolmoTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class MolmoPreTrainedModel(CoherePreTrainedModel):
    _no_split_modules = ["MolmoTextDecoderLayer", "MolmoTextPrenormDecoderLayer"]


class MolmoTextModel(CohereModel):
    def __init__(self, config):
        decoder_layer = MolmoTextDecoderLayer if self.config.use_postnorm else MolmoTextPrenormDecoderLayer
        super().__init__(config)
        self.layers = nn.ModuleList(
            [decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class MolmoForCausalLM(Qwen2ForCausalLM):
    _tp_plan = {"lm_head": "colwise_rep"}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,
        **kwargs,
    ):
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
        >>> from transformers import AutoTokenizer, MolmoForCausalLM

        >>> model = MolmoForCausalLM.from_pretrained("...")
        >>> tokenizer = AutoTokenizer.from_pretrained("...")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
            num_logits_to_keep,
            **kwargs,
        )


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


# Molmo image components inherited from CLIPVision
# We have different attention classes for the txt and the image components, they need to be propagated back correctly


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

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size, patches, height, width = pixel_values.shape
        if height != self.image_size:
            raise ValueError(f"Input image size ({height}) doesn't match model" f" ({self.image_size}).")
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        class_embeds = self.class_embedding.expand(batch_size, patches, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=2)
        embeddings = embeddings + self.position_embedding(self.position_ids).unsqueeze(1)
        return embeddings.flatten(0, 1)  # NOTE: DON'T FLATTEN MORE TO MATCH ORIG IMPL


class MolmoVisionEncoderLayer(CLIPEncoderLayer):
    pass


class MolmoVisionEncoder(CLIPEncoder):
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
        embed_dim = config.hidden_size
        self.encoder = MolmoVisionEncoder(config)  # necessary because of renaming issue in modular
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        del self.post_layernorm
        del self.pre_layrnorm  # old typo in CLIP

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

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

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


class MolmoVisionModel(CLIPVisionModel):
    _no_split_modules = ["MolmoVisionEncoderLayer"]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def pooling_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MolmoPoolingAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**0.5
        self.is_causal = True

        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim // 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        query_hidden_shape = (*input_shape, -1, self.head_dim)
        key_value_shape = key_value_hidden_states.shape[:-1]
        key_value_hidden_shape = (*key_value_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(query_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(key_value_hidden_states).view(key_value_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(key_value_hidden_states).view(key_value_hidden_shape).transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = pooling_eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class MolmoAdapterModel(MolmoPreTrainedModel):
    config_class = MolmoPoolingConfig
    main_input_name = "image_features"

    def __init__(self, config: MolmoPoolingConfig):
        super().__init__(config)

        if config.image_pooling_type == "attention_meanq":
            self.image_pooling_2d = MolmoPoolingAttention(config)
        elif config.image_pooling_type is not None:
            raise NotImplementedError(
                f"Unknown image pooling 2D method: {config.pooling_config.image_pooling_type}, Can be only `attention_meanq`"
            )

        if config.image_padding_embed == "pad_and_partial_pad":
            self.pad_embed = nn.Parameter(torch.zeros((2, config.pad_embed_dim)))
        elif config.image_padding_embed is not None:
            raise ValueError(
                f"Unknown image padding method {config.image_padding_embed}, can be only `pad_and_partial_pad`"
            )

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)
        self.multi_modal_projector = MolmoMultiModalProjector(config)

    def forward(self, image_features, image_masks) -> torch.FloatTensor:
        batch_size, patches = image_features.shape[:2]
        if self.config.image_padding_embed is not None:
            pad_embed = self.pad_embed[:, None, None, None, :]
            all_pad = image_masks == 0
            partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=image_features.dtype)
            all_pad = all_pad.to(dtype=image_features.dtype)
            image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
            image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)

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
        leading_dimension, image_batch_size, patch_height, patch_width, image_embed_dim = image_features.shape

        image_features = image_features.view(
            leading_dimension,
            image_batch_size,
            patch_height // self.config.pooling_height,
            self.config.pooling_height,
            patch_width // self.config.pooling_width,
            self.config.pooling_width,
            image_embed_dim,
        )
        image_features = image_features.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            -1, self.config.pooling_height * self.config.pooling_width, image_embed_dim
        )

        if self.config.image_pooling_type is not None:
            queries = image_features.mean(-2, keepdim=True)
            image_features = self.image_pooling_2d(queries, image_features)[0]

        # Round up in case we need to pad the image features for pooling
        patch_height = (num_patches + self.config.pooling_height - 1) // self.config.pooling_height
        patch_width = (num_patches + self.config.pooling_width - 1) // self.config.pooling_width

        image_features = image_features.reshape(batch_size, patches, patch_height * patch_width, -1)
        image_features = self.multi_modal_projector(image_features)
        return image_features


class MolmoForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = MolmoConfig

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

            all_pixel_values = pixel_values_flat[valid_crops_flat.to(pixel_values_flat.device)]
            all_image_masks = image_masks_flat[valid_crops_flat.to(image_masks_flat.device)]
            all_image_token_indices = image_token_indices_flat[valid_crops_flat.to(image_token_indices_flat.device)]

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
            valid_features = image_features_flat[valid_positions.to(image_features_flat.device)]
            valid_batch_indices = valid_batch_indices_expanded[
                valid_positions.to(valid_batch_indices_expanded.device)
            ].long()

            flat_indices = valid_batch_indices * seq_len + valid_indices.to(valid_batch_indices.device)
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

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


__all__ = [
    "MolmoConfig",
    "MolmoPoolingConfig",
    "MolmoTextConfig",
    "MolmoVisionConfig",
    "MolmoVisionEmbeddings",
    "MolmoVisionModel",
    "MolmoTextAttention",
    "MolmoPoolingAttention",
    "MolmoAdapterModel",
    "MolmoTextModel",
    "MolmoPreTrainedModel",
    "MolmoForCausalLM",
    "MolmoForConditionalGeneration",
]
