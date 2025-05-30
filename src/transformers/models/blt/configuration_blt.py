# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
"""Mllama model configuration"""

from typing import Dict, List, Optional

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class MllamaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaVisionModel`]. It is used to instantiate an
    Mllama vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-11B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_global_layers (`int`, *optional*, defaults to 8):
            Number of global layers in the Transformer encoder.
            Vision model has a second transformer encoder, called global.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        vision_output_dim (`int`, *optional*, defaults to 7680):
            Dimensionality of the vision model output. Includes output of transformer
            encoder with intermediate layers and global transformer encoder.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image *tile*.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        max_num_tiles (`int`, *optional*, defaults to 4):
            Maximum number of tiles for image splitting.
        intermediate_layers_indices (`List[int]`, *optional*, defaults to [3, 7, 15, 23, 30]):
            Indices of intermediate layers of transformer encoder from which to extract and output features.
            These output features are concatenated with final hidden state of transformer encoder.
        supported_aspect_ratios (`List[List[int]]`, *optional*):
            List of supported aspect ratios for image splitting. If not specified, the default supported aspect ratios
            are [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]] for `max_num_tiles=4`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import MllamaVisionConfig, MllamaVisionModel

    >>> # Initializing a Llama config
    >>> config = MllamaVisionConfig()

    >>> # Initializing a vision model from the mllama-11b style configuration
    >>> model = MllamaVisionModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1280,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 32,
        num_global_layers: int = 8,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5120,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        max_num_tiles: int = 4,
        intermediate_layers_indices: Optional[List[int]] = None,
        supported_aspect_ratios: Optional[List[List[int]]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if supported_aspect_ratios is None:
            if max_num_tiles != 4:
                raise ValueError("max_num_tiles must be 4 for default supported aspect ratios")
            supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if intermediate_layers_indices is None:
            intermediate_layers_indices = [3, 7, 15, 23, 30]

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.vision_output_dim = vision_output_dim
        self.patch_size = patch_size
        self.intermediate_layers_indices = intermediate_layers_indices
        self.num_global_layers = num_global_layers
        self.max_num_tiles = max_num_tiles
        self.norm_eps = norm_eps
        self.attention_heads = num_attention_heads
        self.supported_aspect_ratios = supported_aspect_ratios
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def max_aspect_ratio_id(self) -> int:
        return len(self.supported_aspect_ratios)


class MllamaTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaTextModel`]. It is used to instantiate an
    Mllama text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-11B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Mllama text model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MllamaTextModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If not
            specified, will default to `num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        rope_theta (`float`, *optional*, defaults to `500000.0`):
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
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        cross_attention_layers (`List[int]`, *optional*):
            Indices of the cross attention layers. If not specified, will default to [3, 8, 13, 18, 23, 28, 33, 38].
        dropout (`float`, *optional*, defaults to 0):
            The dropout probability for self- and cross-attention layers.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 128001):
            The id of the end of sentence token.
        pad_token_id (`int`, *optional*, defaults to 128004):
            The id of the padding token.

    Example:

    ```python
    >>> from transformers import MllamaTextModel, MllamaTextConfig

    >>> # Initializing a Mllama text config
    >>> config = MllamaTextConfig()

    >>> # Initializing a model from the Mllama text configuration
    >>> model = MllamaTextModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14_336,
        rope_theta: float = 500_000,
        rope_scaling: Optional[Dict] = None,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        cross_attention_layers: Optional[List[int]] = None,
        dropout: float = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: Optional[int] = 128004,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

# coding=utf-8
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""BLT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation




class BLTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BLTModel`]. It is used to instantiate an BLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BLT-7B.
    e.g. [meta-blt/BLT-2-7b-hf](https://huggingface.co/meta-blt/BLT-2-7b-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the BLT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BLTModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. BLT 1 supports up to 2048 tokens,
            BLT 2 up to 4096, CodeBLT up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads
        multiple_of (`int`, *optional*, defaults to 256):
            The hidden size will be a multiple of this value. This is used to ensure efficient computation.
        num_patches (`int`, *optional*):
            Number of patches to use in the local encoder.
        cross_attn_encoder (`bool`, *optional*):
            Whether to use cross attention in the encoder.
        cross_attn_all_layers_encoder (`bool`, *optional*):
            Whether to use cross attention in all encoder layers.
        cross_attn_nheads (`int`, *optional*):
            Number of attention heads for cross attention.
        cross_attn_k (`int`, *optional*):
            Number of key-value heads for cross attention.
        dropout (`float`, *optional*):
            The dropout probability for all fully connected layers.
        attn_impl (`str`, *optional*):
            The attention implementation to use.
        attn_bias_type (`str`, *optional*):
            The type of attention bias to use.
        patch_size (`int`, *optional*):
            Size of each patch.
        dim_token_emb (`int`, *optional*):
            Dimension of token embeddings.
        dim_patch_emb (`int`, *optional*):
            Dimension of patch embeddings.
        max_seqlen (`int`, *optional*):
            Maximum sequence length.
        init_std_factor (`float`, *optional*):
            Factor to scale the initialization standard deviation.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function to use in the feed forward layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the activation function.
        decoder_ffn_dim (`int`, *optional*):
            Dimension of the feed-forward network in the decoder.
        sliding_window (`int`, *optional*):
            Size of the sliding window for local attention.
        cross_attn_init_by_pooling (`bool`, *optional*, defaults to `False`):
            Whether to initialize cross attention by pooling.
        cross_attn_decoder (`bool`, *optional*, defaults to `False`):
            Whether to use cross attention in the decoder.
        cross_attn_all_layers_decoder (`bool`, *optional*, defaults to `False`):
            Whether to use cross attention in all decoder layers.
        use_local_encoder_transformer (`bool`, *optional*, defaults to `False`):
            Whether to use transformer in the local encoder.
        downsampling_by_pooling (`str`, *optional*):
            Method to use for patch downsampling.
    """

    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `BLTModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        multiple_of=256,
        # BLT specific attributes
        num_patches=4,
        cross_attn_encoder=False,
        cross_attn_all_layers_encoder=False,
        cross_attn_nheads=None,
        cross_attn_k=None,
        dropout=0.0,
        attn_impl="eager",
        attn_bias_type="causal",
        patch_size=16,
        dim_token_emb=None,
        dim_patch_emb=None,
        max_seqlen=2048,
        init_std_factor=1.0,
        activation_function="silu",
        activation_dropout=0.0,
        decoder_ffn_dim=None,
        sliding_window=None,
        cross_attn_init_by_pooling=False,
        cross_attn_decoder=False,
        cross_attn_all_layers_decoder=False,
        use_local_encoder_transformer=False,
        downsampling_by_pooling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.multiple_of = multiple_of

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        
        # BLT specific attributes
        self.num_patches = num_patches
        self.cross_attn_encoder = cross_attn_encoder
        self.cross_attn_all_layers_encoder = cross_attn_all_layers_encoder
        self.cross_attn_nheads = cross_attn_nheads
        self.cross_attn_k = cross_attn_k
        self.dropout = dropout
        self.attn_impl = attn_impl
        self.attn_bias_type = attn_bias_type
        self.patch_size = patch_size
        self.dim_token_emb = dim_token_emb if dim_token_emb is not None else hidden_size
        self.dim_patch_emb = dim_patch_emb if dim_patch_emb is not None else hidden_size
        self.max_seqlen = max_seqlen
        self.init_std_factor = init_std_factor
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.decoder_ffn_dim = decoder_ffn_dim if decoder_ffn_dim is not None else intermediate_size
        self.sliding_window = sliding_window
        self.cross_attn_init_by_pooling = cross_attn_init_by_pooling
        self.cross_attn_decoder = cross_attn_decoder
        self.cross_attn_all_layers_decoder = cross_attn_all_layers_decoder
        self.use_local_encoder_transformer = use_local_encoder_transformer
        self.downsampling_by_pooling = downsampling_by_pooling

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["BLTConfig"]
