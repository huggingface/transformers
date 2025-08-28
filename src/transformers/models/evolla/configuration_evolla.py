# coding=utf-8
# Copyright 2025 Westlake Representational Learning Lab (Fajie Yuan Lab) team and the HuggingFace Inc. team. All rights reserved.
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
"""Evolla model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class SaProtConfig(PretrainedConfig):
    r"""This is the configuration class to store the configuration of a [`EvollaSaProtProteinEncoder`]. It is used to instantiate a
    SaProt model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 446):
            Vocabulary size of the protein sequence model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`EvollaModel`].
        mask_token_id (`int`, *optional*, defaults to 4):
            The id of the *mask* token in the protein sequence model.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the *padding* token in the protein sequence model.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the protein sequence model layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 33):
            Number of hidden layers in the protein sequence model.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the protein sequence model.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the intermediate layers in the protein sequence model.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the hidden layers in the protein sequence model.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in the protein sequence model.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that the protein sequence model might ever be used with. Typically set this to
            something large just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for the layer normalization layer in the protein sequence model.
        position_embedding_type (`str`, *optional*, defaults to `"rotary"`):
            The type of position embedding to use in the protein sequence model. Currently only `"rotary"` is supported.
        emb_layer_norm_before (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization before the position embedding in the protein sequence model.
        token_dropout (`bool`, *optional*, defaults to `True`):
            Whether to apply dropout to the tokens in the protein sequence model."""

    def __init__(
        self,
        vocab_size=446,
        mask_token_id=4,
        pad_token_id=1,
        hidden_size=1280,
        num_hidden_layers=33,
        num_attention_heads=20,
        intermediate_size=5120,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        position_embedding_type="rotary",
        emb_layer_norm_before=False,
        token_dropout=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout


class EvollaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        protein_encoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SaProtConfig`].
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Evolla llama model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`EvollaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the llama layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the intermediate layers in the llama model.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the llama model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the llama model.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value pairs for each attention layer in the llama model.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the llama model. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"silu"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for the RMS-norm layer in the llama model.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The threshold value for the RoPE layer in the llama model.
        rope_scaling (`float`, *optional*):
            The scaling factor for the RoPE layer in the llama model.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention layer.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layer.
        aligner_ffn_mult (`int`, *optional*, defaults to 4):
            The FFN multiplier for the aligner layer.
        aligner_enable_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the aligner layer.
        aligner_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in the aligner layer.
        aligner_num_add_layers (`int`, *optional*, defaults to 8):
            The number of additional layers for the aligner layer.
        resampler_depth (`int`, *optional*, defaults to 6):
            The depth of the resampler layer in the llama model.
        resampler_dim_head (`int`, *optional*, defaults to 64):
            The dimension of the heads in the resampler layer in the llama model.
        resampler_heads (`int`, *optional*, defaults to 8):
            The number of heads in the resampler layer in the llama model.
        resampler_num_latents (`int`, *optional*, defaults to 64):
            The number of latents in the resampler layer in the llama model.
        resampler_ff_mult (`int`, *optional*, defaults to 4):
            The FFN multiplier for the resampler layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 128009):
            The id of the *end-of-sequence* token.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the input and output word embeddings.

    Example:

    ```python
    >>> from transformers import EvollaModel, EvollaConfig

    >>> # Initializing a Evolla evolla-10b style configuration
    >>> configuration = EvollaConfig()

    >>> # Initializing a model from the evolla-10b style configuration
    >>> model = EvollaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "EvollaModel"
    sub_configs = {"protein_encoder_config": SaProtConfig}

    def __init__(
        self,
        protein_encoder_config=None,
        vocab_size=128256,  # llama vocab size
        hidden_size=4096,  # llama hidden size
        intermediate_size=14336,  # llama intermediate size
        num_hidden_layers=32,  # llama num layers
        num_attention_heads=32,  # llama num heads
        num_key_value_heads=8,  # llama num key-value heads
        hidden_act="silu",  # llama activation function
        max_position_embeddings=8192,  # llama rope max length
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        aligner_ffn_mult=4,
        aligner_enable_bias=True,
        aligner_attention_probs_dropout_prob=0.1,
        aligner_num_add_layers=8,
        resampler_depth=6,
        resampler_dim_head=64,
        resampler_heads=8,
        resampler_num_latents=64,
        resampler_ff_mult=4,
        initializer_range=0.02,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128009,
        use_cache=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.aligner_ffn_mult = aligner_ffn_mult
        self.aligner_enable_bias = aligner_enable_bias
        self.aligner_attention_probs_dropout_prob = aligner_attention_probs_dropout_prob
        self.aligner_num_add_layers = aligner_num_add_layers
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head
        self.resampler_heads = resampler_heads
        self.resampler_num_latents = resampler_num_latents
        self.resampler_ff_mult = resampler_ff_mult

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Subconfig
        if protein_encoder_config is None:
            protein_encoder_config = {}
            logger.info("`protein_encoder_config` is `None`. Initializing the `SaProtConfig` with default values.")
        self.protein_encoder_config = SaProtConfig(**protein_encoder_config)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["EvollaConfig"]
