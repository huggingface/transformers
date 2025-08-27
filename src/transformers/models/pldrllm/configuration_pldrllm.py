# coding=utf-8
# Copyright 2025 Fromthesky Research Labs, LLC. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code uses the Llama model implementation by Eleuther AI 
# and Huggingface teams in this library as a starting point and implements 
# the PLDR-LLM (Large Language Model from Power Law Decoder Representations)
#  architecture based on its implementation by the Fromthesky Research Labs team.
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

"""PLDR-LLM model configuration"""

import numpy as np
from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class PldrllmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PldrllmModel`]. It is used to instantiate a PLDR-LLM
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the PLDR-LLM-v51-110M-3.
    e.g. [fromthesky/PLDR-LLM-v51-110M-3](https://huggingface.co/fromthesky/PLDR-LLM-v51-110M-3)
    Check out these papers for the details of PLDR-LLM architecture:
    [Paper-1](https://huggingface.co/papers/2107.02039) [Paper-2](https://huggingface.co/papers/2410.16703) [Paper-3](https://huggingface.co/papers/2502.13502)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the PLDR-LLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`PldrllmModel`]
        hidden_size (`int`, *optional*, defaults to 896):
            Dimension of the hidden representations. if set to None, hidden_size is calculated from
            num_attention_heads and head_dim.
        intermediate_size (`int`, *optional*, defaults to 2389):
            Dimension of the Pointwise Feed Forward Network representations. if set to None, intermediate_size is calculated from
            num_attention_heads and head_dim.
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 14):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length (context length) for the PLDR-LLM. PLDR-LLM-v51-110M-3 supports up to 1024.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Intended as the standard deviation of the truncated_normal_initializer for initializing all weight matrices. 
            This parameter is not used for initialization of the PLDR-LLM module weigths in favor of xavier_uniform_ initialization.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 3):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
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
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        glu_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in Gated Linear Units used in Pointwise Feedforward Network and Residual Layers for
            the metric learner.
        final_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the LM head layer of the PldrllmForCausalLM implementation.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 64):
            The attention head dimension.
        reference_rope (`bool`, *optional*, defaults to `True`):
            Whether to use the rotary positional embedding implementation used in the reference paper implementing the
            PLDR-LLM in pytorch. Check out [this paper](https://huggingface.co/papers/2502.13502).
        num_reslayerA (`int`, *optional*, defaults to 8):
            Number of residual layers in the metric learner section of the power law graph attention layer.
        num_denseA (`int`, *optional*, defaults to 2):
            Number of gated linear units in each residual layer in the metric learner section of the power law graph attention layer.
        A_dff (`int`, *optional*, defaults to 170):
            The dimension of hidden layer in the gated linear unit for the residual metric learner. Input and output dimensions
            are set at head_dim. 
        custom_G_type (`str`, *optional*, defaults to None):
            PLDR-LLM supports predefined energy-curvature tensor (G) values that can bypass the metric learner section during training and
            inference. This assigns the decoder.past_G_values attribute to a predefined value. This is useful for experimentation and assigning
            an already learned energy-curvature tensor. The StaticCache is supported only for predefined past_G_values.
            None: G values are learned during training and inferred by the residual metric learner at least once (depending on use_cache status).
                 past_G_values has shape (num_layers, 3, batch_size, num_heads, head_dim, head_dim).
            'identity': decoder.past_G_values are assigned to identity matrix and metric learner layer is not part of the model. This setting is equivalent to
                        an LLM with Scaled Dot Product Attention (SDPA). The decoder.past_G_values are saved with the model.
            'random': decoder.past_G_values are assigned to randomly initialized matrix from a normal distribution. This setting is equivalent to
                        an LLM with Scaled Dot Product Attention (SDPA). The decoder.past_G_values are saved with the model.
            'external': decoder.past_G_values are expected to be assigned after initializing/loading the PLDR-LLM weights. decoder.past_G_values[:, 2,...].
                        are initialized to identity matrix by default. The expected shape of input is (num_layers, 3, 1, num_heads, head_dim, head_dim) and
                        [:, 2,...] must have the predefined energy-curvature tensor values. Other entries are set to zero tensor by default.
        cache_first_G (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the G values from first sample in a batch or G values from all samples for past_G_values initialization. 
            When `cache_first_G=true`, the batch_size of past_G_values is 1. This argument should be set to True for contrastive text generation 
            with learned G values.

        output_pldr_attentions (`bool`, *optional*, defaults to `False`):
            Whether to return the deductive outputs and learnable parameters of power law graph attention module as tuple containing:
            the output of the residual metric learner (metric tensor, A), output (A_LM) after application of iSwiGLU on metric tensor, learned 
            exponents of potential tensor, learned weights for energy-curvature tensor, learned bias for
            energy-curvature tensor, energy-curvature tensor (G_LM), and attention weights.       

    ```python
    >>> from transformers import PldrllmModel, PldrllmConfig

    >>> # Initializing a PLDR-LLM PLDR-LLM-v51-110M-3 style configuration
    >>> configuration = PldrllmConfig()

    >>> # Initializing a model from the PLDR-LLM-v51-110M-3 style configuration
    >>> model = PldrllmModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pldrllm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=2389,
        num_hidden_layers=5,
        num_attention_heads=14,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-6, #hard coded
        use_cache=True, 
        output_pldr_attentions=False,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=False, #hard coded
        rope_theta=10000.0, #hard coded
        rope_scaling=None, #hard coded
        attention_bias=True, #hard coded
        glu_bias=True, #hard coded
        final_bias=True, #hard coded
        reference_rope=True,
        attention_dropout=0.0, #hard coded
        head_dim=64,
        num_reslayerA=8,
        num_denseA=2,
        A_dff=170,
        custom_G_type=None,
        cache_first_G=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size if hidden_size is not None else int(num_attention_heads*head_dim)
        self.intermediate_size = intermediate_size if intermediate_size is not None else int(np.floor(num_attention_heads*head_dim*4*2/3))
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_reslayerA=num_reslayerA
        self.num_denseA=num_denseA
        self.A_dff=A_dff
        self.glu_bias=glu_bias
        self.attention_bias = attention_bias
        self.final_bias=final_bias
        self.initializer_range=initializer_range

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.output_pldr_attentions=output_pldr_attentions
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.reference_rope=reference_rope
        self.custom_G_type=custom_G_type
        self.cache_first_G=cache_first_G
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)




__all__ = ["PldrllmConfig"]
