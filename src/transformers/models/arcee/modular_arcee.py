# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Arcee model."""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import auto_docstring, is_torch_available, logging
from transformers.modeling_rope_utils import rope_config_validation
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


logger = logging.get_logger(__name__)


class ArceeConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ArceeModel`]. It is used to instantiate an Arcee
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the AFM-4.5B-Base.

    Pre-trained weights are available at
    [arcee-ai/AFM-4.5B](https://huggingface.co/arcee-ai/AFM-4.5B)
    and were used to build the examples below.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Arcee model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ArceeModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
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
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. AFM-4.5B-Base supports up to 16384 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 128000):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 128001):
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
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'yarn'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'yarn'. The original max position embeddings used during pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn'. The scaling factor to be applied on the attention computation. If unspecified,
                    it defaults to value recommended by the implementation, using the `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads

    ```python
    >>> from transformers import ArceeModel, ArceeConfig

    >>> # Initializing an Arcee AFM-4.5B-Base style configuration
    >>> configuration = ArceeConfig()

    >>> # Initializing a model from the AFM-4.5B-Base style configuration
    >>> model = ArceeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "arcee"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2560,
        intermediate_size=18432,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="relu2",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )
        
        # Validate the correctness of rotary position embeddings parameters using Arcee's custom validation
        # BC: if there is a 'type' field, copy it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)


class ArceeRMSNorm(LlamaRMSNorm):
    """ArceeRMSNorm is identical to LlamaRMSNorm"""
    pass



class ArceeMLP(nn.Module):
    """Arcee MLP with configurable activation function (typically relu2)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.up_proj(x)))
        return down_proj


class ArceeAttention(LlamaAttention):
    """Multi-headed attention for Arcee - identical to Llama attention"""
    pass


class ArceeDecoderLayer(LlamaDecoderLayer):
    """Arcee decoder layer with custom MLP"""
    
    def __init__(self, config: ArceeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the Llama MLP with Arcee MLP to use the correct activation
        self.mlp = ArceeMLP(config)


class ArceePreTrainedModel(LlamaPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    config_class = ArceeConfig


class ArceeModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ArceeDecoderLayer`]

    Args:
        config: ArceeConfig
    """

    def __init__(self, config: ArceeConfig):
        super().__init__(config)
        # The parent init handles most setup, we just need to ensure our decoder layers are used
        self.layers = nn.ModuleList(
            [ArceeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Initialize weights and apply final processing
        self.post_init()


class ArceeForCausalLM(LlamaForCausalLM):
    """Arcee Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings)."""
    
    def __init__(self, config):
        # We need to set config_class before calling super().__init__
        self.config_class = ArceeConfig
        super().__init__(config)
        self.model = ArceeModel(config)
        # Initialize weights and apply final processing
        self.post_init()


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForSequenceClassification(LlamaForSequenceClassification):
    """
    The Arcee Model transformer with a sequence classification head on top (linear layer).
    """
    
    def __init__(self, config):
        self.config_class = ArceeConfig
        super().__init__(config)
        self.model = ArceeModel(config)
        # Initialize weights and apply final processing
        self.post_init()


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForQuestionAnswering(LlamaForQuestionAnswering):
    """
    The Arcee Model transformer with a span classification head on top for extractive question-answering tasks.
    """
    
    def __init__(self, config):
        self.config_class = ArceeConfig
        super().__init__(config)
        # Note: LlamaForQuestionAnswering uses self.transformer, not self.model
        self.transformer = ArceeModel(config)
        # Initialize weights and apply final processing
        self.post_init()


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForTokenClassification(LlamaForTokenClassification):
    """
    The Arcee Model transformer with a token classification head on top.
    """
    
    def __init__(self, config):
        self.config_class = ArceeConfig
        super().__init__(config)
        self.model = ArceeModel(config)
        # Initialize weights and apply final processing
        self.post_init()


__all__ = [
    "ArceeConfig",
    "ArceeForCausalLM",
    "ArceeForQuestionAnswering",
    "ArceeForSequenceClassification", 
    "ArceeForTokenClassification",
    "ArceeModel",
    "ArceePreTrainedModel",
]