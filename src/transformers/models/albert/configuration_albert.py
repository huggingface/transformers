# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""ALBERT model configuration"""

from collections import OrderedDict
from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig


class AlbertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlbertModel`] or a [`TFAlbertModel`]. It is used
    to instantiate an ALBERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ALBERT
    [albert/albert-xxlarge-v2](https://huggingface.co/albert/albert-xxlarge-v2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30000):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AlbertModel`] or [`TFAlbertModel`].
        embedding_size (`int`, *optional*, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_hidden_groups (`int`, *optional*, defaults to 1):
            Number of groups for the hidden layers, parameters in the same group are shared.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 16384):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        inner_group_num (`int`, *optional*, defaults to 1):
            The number of inner repetition of attention and ffn.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`AlbertModel`] or [`TFAlbertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for attached classifiers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 3):
            End of stream token id.

    Examples:

    ```python
    >>> from transformers import AlbertConfig, AlbertModel

    >>> # Initializing an ALBERT-xxlarge style configuration
    >>> albert_xxlarge_configuration = AlbertConfig()

    >>> # Initializing an ALBERT-base style configuration
    >>> albert_base_configuration = AlbertConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     intermediate_size=3072,
    ... )

    >>> # Initializing a model (with random weights) from the ALBERT-base style configuration
    >>> model = AlbertModel(albert_xxlarge_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "albert"

    def __init__(
        self,
        vocab_size=30000,
        embedding_size=128,
        hidden_size=4096,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=64,
        intermediate_size=16384,
        inner_group_num=1,
        hidden_act="gelu_new",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout_prob=0.1,
        position_embedding_type="absolute",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.position_embedding_type = position_embedding_type


# Copied from transformers.models.bert.configuration_bert.BertOnnxConfig with Roberta->Albert
class AlbertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


__all__ = ["AlbertConfig", "AlbertOnnxConfig"]
