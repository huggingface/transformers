# coding=utf-8
# Copyright 2023 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang and The HuggingFace Inc. team. All rights reserved.
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
"""ErnieM model configuration"""
# Adapted from original paddlenlp repository.(https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie_m/configuration.py)

from __future__ import annotations

from typing import Dict

from ....configuration_utils import PretrainedConfig


class ErnieMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieMModel`]. It is used to instantiate a
    Ernie-M model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `Ernie-M`
    [susnato/ernie-m-base_pytorch](https://huggingface.co/susnato/ernie-m-base_pytorch) architecture.


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of `inputs_ids` in [`ErnieMModel`]. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling
            [`ErnieMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embedding layer, encoder layers and pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors to feed-forward layers are
            firstly projected from hidden_size to intermediate_size, and then projected back to hidden_size. Typically
            intermediate_size is larger than hidden_size.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the feed-forward layer. `"gelu"`, `"relu"` and any other torch
            supported activation functions are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability used in `MultiHeadAttention` in all encoder layers to drop some attention target.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length
            of an input sequence.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the normal initializer for initializing all weight matrices. The index of padding
            token in the token vocabulary.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        act_dropout (`float`, *optional*, defaults to 0.0):
            This dropout probability is used in `ErnieMEncoderLayer` after activation.

    A normal_initializer initializes weight matrices as normal distributions. See
    `ErnieMPretrainedModel._init_weights()` for how weights are initialized in `ErnieMModel`.
    """

    model_type = "ernie_m"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
        self,
        vocab_size: int = 250002,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 514,
        initializer_range: float = 0.02,
        pad_token_id: int = 1,
        layer_norm_eps: float = 1e-05,
        classifier_dropout=None,
        act_dropout=0.0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.act_dropout = act_dropout


__all__ = ["ErnieMConfig"]
