# coding=utf-8
# Copyright 2023 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Mega configuration"""
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mnaylor/mega-base-wikitext": "https://huggingface.co/mnaylor/mega-base-wikitext/resolve/main/config.json",
}



class MegaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegaModel`] or a [`TFMegaModel`]. It is
    used to instantiate a Mega model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Mega
    [mnaylor/mega-base-wikitext](https://huggingface.co/mnaylor/mega-base-wikitext) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Mega model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MegaModel`] or [`TFMegaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MegaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import MegaConfig, MegaModel

    >>> # Initializing a Mega configuration
    >>> configuration = MegaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MegaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mega"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256, # encoder hidden size (H in the paper)
        ema_projection_size=16, 
        bidirectional=True,
        shared_representation_size=64, # linear projection for shared representation after silu gating (Z in the paper)
        use_chunking=False,
        chunk_size=-1, 
        truncation=None,
        normalize_before_mega=True,
        normalization_type="scalenorm", # scalenorm, layernorm, rmsnorm, batchnorm, syncbatchnorm
        norm_affine=True,
        activation="silu", # silu, relu, linear, gelu, or gelu_accurate
        attention_activation="softmax", # softmax, laplace, or relu2
        dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_feature_dropout=False, # feature-based dropout or standard dropout
        use_normalized_ffn=True,
        nffn_hidden_size=256, # if use_normalized_ffn is True, this will be used to construct the linear layer
        normalize_before_ffn=True, # when to apply norm in the NFFN
        nffn_activation_dropout_prob=0.1,
        max_positions=1024,
        add_token_type_embeddings=False,
        type_vocab_size=2, # if add_token_type_embeddings is True, this will be used to construct the token type embeddings
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        relative_positional_bias="rotary", # rotary or simple
        classifier_dropout=None,
        use_cache=True, # unsure if i'll need this or not
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.attention_activation = attention_activation
        self.intermediate_size = intermediate_size
        self.ema_projection_size = ema_projection_size
        self.bidirectional = bidirectional
        self.shared_representation_size = shared_representation_size
        self.use_chunking = use_chunking 
        self.chunk_size = chunk_size
        self.truncation = truncation
        self.normalize_before_mega = normalize_before_mega
        self.normalization_type = normalization_type
        self.norm_affine = norm_affine
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_feature_dropout = use_feature_dropout
        self.use_normalized_ffn = use_normalized_ffn
        self.nffn_hidden_size = nffn_hidden_size
        self.normalize_before_ffn = normalize_before_ffn
        self.nffn_activation_dropout_prob = nffn_activation_dropout_prob
        self.max_positions = max_positions
        self.add_token_type_embeddings = add_token_type_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.relative_positional_bias = relative_positional_bias
        # self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class MegaOnnxConfig(OnnxConfig):
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
            ]
        )


# The config class I put together in my Colab implementation
class MegaConfigOG:
  "We'll need tokenizer configs too"
  def __init__(
      self,
      embed_dim, # hidden_size
      zdim, # shared_representation_size
      hdim, # intermediate_size
      ndim, # ema_projection_size
      dropout_prob=0.0, # dropout_prob
      attention_dropout_prob=0.0, # attention_probs_dropout_prob
      hidden_dropout_prob=0.0, # hidden_dropout_prob
      activation='silu', # activation
      attention_activation='softmax', # attention_activation
      bidirectional=True, # bidirectional
      chunk_size=-1, # chunk size
      truncation=None, # truncation
      norm_type='layernorm', # normalization_type
      prenorm=True, # normalize_before_mega
      norm_affine=True, # norm_affine
      feature_dropout=False, # use_feature_dropout
      rel_pos_bias='rotary', # relative_positional_bias
      max_positions=1024, # max_positions
      ffn_hidden_size=256, # nffn_hidden_size
      ffn_activation_dropout_prob=0.0, # nffn_activation_dropout_prob
      normalize_before_ffn = True # normalize_before_ffn
  ):
    self.embed_dim = embed_dim 
    self.zdim = zdim
    self.hdim = hdim 
    self.ndim = ndim
    self.dropout_prob = dropout_prob
    self.attention_dropout_prob = attention_dropout_prob
    self.hidden_dropout_prob = hidden_dropout_prob
    self.activation = activation 
    self.attention_activation = attention_activation
    self.bidirectional = bidirectional 
    self.chunk_size = chunk_size 
    self.truncation = truncation 
    self.norm_type = norm_type 
    self.prenorm = prenorm 
    self.norm_affine = norm_affine 
    self.feature_dropout = feature_dropout 
    self.rel_pos_bias = rel_pos_bias 
    self.max_positions = max_positions
    self.ffn_hidden_size = ffn_hidden_size
    self.ffn_activation_dropout_prob = ffn_activation_dropout_prob
    self.normalize_before_ffn = normalize_before_ffn