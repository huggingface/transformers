# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
""" VGCN-BERT model configuration"""
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VGCNBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "zhibinlu/vgcn-distilbert-base-uncased": "https://huggingface.co/zhibinlu/vgcn-distilbert-base-uncased/resolve/main/config.json",
}


class VGCNBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VGCNBertModel`] or a [`TFVGCNBertModel`]. It
    is used to instantiate a VGCN-BERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the VGCN-BERT
    [zhibinlu/vgcn-distilbert-base-uncased](https://huggingface.co/zhibinlu/vgcn-distilbert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vgcn_graph_embedding_dim (`int`, *optional*, defaults to 16):
            Dimensionality of the number of output embedding from VGCN model.
        vgcn_hidden_dim (`int`, *optional*, defaults to 128):
            Dimensionality of the VGCN model hidden layer.
        vgcn_activation (`str` or `Callable`, *optional*, defaults to `"None"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        vgcn_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the graph convolutional network in the embeddings, encoder, and pooler.
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the VGCN-BERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`VGCNBertModel`] or [`TFVGCNBertModel`].
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        sinusoidal_pos_embds (`boolean`, *optional*, defaults to `False`):
            Whether to use sinusoidal positional embeddings.
        n_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        n_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dim (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_dim (`int`, *optional*, defaults to 3072):
            The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qa_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probabilities used in the question answering model [`VGCNBertForQuestionAnswering`].
        seq_classif_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probabilities used in the sequence classification and the multiple choice model
            [`VGCNBertForSequenceClassification`].

    Examples:

    ```python
    >>> from transformers import VGCNBertConfig, VGCNBertModel

    >>> # Initializing a VGCN-BERT configuration
    >>> configuration = VGCNBertConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VGCNBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vgcn-bert"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vgcn_graph_embds_dim=16,
        vgcn_hidden_dim=128,
        vgcn_activation="relu",
        vgcn_dropout=0.1,
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        **kwargs,
    ):
        self.vgcn_graph_embds_dim = vgcn_graph_embds_dim
        self.vgcn_hidden_dim = vgcn_hidden_dim
        self.vgcn_activation = vgcn_activation
        self.vgcn_dropout = vgcn_dropout
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        super().__init__(**kwargs, pad_token_id=pad_token_id)


class VGCNBertOnnxConfig(OnnxConfig):
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
