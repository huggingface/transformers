# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Emformer model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

EMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "anton-l/emformer-base-librispeech": (
        "https://huggingface.co/anton-l/emformer-base-librispeech/resolve/main/config.json"
    ),
}


class EmformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EmformerModel`]. It is used to instantiate an
    Emformer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Emformer
    [anton-l/emformer-base-librispeech](https://huggingface.co/anton-l/emformer-base-librispeech) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 4096):
            Vocabulary size of the Emformer model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`EmformerModel`]. Vocabulary size of the model. Defines the different
            tokens that can be represented by the *inputs_ids* passed to the forward method of [`EmformerModel`].
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        blank_token_id (`int`, *optional*, defaults to 4096):
            The ID of a blank token used in RNN-T decoding.
        pad_token_id (`int`, *optional*, defaults to 1):
            The ID of a padding token used to pad [`EmformerForRNNT`] outputs.
        input_dim (`int`, *optional*, defaults to 80):
            The feature dimension of input features (Mel spectrogram).
        time_reduction_input_dim (`int`, *optional*, defaults to 128):
            The dimension that the input features are projected to before time shuffling.
        time_reduction_stride (`int`, *optional*, defaults to 4):
            The stride of [`EmformerTimeReduction`] used to compress input features into fewer frames.
        left_context_length (`int`, *optional*, defaults to 30):
            The length of Emformer attention context on the left.
        right_context_length (`int`, *optional*, defaults to 4):
            The length of Emformer attention context on the right.
        segment_length (`int`, *optional*, defaults to 16):
            The length of Emformer attention segments.
        output_dim (`int`, *optional*, defaults to 1024):
            The dimension that the Transformer output features are projected to.
        token_embedding_dim (`int`, *optional*, defaults to 512):
            The dimensionality of RNN-T token embeddings.
        num_lstm_layers (`int`, *optional*, defaults to 3):
            The number of LSTM layers in the RNN-T predictor.

    Example:

    ```python
    >>> from transformers import EmformerModel, EmformerConfig

    >>> # Initializing a Emformer anton-l/emformer-base-librispeech style configuration
    >>> configuration = EmformerConfig()

    >>> # Initializing a model from the anton-l/emformer-base-librispeech style configuration
    >>> model = EmformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "emformer"

    def __init__(
        self,
        vocab_size=4096,
        num_hidden_layers=20,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        initializer_range=0.02,
        blank_token_id=4096,
        pad_token_id=1,
        input_dim=80,
        time_reduction_input_dim=128,
        time_reduction_stride=4,
        left_context_length=30,
        right_context_length=4,
        segment_length=16,
        output_dim=1024,
        token_embedding_dim=512,
        num_lstm_layers=3,
        lstm_hidden_dim=512,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
        joiner_activation="relu",
        max_output_length=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        self.input_dim = input_dim
        self.time_reduction_input_dim = time_reduction_input_dim
        self.time_reduction_stride = time_reduction_stride
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.segment_length = segment_length
        self.output_dim = output_dim

        self.token_embedding_dim = token_embedding_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer_norm_epsilon = lstm_layer_norm_epsilon
        self.lstm_dropout = lstm_dropout

        self.joiner_activation = joiner_activation

        self.blank_token_id = blank_token_id
        self.pad_token_id = pad_token_id
        self.max_output_length = max_output_length
