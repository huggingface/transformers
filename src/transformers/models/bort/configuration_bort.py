# coding=utf-8
# Copyright 2020, The HuggingFace Inc. team.
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
""" BORT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BORT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # See all BORT models at https://huggingface.co/models?filter=bort
}


class BortConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BortModel` or a
    :class:`~transformers.TFBortModel`. It is used to instantiate a BORT model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        hidden_size (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        intermediate_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BORT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BortModel` or
            :class:`~transformers.TFBortModel`.

    Examples::

        >>> from transformers import BortModel, BortConfig

        >>> # Initializing a BORT bert-base-uncased style configuration
        >>> configuration = BortConfig()

        >>> # Initializing a model from the bort style configuration
        >>> model = BortModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bort"

    def __init__(
        self,
        attention_probs_dropout_prob=0.1,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=768,
        layer_norm_eps=1e-05,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=8,
        num_hidden_layers=4,
        pad_token_id=1,
        type_vocab_size=1,
        vocab_size=50265,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
