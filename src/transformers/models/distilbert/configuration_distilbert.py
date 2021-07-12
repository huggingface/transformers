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
""" DistilBERT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
    "distilbert-base-uncased-distilled-squad": "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/config.json",
    "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/config.json",
    "distilbert-base-cased-distilled-squad": "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
    "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/config.json",
    "distilbert-base-multilingual-cased": "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/config.json",
    "distilbert-base-uncased-finetuned-sst-2-english": "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
}


class DistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DistilBertModel` or a
    :class:`~transformers.TFDistilBertModel`. It is used to instantiate a DistilBERT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DistilBERT `distilbert-base-uncased
    <https://huggingface.co/distilbert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.DistilBertModel` or
            :class:`~transformers.TFDistilBertModel`.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        sinusoidal_pos_embds (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to use sinusoidal positional embeddings.
        n_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        n_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dim (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_dim (:obj:`int`, `optional`, defaults to 3072):
            The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qa_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilities used in the question answering model
            :class:`~transformers.DistilBertForQuestionAnswering`.
        seq_classif_dropout (:obj:`float`, `optional`, defaults to 0.2):
            The dropout probabilities used in the sequence classification and the multiple choice model
            :class:`~transformers.DistilBertForSequenceClassification`.

    Examples::

        >>> from transformers import DistilBertModel, DistilBertConfig

        >>> # Initializing a DistilBERT configuration
        >>> configuration = DistilBertConfig()

        >>> # Initializing a model from the configuration
        >>> model = DistilBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "distilbert"

    def __init__(
        self,
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
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id)
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

    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers
