# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" REALM model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

REALM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "realm-cc-news-pretrained-bert": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-bert/resolve/main/config.json",
    "realm-cc-news-pretrained-embedder": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-embedder/resolve/main/config.json",
    "realm-cc-news-pretrained-retriever": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-retriever/resolve/main/config.json",
    # See all REALM models at https://huggingface.co/models?filter=realm
}


class RealmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of :class:`~transformers.RealmEmbedder`,
    :class:`~transformers.RealmRetriever`, and :class:`~transformers.RealmEncoder`. It is used to instantiate an REALM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the REALM `realm-cc-news-pretrained
    <https://huggingface.co/realm-cc-news-pretrained>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the REALM model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.RealmEmbedder`,
            :class:`~transformers.RealmRetriever`, or :class:`~transformers.RealmEncoder`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        retriever_proj_size (:obj:`int`, `optional`, defaults to 128):
            Dimension of the retriever(embedder) projection.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_candidates (:obj:`int`, `optional`, defaults to 8):
            Number of candidates inputted to the RealmRetriever or RealmEncoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.RealmEmbedder`,
            :class:`~transformers.RealmRetriever`, or :class:`~transformers.RealmEncoder`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    Example::

        >>> from transformers import RealmEmbedder, RealmConfig

        >>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
        >>> configuration = RealmConfig()

        >>> # Initializing a model from the qqaatw/realm-cc-news-pretrained-embedder style configuration
        >>> model = RealmEmbedder(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "realm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        retriever_proj_size=128,
        span_hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_candidates=8,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.retriever_proj_size = retriever_proj_size
        self.span_hidden_size = span_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_candidates = num_candidates
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
