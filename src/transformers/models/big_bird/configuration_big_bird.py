# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
""" BigBird model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/config.json",
    "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/config.json",
    "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/config.json",
    # See all BigBird models at https://huggingface.co/models?filter=big_bird
}


class BigBirdConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BigBirdModel`. It is used to
    instantiate an BigBird model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BigBird
    `google/bigbird-roberta-base <https://huggingface.co/google/bigbird-roberta-base>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50358):
            Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BigBirdModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 1024 or 2048 or 4096).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BigBirdModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"block_sparse"`)
            Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
            layer (with n^2 complexity). Possible values are :obj:`"original_full"` and :obj:`"block_sparse"`.
        use_bias (:obj:`bool`, `optional`, defaults to :obj:`True`)
            Whether to use bias in query, key, value.
        rescale_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether to rescale embeddings with (hidden_size ** 0.5).
        block_size (:obj:`int`, `optional`, defaults to 64)
            Size of each block. Useful only when :obj:`attention_type == "block_sparse"`.
        num_random_blocks (:obj:`int`, `optional`, defaults to 3)
            Each query is going to attend these many number of random blocks. Useful only when :obj:`attention_type ==
            "block_sparse"`.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.

        Example::

        >>> from transformers import BigBirdModel, BigBirdConfig

        >>> # Initializing a BigBird google/bigbird-roberta-base style configuration
        >>> configuration = BigBirdConfig()

        >>> # Initializing a model from the google/bigbird-roberta-base style configuration
        >>> model = BigBirdModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "big_bird"

    def __init__(
        self,
        vocab_size=50358,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4096,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sep_token_id=66,
        attention_type="block_sparse",
        use_bias=True,
        rescale_embeddings=False,
        block_size=64,
        num_random_blocks=3,
        gradient_checkpointing=False,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder
        self.gradient_checkpointing = gradient_checkpointing

        self.rescale_embeddings = rescale_embeddings
        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.classifier_dropout = classifier_dropout
