# coding=utf-8
# Copyright 2022 The REALM authors and The HuggingFace Inc. team.
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
"""REALM model configuration."""

from ....configuration_utils import PretrainedConfig
from ....utils import logging


logger = logging.get_logger(__name__)


class RealmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of

    1. [`RealmEmbedder`]
    2. [`RealmScorer`]
    3. [`RealmKnowledgeAugEncoder`]
    4. [`RealmRetriever`]
    5. [`RealmReader`]
    6. [`RealmForOpenQA`]

    It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
    [google/realm-cc-news-pretrained-embedder](https://huggingface.co/google/realm-cc-news-pretrained-embedder)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the REALM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RealmEmbedder`], [`RealmScorer`], [`RealmKnowledgeAugEncoder`], or
            [`RealmReader`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        retriever_proj_size (`int`, *optional*, defaults to 128):
            Dimension of the retriever(embedder) projection.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_candidates (`int`, *optional*, defaults to 8):
            Number of candidates inputted to the RealmScorer or RealmKnowledgeAugEncoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RealmEmbedder`], [`RealmScorer`],
            [`RealmKnowledgeAugEncoder`], or [`RealmReader`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        span_hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the reader's spans.
        max_span_width (`int`, *optional*, defaults to 10):
            Max span width of the reader.
        reader_layer_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the reader's layer normalization layers.
        reader_beam_size (`int`, *optional*, defaults to 5):
            Beam size of the reader.
        reader_seq_len (`int`, *optional*, defaults to 288+32):
            Maximum sequence length of the reader.
        num_block_records (`int`, *optional*, defaults to 13353718):
            Number of block records.
        searcher_beam_size (`int`, *optional*, defaults to 5000):
            Beam size of the searcher. Note that when eval mode is enabled, *searcher_beam_size* will be the same as
            *reader_beam_size*.

    Example:

    ```python
    >>> from transformers import RealmConfig, RealmEmbedder

    >>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
    >>> configuration = RealmConfig()

    >>> # Initializing a model (with random weights) from the google/realm-cc-news-pretrained-embedder style configuration
    >>> model = RealmEmbedder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "realm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        retriever_proj_size=128,
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
        span_hidden_size=256,
        max_span_width=10,
        reader_layer_norm_eps=1e-3,
        reader_beam_size=5,
        reader_seq_len=320,  # 288 + 32
        num_block_records=13353718,
        searcher_beam_size=5000,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # Common config
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.retriever_proj_size = retriever_proj_size
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

        # Reader config
        self.span_hidden_size = span_hidden_size
        self.max_span_width = max_span_width
        self.reader_layer_norm_eps = reader_layer_norm_eps
        self.reader_beam_size = reader_beam_size
        self.reader_seq_len = reader_seq_len

        # Retrieval config
        self.num_block_records = num_block_records
        self.searcher_beam_size = searcher_beam_size


__all__ = ["RealmConfig"]
