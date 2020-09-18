# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
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
""" Reformer model configuration """

from .configuration_utils import PretrainedConfig
from .utils import logging


logger = logging.get_logger(__name__)

REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/reformer-crime-and-punishment": "https://cdn.huggingface.co/google/reformer-crime-and-punishment/config.json",
    "google/reformer-enwik8": "https://cdn.huggingface.co/google/reformer-enwik8/config.json",
}


class ReformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ReformerModel`.
    It is used to instantiate an Reformer model according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        attention_head_size (:obj:`int`, optional, defaults to 64):
            Dimensionality of the projected key, query and value vectors
        attn_layers (:obj:`list(str)`, optional, defaults to ["local", "lsh", "local", "lsh", "local", "lsh"]):
            List of attention layer types in ascending order. It can be chosen between a
            LSHSelfAttention layer ("lsh") and a LocalSelfAttention layer ("local").
            For more information on LSHSelfAttention layer, see `LSH Self Attention <reformer.html#lsh-self-attention>`__ .
            For more information on LocalSelfAttention layer, see `Local Self Attention <reformer.html#local-sensitive-hashing-self-attention>`__ .
        axial_pos_embds (:obj:`bool`, optional, defaults to :obj:`True`):
            If `True` use axial position embeddings. For more information on how axial position embeddings work, see `Axial Position Encodings <reformer.html#axial-positional-encodings>`__
        axial_norm_std (:obj:`float`, optional, defaluts to 1.0):
            The standard deviation of the normal_initializer for initializing the weight matrices of the axial positional encodings.
        axial_pos_shape (:obj:`list(int)`, optional, defaults to `[64, 64]`):
            The position dims of the axial position encodings.
            During training the product of the position dims has to equal the sequence length.
            For more information on how axial position embeddings work, see `Axial Position Encodings <reformer.html#axial-positional-encodings>`__.
        axial_pos_embds_dim (:obj:`list(int)`, optional, defaults to `[64, 192]`):
            The embedding dims of the axial position encodings.
            The sum of the embedding dims has to equal the hidden size.
            For more information on how axial position embeddings work, see `Axial Position Encodings <reformer.html#axial-positional-encodings>`__.
        chunk_size_lm_head (:obj:`int`, optional, defaults to 0):
            The chunk size of the final language model feed forward head layer.
            A chunk size of 0 means that the feed forward layer is not chunked.
            A chunk size of n means that the feed forward layer processes n < sequence_length embeddings at a time.
            For more information on feed forward chunking, see `How does Feed Forward Chunking work? <../glossary.html#feed-forward-chunking>`__ .
        eos_token_id (:obj:`int`, optional, defaults to 2):
            The token id for the <EOS> token.
        feed_forward_size (:obj:`int`, optional, defaults to 512):
            Dimensionality of the "feed_forward" (i.e., feed-forward) layer in the residual attention block.
        hash_seed (:obj:`int`, optional, defaults to `None`):
            Seed that can be used to make local sensitive hashing in LSHSelfAttention deterministic. This should only be set for testing purposed. For evaluation and training purposes `hash_seed` should be set to `None` to ensure fully random rotations in local sensitive hashing scheme.
        hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "relu"):
            The non-linear activation function (function or string) in the feed forward layer in the residual attention block.
            If string, "gelu", "relu", "swish", "gelu_new" and "gelu_fast" are supported.
        hidden_dropout_prob (:obj:`float`, optional, defaults to 0.05):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        hidden_size (:obj:`int`, optional, defaults to 256):
            Dimensionality of the output hidden states of the residual attention blocks.
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_decoder (:obj:`bool`, optional, defaults to :obj:`False`):
            If `is_decoder` is True, a causal mask is used in addition to `attention_mask`.
            When using the Reformer for causal language modeling, `is_decoder` is set to `True`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        local_chunk_length (:obj:`int`, optional, defaults to 64):
            Length of chunk which attends to itself in LocalSelfAttention. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention).
        local_num_chunks_before (:obj:`int`, optional, defaults to 1):
            Number of previous neighbouring chunks to attend to in LocalSelfAttention layer to itself.
        local_num_chunks_after (:obj:`int`, optional, defaults to 0):
            Number of following neighbouring chunks to attend to in LocalSelfAttention layer in addition to itself.
        local_attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities in LocalSelfAttention.
        lsh_attn_chunk_length (:obj:`int`, optional, defaults to 64):
            Length of chunk which attends to itself in LSHSelfAttention. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention).
        lsh_num_chunks_before (:obj:`int`, optional, defaults to 1):
            Number of previous neighbouring chunks to attend to in LSHSelfAttention layer to itself.
        lsh_num_chunks_after (:obj:`int`, optional, defaults to 0):
            Number of following neighbouring chunks to attend to in LSHSelfAttention layer to itself.
        lsh_attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities in LSHSelfAttention.
        max_position_embeddings (:obj:`int`, optional, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        num_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_buckets (:obj:`int` or :obj:`list(int)`, optional, defaults to `None`):
            Number of buckets, the key query vectors can be "hashed into" using the locality sensitive hashing scheme. Each query key vector is hashed into a hash in `1, ..., num_buckets`.
            The number of buckets can also be factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is factorized into two factors.
            The number of buckets (or the product the factors) should approximately equal sequence length / lsh_chunk_length. If `num_buckets` is set to `None`, a good value for `num_buckets` is calculated on the fly.
        num_hashes (:obj:`int`, optional, defaults to 1):
            Number of hashing rounds (e.g. number of random rotations) in Local Sensitive Hashing scheme.
            The higher `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive the hashing becomes.
        pad_token_id (:obj:`int`, optional, defaults to 0):
            The token id for the <PAD> token.
        vocab_size (:obj:`int`, optional, defaults to 320):
            Vocabulary size of the Reformer model. Defines the different tokens that
            can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.ReformerModel`.
        tie_word_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to tie input and output embeddings.

    Example::

        >>> from transformers import ReformerModel, ReformerConfig

        >>> # Initializing a Reformer configuration
        >>> configuration = ReformerConfig()

        >>> # Initializing a Reformer model
        >>> model = ReformerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "reformer"

    def __init__(
        self,
        attention_head_size=64,
        attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
        axial_norm_std=1.0,
        axial_pos_embds=True,
        axial_pos_shape=[64, 64],
        axial_pos_embds_dim=[64, 192],
        chunk_size_lm_head=0,
        eos_token_id=2,
        feed_forward_size=512,
        hash_seed=None,
        hidden_act="relu",
        hidden_dropout_prob=0.05,
        hidden_size=256,
        initializer_range=0.02,
        is_decoder=False,
        layer_norm_eps=1e-12,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        local_attention_probs_dropout_prob=0.05,
        local_attn_chunk_length=64,
        lsh_attn_chunk_length=64,
        lsh_attention_probs_dropout_prob=0.0,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        max_position_embeddings=4096,
        num_attention_heads=12,
        num_buckets=None,
        num_hashes=1,
        pad_token_id=0,
        vocab_size=320,
        tie_word_embeddings=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.hash_seed = hash_seed
        self.vocab_size = vocab_size
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hashes = num_hashes
        self.num_hidden_layers = len(attn_layers)
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.local_attn_chunk_length = local_attn_chunk_length
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.local_num_chunks_after = local_num_chunks_after
        self.local_num_chunks_before = local_num_chunks_before
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.axial_norm_std = axial_norm_std
        self.chunk_size_lm_head = chunk_size_lm_head
        self.attn_layers = attn_layers
