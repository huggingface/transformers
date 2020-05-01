# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class ReformerConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers.ReformerModel`.
        It is used to instantiate an Reformer model according to the specified arguments, defining the model
        architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        Args:
            attention_head_size=64,
            attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
            axial_pos_embds=True,
            axial_norm_std=1.0,
            axial_pos_shape=[512, 1024],
            axial_pos_embds_dim=[512, 1024],
                TODO (PVP)
                TODO (PVP)
            chunk_size_lm_head=0,
                TODO (PVP)
            chunk_size_feed_forward=0,
            feed_forward_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "feed_forward" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            local_chunk_length (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            local_num_chunks_before (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            local_num_chunks_after (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            local_attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            lsh_chunk_length (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            lsh_num_chunks_before (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            lsh_num_chunks_after (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            lsh_attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_buckets (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            num_hashes (:obj:`int`, optional, defaults to ):
                TODO (PVP)
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the Reformer model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.ReformerModel`.

        Example::

            from transformers import ReformerModel, ReformerConfig

            # Initializing a Reformer configuration
            configuration = ReformerConfig()

            # Initializing a Reformer model
            model = ReformerModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """
    pretrained_config_archive_map = REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "reformer"

    def __init__(
        self,
        attention_head_size=64,
        attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
        axial_norm_std=1.0,
        axial_pos_embds=True,
        axial_pos_shape=[512, 1024],
        axial_pos_embds_dim=[512, 1024],
        chunk_size_lm_head=0,
        chunk_size_feed_forward=0,
        eos_token_id=2,
        feed_forward_size=512,
        hash_seed=None,
        hidden_act="relu",
        hidden_dropout_prob=0.05,
        hidden_size=256,
        initializer_range=0.02,
        is_decoder=True,
        layer_norm_eps=1e-12,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        local_attention_probs_dropout_prob=0.05,
        local_attn_chunk_length=64,
        lsh_attn_chunk_length=64,
        lsh_attention_probs_dropout_prob=0.0,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        max_position_embeddings=524288,
        num_attention_heads=2,
        num_buckets=[64, 128],
        num_hashes=1,
        pad_token_id=0,
        vocab_size=320,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_decoder=is_decoder, **kwargs)

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
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.attn_layers = attn_layers
