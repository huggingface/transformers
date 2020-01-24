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
""" RoBERTa configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

PRETRAINED_CONFIG_ARCHIVE_MAP = {
    #"transformer-base": "https://s3.amazonaws.com/models.huggingface.co/bert/transformer-base-config.json",

}
from easydict import EasyDict as edict


# def bart_large_config(args):
#     #args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4 * 1024)
#     args.encoder_layers = getattr(args, 'encoder_layers', 12)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
#     args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
#
#     args.decoder_layers = getattr(args, 'decoder_layers', 12)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
#     args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
#
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.)
#     args.relu_dropout = getattr(args, 'relu_dropout', 0.)
#     args.dropout = getattr(args, 'dropout', 0.1)
#     args.max_target_positions = getattr(args, 'max_target_positions', 1024)
#     args.max_source_positions = getattr(args, 'max_source_positions', 1024)
#     args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
#     args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
#     args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed',
#                                                     True)
#     args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
#
#     args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
#     args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
#
#     args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
#     args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)
#
#     args.activation_fn = getattr(args, 'activation_fn', 'gelu')
#     args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
#     args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)




class BertConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers.BertModel`.
        It is used to instantiate an BERT model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the BERT model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.

        Example::

            from transformers import BertModel, BertConfig

            # Initializing a BERT bert-base-uncased style configuration
            configuration = BertConfig()

            # Initializing a model from the bert-base-uncased style configuration
            model = BertModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bart"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


_FAIRSEQ_DEFAULTS = dict(
    encoder_embed_dim=1024,
    encoder_ffn_embed_dim=4096,
    encoder_layers=12,
    encoder_attention_heads=16,
    encoder_normalize_before=False,
    encoder_learned_pos=True,
    decoder_embed_path=None,
    decoder_embed_dim=1024,
    decoder_ffn_embed_dim=4096,
    decoder_layers=12,
    decoder_attention_heads=16,
    decoder_normalize_before=False,
    decoder_learned_pos=True,
    attention_dropout=0.0,
    relu_dropout=0.0,
    dropout=0.1,
    max_target_positions=1024,
    max_source_positions=1024,
    adaptive_softmax_cutoff=None,
    adaptive_softmax_dropout=0,
    share_decoder_input_output_embed=True,
    share_all_embeddings=True,
    decoder_output_dim=1024,
    decoder_input_dim=1024,
    no_scale_embedding=True,
    layernorm_embedding=True,
    activation_fn='gelu',
    pooler_activation_fn='tanh',
    pooler_dropout=0.0
)

class BartConfig(PretrainedConfig):
    def __init__(self,
                 encoder_embed_dim=1024,
                 encoder_ffn_embed_dim=4096,
                 encoder_layers=12,
                 encoder_attention_heads=16,
                 encoder_normalize_before=False,
                 encoder_learned_pos=True,
                 decoder_embed_path=None,
                 decoder_embed_dim=1024,
                 decoder_ffn_embed_dim=4096,
                 decoder_layers=12,
                 decoder_attention_heads=16,
                 decoder_normalize_before=False,
                 decoder_learned_pos=True,
                 attention_dropout=0.0,
                 relu_dropout=0.0,
                 dropout=0.1,
                 max_target_positions=1024,
                 max_source_positions=1024,
                 adaptive_softmax_cutoff=None,
                 adaptive_softmax_dropout=0,
                 share_decoder_input_output_embed=True,
                 share_all_embeddings=True,
                 decoder_output_dim=1024,
                 decoder_input_dim=1024,
                 no_scale_embedding=True,
                 layernorm_embedding=True,
                 activation_fn='gelu',
                 pooler_activation_fn='tanh',
                 pooler_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        for k in _FAIRSEQ_DEFAULTS:
            setattr(self, k, locals()[k])   # hack to avoid 1 million LOC
