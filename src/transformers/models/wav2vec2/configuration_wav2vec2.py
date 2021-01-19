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
""" Wav2Vec2 model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "fill_later": "https://huggingface.co/fill_later/resolve/main/config.json",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
}


class Wav2Vec2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.Wav2Vec2Model`. It is used to
    instantiate an Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2 `fill_later
    <https://huggingface.co/fill_later>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the Wav2Vec2 model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.Wav2Vec2Model` or
            :class:`~transformers.TFWav2Vec2Model`. Vocabulary size of the model. Defines the different tokens that can
            be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.Wav2Vec2Model`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
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
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.Wav2Vec2Model`
            or :class:`~transformers.TFWav2Vec2Model`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

        >>> from transformers import Wav2Vec2Model, Wav2Vec2Config

        >>> # Initializing a Wav2Vec2 fill_later style configuration
        >>> configuration = Wav2Vec2Config()

        >>> # Initializing a model from the fill_later style configuration
        >>> model = Wav2Vec2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "wav2vec2"

    def __init__(
        self,
        hidden_size=768,  # encoder_embed_dim
        feat_extract_layer_norm="group_norm",  # extractor_mode default => group_norm
        feat_extract_dropout=0.0,  # hard-coded
        feat_extract_activation="gelu",  # hard-coded
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # conv_feature_layers [0]
        conv_stride=(10, 2, 2, 2, 2, 2, 2),  # conv_feature_layers [2]
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # conv_feature_layers [1]
        conv_bias=False,
        num_conv_pos_embeddings=128,  # conv_pos
        num_conv_pos_embedding_groups=16,  # conv_pos_groups
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        intermediate_size=3072,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        vocab_size=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.feat_extract_layer_norm = feat_extract_layer_norm
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect."
                "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`,"
                f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )
