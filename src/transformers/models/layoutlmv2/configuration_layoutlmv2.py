# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" LayoutLMv2 model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LayoutLMv2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/config.json",
    "layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/config.json",
    # See all LayoutLMv2 models at https://huggingface.co/models?filter=layoutlmv2
}


class LayoutLMv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LayoutLMv2Model`.
    It is used to instantiate an LayoutLMv2 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the LayoutLMv2 `microsoft/layoutlmv2-base-uncased <https://huggingface.co/microsoft/layoutlmv2-base-uncased>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the LayoutLMv2 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.LayoutLMv2Model` or
            :class:`~transformers.TFLayoutLMv2Model`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.LayoutLMv2Model` or
            :class:`~transformers.TFLayoutLMv2Model`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
        ...
            ...

        Example::

        >>> from transformers import LayoutLMv2Model, LayoutLMv2Config

        >>> # Initializing a LayoutLMv2 microsoft/layoutlmv2-base-uncased style configuration
        >>> configuration = LayoutLMv2Config()

        >>> # Initializing a model from the microsoft/layoutlmv2-base-uncased style configuration
        >>> model = LayoutLMv2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "layoutlmv2"

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
        pad_token_id=0,
        gradient_checkpointing=False,
        max_2d_position_embeddings=1024,
        max_rel_pos=128,
        rel_pos_bins=32,
        fast_qkv=True,
        max_rel_2d_pos=256,
        rel_2d_pos_bins=64,
        convert_sync_batchnorm=True,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        has_visual_segment_embedding=False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
