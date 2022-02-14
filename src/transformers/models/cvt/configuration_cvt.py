# coding=utf-8
# Copyright 2022 AnugunjNaman and The HuggingFace Inc. team. All rights reserved.
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
""" Cvt model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CVT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "msft/cvt": "https://huggingface.co/msft/cvt/resolve/main/config.json",
    # See all Cvt models at https://huggingface.co/models?filter=cvt
}


class CvtConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~CvtModel`].
    It is used to instantiate an Cvt model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the Cvt [msft/cvt](https://huggingface.co/msft/cvt) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Cvt model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~CvtModel`] or
            [`~TFCvtModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~CvtModel`] or
            [`~TFCvtModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import CvtModel, CvtConfig

    >>> # Initializing a Cvt msft/cvt style configuration
    >>> configuration = CvtConfig()

    >>> # Initializing a model from the msft/cvt style configuration
    >>> model = CvtModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "cvt"
    

    def __init__(
        self,
        num_labels = 1000,
        num_channels = 3,
        num_stages = 3,
        patch_sizes = [7, 3, 3],
        patch_stride = [4, 2, 2],
        patch_padding = [2, 1, 1],
        embed_dim = [64, 192, 384],
        num_heads = [1, 3, 6],
        depth = [1, 2, 10],
        mlp_ratio = [4.0, 4.0, 4.0],
        attention_drop_rate = [0.0, 0.0, 0.0],
        drop_rate = [0.0, 0.0, 0.0],
        drop_path_rate = [0.0, 0.0, 0.1],
        qkv_bias = [True, True, True],
        cls_token = [False, False, False],
        pos_embed = [False, False, False],
        qkv_projection_method = ['dw_bn', 'dw_bn', 'dw_bn'],
        kernel_qkv = [3, 3, 3],
        padding_kv = [1, 1, 1],
        stride_kv = [2, 2, 2],
        padding_q = [1, 1, 1],
        stride_q = [1, 1, 1],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.patch_sizes = patch_sizes
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.attention_drop_rate = attention_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.qkv_projection_method = qkv_projection_method
        self.kernel_qkv = kernel_qkv
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.padding_q = padding_q
        self.stride_q = stride_q
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

    