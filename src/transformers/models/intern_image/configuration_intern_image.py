# coding=utf-8
# Copyright 2022 OpenGVLab and The HuggingFace Inc. team. All rights reserved.
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
""" internimage model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

INTERN_IMAGE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "OpenGVLab/internimage": "https://huggingface.co/OpenGVLab/internimage/resolve/main/config.json",
    # See all internimage models at https://huggingface.co/models?filter=intern_image
}


class InternImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~InternImageModel`].
    It is used to instantiate an internimage model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the internimage [OpenGVLab/internimage](https://huggingface.co/OpenGVLab/internimage) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        # vocab_size (`int`, *optional*, defaults to 30522):
        #     Vocabulary size of the internimage model. Defines the number of different tokens that can be represented by the
        #     `inputs_ids` passed when calling [`~InternImageModel`] or
        #     [`~TFInternImageModel`].
        # hidden_size (`int`, *optional*, defaults to 768):
        #     Dimension of the encoder layers and the pooler layer.
        # num_hidden_layers (`int`, *optional*, defaults to 12):
        #     Number of hidden layers in the Transformer encoder.
        # num_attention_heads (`int`, *optional*, defaults to 12):
        #     Number of attention heads for each attention layer in the Transformer encoder.
        # intermediate_size (`int`, *optional*, defaults to 3072):
        #     Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        # hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        #     The non-linear activation function (function or string) in the encoder and pooler.
        #     If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        # hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        #     The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        # attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        #     The dropout ratio for the attention probabilities.
        # max_position_embeddings (`int`, *optional*, defaults to 512):
        #     The maximum sequence length that this model might ever be used with.
        #     Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        # type_vocab_size (`int`, *optional*, defaults to 2):
        #     The vocabulary size of the `token_type_ids` passed when calling [`~InternImageModel`] or
        #     [`~TFInternImageModel`].
        # initializer_range (`float`, *optional*, defaults to 0.02):
        #     The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        # layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        #     The epsilon used by the layer normalization layers.
        # use_cache (`bool`, *optional*, defaults to `True`):
        #     Whether or not the model should return the last key/values attentions (not used by all models). Only
        #     relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import InternImageModel, InternImageConfig

    >>> # Initializing a internimage OpenGVLab/internimage style configuration
    >>> configuration = InternImageConfig()

    >>> # Initializing a model from the OpenGVLab/internimage style configuration
    >>> model = InternImageModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "intern_image"

    def __init__(
        self,
        core_op="DCNv3_pytorch",
        channels=64,
        depths=(4, 4, 18, 4),
        groups=(4, 8, 16, 32),
        num_classes=1000,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="linear",
        act_layer="GELU",
        norm_layer="LN",
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        cls_scale=1.5,
        with_cp=False,
        **kwargs,
    ):
        self.core_op = core_op
        self.channels = channels
        self.depths = depths
        self.groups = groups
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.drop_path_type = drop_path_type
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.layer_scale = layer_scale
        self.offset_scale = offset_scale
        self.post_norm = post_norm
        self.cls_scale = cls_scale
        self.with_cp = with_cp
        super().__init__(**kwargs)
