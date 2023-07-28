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
""" Kosmos-2 model configuration"""

import copy
from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/kosmos-2-patch14-224": (
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/config.json"
    ),
    # See all Kosmos-2 models at https://huggingface.co/models?filter=kosmos-2
}


class Kosmos2TextConfig(PretrainedConfig):

    model_type = "kosmos_2_text_model"

    def __init__(
        self,
        vocab_size=65037,
        max_position_embeddings=2048,
        embed_dim=2048,
        layers=24,
        ffn_dim=8192,
        attention_heads=32,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
        scale_embedding=True,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        # TODO: should be in generation_config
        no_repeat_ngram_size=3,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.attention_heads = attention_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # TODO: should be in generation_config
        self.no_repeat_ngram_size = no_repeat_ngram_size


class Kosmos2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2Model`]. It is used to instantiate an Kosmos-2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Kosmos-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 64):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "kosmos-2"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `Kosmos2TextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `Kosmos2VisionConfig` with default values.")

        self.text_config = Kosmos2TextConfig(**text_config)
        # TODO: Remove this by adding Kosmos2VisionConfig
        from transformers import CLIPVisionConfig as Kosmos2VisionConfig
        self.vision_config = Kosmos2VisionConfig(**vision_config)

        self.latent_query_num = latent_query_num

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
