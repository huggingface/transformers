# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Clip model configuration """

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "clip_ViT_B_32": "https://huggingface.co/clip_ViT_B_32/resolve/main/config.json",
    # See all Clip models at https://huggingface.co/models?filter=clip
}


class ClipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ClipModel`. It is used to
    instantiate an Clip model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Clip `clip_ViT_B_32
    <https://huggingface.co/clip_ViT_B_32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 49408):
            Vocabulary size of the Clip text model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.ClipModel`. Defines the different tokens
            that can be represented by the `inputs_ids` passed to the forward method of
            :class:`~transformers.ClipTextModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

        >>> from transformers import ClipTextModel, ClipTextConfig

        >>> # Initializing a ClipTextModel with clip_ViT_B_32 style configuration
        >>> configuration = ClipConfig()

        >>> # Initializing a model from the clip_ViT_B_32 style configuration
        >>> model = ClipTextModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "clip"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        output_dim=512,
        dropout=0.0,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        attention_dropout=0.0,
        initializer_range=0.02,  # TODO(PS): this should be changed
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.gradient_checkpointing = gradient_checkpointing


class ClipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ClipModel`. It is used to
    instantiate an Clip model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Clip `clip_ViT_B_32
    <https://huggingface.co/clip_ViT_B_32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`32`):
            The size (resolution) of each patch.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example::

        >>> from transformers import ClipVisionModel, ClipTextConfig

        >>> # Initializing a ClipVisionModel with clip_ViT_B_32 style configuration
        >>> configuration = ClipConfig()

        >>> # Initializing a model from the clip_ViT_B_32 style configuration
        >>> model = ClipVisionModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "clip"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        output_dim=512,
        dropout=0.0,
        num_hidden_layers=12,
        num_attention_heads=12,
        patch_size=32,
        image_size=224,
        initializer_range=0.02,
        attention_dropout=0.0,
        layer_norm_eps=1e-5,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing


class ClipConfig(PretrainedConfig):
    r"""
    :class:`~transformers.ClipConfig` is the configuration class to store the configuration of a
    :class:`~transformers.ClipModel`. It is used to instantiate CLIP model according to the specified arguments,
    defining the text model and vision model configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        output_dim: (:obj:`int`, `optional`, defaults to 512):
            some explanatiom
        kwargs (`optional`):
            Dictionary of keyword arguments. Notably:

                - **text_config** (:class:`~transformers.ClipTextConfig`, `optional`) -- An instance of a configuration
                  object that defines the text model config.
                - **vision_config** (:class:`~transformers.ClipVisionConfig`, `optional`) -- An instance of a
                  configuration object that defines the vision model config.
    """

    model_type = "clip"
    is_composition = True

    def __init__(self, output_dim=512, **kwargs):
        super().__init__(**kwargs)
        text_config_dict = kwargs.pop("text_config")
        vision_config_dict = kwargs.pop("vision_config")

        self.text_config = ClipTextConfig(**text_config_dict)
        self.vision_config = ClipVisionConfig(**vision_config_dict)

        self.output_dim = output_dim

    @classmethod
    def from_text_vision_configs(cls, text_config: ClipTextConfig, vision_config: ClipVisionConfig, **kwargs):
        r"""
        Instantiate a :class:`~transformers.ClipConfig` (or a derived class) from clip text model configuration and
        clip vision model configuration.

        Returns:
            :class:`ClipConfig`: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
