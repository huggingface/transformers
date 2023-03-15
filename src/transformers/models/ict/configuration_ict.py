# coding=utf-8
# Copyright 2023 Authors at City University of Hong Kong, Microsoft Cloud + AI, 
# The HuggingFace Inc. team. All rights reserved.
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
""" ICT model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

import copy

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

ICT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sheonhan/ict-imagenet-32": "https://huggingface.co/sheonhan/ict-imagenet-32/resolve/main/config.json",
}


class ICTGuidedUpsamplerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ICTGuidedUpsampler`]. It is used to instantiate an 
    [`ICTGuidedUpsampler`] model according to the specified arguments, defining the model architecture. Instantiating a 
    configuration with the defaults will yield a similar configuration to that of the [ICTGuidedUpsampler model trained with the ImageNet dataset](https://huggingface.co/sheonhan/ict-imagenet-32).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        learning_rate (`float`, *optional*, defaults to 0.0001):
            The desired learning rate of the [`ICTGuidedUpsampler`] model.
        dis_gen_learning_rate (`float`, *optional*, defaults to 0.1):
            The discriminator/generator learning rate ratio.
        adam_beta1 (`float`, *optional*, defaults to 0.0):
            The beta1 to use in Adam.
        adam_beta2 (`float`, *optional*, defaults to 0.9):
            The beta2 to use in Adam.
        batch_size (`int`, *optional*, defaults to 64):
            The batch size for training.
        input_size (`int`, *optional*, defaults to 256):
            The input image size for training. (0 for the original size.)
        max_iteration (`float`, *optional*, defaults to 5e7):
            The maximum number of iterations to train the model.
        residual_blocks (`int`, *optional*, defaults to 8):
            The number of residual blocks.
        l1_loss_weight (`float`, *optional*, defaults to 1.0):
            The weight of the L1 loss function.
        style_loss_weight (`float`, *optional*, defaults to 250.):
            The weight of the style loss module.
        content_loss_weight (`float`, *optional*, defaults to 0.1):
            The weight of the content loss module.
        inpaint_adv_loss_weight (`float`, *optional*, defaults to 0.1):
            The weight of the adversial loss module.
        gan_loss (`str`, *optional*, defaults to`"nsgan"`):
            GAN's loss function can be either "nsgan", "lsgan", or "hinge". 


    Example:

    ```python
    >>> from transformers import ICTGuidedUpsamplerConfig, ICTGuidedUpsampler

    >>> # Initializing a ICT ict-imagenet-32 style configuration
    >>> configuration = ICTGuidedUpsamplerConfig()

    >>> # Initializing a model (with random weights) from the ict-imagenet-32 style configuration
    >>> upsampler = ICTGuidedUpsampler(configuration)

    >>> # Accessing the model configuration
    >>> configuration = upsampler.config
    ```"""
    model_type = "ict-guided-upsampler"

    def __init__(
        self,
        learning_rate=0.0001,
        dis_gen_learning_rate=0.1,
        adam_beta1=0.0,
        adam_beta2=0.9,
        batch_size=64,
        input_size=256,
        max_iteration=5e7,
        residual_blocks=8,
        l1_loss_weight=1.0,
        style_loss_weight=25.0,
        content_loss_weight=0.1,
        inpaint_adv_loss_weight=0.1,
        gan_loss="nsgan",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.dis_gen_learning_rate = dis_gen_learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.batch_size = batch_size
        self.input_size = input_size
        self.max_iteration = max_iteration
        self.residual_blocks = residual_blocks
        self.l1_loss_weight = l1_loss_weight
        self.style_loss_weight = style_loss_weight
        self.content_loss_weight = content_loss_weight
        self.inpaint_adv_loss_weight = inpaint_adv_loss_weight
        self.gan_loss = gan_loss

class ICTTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ICTTransformer`]. It is used to instantiate an ICT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [ICT model trained with the ImageNet dataset](https://huggingface.co/sheonhan/ict-imagenet-32).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the ICT model. Defines the number of different tokens that can be represented by the
            `pixel_values` passed when calling [`ICTTransformer`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 35):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function (can be one of the activation functions defined in src/transformers/activations.py).
            Defaults to "quick_gelu".
        embedding_dropout_prob (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        residual_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `32`):
            The size (resolution) of each image.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.

    Example:

    ```python
    >>> from transformers import ICTTransformerConfig, ICTTransformer

    >>> # Initializing a ICT ict-imagenet-32 style configuration
    >>> configuration = ICTTransformerConfig()

    >>> # Initializing a model (with random weights) from the ict-imagenet-32 style configuration
    >>> model = ICTTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ict-transformer"

    def __init__(
        self,
        vocab_size=512,
        hidden_size=768,
        num_hidden_layers=35,
        num_attention_heads=8,
        intermediate_size=4096,
        activation_function="gelu",
        embedding_dropout_prob=0.0,
        residual_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=32,
        num_channels=3,
        qkv_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.embedding_dropout_prob = embedding_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.block_size = self.image_size * self.image_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias

class ICTConfig(PretrainedConfig):
    r"""
    [`ICTConfig`] is the configuration class to store the configuration of a [`ICTMdel`]. It is used to
    instantiate an ICT model according to the specified arguments, defining the transformer model and guided upsampler configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ICT
    [sheonhan/ict-imagenet-32](https://huggingface.co/sheonhan/ict-imagenet-32) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        transformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ICTTransformerConfig`].
        guided_upsampler_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ICTGuidedUpsamplerConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.
    Example:
    ```python
    >>> from transformers import ICTConfig, ICTModel
    >>> # Initializing a ICTConfig with sheonhan/ict-imagenet-32 style configuration
    >>> configuration = ICTConfig()
    >>> # Initializing a ICTModel (with random weights) from the sheonhan/ict-imagenet-32 style configuration
    >>> model = ICTModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> # We can also initialize a ICTConfig from a ICTTransformerConfig and a ICTGuidedUpsamplerConfig
    >>> from transformers import ICTTransformerConfig, ICTGuidedUpsamplerConfig
    >>> # Initializing ALIGN Text and Vision configurations
    >>> config_transformer = ICTTransformerConfig()
    >>> config_guided_upsampler = ICTGuidedUpsamplerConfig()
    >>> config = ICTConfig.from_text_guided_upsampler_configs(config_transformer, config_guided_upsampler)
    ```"""

    model_type = "ict"

    def __init__(
        self,
        transformer_config=None,
        guided_upsampler_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if transformer_config is None:
            transformer_config = {}
            logger.info("transformer_config is None. Initializing the ICTTransformerConfig with default values.")

        if guided_upsampler_config is None:
            guided_upsampler_config = {}
            logger.info("guided_upsampler_config is None. Initializing the ICTGuidedUpsamplerConfig with default values.")

        self.transformer_config = ICTTransformerConfig(**transformer_config)
        self.guided_upsampler_config = ICTGuidedUpsamplerConfig(**guided_upsampler_config)


    @classmethod
    def from_transformer_and_guided_upsampler_configs(cls, transformer_config: ICTTransformerConfig, guided_upsampler_config: ICTGuidedUpsamplerConfig, **kwargs):
        r"""
        Instantiate a [`ICTConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.
        Returns:
            [`ICTConfig`]: An instance of a configuration object
        """

        return cls(transformer_config=transformer_config.to_dict(), guided_upsampler_config=guided_upsampler_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["transformer_config"] = self.transformer_config.to_dict()
        output["guided_upsampler_config"] = self.guided_upsampler_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
