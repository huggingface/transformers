# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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


from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


class EncoderDecoderConfig(PretrainedConfig):
    r"""
    [`EncoderDecoderConfig`] is the configuration class to store the configuration of a [`EncoderDecoderModel`]. It is
    used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> config_encoder = BertConfig()
    >>> config_decoder = BertConfig()

    >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a Bert2Bert model (with random weights) from the google-bert/bert-base-uncased style configurations
    >>> model = EncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
    >>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""

    model_type = "encoder-decoder"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because "
                f"both `encoder` and `decoder` sub-configurations were not passed, only {kwargs}"
            )
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


__all__ = ["EncoderDecoderConfig"]
