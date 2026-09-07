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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="")
@strict
class EncoderDecoderConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> config = EncoderDecoderConfig()

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

    encoder: PreTrainedConfig | dict | None = None
    decoder: PreTrainedConfig | dict | None = None
    pad_token_id: int | None = None
    decoder_start_token_id: int | None = None
    is_encoder_decoder: bool | None = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder, dict):
            self.encoder["model_type"] = self.encoder.get("model_type", "bert")
            self.encoder = CONFIG_MAPPING[self.encoder["model_type"]](**self.encoder)
        elif self.encoder is None:
            self.encoder = CONFIG_MAPPING["bert"]()

        if isinstance(self.decoder, dict):
            self.decoder["model_type"] = self.decoder.get("model_type", "bert")
            self.decoder = CONFIG_MAPPING[self.decoder["model_type"]](**self.decoder)
        elif self.decoder is None:
            self.decoder = CONFIG_MAPPING["bert"](is_decoder=True, add_cross_attention=True)

        super().__post_init__(**kwargs)

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PreTrainedConfig, decoder_config: PreTrainedConfig, **kwargs
    ) -> PreTrainedConfig:
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
