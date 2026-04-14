# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/nougat-base")
@strict
class NougatConfig(PreTrainedConfig):
    r"""
    encoder (`dict | PreTrainedConfig`):
        The config object or dictionary of the encoder backbone.
    decoder (`dict | PreTrainedConfig`):
        The config object or dictionary of the decoder backbone.

    Examples:

    ```python
    >>> from transformers import NougatConfig, VisionEncoderDecoderModel

    >>> # Initializing a Nougat configuration
    >>> config = NougatConfig()

    >>> # Initializing a VisionEncoderDecoder model (with random weights) from a Nougat configurations
    >>> model = VisionEncoderDecoderModel(config=config)
    ```"""

    model_type = "nougat"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}

    encoder: dict | PreTrainedConfig | None = None
    decoder: dict | PreTrainedConfig | None = None
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if self.encoder is None or self.decoder is None:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because "
                f"one of both `encoder` or `decoder` sub-configurations is not passed."
            )

        if isinstance(self.encoder, dict):
            encoder_model_type = self.encoder.pop("model_type")
            self.encoder = AutoConfig.for_model(encoder_model_type, **self.encoder)
        if isinstance(self.decoder, dict):
            decoder_model_type = self.decoder.pop("model_type")
            self.decoder = AutoConfig.for_model(decoder_model_type, **self.decoder)
        super().__post_init__(**kwargs)

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PreTrainedConfig, decoder_config: PreTrainedConfig, **kwargs
    ) -> PreTrainedConfig:
        r"""
        Instantiate a [`VisionEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`VisionEncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


__all__ = ["NougatConfig"]
