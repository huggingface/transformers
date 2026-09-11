# Copyright 2021 The HuggingFace Inc. team.
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

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring, logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="")
@strict
class SpeechEncoderDecoderConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

    >>> # Initializing a Wav2Vec2 & BERT style configuration
    >>> config = SpeechEncoderDecoderConfig()

    >>> # Initializing a Wav2Vec2Bert model from a Wav2Vec2 & google-bert/bert-base-uncased style configurations
    >>> model = SpeechEncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = SpeechEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = SpeechEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""

    model_type = "speech-encoder-decoder"
    sub_configs_defaults = {
        "encoder": SubConfigSpec(config_class=AutoConfig, model_type="wav2vec2"),
        "decoder": SubConfigSpec(
            config_class=AutoConfig, model_type="bert", init_kwargs={"is_decoder": True, "add_cross_attention": True}
        ),
    }

    encoder: PreTrainedConfig | dict | None = None
    decoder: PreTrainedConfig | dict | None = None
    is_encoder_decoder: bool | None = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PreTrainedConfig, decoder_config: PreTrainedConfig, **kwargs
    ) -> PreTrainedConfig:
        r"""
        Instantiate a [`SpeechEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`SpeechEncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


__all__ = ["SpeechEncoderDecoderConfig"]
