# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Dict
import copy
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class StyleTextToSpeech2AcousticTextEncoderConfig(PretrainedConfig):
    model_type = "acoustic_text_encoder"
    base_config_key = "acoustic_text_encoder_config"

    def __init__(
        self,
        hidden_size=512,
        vocab_size=178,
        num_hidden_layers=3,
        dropout=0.2,
        kernel_size=5,
        leaky_relu_slope=0.2,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.leaky_relu_slope = leaky_relu_slope

        super().__init__(**kwargs)


class StyleTextToSpeech2PredictorConfig(PretrainedConfig):
    model_type = "predictor"
    base_config_key = "predictor_config"

    def __init__(
        self,
        hidden_size=512,
        style_hidden_size=128,
        vocab_size=178,
        prosodic_text_encoder_sub_model_type="albert",
        prosodic_text_encoder_sub_model_config=None,
        prosodic_text_encoder_hidden_size=768,
        prosodic_text_encoder_num_attention_heads=12,
        prosodic_text_encoder_intermediate_size=2048,
        prosodic_text_encoder_max_position_embeddings=512,
        prosodic_text_encoder_dropout=0.1,
        prosody_encoder_num_layers=3,
        prosody_encoder_dropout=0.2,
        duration_projector_max_duration=50,
        prosody_predictor_dropout=0.2,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.style_hidden_size = style_hidden_size
        self.vocab_size = vocab_size

        if prosodic_text_encoder_sub_model_config is None:
            self.prosodic_text_encoder_sub_model_config = CONFIG_MAPPING[prosodic_text_encoder_sub_model_type](
                vocab_size=vocab_size,
                hidden_size=prosodic_text_encoder_hidden_size,
                num_attention_heads=prosodic_text_encoder_num_attention_heads,
                intermediate_size=prosodic_text_encoder_intermediate_size,
                max_position_embeddings=prosodic_text_encoder_max_position_embeddings,
                dropout=prosodic_text_encoder_dropout,
            )
        else:
            self.prosodic_text_encoder_sub_model_config = CONFIG_MAPPING[prosodic_text_encoder_sub_model_type](**prosodic_text_encoder_sub_model_config)

        self.prosody_encoder_num_layers = prosody_encoder_num_layers
        self.prosody_encoder_dropout = prosody_encoder_dropout
        self.duration_projector_max_duration = duration_projector_max_duration
        self.prosody_predictor_dropout = prosody_predictor_dropout

        super().__init__(**kwargs)


class StyleTextToSpeech2DecoderConfig(PretrainedConfig):
    model_type = "decoder"
    base_config_key = "decoder_config"

    def __init__(
        self,
        hidden_size=512,
        vocab_size=178,
        style_hidden_size=128,
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[10, 6],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[20, 12],
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
        sampling_rate=24000,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.style_hidden_size = style_hidden_size
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.sampling_rate = sampling_rate
    
        super().__init__(**kwargs)


class StyleTextToSpeech2Config(PretrainedConfig):
    model_type = "style_text_to_speech_2"

    def __init__(
        self,
        style_size=256,
        acoustic_text_encoder_config: Dict=None,
        predictor_config: Dict=None,
        decoder_config: Dict=None,
        initializer_range=0.02,
        **kwargs,
    ):
        if acoustic_text_encoder_config is None:
            acoustic_text_encoder_config = {}
            logger.info("acoustic_text_encoder_config is None. Initializing the acoustic text encoder with default values.")

        if predictor_config is None:
            predictor_config = {}
            logger.info("predictor_config is None. Initializing the predictor with default values.")

        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing the decoder with default values.")

        self.style_size = style_size

        self.acoustic_text_encoder_config = StyleTextToSpeech2AcousticTextEncoderConfig(**acoustic_text_encoder_config)
        self.predictor_config = StyleTextToSpeech2PredictorConfig(**predictor_config)
        self.decoder_config = StyleTextToSpeech2DecoderConfig(**decoder_config)

        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    @classmethod
    def from_sub_model_configs(
        cls,
        acoustic_text_encoder_config: StyleTextToSpeech2AcousticTextEncoderConfig,
        predictor_config: StyleTextToSpeech2PredictorConfig,
        decoder_config: StyleTextToSpeech2DecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`StyleTextToSpeech2Config`] (or a derived class) from StyleTextToSpeech2 sub-models configuration.

        Returns:
            [`StyleTextToSpeech2Config`]: An instance of a configuration object
        """
        return cls(
            acoustic_text_encoder_config=acoustic_text_encoder_config.to_dict(),
            predictor_config=predictor_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
            **kwargs,
        )


__all__ = [
    "StyleTextToSpeech2AcousticTextEncoderConfig",
    "StyleTextToSpeech2PredictorConfig",
    "StyleTextToSpeech2DecoderConfig",
    "StyleTextToSpeech2Config"
]