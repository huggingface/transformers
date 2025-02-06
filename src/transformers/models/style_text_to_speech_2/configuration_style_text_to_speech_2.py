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


class StyleTextToSpeech2SubModelConfig(PretrainedConfig):
    def __init__(
        self, 
        vocab_size=178,
        hidden_size=512,
        style_hidden_size=128,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.style_hidden_size = style_hidden_size
        super().__init__(**kwargs)


class StyleTextToSpeech2AcousticTextEncoderConfig(StyleTextToSpeech2SubModelConfig):
    model_type = "acoustic_text_encoder"
    base_config_key = "acoustic_text_encoder_config"

    def __init__(
        self,
        num_hidden_layers=3,
        dropout=0.2,
        kernel_size=5,
        leaky_relu_slope=0.2,
        **kwargs
    ):
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.leaky_relu_slope = leaky_relu_slope

        super().__init__(**kwargs)


class StyleTextToSpeech2ProsodicTextEncoderConfig(StyleTextToSpeech2SubModelConfig):
    model_type = "prosodic_encoder"
    base_config_key = "prosodic_encoder_config"

    def __init__(
        self,
        sub_model_type="albert",
        bert_vocab_size=178,
        bert_hidden_size=768,
        num_attention_heads=12,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1,
        bert_config=None,
        **kwargs
    ):
        if bert_config is None:
            self.bert_config = CONFIG_MAPPING[sub_model_type](
                vocab_size = bert_vocab_size,
                hidden_size = bert_hidden_size,
                num_attention_heads = num_attention_heads,
                intermediate_size = intermediate_size,
                max_position_embeddings = max_position_embeddings,
                dropout = dropout,
            )
        else:
            self.bert_config = CONFIG_MAPPING[sub_model_type](**bert_config)
    
        self.bert_hidden_size = bert_hidden_size
        self.dropout = dropout

        super().__init__(**kwargs)


class StyleTextToSpeech2DurationEncoderConfig(StyleTextToSpeech2SubModelConfig):
    model_type = "duration_encoder"
    base_config_key = "duration_encoder_config"

    def __init__(
        self,
        num_layers=3,
        dropout=0.2,
        max_duration=50,
        **kwargs
    ):
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_duration = max_duration

        super().__init__(**kwargs)


class StyleTextToSpeech2DurationPredictorConfig(StyleTextToSpeech2SubModelConfig):
    model_type = "duration_predictor"
    base_config_key = "duration_predictor_config"

    def __init__(
        self,
        max_duration=50,
        **kwargs
    ):
        self.max_duration = max_duration
        
        super().__init__(**kwargs)


class StyleTextToSpeech2ProsodyPredictorConfig(StyleTextToSpeech2SubModelConfig):
    model_type = "prosody_predictor"
    base_config_key = "prosody_predictor_config"

    def __init__(
        self,
        dropout=0.2,
        **kwargs
    ):
        self.dropout = dropout

        super().__init__(**kwargs)


class StyleTextToSpeech2DecoderConfig(StyleTextToSpeech2SubModelConfig):
    model_type = "decoder"
    base_config_key = "decoder_config"

    def __init__(
        self,
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[10, 6],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[20, 12],
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
        **kwargs
    ):
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        super().__init__(**kwargs)


class StyleTextToSpeech2Config(PretrainedConfig):
    model_type = "style_text_to_speech_2"
    sub_configs = {
        "acoustic_text_encoder_config": StyleTextToSpeech2AcousticTextEncoderConfig,
        "prosodic_text_encoder_config": StyleTextToSpeech2ProsodicTextEncoderConfig,
        "duration_encoder_config": StyleTextToSpeech2DurationEncoderConfig,
        "duration_predictor_config": StyleTextToSpeech2DurationPredictorConfig,
        "prosody_predictor_config": StyleTextToSpeech2ProsodyPredictorConfig,
        "decoder_config": StyleTextToSpeech2DecoderConfig,
    }

    def __init__(
        self,
        acoustic_text_encoder_config: Dict=None,
        prosodic_text_encoder_config: Dict=None,
        duration_encoder_config: Dict=None,
        duration_predictor_config: Dict=None,
        prosody_predictor_config: Dict=None,
        decoder_config: Dict=None,
        initializer_range=0.02,
        **kwargs,
    ):
        if acoustic_text_encoder_config is None:
            acoustic_text_encoder_config = {}
            logger.info("acoustic_text_encoder_config is None. initializing the acoustic text encoder with default values.")

        if prosodic_text_encoder_config is None:
            prosodic_text_encoder_config = {}
            logger.info("prosodic_text_encoder_config is None. initializing the prosodic text encoder with default values.")

        if duration_encoder_config is None:
            duration_encoder_config = {}
            logger.info("duration_encoder_config is None. initializing the duration encoder with default values.")

        if duration_predictor_config is None:
            duration_predictor_config = {}
            logger.info("duration_predictor_config is None. initializing the duration predictor with default values.")
        
        if prosody_predictor_config is None:
            prosody_predictor_config = {}
            logger.info("prosody_predictor_config is None. initializing the prosody predictor with default values.")

        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. initializing the decoder with default values.")

        self.acoustic_text_encoder_config = StyleTextToSpeech2AcousticTextEncoderConfig(**acoustic_text_encoder_config)
        self.prosodic_text_encoder_config = StyleTextToSpeech2ProsodicTextEncoderConfig(**prosodic_text_encoder_config)
        self.duration_encoder_config = StyleTextToSpeech2DurationEncoderConfig(**duration_encoder_config)
        self.duration_predictor_config = StyleTextToSpeech2DurationPredictorConfig(**duration_predictor_config)
        self.prosody_predictor_config = StyleTextToSpeech2ProsodyPredictorConfig(**prosody_predictor_config)
        self.decoder_config = StyleTextToSpeech2DecoderConfig(**decoder_config)

        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    @classmethod
    def from_sub_model_configs(
        cls,
        acoustic_text_encoder_config: StyleTextToSpeech2AcousticTextEncoderConfig,
        prosodic_text_encoder_config: StyleTextToSpeech2ProsodicTextEncoderConfig,
        duration_encoder_config: StyleTextToSpeech2DurationEncoderConfig,
        duration_predictor_config: StyleTextToSpeech2DurationPredictorConfig,
        prosody_predictor_config: StyleTextToSpeech2ProsodyPredictorConfig,
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
            prosodic_text_encoder_config=prosodic_text_encoder_config.to_dict(),
            duration_encoder_config=duration_encoder_config.to_dict(),
            duration_predictor_config=duration_predictor_config.to_dict(),
            prosody_predictor_config=prosody_predictor_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
            **kwargs,
        )


__all__ = ["StyleTextToSpeech2AcousticTextEncoderConfig", "StyleTextToSpeech2ProsodicTextEncoderConfig", "StyleTextToSpeech2DurationEncoderConfig", "StyleTextToSpeech2DurationPredictorConfig", "StyleTextToSpeech2ProsodyPredictorConfig", "StyleTextToSpeech2DecoderConfig", "StyleTextToSpeech2Config"]