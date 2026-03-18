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


import numpy as np
from transformers import Wav2Vec2BertConfig
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class Xcodec2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`Xcodec2Model`]. It is used to instantiate a
    Xcodec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size for the audio encoder model.
        downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2, 4, 4, 5]`):
            Ratios for downsampling in the encoder.
        decoder_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size for the audio decoder model.
        semantic_model_config (`Union[Dict, Wav2Vec2BertConfig]`, *optional*):
            An instance of the configuration object for the semantic (Wav2Vec2BertConfig) model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for the model.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key value heads for the model.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        resnet_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate for the ResNet blocks in the decoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for the attention layer.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for RMS normalization.
        head_dim (`int`, *optional*, defaults to 64):
            Head dimension for the model.
        quantization_dim (`int`, *optional*, defaults to 2048):
            Dimension for the VQ codebook.
        quantization_levels (`list[int]`, *optional*, defaults to `[4, 4, 4, 4, 4, 4, 4, 4]`):
            Levels for the codebook.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
    """

    model_type = "xcodec2"

    sub_configs = {
        "semantic_model_config": Wav2Vec2BertConfig,
    }

    def __init__(
        self,
        encoder_hidden_size=48,
        downsampling_ratios=[2, 2, 4, 4, 5],
        decoder_hidden_size=1024,
        semantic_model_config=None,
        initializer_range=0.02,
        sampling_rate=16000,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=12,
        resnet_dropout=0.1,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        head_dim=64,
        quantization_dim=2048,
        quantization_levels=[4, 4, 4, 4, 4, 4, 4, 4],
        max_position_embeddings=4096,
        rope_parameters=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(semantic_model_config, dict):
            semantic_model_config["model_type"] = semantic_model_config.get("model_type", "wav2vec2-bert")
            semantic_model_config = CONFIG_MAPPING[semantic_model_config["model_type"]](**semantic_model_config)
        elif semantic_model_config is None:
            semantic_model_config = CONFIG_MAPPING["wav2vec2-bert"]()
        self.semantic_model_config = semantic_model_config

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.initializer_range = initializer_range
        self.sampling_rate = sampling_rate
        self.decoder_hidden_size = decoder_hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.resnet_dropout = resnet_dropout
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else {"rope_theta": 10000.0, "rope_type": "default"}
        )
        self.quantization_dim = quantization_dim
        self.quantization_levels = quantization_levels

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.downsampling_ratios))

    @property
    def n_fft(self) -> int:
        return self.hop_length * 4

    @property
    def hidden_size(self) -> int:
        # NOTE: for modular usage of LlamaAttention
        return self.decoder_hidden_size
    

__all__ = ["Xcodec2Config"]
