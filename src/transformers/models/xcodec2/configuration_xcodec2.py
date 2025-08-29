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
"""Xcodec2 model configuration"""


import math
from typing import Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging

import numpy as np

from transformers import AutoConfig, Wav2Vec2BertConfig


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
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size for the audio decoder model.
        semantic_model_config (`Union[Dict, Wav2Vec2BertConfig]`, *optional*):
            An instance of the configuration object for the semantic (Wav2Vec2BertConfig) model.

        
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        use_vocos (`bool`, *optional*, defaults to `True`):
            Whether to use VOCOS.
        hidden_size (`int`, *optional*, defaults to 1024):
            Intermediate representation dimension.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for the model.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key value heads for the model.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for the attention layer.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for RMS normalization.
        head_dim (`int`, *optional*, defaults to 64):
            Head dimension for the model.
        dilations (`tuple`, *optional*, defaults to `[1, 3, 9]`):
            Dilation values for the model.
        depth (`int`, *optional*, defaults to 12):
            Depth for the model.
        hop_length (`int`, *optional*, defaults to 320):
            The hop length for STFT.
        vq_dim (`int`, *optional*, defaults to 2048):
            Dimension for the VQ codebook.
        vq_commit_weight (`float`, *optional*, defaults to 0.25):
            Commit weight for the VQ.
        vq_weight_init (`bool`, *optional*, defaults to `False`):
            Whether to initialize VQ weights.
        vq_full_commit_loss (`bool`, *optional*, defaults to `False`):
            Whether to use full commit loss for the VQ.
        codebook_size (`int`, *optional*, defaults to 16384):
            Number of discrete codes that make up VQVAE.
        codebook_dim (`int`, *optional*, defaults to 16):
            Dimension of the codebook vectors. If not defined, uses `hidden_size`.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the rotary position embeddings.
    """

    model_type = "xcodec2"

    sub_configs = {
        "semantic_model_config": Wav2Vec2BertConfig,
    }

    def __init__(
        self,
        encoder_hidden_size: int = 1024,
        downsampling_ratios: tuple = [2, 2, 4, 4, 5],
        decoder_hidden_size: int = 1024,
        semantic_model_config: Union[dict, Wav2Vec2BertConfig] = None,

        initializer_range: float = 0.02,
        sampling_rate: int = 16000,
        use_vocos: bool = True,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        num_hidden_layers: int = 12,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 64,
        dilations: tuple = [1, 3, 9],
        depth: int = 12,
        vq_dim: int = 2048,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 16384,
        codebook_dim: int = 16,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.downsampling_ratios = downsampling_ratios

        if semantic_model_config is None:
            self.semantic_model_config = Wav2Vec2BertConfig()
        elif isinstance(semantic_model_config, dict):
            if "_name_or_path" in semantic_model_config:
                # If the config is a path, load it using AutoConfig
                self.semantic_model_config = AutoConfig.from_pretrained(semantic_model_config["_name_or_path"])
            else:
                # assume HubertConfig as probably created from scratch
                logger.warning(
                    "Could not determine semantic model type from config architecture. Defaulting to `Wav2Vec2BertConfig`."
                )
                self.semantic_model_config = Wav2Vec2BertConfig(**semantic_model_config)
        elif isinstance(semantic_model_config, Wav2Vec2BertConfig):
            self.semantic_model_config = semantic_model_config
        else:
            raise ValueError(
                f"semantic_model_config must be a dict or Wav2Vec2BertConfig instance, but got {type(semantic_model_config)}"
            )

        self.initializer_range = initializer_range
        self.sampling_rate = sampling_rate
        self.use_vocos = use_vocos
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.head_dim = head_dim
        self.dilations = dilations
        self.depth = depth
        # single codebook is main feature of xcodec2
        self.num_quantizers = 1
        self.vq_dim = vq_dim
        self.vq_commit_weight = vq_commit_weight
        self.vq_weight_init = vq_weight_init
        self.vq_full_commit_loss = vq_full_commit_loss
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sampling_rate / self.hop_length)
    
    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_model_config.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.encoder_hidden_size + self.decoder_hidden_size + self.semantic_hidden_size + self.hidden_size

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.downsampling_ratios))
    

__all__ = ["Xcodec2Config"]
