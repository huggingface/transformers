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
""" VibeVoice Semantic Tokenizer model configuration"""

from typing import Optional
import numpy as np

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceSemanticTokenizerModel`]. It is used to
    instantiate a VibeVoice semantic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    semantic tokenizer of [VibeVoice](https://hf.co/papers/2508.19205).

    Args:
    TODO list and remove type hints
    
    """
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        sample_latent: bool = False,
        pad_mode: str = 'constant',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[list[int]] = [8,5,5,4,2,2],
        encoder_depths: list[int] = [3,3,3,3,3,3,8],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.sample_latent = sample_latent

        # common parameters
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths

    @property
    def hop_length(self) -> int:
        return np.prod(self.encoder_ratios)

__all__ = ["VibeVoiceSemanticTokenizerConfig"]
