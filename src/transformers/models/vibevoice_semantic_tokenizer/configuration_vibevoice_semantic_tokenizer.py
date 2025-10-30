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

import numpy as np

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceSemanticTokenizerModel`]. It is used to
    instantiate a VibeVoice semantic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    semantic tokenizer of [VibeVoice-1.5B](https://hf.co/microsoft/VibeVoice-1.5B).

    Args:
        channels (`int`, *optional*, defaults to `1`):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to `128`):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to `7`):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to `1e-5`):
            Epsilon value for RMSNorm layers.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in convolution and feed-forward layers.
        layer_scale_init_value (`float`, *optional*, defaults to `1e-6`):
            Initial value for layer scaling.
        weight_init_value (`float`, *optional*, defaults to `1e-2`):
            Standard deviation for weight initialization.
        n_filters (`int`, *optional*, defaults to `32`):
            Number of filters in initial convolutional layer, and doubles after each downsampling.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each layer.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to `4`):
            Expansion factor for feed-forward networks.

    """
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels=1,
        hidden_size=128,
        kernel_size=7,
        rms_norm_eps=1e-5,
        bias=True,
        layer_scale_init_value=1e-6,
        weight_init_value=1e-2,
        n_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.kernel_size = kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.bias = bias
        self.layer_scale_init_value = layer_scale_init_value
        self.ffn_expansion = ffn_expansion
        self.weight_init_value = weight_init_value
        self.n_filters = n_filters
        self.downsampling_ratios = downsampling_ratios
        self.depths = depths

    @property
    def hop_length(self) -> int:
        return np.prod(self.downsampling_ratios)

__all__ = ["VibeVoiceSemanticTokenizerConfig"]
