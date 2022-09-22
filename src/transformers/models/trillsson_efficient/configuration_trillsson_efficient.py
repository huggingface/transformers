# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Trillsson_efficient model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TRILLSSON_EFFICIENT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "vumichien/nonsemantic-speech-trillsson3": "https://huggingface.co/vumichien/nonsemantic-speech-trillsson3"
                                               "/resolve/main/config.json",
}


class Trillsson_efficientConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Trillsson_efficientModel`]. It is used to
    instantiate an Trillsson_efficient model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Trillson_efficient
    [vumichien/nonsemantic-speech-trillsson3](https://huggingface.co/vumichien/nonsemantic-speech-trillsson3)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        output_size (:obj:`int`, `optional`, defaults to 1024):
            The number of output dimensions.
        depth_multiplier (:obj:`float`, `optional`, defaults to 1.0):
            The depth multiplier for controling the size of the network.
        depth_divisible_by (:obj:`int`, `optional`, defaults to 8):
            The value that the network depth should be divisible by.
        min_depth (:obj:`int`, `optional`, defaults to 8):
            The minimum depth of the network.
        hidden_act (:obj:`str`, `optional`, defaults to "swish"):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            "gelu", "relu", "silu", "swish", "gelu_new" and "tanh" are supported.
        drop_connect_rate (:obj:`float`, `optional`, defaults to 0.2):
            The dropout ratio for drop connect.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (:obj:`float`, `optional`, defaults to 0.001):
            The epsilon used by the layer normalization layers.
        norm_momentum (:obj:`float`, `optional`, defaults to 0.1):
            The momentum used by the layer normalization layers.
        log_floor (:obj:`float`, `optional`, defaults to 1e-12):
            The floor value for the log operation.
        log_additive_offset (:obj:`float`, `optional`, defaults to 0.001):
            The additive offset for the log operation.
        window_length_secs (:obj:`float`, `optional`, defaults to 0.025):
            The window length in seconds.
        hop_length_secs (:obj:`float`, `optional`, defaults to 0.010):
            The hop length in seconds.
        f_max (:obj:`float`, `optional`, defaults to 7500.0):
            The maximum frequency in Hz.
        f_min (:obj:`float`, `optional`, defaults to 125.0):
            The minimum frequency in Hz.
        fft_length (:obj:`int`, `optional`, defaults to None):
            The FFT length. If None, it will be computed from the window length.

    Example:
    ```python
    >>> from transformers import Trillsson_efficientConfig, Trillsson_efficientModel
    >>> # Initializing a "nonsemantic-speech-trillsson3" style configuration
    >>> configuration = Trillsson_efficientConfig()
    >>> # Initializing a model from the "nonsemantic-speech-trillsson3" style configuration
    >>> model = Trillsson_efficientModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "trillsson_efficient"

    def __init__(
        self,
        output_size=1024,
        depth_multiplier=1.0,
        depth_divisible_by=8,
        min_depth=8,
        hidden_act="swish",
        drop_connect_rate=0.2,
        initializer_range=0.02,
        norm_eps=0.001,
        norm_momentum=0.1,
        log_floor=1e-12,
        log_additive_offset=0.001,
        window_length_secs=0.025,
        hop_length_secs=0.010,
        f_max=7500.0,
        f_min=125.0,
        fft_length=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")

        self.output_size = output_size
        self.depth_multiplier = depth_multiplier
        self.depth_divisible_by = depth_divisible_by
        self.min_depth = min_depth
        self.hidden_act = hidden_act
        self.drop_connect_rate = drop_connect_rate
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.log_floor = log_floor
        self.log_additive_offset = log_additive_offset
        self.window_length_secs = window_length_secs
        self.hop_length_secs = hop_length_secs
        self.f_max = f_max
        self.f_min = f_min
        self.fft_length = fft_length
