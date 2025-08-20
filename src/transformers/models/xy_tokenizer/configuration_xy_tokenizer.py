# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. teams. All rights reserved.
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
"""XY-Tokenizer model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class XYTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XYTokenizer`]. It is used to instantiate a
    XY-Tokenizer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the XY-Tokenizer
    [fnlp/XY_Tokenizer_TTSD_V0_hf](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0_hf) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        input_sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the input audio.
        output_sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the output audio.
        encoder_downsample_rate (`int`, *optional*, defaults to 1280):
            The total downsampling factor of the encoder part.
        decoder_upsample_rate (`int`, *optional*, defaults to 1920):
            The total upsampling factor of the decoder part.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for weight initialization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    model_type = "xy_tokenizer"

    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        encoder_downsample_rate: int = 1280,
        decoder_upsample_rate: int = 1920,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.encoder_downsample_rate = encoder_downsample_rate
        self.decoder_upsample_rate = decoder_upsample_rate
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Store complex nested parameters dynamically for backward compatibility
        self.params = kwargs

        super().__init__(**kwargs)


__all__ = ["XYTokenizerConfig"]
