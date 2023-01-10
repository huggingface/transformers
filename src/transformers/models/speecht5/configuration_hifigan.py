# coding=utf-8
# Copyright 2022-2023 The HuggingFace Inc. team. All rights reserved.
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
"""Hi-Fi GAN model configuration"""

from transformers import PretrainedConfig


SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "Matthijs/speecht5_hifigan": "https://huggingface.co/Matthijs/speecht5_hifigan/resolve/main/config.json",
}


class SpeechT5HiFiGANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SpeechT5HiFiGANModel`]. It is used to instantiate
    a SpeechT5 Hi-Fi GAN vocoder model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5
    [Matthijs/speecht5_hifigan](https://huggingface.co/Matthijs/speecht5_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    model_type = "hifigan"

    def __init__(
        self,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[4, 4, 4, 4],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[8, 8, 8, 8],
        model_in_dim=80,
        sampling_rate=16000,
        initializer_range=0.01,
        leaky_relu_slope=0.1,
        normalize_before=True,
        **kwargs,
    ):
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.sampling_rate = sampling_rate
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
        super().__init__(**kwargs)
