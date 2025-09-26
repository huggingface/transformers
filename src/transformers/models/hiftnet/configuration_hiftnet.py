# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HiFTNetConfig(PretrainedConfig):
    # TODO: @eustlb, check this
    model_type = "generator"
    base_config_key = "generator_config"

    def __init__(
        self,
        hidden_size=512,
        vocab_size=178,
        num_mel_bins=80,
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[8, 5, 3],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[16, 11, 7],  # TODO: is it better with a
        n_fft=16,
        hop_size=5,
        sampling_rate=24000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate

        super().__init__(**kwargs)


__all__ = ["HiFTNetConfig"]
