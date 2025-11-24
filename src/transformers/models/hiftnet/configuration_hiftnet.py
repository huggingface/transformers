# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""HiFTNet model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HiFTNetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HiFTNetModel`]. It is used to instantiate a
    HiFTNet vocoder model according to the specified arguments, defining the model architecture.

    HiFTNet is a neural vocoder that combines Neural Source Filter with ISTFTNet for high-quality speech synthesis.
    It was introduced in the paper "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
    and further improved with the HiFT architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        in_channels (`int`, *optional*, defaults to 80):
            Number of input channels (mel spectrogram bins).
        base_channels (`int`, *optional*, defaults to 512):
            Base number of channels for the generator network.
        nb_harmonics (`int`, *optional*, defaults to 8):
            Number of harmonic overtones for the neural source filter.
        sampling_rate (`int`, *optional*, defaults to 22050):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        nsf_alpha (`float`, *optional*, defaults to 0.1):
            Amplitude of sine-waveform in the neural source filter.
        nsf_sigma (`float`, *optional*, defaults to 0.003):
            Standard deviation of Gaussian noise in the neural source filter.
        nsf_voiced_threshold (`float`, *optional*, defaults to 10.0):
            F0 threshold for voiced/unvoiced classification.
        upsample_rates (`list[int]`, *optional*, defaults to `[8, 8]`):
            A list of integers defining the stride of each 1D convolutional layer in the upsampling network.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[16, 16]`):
            A list of integers defining the kernel size of each 1D convolutional layer in the upsampling network.
        istft_n_fft (`int`, *optional*, defaults to 16):
            FFT size for inverse STFT.
        istft_hop_len (`int`, *optional*, defaults to 4):
            Hop length for inverse STFT.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            A list of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested list of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        source_resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[7, 11]`):
            A list of integers defining the kernel sizes for source residual blocks.
        source_resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5]]`):
            A nested list of integers defining the dilation rates for source residual blocks.
        lrelu_slope (`float`, *optional*, defaults to 0.1):
            The slope of the leaky ReLU activation.
        audio_limit (`float`, *optional*, defaults to 0.99):
            Maximum absolute value for output audio clipping.
        f0_predictor_in_channels (`int`, *optional*, defaults to 80):
            Input channels for the F0 predictor (should match in_channels).
        f0_predictor_cond_channels (`int`, *optional*, defaults to 512):
            Conditional channels for the F0 predictor.

    Example:

    ```python
    >>> from transformers import HiFTNetModel, HiFTNetConfig

    >>> # Initializing a HiFTNet configuration
    >>> configuration = HiFTNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = HiFTNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hiftnet"

    def __init__(
        self,
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=22050,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10.0,
        upsample_rates=None,
        upsample_kernel_sizes=None,
        istft_n_fft=16,
        istft_hop_len=4,
        resblock_kernel_sizes=None,
        resblock_dilation_sizes=None,
        source_resblock_kernel_sizes=None,
        source_resblock_dilation_sizes=None,
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor_in_channels=80,
        f0_predictor_cond_channels=512,
        **kwargs,
    ):
        # Set default values for list parameters
        if upsample_rates is None:
            upsample_rates = [8, 8]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = [7, 11]
        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.nsf_alpha = nsf_alpha
        self.nsf_sigma = nsf_sigma
        self.nsf_voiced_threshold = nsf_voiced_threshold
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.istft_n_fft = istft_n_fft
        self.istft_hop_len = istft_hop_len
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.source_resblock_kernel_sizes = source_resblock_kernel_sizes
        self.source_resblock_dilation_sizes = source_resblock_dilation_sizes
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.f0_predictor_in_channels = f0_predictor_in_channels
        self.f0_predictor_cond_channels = f0_predictor_cond_channels
        # Add hidden_size for compatibility with common tests
        self.hidden_size = base_channels

        super().__init__(**kwargs)


__all__ = ["HiFTNetConfig"]
