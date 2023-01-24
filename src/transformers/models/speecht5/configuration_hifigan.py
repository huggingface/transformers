# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import SpeechT5HiFiGAN, SpeechT5HiFiGANConfig

    >>> # Initializing a "Matthijs/speecht5_hifigan" style configuration
    >>> configuration = SpeechT5HiFiGANConfig()

    >>> # Initializing a model (with random weights) from the "Matthijs/speecht5_hifigan" style configuration
    >>> model = SpeechT5HiFiGAN(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "hifigan"

    def __init__(
        self,
        model_in_dim=80,
        sampling_rate=16000,
        upsample_initial_channel=512,
        upsample_rates=[4, 4, 4, 4],
        upsample_kernel_sizes=[8, 8, 8, 8],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initializer_range=0.01,
        leaky_relu_slope=0.1,
        normalize_before=True,
        **kwargs,
    ):
        self.model_in_dim = model_in_dim
        self.sampling_rate = sampling_rate
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
        super().__init__(**kwargs)
