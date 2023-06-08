# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
""" EnCodec model configuration"""


from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Matthijs/encodec_24khz": "https://huggingface.co/Matthijs/encodec_24khz/resolve/main/config.json",
    "Matthijs/encodec_48khz": "https://huggingface.co/Matthijs/encodec_48khz/resolve/main/config.json",
}


class EncodecConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EncodecModel`]. It is used to instantiate a
    Encodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [Matthijs/encodec_24khz](https://huggingface.co/Matthijs/encodec_24khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        target_bandwidths (`List[float]`, *optional*):
            TODO
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        normalize=False,
            TODO
        segment=None,
            TODO
        overlap=0.01,
            TODO
        dimension (int):
            Intermediate representation dimension.
        num_filters (int):
            Base width for the model.
        num_residual_layers (int):
            Number of residual layers.
        ratios (Sequence[int]):
            Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
            will use the ratios in the reverse order to the ones specified here that must match the decoder order.
        activation (str):
            Activation function.
        activation_params (dict):
            Parameters to provide to the activation function.
        norm (`str`, *optional*, defaults to `"weight_norm"`):
            Normalization method.
        norm_params (`Dict[str, Any]`, *optional*):
            Parameters to provide to the underlying normalization used along with the convolution.
        final_activation (str):
            Final activation function after all convolutions.
        final_activation_params (dict):
            Parameters to provide to the activation function.
        kernel_size (int):
            Kernel size for the initial convolution.
        last_kernel_size (int):
            Kernel size for the initial convolution.
        residual_kernel_size (int):
            Kernel size for the residual layers.
        dilation_base (int):
            How much to increase the dilation with each layer.
        causal (`bool`, *optional*, defaults to True):
            Whether to use fully causal convolution.
        pad_mode (str):
            Padding mode for the convolutions.
        true_skip (bool):
            Whether to use true skip connection or a simple (streamable) convolution as the skip connection in the
            residual network blocks.
        compress (int):
            Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int):
            Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float):
            Ratio for trimming at the right of the transposed convolution under the causal setup. If equal to 1.0, it
            means that all the trimming is done at the right.
        bins (int): Codebook size.
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with randomly
            selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.

    Example:

    ```python
    >>> from transformers import EncodecModel, EncodecConfig

    >>> # Initializing a "Matthijs/encodec_24khz" style configuration
    >>> configuration = EncodecConfig()

    >>> # Initializing a model (with random weights) from the "Matthijs/encodec_24khz" style configuration
    >>> model = EncodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "encodec"

    def __init__(
        self,
        target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
        sampling_rate=24_000,
        audio_channels=1,
        normalize=False,
        segment=None,  # TODO: segment length in seconds
        overlap=0.01,
        dimension=128,  # TODO: hidden_size?
        num_filters=32,
        num_residual_layers=1,
        ratios=[8, 5, 4, 2],
        activation="ELU",
        activation_params={"alpha": 1.0},
        norm="weight_norm",
        norm_params={},
        final_activation=None,
        final_activation_params=None,
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        causal=True,
        pad_mode="reflect",
        true_skip=False,
        compress=2,
        lstm=2,
        trim_right_ratio=1.0,
        bins=1024,  # TODO: rename to codebook_size
        codebook_dim=None,
        decay=0.99,
        epsilon=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2,
        commitment_weight=1.0,
        is_encoder_decoder=True,
        **kwargs,
    ):
        self.target_bandwidths = target_bandwidths
        self.sampling_rate = sampling_rate
        self.audio_channels = audio_channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.dimension = dimension
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.ratios = ratios
        self.activation = activation
        self.activation_params = activation_params
        self.norm = norm
        self.norm_params = norm_params
        self.final_activation = final_activation
        self.final_activation_params = final_activation_params
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.dilation_base = dilation_base
        self.causal = causal
        self.pad_mode = pad_mode
        self.true_skip = true_skip
        self.compress = compress
        self.lstm = lstm
        self.trim_right_ratio = trim_right_ratio
        self.bins = bins
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.commitment_weight = commitment_weight
        self.is_encoder_decoder = is_encoder_decoder

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

    @property
    def segment_length(self) -> Optional[int]:
        if self.segment is None:
            return None
        else:
            return int(self.segment * self.sampling_rate)

    @property
    def segment_stride(self) -> Optional[int]:
        if self.segment is None:
            return None
        else:
            return max(1, int((1.0 - self.overlap) * self.segment_length))
