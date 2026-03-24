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
"""EnCodec model configuration"""

import math

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/encodec_24khz")
@strict
class EncodecConfig(PreTrainedConfig):
    r"""
    target_bandwidths (`list[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0, 24.0]`):
        The range of different bandwidths the model can encode audio with.
    normalize (`bool`, *optional*, defaults to `False`):
        Whether the audio shall be normalized when passed.
    chunk_length_s (`float`, *optional*):
        If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
    overlap (`float`, *optional*):
        Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
        formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
    num_filters (`int`, *optional*, defaults to 32):
        Number of convolution kernels of first `EncodecConv1d` down sampling layer.
    num_residual_layers (`int`,  *optional*, defaults to 1):
        Number of residual layers.
    upsampling_ratios (`Sequence[int]` , *optional*, defaults to `[8, 5, 4, 2]`):
        Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
        will use the ratios in the reverse order to the ones specified here that must match the decoder order.
    norm_type (`str`, *optional*, defaults to `"weight_norm"`):
        Normalization method. Should be in `["weight_norm", "time_group_norm"]`
    kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for the initial convolution.
    last_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for the last convolution layer.
    residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the residual layers.
    dilation_growth_rate (`int`, *optional*, defaults to 2):
        How much to increase the dilation with each layer.
    use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether to use fully causal convolution.
    pad_mode (`str`, *optional*, defaults to `"reflect"`):
        Padding mode for the convolutions.
    compress (`int`, *optional*, defaults to 2):
        Reduced dimensionality in residual branches (from Demucs v3).
    num_lstm_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers at the end of the encoder.
    trim_right_ratio (`float`, *optional*, defaults to 1.0):
        Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
        equal to 1.0, it means that all the trimming is done at the right.
    use_conv_shortcut (`bool`, *optional*, defaults to `True`):
        Whether to use a convolutional layer as the 'skip' connection in the `EncodecResnetBlock` block. If False,
        an identity function will be used, giving a generic residual connection.

    Example:

    ```python
    >>> from transformers import EncodecModel, EncodecConfig

    >>> # Initializing a "facebook/encodec_24khz" style configuration
    >>> configuration = EncodecConfig()

    >>> # Initializing a model (with random weights) from the "facebook/encodec_24khz" style configuration
    >>> model = EncodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "encodec"

    target_bandwidths: list[float] | tuple[float, ...] = (1.5, 3.0, 6.0, 12.0, 24.0)
    sampling_rate: int = 24_000
    audio_channels: int = 1
    normalize: bool = False
    chunk_length_s: int | float | None = None
    overlap: float | None = None
    hidden_size: int = 128
    num_filters: int = 32
    num_residual_layers: int = 1
    upsampling_ratios: list[int] | tuple[int, ...] = (8, 5, 4, 2)
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "reflect"
    compress: int = 2
    num_lstm_layers: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 1024
    codebook_dim: int | None = None
    use_conv_shortcut: bool = True

    def __post_init__(self, **kwargs):
        self.codebook_dim = self.codebook_dim if self.codebook_dim is not None else self.hidden_size
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

    # This is a property because you might want to change the chunk_length_s on the fly
    @property
    def chunk_length(self) -> int | None:
        if self.chunk_length_s is None:
            return None
        else:
            return int(self.chunk_length_s * self.sampling_rate)

    # This is a property because you might want to change the chunk_length_s on the fly
    @property
    def chunk_stride(self) -> int | None:
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.upsampling_ratios))

    @property
    def codebook_nbits(self) -> int:
        return math.ceil(math.log2(self.codebook_size))

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sampling_rate / self.hop_length)

    @property
    def num_quantizers(self) -> int:
        return int(1000 * self.target_bandwidths[-1] // (self.frame_rate * self.codebook_nbits))


__all__ = ["EncodecConfig"]
