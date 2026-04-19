# Copyright 2024 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
"""Mimi model configuration"""

import math

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="kyutai/mimi")
@strict
class MimiConfig(PreTrainedConfig):
    r"""
    audio_channels (`int`, *optional*, defaults to 1):
        Number of channels in the audio data. Either 1 for mono or 2 for stereo.
    num_filters (`int`, *optional*, defaults to 64):
        Number of convolution kernels of first `MimiConv1d` down sampling layer.
    num_residual_layers (`int`,  *optional*, defaults to 1):
        Number of residual layers.
    upsampling_ratios (`Sequence[int]`, *optional*):
        Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
        will use the ratios in the reverse order to the ones specified here that must match the decoder order.
        If not specified, will defaults to `[8, 6, 5, 4]`
    last_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the last convolution layer.
    residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the residual layers.
    dilation_growth_rate (`int`, *optional*, defaults to 2):
        How much to increase the dilation with each layer.
    use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether to use fully causal convolution.
    pad_mode (`str`, *optional*, defaults to `"constant"`):
        Padding mode for the convolutions.
    compress (`int`, *optional*, defaults to 2):
        Reduced dimensionality in residual branches.
    trim_right_ratio (`float`, *optional*, defaults to 1.0):
        Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
        equal to 1.0, it means that all the trimming is done at the right.
    num_quantizers (`int`, *optional*, defaults to 32):
        Number of quantizer channels, or codebooks, in the quantizer.
    use_conv_shortcut (`bool`, *optional*, defaults to `False`):
        Whether to use a convolutional layer as the 'skip' connection in the `MimiResnetBlock` block. If False,
        an identity function will be used, giving a generic residual connection.
    vector_quantization_hidden_dimension (`int`, *optional*, defaults to 256):
        Intermediate representation dimension in the residual vector quantization space.
    num_semantic_quantizers (`int`, *optional*, defaults to 1):
        Number of semantic quantizer channels, or codebooks, in the semantic quantizer. Must be lower than `num_quantizers`.
    upsample_groups (`int`, *optional*, defaults to 512):
        If `frame_rate!=encodec_frame_rate`, indicates the number of groups used in the upsampling operation to go from one rate to another.
    use_streaming (`bool`, *optional*, defaults to `False`):
        Whether to use streaming mode. If `True`, the model encode method will return the padding cache that can be used in a subsequent call to the encode method.

    Example:

    ```python
    >>> from transformers import MimiModel, MimiConfig

    >>> # Initializing a "kyutai/mimi" style configuration
    >>> configuration = MimiConfig()

    >>> # Initializing a model (with random weights) from the "kyutai/mimi" style configuration
    >>> model = MimiModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mimi"

    sampling_rate: int = 24_000
    audio_channels: int = 1
    hidden_size: int = 512
    num_filters: int = 64
    num_residual_layers: int = 1
    upsampling_ratios: list[int] | None = None
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "constant"
    compress: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_quantizers: int = 32
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256
    num_semantic_quantizers: int = 1
    upsample_groups: int = 512
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int | None = None
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8000
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_cache: bool = False
    use_streaming: bool = False
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int = 250
    attention_dropout: float | int = 0.0
    layer_scale_initial_scale: float = 0.01
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.upsampling_ratios if self.upsampling_ratios else [8, 6, 5, 4]
        self.codebook_dim = self.codebook_dim if self.codebook_dim is not None else self.hidden_size
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        # Handle backward compatibility for frame_rate:
        # If frame_rate is explicitly provided, use it (backward compatibility)
        # Otherwise, compute it from other parameters (correctly)
        self._frame_rate = kwargs.pop("frame_rate", None)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_semantic_quantizers >= self.num_quantizers:
            raise ValueError(
                f"The number of semantic quantizers should be lower than the total number of quantizers {self.num_quantizers}, but is currently {self.num_semantic_quantizers}."
            )

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.upsampling_ratios)
        return math.ceil(self.sampling_rate / hop_length)

    @property
    def num_codebooks(self) -> int:
        # alias to num_quantizers
        return self.num_quantizers

    @property
    def frame_size(self) -> int:
        # 1. we need each encoder conv stride
        # first conv
        strides = [1]

        # layer convs
        for ratio in reversed(self.upsampling_ratios):
            for j in range(self.num_residual_layers):
                len_kernel_sizes = len(self.residual_kernel_size) if isinstance(self.residual_kernel_size, list) else 1
                strides.extend([1] * (len_kernel_sizes + 1))
                if self.use_conv_shortcut:  # skip connection
                    strides.append(1)

            strides.append(ratio)

        # last conv
        strides.append(1)

        # downsampling layer
        strides.append(2)

        return math.prod(strides)

    @property
    def frame_rate(self) -> float:
        # handle backward compatibility
        if self._frame_rate is not None:
            return self._frame_rate
        return self.sampling_rate / self.frame_size


__all__ = ["MimiConfig"]
