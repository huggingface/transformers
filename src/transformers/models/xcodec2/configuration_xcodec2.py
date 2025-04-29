# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
"""X-codec model configuration"""

import math
from typing import Optional

import numpy as np

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class XCodec2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`XCodec2Model`]. It is used to instantiate a
    XCodec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        target_bandwidths (`List[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0, 24.0]`):
            The range of diffent bandwiths the model can encode audio with.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        normalize (`bool`, *optional*, defaults to `False`):
            Whether the audio shall be normalized when passed.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
        hidden_size (`int`, *optional*, defaults to 128):
            Intermediate representation dimension.
        num_filters (`int`, *optional*, defaults to 32):
            Number of convolution kernels of first `XCodec2Conv1d` down sampling layer.
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
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of discret codes that make up VQVAE.
        codebook_dim (`int`, *optional*):
            Dimension of the codebook vectors. If not defined, uses `hidden_size`.
        use_conv_shortcut (`bool`, *optional*, defaults to `True`):
            Whether to use a convolutional layer as the 'skip' connection in the `XCodec2ResnetBlock` block. If False,
            an identity function will be used, giving a generic residual connection.
        semantic_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size for the semantic model.
        codec_encoder_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size for the codec encoder model.
        codec_decoder_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size for the codec decoder model.
        use_vocos (`bool`, *optional*, defaults to `True`):
            Whether to use VOCOS.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Intermediate size for the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for the model.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key value heads for the model.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for the attention layer.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        head_dim (`int`, *optional*, defaults to 64):
            Head dimension for the model.
        up_ratios (`tuple`, *optional*, defaults to `(2, 2, 4, 4, 5)`):
            Up sampling ratios for the model.
        dilations (`tuple`, *optional*, defaults to `(1, 3, 9)`):
            Dilation values for the model.
        depth (`int`, *optional*, defaults to 12):
            Depth for the model.
        pos_meb_dim (`int`, *optional*, defaults to 64):
            Position MELB dimension for the model.
        vq_num_quantizers (`int`, *optional*, defaults to 1):
            Number of VQ quantizers for the model.
        vq_dim (`int`, *optional*, defaults to 2048):
            Dimension for the VQ codebook.
        vq_commit_weight (`float`, *optional*, defaults to 0.25):
            Commit weight for the VQ.
        vq_weight_init (`bool`, *optional*, defaults to `False`):
            Whether to initialize VQ weights.
        vq_full_commit_loss (`bool`, *optional*, defaults to `False`):
            Whether to use full commit loss for the VQ.
    """

    model_type = "xcodec2"

    def __init__(
        self,
        semantic_hidden_size: int = 1024,
        codec_encoder_hidden_size: int = 1024,
        codec_decoder_hidden_size: int = 1024,
        use_vocos: bool = True,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        num_hidden_layers: int = 12,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 64,
        up_ratios: tuple = (2, 2, 4, 4, 5),
        dilations: tuple = (1, 3, 9),
        depth: int = 12,
        vq_num_quantizers: int = 1,
        hop_length: int = 320,
        vq_dim: int = 2048,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 16384,
        codebook_dim: int = 16,
        max_position_embeddings: int = 64,
        rope_theta: float = 10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.semantic_hidden_size = semantic_hidden_size
        self.codec_encoder_hidden_size = codec_encoder_hidden_size
        self.codec_decoder_hidden_size = codec_decoder_hidden_size
        self.use_vocos = use_vocos
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.head_dim = head_dim
        self.up_ratios = up_ratios
        self.dilations = dilations
        self.depth = depth
        self.vq_num_quantizers = vq_num_quantizers
        self.hop_length = hop_length
        self.vq_dim = vq_dim
        self.vq_commit_weight = vq_commit_weight
        self.vq_weight_init = vq_weight_init
        self.vq_full_commit_loss = vq_full_commit_loss
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
__all__ = ["XCodec2Config"]
