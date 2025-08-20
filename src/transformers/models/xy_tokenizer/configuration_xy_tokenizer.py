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
        code_dim (`int`, *optional*, defaults to 1280):
            The dimension of the code embeddings.
        semantic_encoder_d_model (`int`, *optional*, defaults to 1280):
            Hidden dimension for the semantic encoder.
        acoustic_encoder_d_model (`int`, *optional*, defaults to 1280):
            Hidden dimension for the acoustic encoder.
        num_quantizers (`int`, *optional*, defaults to 32):
            Number of residual quantizers.
        codebook_size (`int`, *optional*, defaults to 1024):
            Size of each codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Dimension of each codebook entry.
        hidden_size (`int`, *optional*, defaults to 1280):
            Hidden size for transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in transformer layers.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Intermediate size in feed-forward networks.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in transformers.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The activation function to use.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for attention layers.
        max_position_embeddings (`int`, *optional*, defaults to 1500):
            Maximum position embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for weight initialization.
        use_cache (`bool`, *optional*, defaults to `True`): <fill_docstring>
    """

    model_type = "xy_tokenizer"

    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        encoder_downsample_rate: int = 1280,
        decoder_upsample_rate: int = 1920,
        code_dim: int = 1280,
        semantic_encoder_d_model: int = 1280,
        acoustic_encoder_d_model: int = 1280,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        hidden_size: int = 1280,
        num_attention_heads: int = 20,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        activation_function: str = "gelu",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 1500,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.encoder_downsample_rate = encoder_downsample_rate
        self.decoder_upsample_rate = decoder_upsample_rate
        self.code_dim = code_dim
        self.semantic_encoder_d_model = semantic_encoder_d_model
        self.acoustic_encoder_d_model = acoustic_encoder_d_model
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Store complex nested parameters dynamically for backward compatibility
        self.params = kwargs

        super().__init__(**kwargs)


__all__ = ["XYTokenizerConfig"]
