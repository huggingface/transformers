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
"""FastConformer model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class FastConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastConformerModel`]. It is used to instantiate a
    FastConformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FastConformer
    [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 1024):
            Vocabulary size of the FastConformer model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FastConformerModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the hidden states.
        encoder_layers (`int`, *optional*, defaults to 24):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see
            https://arxiv.org/abs/1909.11556) for more details.
        activation_function (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        conv_kernel_size (`int`, *optional*, defaults to 9):
            The kernel size of the convolution layers in the Conformer block.
        subsampling_factor (`int`, *optional*, defaults to 8):
            The factor by which the input sequence is subsampled.
        subsampling_conv_channels (`int`, *optional*, defaults to 256):
            The number of channels in the subsampling convolution layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the linear layers.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features
    """
    model_type = "fastconformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model", "num_hidden_layers": "encoder_layers"}

    def __init__(
        self,
        vocab_size=1024,
        d_model=1024,
        encoder_layers=24,
        encoder_attention_heads=8,
        encoder_ffn_dim=4096,
        encoder_layerdrop=0.1,
        activation_function="silu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        initializer_range=0.02,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        use_bias=False,
        num_mel_bins=128,
        xscaling=False,
        dropout_emb=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.use_bias = use_bias
        self.num_mel_bins = num_mel_bins
        self.xscaling = xscaling
        self.dropout_emb = dropout_emb

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        ) 