# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" HtDemucs model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/htdemucs": "https://huggingface.co/facebook/htdemucs/resolve/main/config.json",
    # See all Htdemucs models at https://huggingface.co/models?filter=htdemucs
}


class HtdemucsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HtdemucsModel`]. It is used to instantiate a
    HtDemucs model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the HtDemucs
    [facebook/htdemucs](https://huggingface.co/facebook/htdemucs/resolve/main/config.json) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer block.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-4):
            Initialization values for the post-attention block scaling weights.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        num_stems (`int`, *optional*, defaults to 4):
            The number of stems the audio is split into.
        audio_channels (`int`, *optional*, defaults to 2):
            The number of input/output audio channels.
        hidden_channels (`int`, *optional*, defaults to 48):
            The initial number of hidden channels.
        channel_growth (`int`, *optional*, defaults to 2):
            Factor by which to increase the number of hidden channels with each layer.
        num_conv_layers (`int`, *optional*, defaults to 4):
            Number of convolutional layers in the temporal and frequency branches of the encoder and decoder.
        residual_conv_depth (`int`, *optional*, defaults to 2):
            Depth of the convolutional residual branch in the temporal and frequency branches of the encoder and decoder.
        bottom_channels (`int`, *optional*, defaults to 512):
            Dimension of the linear conv layer that is applied before and after the transformer model to
            upsample/downsample the number of channels.
        n_fft (`int`, *optional* defaults to 4096):
            Size of the Fourier transform applied to the input audio. Should match that used in the
            `HtDemucsFeatureExtractor` class.
        stride (`int`, *optional*, defaults to 4):
            Stride for encoder and decoder layers.
        freq_embedding_lr_scale (`float`, *optional*, defaults to 10):
            Factor by which to boost the learning rate in the scaled embedding layer.
        freq_embedding_scale (`float`, *optional*, defaults to 0.2):
            Factor by which to scale the weights of the frequency embeddings.
    """

    model_type = "htdemucs"

    def __init__(
        self,
        max_position_embeddings=2048,
        num_hidden_layers=5,
        ffn_dim=2048,
        num_attention_heads=8,
        layerdrop=0.0,
        use_cache=True,
        init_std=0.02,
        activation_function="gelu",
        hidden_size=512,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        layer_scale_init_value=1e-4,
        num_stems=4,
        audio_channels=2,
        hidden_channels=48,
        channel_growth=2,
        num_conv_layers=4,
        residual_conv_depth=2,
        bottom_channels=512,
        n_fft=4096,
        stride=4,
        freq_embedding_lr_scale=10,
        freq_embedding_scale=0.2,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layer_scale_init_value = layer_scale_init_value
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.init_std = init_std
        self.num_stems = num_stems
        self.audio_channels = audio_channels
        self.hidden_channels = hidden_channels
        self.channel_growth = channel_growth
        self.num_conv_layers = num_conv_layers
        self.residual_conv_depth = residual_conv_depth
        self.bottom_channels = bottom_channels
        self.n_fft = n_fft
        self.stride = stride
        self.freq_embedding_lr_scale = freq_embedding_lr_scale
        self.freq_embedding_scale = freq_embedding_scale

        super().__init__(**kwargs)
