# coding=utf-8
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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
""" VITS model configuration"""

import functools
import operator

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VITS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "TODO": "https://huggingface.co/TODO/resolve/main/config.json",
}


class VitsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitsModel`]. It is used to instantiate a
    VITS model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VITS
    [TODO](https://huggingface.co/TODO) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 38):
            Vocabulary size of the VITS model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed to the forward method of [`VitsModel`].
        hidden_size (`int`, *optional*, defaults to 192):
            Dimensionality of the text encoder layers.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        ffn_kernel_size (`int`, *optional*, defaults to 3):
            TODO
        inter_channels (`int`, *optional*, defaults to 192):
            TODO
            Should be divisible by two.
        spec_channels (`int`, *optional*, defaults to 513):
            Number of channels in the target spectrograms.
        segment_size (`int`, *optional*, defaults to 32):
            TODO
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        use_stochastic_duration_prediction (`bool`, *optional*, defaults to `True`):
            Whether to use the stochastic duration prediction module or the regular duration predictor.
        num_speakers (`int`, *optional*, defaults to 1):
            Number of speakers if this is a multi-speaker model.
        speaker_embedding_channels (`int`, *optional*, defaults to 0):
            Number of channels used by the speaker embeddings. Is zero for single-speaker models.
        resblock="1",
            TODO: is this HifiGan?
        resblock_kernel_sizes=[3, 7, 11],
            TODO: is this HifiGan?
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            TODO: is this HifiGan?
        upsample_rates=[8, 8, 2, 2],
            TODO: is this HifiGan?
        upsample_initial_channel=512,
            TODO: is this HifiGan?
        upsample_kernel_sizes=[16, 16, 4, 4],
            TODO: is this HifiGan?

        duration_predictor_kernel_size=3
        duration_predictor_dropout=0.5
        duration_predictor_num_flows=4
        duration_predictor_filter_channels=256


        prior_encoder_num_flows=4,
        wavenet_kernel_size=5,
            This must be an odd number.
        wavenet_dilation_rate=1,
        wavenet_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the WaveNet layers.

        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import VitsModel, VitsConfig

    >>> # Initializing a "TODO" style configuration
    >>> configuration = VitsConfig()

    >>> # Initializing a model (with random weights) from the "TODO" style configuration
    >>> model = VitsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vits"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "num_hidden_layers": "encoder_layers"}

    def __init__(
        self,
        vocab_size=38,
        hidden_size=192,
        encoder_layers=6,
        encoder_attention_heads=2,
        encoder_ffn_dim=768,
        encoder_layerdrop=0.1,
        ffn_kernel_size=3,
        inter_channels=192,  # TODO: better name?  intermediate_size?
        spec_channels=513,   # TODO: spectrogram_channels?
        segment_size=32,     # TODO: hps.train.segment_size // hps.data.hop_length
        hidden_act="relu",  # or quick_gelu
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        leaky_relu_slope=0.1,
        use_stochastic_duration_prediction=True,
        num_speakers=1,
        speaker_embedding_channels=0,

        # TODO: is this HifiGan?
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],

        prior_encoder_num_flows=4,
        wavenet_kernel_size=5,
        wavenet_dilation_rate=1,
        wavenet_dropout=0.0,

        duration_predictor_kernel_size=3,
        duration_predictor_dropout=0.5,
        duration_predictor_num_flows=4,
        duration_predictor_filter_channels=256,

        use_cache=False,
        is_encoder_decoder=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.ffn_kernel_size = ffn_kernel_size
        self.inter_channels = inter_channels
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.leaky_relu_slope = leaky_relu_slope
        self.use_stochastic_duration_prediction = use_stochastic_duration_prediction
        self.num_speakers = num_speakers
        self.speaker_embedding_channels = speaker_embedding_channels
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.prior_encoder_num_flows = prior_encoder_num_flows
        self.wavenet_kernel_size = wavenet_kernel_size
        self.wavenet_dilation_rate = wavenet_dilation_rate
        self.wavenet_dropout = wavenet_dropout
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_dropout = duration_predictor_dropout
        self.duration_predictor_num_flows = duration_predictor_num_flows
        self.duration_predictor_filter_channels = duration_predictor_filter_channels

        self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
