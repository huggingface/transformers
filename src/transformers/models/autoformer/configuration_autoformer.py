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
"""Autoformer model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class AutoformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AutoformerModel`]. It is used to instantiate an
    Autoformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Autoformer
    [huggingface/autoformer-tourism-monthly](https://huggingface.co/huggingface/autoformer-tourism-monthly)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        prediction_length (`int`):
            The prediction length for the decoder. In other words, the prediction horizon of the model.
        context_length (`int`, *optional*, defaults to `prediction_length`):
            The context length for the encoder. If unset, the context length will be the same as the
            `prediction_length`.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model. Could be either "student_t", "normal" or "negative_binomial".
        loss (`string`, *optional*, defaults to `"nll"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood (nll) - which currently is the only supported one.
        input_size (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        lags_sequence (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`):
            The lags of the input time series as covariates often dictated by the frequency. Default is `[1, 2, 3, 4,
            5, 6, 7]`.
        scaling (`bool`, *optional* defaults to `True`):
            Whether to scale the input targets.
        num_time_features (`int`, *optional*, defaults to 0):
            The number of time features in the input time series.
        num_dynamic_real_features (`int`, *optional*, defaults to 0):
            The number of dynamic real valued features.
        num_static_categorical_features (`int`, *optional*, defaults to 0):
            The number of static categorical features.
        num_static_real_features (`int`, *optional*, defaults to 0):
            The number of static real valued features.
        cardinality (`list[int]`, *optional*):
            The cardinality (number of different values) for each of the static categorical features. Should be a list
            of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        embedding_dimension (`list[int]`, *optional*):
            The dimension of the embedding for each of the static categorical features. Should be a list of integers,
            having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        d_model (`int`, *optional*, defaults to 64):
            Dimensionality of the transformer layers.
        encoder_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 2):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
            `"relu"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for each time step of inference.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.
        label_length (`int`, *optional*, defaults to 10):
            Start token length of the Autoformer decoder, which is used for direct multi-step prediction (i.e.
            non-autoregressive generation).
        moving_average (`int`, *optional*, defaults to 25):
            The window size of the moving average. In practice, it's the kernel size in AvgPool1d of the Decomposition
            Layer.
        autocorrelation_factor (`int`, *optional*, defaults to 3):
            "Attention" (i.e. AutoCorrelation mechanism) factor which is used to find top k autocorrelations delays.
            It's recommended in the paper to set it to a number between 1 and 5.


        Example:

    ```python
    >>> from transformers import AutoformerConfig, AutoformerModel

    >>> # Initializing a default Autoformer configuration
    >>> configuration = AutoformerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = AutoformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "autoformer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    def __init__(
        self,
        prediction_length: Optional[int] = None,
        context_length: Optional[int] = None,
        distribution_output: str = "student_t",
        loss: str = "nll",
        input_size: int = 1,
        lags_sequence: list[int] = [1, 2, 3, 4, 5, 6, 7],
        scaling: bool = True,
        num_time_features: int = 0,
        num_dynamic_real_features: int = 0,
        num_static_categorical_features: int = 0,
        num_static_real_features: int = 0,
        cardinality: Optional[list[int]] = None,
        embedding_dimension: Optional[list[int]] = None,
        d_model: int = 64,
        encoder_attention_heads: int = 2,
        decoder_attention_heads: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_ffn_dim: int = 32,
        decoder_ffn_dim: int = 32,
        activation_function: str = "gelu",
        dropout: float = 0.1,
        encoder_layerdrop: float = 0.1,
        decoder_layerdrop: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        num_parallel_samples: int = 100,
        init_std: float = 0.02,
        use_cache: bool = True,
        is_encoder_decoder=True,
        # Autoformer arguments
        label_length: int = 10,
        moving_average: int = 25,
        autocorrelation_factor: int = 3,
        **kwargs,
    ):
        # time series specific configuration
        self.prediction_length = prediction_length
        self.context_length = context_length if context_length is not None else prediction_length
        self.distribution_output = distribution_output
        self.loss = loss
        self.input_size = input_size
        self.num_time_features = num_time_features
        self.lags_sequence = lags_sequence
        self.scaling = scaling
        self.num_dynamic_real_features = num_dynamic_real_features
        self.num_static_real_features = num_static_real_features
        self.num_static_categorical_features = num_static_categorical_features
        if cardinality is not None and num_static_categorical_features > 0:
            if len(cardinality) != num_static_categorical_features:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality
        else:
            self.cardinality = [0]
        if embedding_dimension is not None and num_static_categorical_features > 0:
            if len(embedding_dimension) != num_static_categorical_features:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]
        self.num_parallel_samples = num_parallel_samples

        # Transformer architecture configuration
        self.feature_size = input_size * len(self.lags_sequence) + self._number_of_features
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop

        self.activation_function = activation_function
        self.init_std = init_std

        self.use_cache = use_cache

        # Autoformer
        self.label_length = label_length
        self.moving_average = moving_average
        self.autocorrelation_factor = autocorrelation_factor

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_dynamic_real_features
            + self.num_time_features
            + self.num_static_real_features
            + self.input_size * 2  # the log1p(abs(loc)) and log(scale) features
        )


__all__ = ["AutoformerConfig"]
