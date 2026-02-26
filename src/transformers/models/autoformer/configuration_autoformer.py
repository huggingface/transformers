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

from ...configuration_utils import PreTrainedConfig
from ...utils import logging, auto_docstring


logger = logging.get_logger(__name__)

@auto_docstring(checkpoint="huggingface/autoformer-tourism-monthly")
class AutoformerConfig(PreTrainedConfig):
    r"""
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
        prediction_length: int | None = None,
        context_length: int | None = None,
        distribution_output: str = "student_t",
        loss: str = "nll",
        input_size: int = 1,
        lags_sequence: list[int] = [1, 2, 3, 4, 5, 6, 7],
        scaling: bool = True,
        num_time_features: int = 0,
        num_dynamic_real_features: int = 0,
        num_static_categorical_features: int = 0,
        num_static_real_features: int = 0,
        cardinality: list[int] | None = None,
        embedding_dimension: list[int] | None = None,
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
