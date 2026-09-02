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
"""Informer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="huggingface/informer-tourism-monthly")
@strict
class InformerConfig(PreTrainedConfig):
    r"""
    prediction_length (`int`):
        The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
        typically dictated by the dataset and we recommend to set it appropriately.
    context_length (`int`, *optional*, defaults to `prediction_length`):
        The context length for the encoder. If `None`, the context length will be the same as the
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
        The lags of the input time series as covariates often dictated by the frequency of the data. Default is
        `[1, 2, 3, 4, 5, 6, 7]` but we recommend to change it based on the dataset appropriately.
    scaling (`string` or `bool`, *optional* defaults to `"mean"`):
        Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
        scaler is set to "mean".
    num_dynamic_real_features (`int`, *optional*, defaults to 0):
        The number of dynamic real valued features.
    num_static_real_features (`int`, *optional*, defaults to 0):
        The number of static real valued features.
    num_static_categorical_features (`int`, *optional*, defaults to 0):
        The number of static categorical features.
    num_time_features (`int`, *optional*, defaults to 0):
        The number of time features in the input time series.
    cardinality (`list[int]`, *optional*):
        The cardinality (number of different values) for each of the static categorical features. Should be a list
        of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if
        `num_static_categorical_features` is > 0.
    embedding_dimension (`list[int]`, *optional*):
        The dimension of the embedding for each of the static categorical features. Should be a list of integers,
        having the same length as `num_static_categorical_features`. Cannot be `None` if
        `num_static_categorical_features` is > 0.
    activation_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability used between the two layers of the feed-forward networks.
    num_parallel_samples (`int`, *optional*, defaults to 100):
        The number of samples to generate in parallel for each time step of inference.
    attention_type (`str`, *optional*, defaults to "prob"):
        Attention used in encoder. This can be set to "prob" (Informer's ProbAttention) or "full" (vanilla
        transformer's canonical self-attention).
    sampling_factor (`int`, *optional*, defaults to 5):
        ProbSparse sampling factor (only makes affect when `attention_type`="prob"). It is used to control the
        reduced query matrix (Q_reduce) input length.
    distil (`bool`, *optional*, defaults to `True`):
        Whether to use distilling in encoder.

    Example:

    ```python
    >>> from transformers import InformerConfig, InformerModel

    >>> # Initializing an Informer configuration with 12 time steps for prediction
    >>> configuration = InformerConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = InformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "informer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
        "initializer_range": "init_std",
    }

    prediction_length: int | None = None
    context_length: int | None = None
    distribution_output: str = "student_t"
    loss: str = "nll"
    input_size: int = 1
    lags_sequence: list[int] | None = None
    scaling: str | bool | None = "mean"
    num_dynamic_real_features: int = 0
    num_static_real_features: int = 0
    num_static_categorical_features: int = 0
    num_time_features: int = 0
    cardinality: list[int] | None = None
    embedding_dimension: list[int] | None = None
    d_model: int = 64
    encoder_ffn_dim: int = 32
    decoder_ffn_dim: int = 32
    encoder_attention_heads: int = 2
    decoder_attention_heads: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 2
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    dropout: float | int = 0.05
    encoder_layerdrop: float | int = 0.1
    decoder_layerdrop: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    num_parallel_samples: int = 100
    init_std: float = 0.02
    use_cache: bool = True
    attention_type: str = "prob"
    sampling_factor: int = 5
    distil: bool = True

    def __post_init__(self, **kwargs):
        self.context_length = self.context_length or self.prediction_length
        self.lags_sequence = self.lags_sequence if self.lags_sequence is not None else [1, 2, 3, 4, 5, 6, 7]

        if not (self.cardinality and self.num_static_categorical_features > 0):
            self.cardinality = [0]

        if not (self.embedding_dimension and self.num_static_categorical_features > 0):
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.feature_size = self.input_size * len(self.lags_sequence) + self._number_of_features
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if (
            self.cardinality
            and self.num_static_categorical_features > 0
            and len(self.cardinality) != self.num_static_categorical_features
        ):
            raise ValueError(
                "The cardinality should be a list of the same length as `num_static_categorical_features`"
            )

        if (
            self.embedding_dimension
            and self.num_static_categorical_features > 0
            and len(self.embedding_dimension) != self.num_static_categorical_features
        ):
            raise ValueError(
                "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
            )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_dynamic_real_features
            + self.num_time_features
            + self.num_static_real_features
            + self.input_size * 2  # the log1p(abs(loc)) and log(scale) features
        )


__all__ = ["InformerConfig"]
