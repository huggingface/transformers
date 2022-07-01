# coding=utf-8
# Copyright 2022 kashif and The HuggingFace Inc. team. All rights reserved.
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
""" TimeSeriesTransformer model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/tst-ett": "https://huggingface.co/huggingface/tst-ett/resolve/main/config.json",
    # See all TimeSeriesTransformer models at https://huggingface.co/models?filter=time_series_transformer
}


class TimeSeriesTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~TimeSeriesTransformerModel`].
    It is used to instantiate an TimeSeriesTransformer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the TimeSeriesTransformer [huggingface/tst-ett](https://huggingface.co/huggingface/tst-ett) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        prediction_length (`int`):
            The prediction horizon for the model.
        context_length (`int`, *optional*, default to `None`):
            The context length for the encoder. If  `None`, the context length will be the same as the prediction length.
        distr_output (`DistributionOutput` default to `StudentTOutput()`):
            The distribution emission head for the model.
        scaling (`bool` default to `True`):
            Whether to scale the input targets.
        freq (`str`, *optional* default to `None`):
            The frequency of the input time series. If `None`, the `lag_seq` and `time_features` must be provided.
        lags_seq (`list` of `int`, *optional* default to `None`):
            The lags of the input time series. Cannot be `None` if `freq` is `None`.
        time_features (`list` of `TimeFeature`, *optional* default to `None`):
            The time features transformations to apply to the input time series. Cannot be `None` if `freq` is `None`.
        encoder_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 2):
            Number of decoder layers.
        nhead (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder and decoder.
        ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder and decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string,
            `"gelu"` and `"relu"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder, and decoder.

        Example:

    ```python
    >>> from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig

    >>> # Initializing a TimeSeriesTransformer huggingface/tst-ett style configuration
    >>> configuration = TimeSeriesTransformerConfig()

    >>> # Initializing a model from the huggingface/tst-ett style configuration
    >>> model = TimeSeriesTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "time_series_transformer"
    # keys_to_ignore_at_inference = ["past_key_values"]
    
    # attribute_map = {
    #     "num_attention_heads": "encoder_attention_heads",
    #     "hidden_size": "d_model"
    # }

    def __init__(
        self,
        prediction_length,
        context_length=None,
        ffn_dim=32,
        nhead=2,
        freq=None,
        encoder_layers=2,
        decoder_layers=2,
        is_encoder_decoder=True,
        activation_function="gelu",
        dropout=0.1,
        init_std=0.02,
        decoder_start_token_id=2,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs
        )

    