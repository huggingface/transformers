# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Dia model configuration"""

from typing import TYPE_CHECKING, List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


class DiaEncoderConfig(PretrainedConfig):
    model_type = "dia_encoder"

    def __init__(
        self,
        max_length: int = 1024,
        num_hidden_layers: int = 12,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        norm_eps: float = 1e-5,
        vocab_size: int = 256,
        dropout: float = 0,
        hidden_act: str = "silu",
        rope_min_timescale: int = 1,
        rope_max_timescale: int = 10_000,
        **kwargs,
    ):
        self.max_length = max_length
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_min_timescale = rope_min_timescale
        self.rope_max_timescale = rope_max_timescale
        super().__init__(**kwargs)


class DiaDecoderConfig(PretrainedConfig):
    model_type = "dia_decoder"

    def __init__(
        self,
        max_length: int = 3072,
        num_hidden_layers: int = 18,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        cross_num_attention_heads: int = 16,
        cross_head_dim: int = 128,
        cross_num_key_value_heads: int = 8,
        cross_hidden_size: int = 1024,
        norm_eps: float = 1e-5,
        vocab_size: int = 1028,
        dropout: float = 0,
        hidden_act: str = "silu",
        num_channels: int = 9,
        rope_min_timescale: int = 1,
        rope_max_timescale: int = 10_000,
        **kwargs,
    ):
        self.max_length = max_length
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.cross_num_key_value_heads = cross_num_key_value_heads
        self.cross_num_attention_heads = cross_num_attention_heads
        self.cross_head_dim = cross_head_dim
        self.cross_hidden_size = cross_hidden_size
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.num_channels = num_channels
        self.rope_min_timescale = rope_min_timescale
        self.rope_max_timescale = rope_max_timescale
        super().__init__(**kwargs)


class DiaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DiaModel`]. It is used to instantiate a
    Dia model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [google/dia](https://huggingface.co/google/dia) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config (`DiaEncoderConfig`, *optional*):
            Configuration for the encoder part of the model. If not provided, a default `DiaEncoderConfig` will be used.
        decoder_config (`DiaDecoderConfig`, *optional*):
            Configuration for the decoder part of the model. If not provided, a default `DiaDecoderConfig` will be used.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1025):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1024):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 1026):
            Beginning of stream token id.
        delay_pattern (`List[int]`, *optional*, defaults to `[0, 8, 9, 10, 11, 12, 13, 14, 15]`):
            The delay pattern for the decoder. The length of this list must match `decoder_config.num_channels`.

    Example:

    ```python
    >>> from transformers import DiaConfig, DiaModel

    >>> # Initializing a DiaConfig with default values
    >>> configuration = DiaConfig()

    >>> # Initializing a DiaModel (with random weights) from the configuration
    >>> model = DiaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dia"
    sub_configs = {"encoder_config": DiaEncoderConfig, "decoder_config": DiaDecoderConfig}

    def __init__(
        self,
        encoder_config: Optional[DiaEncoderConfig] = None,
        decoder_config: Optional[DiaDecoderConfig] = None,
        norm_eps: float = 1e-5,
        pad_token_id: int = 1025,
        eos_token_id: int = 1024,
        bos_token_id: int = 1026,
        delay_pattern: Optional[List[int]] = None,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config = DiaEncoderConfig(**encoder_config)
        if isinstance(decoder_config, dict):
            decoder_config = DiaDecoderConfig(**decoder_config)
        self.encoder_config = encoder_config if encoder_config is not None else DiaEncoderConfig()
        self.decoder_config = decoder_config if decoder_config is not None else DiaDecoderConfig()
        self.norm_eps = norm_eps
        self.delay_pattern = delay_pattern if delay_pattern is not None else [0, 8, 9, 10, 11, 12, 13, 14, 15]
        self.is_encoder_decoder = True

        assert self.decoder_config.num_channels == len(self.delay_pattern), (
            "Number of channels must match delay pattern length."
        )

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_encoder_decoder=True,
            **kwargs,
        )


__all__ = ["DiaConfig", "DiaEncoderConfig", "DiaDecoderConfig"]
