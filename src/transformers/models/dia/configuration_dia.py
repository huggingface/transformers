# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nari-labs/Dia-1.6B")
class DiaEncoderConfig(PreTrainedConfig):
    model_type = "dia_encoder"

    def __init__(
        self,
        max_position_embeddings: int = 1024,
        num_hidden_layers: int = 12,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        norm_eps: float = 1e-5,
        vocab_size: int = 256,
        hidden_act: str = "silu",
        rope_parameters: RopeParameters | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


@auto_docstring(checkpoint="nari-labs/Dia-1.6B")
class DiaDecoderConfig(PreTrainedConfig):
    r"""
    cross_num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each cross-attention layer in the Transformer decoder.
    cross_head_dim (`int`, *optional*, defaults to 128):
        Dimensionality of the cross-attention head.
    cross_num_key_value_heads (`int`, *optional*, defaults to 16):
        Number of key and value heads for each cross-attention layer in the Transformer decoder.
    cross_hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the cross-attention layers.
    """

    model_type = "dia_decoder"

    def __init__(
        self,
        max_position_embeddings: int = 3072,
        num_hidden_layers: int = 18,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        cross_num_attention_heads: int = 16,
        cross_head_dim: int = 128,
        cross_num_key_value_heads: int = 16,
        cross_hidden_size: int = 1024,
        norm_eps: float = 1e-5,
        vocab_size: int = 1028,
        hidden_act: str = "silu",
        num_channels: int = 9,
        rope_parameters: RopeParameters | None = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        is_encoder_decoder: bool = True,
        pad_token_id: int = 1025,
        eos_token_id: int = 1024,
        bos_token_id: int = 1026,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
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
        self.hidden_act = hidden_act
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


@auto_docstring(checkpoint="nari-labs/Dia-1.6B")
class DiaConfig(PreTrainedConfig):
    r"""
    delay_pattern (`list[int]`, *optional*, defaults to `[0, 8, 9, 10, 11, 12, 13, 14, 15]`):
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
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": DiaEncoderConfig, "decoder_config": DiaDecoderConfig}

    def __init__(
        self,
        encoder_config: DiaEncoderConfig | None = None,
        decoder_config: DiaDecoderConfig | None = None,
        norm_eps: float = 1e-5,
        is_encoder_decoder: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        delay_pattern: list[int] | None = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
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
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # TODO: Remove token ID forwarding once the `nari-labs/Dia-1.6B`
        # checkpoint is updated
        if pad_token_id is not None:
            logger.warning_once(
                "Passing `pad_token_id` to `DiaConfig` is deprecated. "
                "Please set it directly on `DiaDecoderConfig` instead."
            )
            self.decoder_config.pad_token_id = pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "Passing `eos_token_id` to `DiaConfig` is deprecated. "
                "Please set it directly on `DiaDecoderConfig` instead."
            )
            self.decoder_config.eos_token_id = eos_token_id
        if bos_token_id is not None:
            logger.warning_once(
                "Passing `bos_token_id` to `DiaConfig` is deprecated. "
                "Please set it directly on `DiaDecoderConfig` instead."
            )
            self.decoder_config.bos_token_id = bos_token_id

        assert self.decoder_config.num_channels == len(self.delay_pattern), (
            "Number of channels must match delay pattern length."
        )

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    def get_text_config(self, *args, **kwargs):
        """Defaulting to audio config as it's the decoder in this case which is usually the text backbone"""
        return self.decoder_config


__all__ = ["DiaConfig", "DiaEncoderConfig", "DiaDecoderConfig"]
