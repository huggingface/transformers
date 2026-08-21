# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="harshaljanjani/canary-1b-v2-hf")
@strict
class CanaryDecoderConfig(PreTrainedConfig):
    model_type = "canary_decoder"

    vocab_size: int = 16384
    hidden_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    intermediate_size: int = 4096
    hidden_act: str = "relu"
    max_position_embeddings: int = 1024
    pad_token_id: int | None = 2
    eos_token_id: int | None = 3
    bos_token_id: int | None = 4
    is_encoder_decoder: bool = True
    use_cache: bool = True
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    attention_bias: bool = True
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="harshaljanjani/canary-1b-v2-hf")
@strict
class CanaryConfig(PreTrainedConfig):
    r"""
    encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
        The config object or dictionary of the FastConformer encoder ([`ParakeetEncoderConfig`]).
    decoder_config (`Union[dict, CanaryDecoderConfig]`, *optional*):
        The config object or dictionary of the Transformer decoder ([`CanaryDecoderConfig`]).
    decoder_start_token_id (`int`, *optional*, defaults to 7):
        The token id that starts decoding (`<|startofcontext|>`, the first token of the multitask prompt).

    Example:

    ```python
    >>> from transformers import CanaryForConditionalGeneration, CanaryConfig

    >>> # Initializing a Canary configuration
    >>> configuration = CanaryConfig()

    >>> # Initializing a model from the configuration
    >>> model = CanaryForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "canary"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": AutoConfig, "decoder_config": CanaryDecoderConfig}

    encoder_config: dict | PreTrainedConfig | None = None
    decoder_config: CanaryDecoderConfig | dict | None = None
    use_cache: bool = True
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 2
    bos_token_id: int | None = 4
    eos_token_id: int | None = 3
    decoder_start_token_id: int | None = 7
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "parakeet_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["parakeet_encoder"](
                num_hidden_layers=32,
                num_mel_bins=128,
                scale_input=False,
                layerdrop=0.0,
                dropout_positions=0.0,
            )

        if isinstance(self.decoder_config, dict):
            self.decoder_config = CanaryDecoderConfig(**self.decoder_config)
        elif self.decoder_config is None:
            self.decoder_config = CanaryDecoderConfig()

        self.vocab_size = self.decoder_config.vocab_size
        super().__post_init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.decoder_config


__all__ = ["CanaryConfig", "CanaryDecoderConfig"]
