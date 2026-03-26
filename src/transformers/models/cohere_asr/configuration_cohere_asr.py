# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from ..auto import CONFIG_MAPPING
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig


@auto_docstring(checkpoint="CohereLabs/cohere-transcribe-03-2026")
@strict
class CohereAsrConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import CohereAsrForConditionalGeneration, CohereAsrConfig

    >>> configuration = CohereAsrConfig()
    >>> model = CohereAsrForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "cohere_asr"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    _default_encoder_config_kwargs = {
        "hidden_size": 1280,
        "num_hidden_layers": 48,
        "num_attention_heads": 8,
        "intermediate_size": 5120,
        "hidden_act": "silu",
        "attention_bias": True,
        "convolution_bias": True,
        "conv_kernel_size": 9,
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "num_mel_bins": 128,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "dropout": 0.0,
        "dropout_positions": 0.0,
        "layerdrop": 0.0,
        "activation_dropout": 0.0,
        "attention_dropout": 0.0,
        "max_position_embeddings": 5000,
        "scale_input": False,
        "initializer_range": 0.02,
    }

    encoder_config: dict | PreTrainedConfig | None = None
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
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    attention_bias: bool = True
    decoder_start_token_id: int | None = None
    tie_word_embeddings: bool = False
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "parakeet_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](
                **{**self._default_encoder_config_kwargs, **self.encoder_config}
            )
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["parakeet_encoder"](**self._default_encoder_config_kwargs)

        super().__post_init__(**kwargs)


__all__ = ["CohereAsrConfig"]
