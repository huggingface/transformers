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
"""PyTorch Canary model."""

import math

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..cohere_asr.modeling_cohere_asr import (
    CohereAsrDecoder,
    CohereAsrForConditionalGeneration,
    CohereAsrModel,
    CohereAsrPreTrainedModel,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import SinusoidsPositionEmbedding


@auto_docstring(checkpoint="harshaljanjani/canary-1b-v2-hf")
@strict
class CanaryDecoderConfig(PreTrainedConfig):
    model_type = "canary_decoder"

    vocab_size: int = 16384
    hidden_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
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
    head_dim: int = 128


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
    decoder_config: dict | PreTrainedConfig | None = None
    use_cache: bool = True
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 2
    bos_token_id: int | None = 4
    eos_token_id: int | None = 3
    decoder_start_token_id: int | None = 7
    initializer_range: float = 0.02
    vocab_size: int = 16384

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
            )

        if isinstance(self.decoder_config, dict):
            self.decoder_config = CanaryDecoderConfig(**self.decoder_config)
        elif self.decoder_config is None:
            self.decoder_config = CanaryDecoderConfig()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.decoder_config.vocab_size != self.vocab_size:
            raise ValueError(
                f"The decoder config vocabulary size ({self.decoder_config.vocab_size}) does not match the Canary "
                f"config vocabulary size ({self.vocab_size})."
            )

    def get_text_config(self, *args, **kwargs):
        return self.decoder_config


logger = logging.get_logger(__name__)


class CanaryPositionalEmbedding(SinusoidsPositionEmbedding):
    """
    Identical to [`SinusoidsPositionEmbedding`] except that the timescales and the `1 / sqrt(channels)` scaling match
    NeMo's `FixedPositionalEncoding`, and it is indexed by `position_ids`.
    """

    def __init__(self, length: int, channels: int):
        max_timescale = 10000 ** ((channels - 2) / channels)
        super().__init__(length, channels, max_timescale)

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        emb = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        return emb.to(torch.get_default_dtype())

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.positional_embedding[position_ids] / math.sqrt(self.channels)


@auto_docstring
class CanaryPreTrainedModel(CohereAsrPreTrainedModel):
    config: CanaryConfig
    _no_split_modules = ["CanaryDecoderLayer"]

    def _get_feat_extract_output_lengths(self):
        raise AttributeError("Not needed for Canary")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, CanaryPositionalEmbedding):
            init.copy_(module.positional_embedding, module.compute_default_singular_positional_embedding())


class CanaryDecoder(CohereAsrDecoder):
    config: CanaryDecoderConfig

    def __init__(self, config: CanaryDecoderConfig):
        super().__init__(config)
        self.pos_emb = CanaryPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.proj = nn.Identity()


@auto_docstring(
    custom_intro="""
    The bare Canary model (FastConformer encoder + Transformer decoder) outputting raw hidden-states without any
    specific head on top.
    """
)
class CanaryModel(CohereAsrModel):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.decoder = CanaryDecoder(config.decoder_config)


@auto_docstring(
    custom_intro="""
    The Canary model with a language modeling head. Can be used for multilingual automatic speech recognition and
    speech-to-text translation.
    """
)
class CanaryForConditionalGeneration(CohereAsrForConditionalGeneration):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.proj_out = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=True)


__all__ = [
    "CanaryConfig",
    "CanaryDecoderConfig",
    "CanaryForConditionalGeneration",
    "CanaryModel",
    "CanaryPreTrainedModel",
]
