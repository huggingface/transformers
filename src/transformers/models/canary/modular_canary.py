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
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..cohere_asr.modeling_cohere_asr import (
    CohereAsrDecoder,
    CohereAsrForConditionalGeneration,
    CohereAsrModel,
    CohereAsrPreTrainedModel,
)
from ..llama.configuration_llama import LlamaConfig
from ..qwen2_5_omni.modeling_qwen2_5_omni import SinusoidsPositionEmbedding


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/canary-1b-v2")
@strict
class CanaryDecoderConfig(LlamaConfig):
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
    attention_bias: bool = True
    head_dim: int = 128

    rms_norm_eps = AttributeError()
    pretraining_tp = AttributeError()
    rope_parameters = AttributeError()
    mlp_bias = AttributeError()
    tie_word_embeddings = AttributeError()
    base_model_tp_plan = AttributeError()
    base_model_pp_plan = AttributeError()


@auto_docstring(checkpoint="nvidia/canary-1b-v2")
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


class CanaryPositionalEmbedding(SinusoidsPositionEmbedding):
    """
    Identical to [`SinusoidsPositionEmbedding`] except that the timescales and the `1 / sqrt(channels)` scaling match
    NeMo's `FixedPositionalEncoding`, and it is indexed by `position_ids`.
    """

    def __init__(self, length: int, channels: int):
        super().__init__(length, channels)
        self.max_timescale = 10000 ** ((channels - 2) / channels)

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

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
        del self.proj

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions:
        r"""
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding indices in `encoder_hidden_states`. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Fixed sinusoidal position embedding added to token embeddings, then layernorm
        pos_emb = self.pos_emb(position_ids.squeeze(0))
        pos_emb = pos_emb.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = self.embedding_layernorm(inputs_embeds + pos_emb)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


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
