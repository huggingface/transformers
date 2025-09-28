# coding=utf-8
# Copyright 2025
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

from typing import Optional, List, Tuple

import torch
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import logging, TransformersKwargs
from ...masking_utils import create_sliding_window_causal_mask, create_causal_mask
from ...cache_utils import Cache, DynamicCache
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Config (Llama-compatible weights; model_type='cwm' for modular converter routing)
# -----------------------------------------------------------------------------
class CwmTextConfig(LlamaConfig):
    """
    Llama3-compatible configuration with layer-interleaved sliding-window attention
    """

    model_type = "llama"  # for VLLM too


    def __init__(
        self,
        # Llama fields
        vocab_size: int = 128256,
        hidden_size: int = 6144,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 64,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id=(128001, 128008, 128009),
        bos_token_id: int = 128000,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1_000_000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pretraining_tp: int = 1,
        mlp_bias: bool = False,
        rope_scaling: Optional[dict] = None,
        # CWM interleaved sliding window fields
        sliding_window: int = 8192,
        layer_types: Optional[List[str]] = None,  # ["full_attention"|"sliding_attention"] per layer
        window_pattern: Optional[int] = None,
        global_window: Optional[int] = None,  # causal
        **kwargs,
    ):
        if rope_scaling is None:
            rope_scaling = {
                "factor": 16.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }

        if layer_types is None:
            if window_pattern is None or window_pattern <= 0:
                window_pattern = 4
            layer_types = [
                ("full_attention" if (i % window_pattern == 0) else "sliding_attention")
                for i in range(num_hidden_layers)
            ]

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=list(eos_token_id),
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rope_scaling=rope_scaling,
            pretraining_tp=pretraining_tp,
            mlp_bias=mlp_bias,
            **kwargs,
        )

        self.sliding_window = int(sliding_window)
        self.layer_types = list(layer_types)
        self.window_pattern = int(window_pattern) if window_pattern is not None else None
        self.global_window = None if global_window is None else int(global_window)



class CwmConfig(CwmTextConfig):
    pass


class CwmDecoderLayer(LlamaDecoderLayer):
    """
    Same as LlamaDecoderLayer, but we inject an additive mask (local or causal) per layer
    based on config.layer_types / sliding_window / global_window before calling attention
    """

    def __init__(self, config: CwmTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx  # Ensure layer_idx is stored as instance attribute
        self._cwm_layer_types = getattr(config, "layer_types", None)
        self.layer_type = None
        if self._cwm_layer_types is not None:
            self.layer_type = self._cwm_layer_types[self.layer_idx]
        self.sliding_window = int(getattr(config, "sliding_window", 0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sliding_window=self.sliding_window if self.layer_type == "sliding_attention" else None,
            **kwargs,
        )

        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
            return outputs
        else:
            return hidden_states


class CwmMLP(LlamaMLP):
    pass


class CwmRMSNorm(LlamaRMSNorm):
    pass


class CwmRotaryEmbedding(LlamaRotaryEmbedding):
    pass

class CwmModelOutputWithPast(BaseModelOutputWithPast):
    pass


class CwmPreTrainedModel(LlamaPreTrainedModel):
    config_class = CwmTextConfig
    base_model_prefix = "model"


class CwmModel(LlamaModel):
    config_class = CwmTextConfig

    def __init__(self, config: CwmTextConfig):
        super().__init__(config)
        self.layers = torch.nn.ModuleList([
            CwmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CwmModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }


        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return CwmModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class CwmForCausalLM(LlamaForCausalLM):
    config_class = CwmTextConfig

    def __init__(self, config: CwmTextConfig):
        super().__init__(config)
        self.model = CwmModel(config)


__all__ = [
    "CwmTextConfig",
    "CwmConfig",
    "CwmPreTrainedModel",
    "CwmModel",
    "CwmForCausalLM",
    "CwmMLP",
    "CwmRMSNorm",
    "CwmRotaryEmbedding",
    "CwmDecoderLayer",
]
