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

"""Nanbeige4.2, a decoder with a shared stack of transformer layers."""

from copy import deepcopy

import torch
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults, no_inherit_decorator
from ...utils.output_capturing import capture_outputs
from ...utils.type_validators import interval
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


# Keep the interval factory separate for modular config inheritance.
loop_count_field = interval(min=1)


@auto_docstring(checkpoint="Nanbeige/Nanbeige4.2-3B")
@strict
class NanbeigeConfig(LlamaConfig):
    r"""
    num_loops (`int`, *optional*, defaults to 2):
        Number of times the decoder layers run per forward pass, with shared weights. Each pass has its own key-value
        cache entries when caching is enabled. By default, the output RMSNorm runs after every pass.
    loop_loss_weights (`list[float]`, *optional*):
        Legacy setting. A nonempty list determines the number of passes as its length plus one.
    skip_loop_final_norm (`bool`, *optional*, defaults to `False`):
        Apply the output RMSNorm only after the last pass when set to `True`.

    ```python
    >>> from transformers import NanbeigeConfig, NanbeigeModel

    >>> configuration = NanbeigeConfig()
    >>> model = NanbeigeModel(configuration)
    >>> configuration = model.config
    ```
    """

    default_theta = 70_000_000.0

    vocab_size: int = 166144
    hidden_size: int = 3072
    intermediate_size: int = 10752
    num_hidden_layers: int = 22
    num_attention_heads: int = 48
    num_key_value_heads: int | None = 8
    head_dim: int | None = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 0
    bos_token_id: int | None = 166100
    eos_token_id: int | list[int] | None = 166101
    num_loops: int = loop_count_field(default=2)
    loop_loss_weights: list[float] | None = None
    skip_loop_final_norm: bool = False


class NanbeigeAttention(LlamaAttention):
    @no_inherit_decorator
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        cache_offset: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx + cache_offset)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class NanbeigeDecoderLayer(LlamaDecoderLayer):
    @no_inherit_decorator
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


class NanbeigePreTrainedModel(LlamaPreTrainedModel):
    pass


class NanbeigeModel(LlamaModel):
    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        num_layers = len(self.layers)
        loop_loss_weights = self.config.loop_loss_weights
        num_loops = len(loop_loss_weights) + 1 if loop_loss_weights else self.config.num_loops
        if cache is not None and num_loops > 1 and len(cache.layers) == num_layers:
            # Each pass reuses the decoder weights and writes to its own cache slots.
            cache.layers += [deepcopy(layer) for layer in cache.layers * (num_loops - 1)]

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :] + past_seen_tokens
            )

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache=cache,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        hidden_states = inputs_embeds
        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        output_attentions = kwargs.pop("output_attentions", self.config.output_attentions)

        for loop_idx in range(num_loops):
            is_final_loop = loop_idx == num_loops - 1
            outputs = self._run_loop(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache,
                cache_offset=loop_idx * num_layers,
                use_cache=use_cache,
                apply_norm=is_final_loop or not self.config.skip_loop_final_norm,
                output_hidden_states=output_hidden_states if is_final_loop else False,
                output_attentions=output_attentions if is_final_loop else False,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state

        return outputs

    @capture_outputs
    def _run_loop(
        self,
        hidden_states: torch.Tensor,
        apply_norm: bool,
        cache: Cache | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, cache=cache, **kwargs)
        if apply_norm:
            hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


class NanbeigeForCausalLM(LlamaForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""Return the final decoder pass's next-token logits in float32."""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "NanbeigeModel",
    "NanbeigePreTrainedModel",
]
