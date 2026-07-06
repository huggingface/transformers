# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""GraniteSWA: Granite with Sliding Window Attention and learnable attention sinks.

GraniteSWA augments the Granite architecture with two changes:
  * per-layer sliding-window attention (controlled by ``layer_types``), and
  * a learnable per-head attention sink.

The sink rescales the attention output by ``sigmoid(logsumexp(attn_logits) - sink)``. This is
mathematically equivalent to adding a single extra (learnable) logit to the softmax denominator
-- i.e. the ``s_aux`` auxiliary-logit mechanism used by GPT-OSS. The eager path computes the
``sigmoid``-scaling explicitly, while the FlashAttention-3 and FlashAttention-4 backends apply
the same sink through the shared attention dispatch by passing ``s_aux``.
"""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import (
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteModel,
    GranitePreTrainedModel,
)
from ..llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-research/granite-swash-2b")
@strict
class GraniteSWAConfig(GraniteConfig):
    r"""
    sliding_window (`int`, *optional*, defaults to 128):
        Size of the sliding attention window used by layers whose `layer_types` entry is
        `"sliding_attention"`.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type, each either `"full_attention"` or `"sliding_attention"`. When
        `None`, every fourth layer (`i % 4 == 0`) uses full attention and the rest use sliding
        window attention.
    no_rope_layers (`list[int]`, *optional*):
        Per-layer flag for rotary position embeddings, `1` to apply RoPE and `0` for NoPE (no
        positional embedding). When `None`, defaults to all-RoPE (`1` for every layer).

    ```python
    >>> from transformers import GraniteSWAModel, GraniteSWAConfig

    >>> # Initializing a GraniteSWA configuration
    >>> configuration = GraniteSWAConfig()

    >>> # Initializing a model from the configuration
    >>> model = GraniteSWAModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_swa"

    vocab_size: int = 100352
    hidden_size: int = 2560
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 20
    num_key_value_heads: int | None = 4
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    bos_token_id: int | None = 100257
    eos_token_id: int | list[int] | None = 100257
    tie_word_embeddings: bool = True
    sliding_window: int | None = 128
    layer_types: list[str] | None = None
    no_rope_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]

        # Per-layer RoPE vs NoPE (1 = apply RoPE, 0 = NoPE). Default is all-RoPE;
        # set `no_rope_layers` explicitly to make specific layers NoPE.
        if self.no_rope_layers is None:
            self.no_rope_layers = [1] * self.num_hidden_layers

        super().__post_init__(**kwargs)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with a learnable per-head sink applied as post-attention LSE scaling."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Sink scaling: sigmoid(logsumexp(logits) - sink), equivalent to an extra softmax-denominator logit.
    lse = torch.logsumexp(attn_weights, dim=-1)  # (batch, num_heads, q_len)
    sink_scale = torch.sigmoid((lse - module.sinks.view(1, -1, 1)).to(torch.float32))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GraniteSWAAttention(LlamaAttention):
    """Granite attention with per-layer sliding window and a learnable per-head attention sink."""

    def __init__(self, config: GraniteSWAConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.scaling = config.attention_multiplier
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.use_rope = bool(config.no_rope_layers[layer_idx])

        # Learnable per-head attention sink (applied as an auxiliary softmax logit).
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Granite: learnable attention sink (FA3/FA4 backends)
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GraniteSWADecoderLayer(GraniteDecoderLayer):
    def __init__(self, config: GraniteSWAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteSWAAttention(config=config, layer_idx=layer_idx)


class GraniteSWAPreTrainedModel(GranitePreTrainedModel):
    _supports_sdpa = False
    _supports_flex_attn = False
    _compatible_flash_implementations = ["kernels-community/vllm-flash-attn3", "flash_attention_4"]
    _can_record_outputs = {
        "hidden_states": GraniteSWADecoderLayer,
        "attentions": GraniteSWAAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, GraniteSWAAttention):
            init.zeros_(module.sinks)


class GraniteSWAModel(GraniteModel):
    def __init__(self, config: GraniteSWAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteSWADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Create the masks once per layer type (full vs sliding window) and reuse across layers.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteSWAForCausalLM(GraniteForCausalLM):
    pass


__all__ = ["GraniteSWAConfig", "GraniteSWAForCausalLM", "GraniteSWAModel", "GraniteSWAPreTrainedModel"]
