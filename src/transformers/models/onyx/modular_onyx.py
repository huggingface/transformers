# coding=utf-8
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
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    eager_attention_forward,
)
from ..llama4.modeling_llama4 import Llama4TextL2Norm


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class OnyxConfig(Gemma2Config, PreTrainedConfig):
    r"""
    qk_scale_factor (`float`, *optional*, defaults to 43.7840518911):
        Multiplier applied to Q after QK-norm, before the standard `1/sqrt(head_dim)` attention scaling.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply a scaleless RMSNorm to Q and K before rotary.
    use_attn_output_gate (`bool`, *optional*, defaults to `True`):
        Whether to gate the per-head attention output with `sigmoid(output_gate_proj(hidden))`.
    output_multiplier (`float`, *optional*, defaults to 0.19611613513818404):
        Scale applied to logits before the final tanh softcap.
    normalize_tok_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to apply a scaleless RMSNorm to the token embeddings before the decoder stack.
    post_norm_eps (`float`, *optional*, defaults to 1e-8):
        Epsilon used for the post-attention and post-FFN norms (which sit between the sub-layer output and the residual).
    every_n_layers_nope (`int`, *optional*, defaults to 4):
        iRoPE stride. NoPE (no rotary) is applied every N layers, counting backward from the last layer.
    no_rope_layers (`list[int]`, *optional*):
        Explicit per-layer rotary mask: 1 = apply rotary, 0 = NoPE. Derived from `every_n_layers_nope` if unset.
    """

    model_type = "onyx"

    vocab_size: int = 202_048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 16_384
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 200_000
    eos_token_id: int | list[int] | None = 200_001
    pad_token_id: int | None = None
    sliding_window: int | None = 2048
    final_logit_softcapping: float | None = 20.0
    attn_logit_softcapping: float | None = None
    layer_types: list[str] | None = None

    # Onyx-specific fields
    qk_scale_factor: float = 43.7840518911
    use_qk_norm: bool = True
    use_attn_output_gate: bool = True
    output_multiplier: float = 0.19611613513818404
    normalize_tok_embeddings: bool = True
    post_norm_eps: float = 1e-8
    every_n_layers_nope: int = 4
    no_rope_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        # Accept the legacy `hidden_act` alias from checkpoints saved with the trust_remote_code impl.
        if (legacy_act := kwargs.pop("hidden_act", None)) is not None:
            self.hidden_activation = legacy_act

        # iRoPE mask: NoPE layers counted backward from the last layer.
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                0 if (self.num_hidden_layers - 1 - i) % self.every_n_layers_nope == 0 else 1
                for i in range(self.num_hidden_layers)
            ]

        # Full attention for NoPE layers, sliding otherwise (Onyx's default layout matches
        # the sliding_window_pattern [w, w, w, 0] used in the reference config).
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if self.no_rope_layers[i] == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        PreTrainedConfig.__post_init__(self, **kwargs)


class OnyxRMSNorm(Gemma2RMSNorm):
    pass


class OnyxScalelessRMSNorm(Llama4TextL2Norm):
    pass


# Weight-as-scale, no offset — used only for the final norm.
class OnyxFinalRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class OnyxMLP(Gemma2MLP):
    pass


class OnyxRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class OnyxAttention(Gemma2Attention):
    def __init__(self, config: OnyxConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.head_dim**-0.5
        self.attn_logit_softcapping = None

        self.use_rope = config.no_rope_layers[layer_idx] == 1

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = OnyxScalelessRMSNorm(eps=config.rms_norm_eps)
            self.scale_query_by = config.qk_scale_factor / (config.head_dim**0.5)

        self.use_output_gate = config.use_attn_output_gate
        if self.use_output_gate:
            self.output_gate_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.qk_norm(query_states) * self.scale_query_by
            key_states = self.qk_norm(key_states)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_interleave(query_states, key_states, cos, sin)

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
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        # attn_output shape here: (batch, seq, num_heads, head_dim)

        if self.use_output_gate:
            gate = torch.sigmoid(self.output_gate_proj(hidden_states).view(*attn_output.shape))
            attn_output = gate * attn_output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OnyxDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OnyxConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = OnyxAttention(config=config, layer_idx=layer_idx)
        self.mlp = OnyxMLP(config)
        self.input_layernorm = OnyxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = OnyxRMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.post_attention_layernorm = OnyxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_ffn_norm = OnyxRMSNorm(config.hidden_size, eps=config.post_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + self.post_attn_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_ffn_norm(hidden_states)

        return hidden_states


class OnyxPreTrainedModel(Gemma2PreTrainedModel):
    config: OnyxConfig
    base_model_prefix = "model"
    _no_split_modules = ["OnyxDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        # Gemma2's init assumes every RMSNorm has a `weight`, but OnyxScalelessRMSNorm doesn't.
        # Route it to the base PreTrainedModel init (which handles Linear/Embedding via initializer_range)
        # and skip the RMSNorm branch for the scaleless variant.
        if isinstance(module, OnyxScalelessRMSNorm):
            return
        super()._init_weights(module)


class OnyxModel(Gemma2Model):
    """Gemma2 text model, but with a plain (unscaled) embedding + a scaleless RMSNorm on the embedded tokens."""

    config: OnyxConfig

    def __init__(self, config: OnyxConfig):
        super().__init__(config)
        # Replace Gemma2's sqrt(hidden_size)-scaled embedding with a plain one — Onyx uses norm, not scale.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Final norm uses weight-as-scale (no offset), unlike the per-layer OnyxRMSNorm.
        self.norm = OnyxFinalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.normalize_tok_embeddings:
            self.embed_norm = OnyxScalelessRMSNorm(eps=config.rms_norm_eps)
        else:
            self.embed_norm = None
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.embed_norm is not None:
                inputs_embeds = self.embed_norm(inputs_embeds)
            input_ids = None  # avoid double-embedding in the super().forward call
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class OnyxForCausalLM(Gemma2ForCausalLM):
    """Gemma2 causal LM, with an extra `output_multiplier` factor applied before the tanh softcap."""

    config: OnyxConfig

    def __init__(self, config: OnyxConfig):
        super().__init__(config)
        self.model = OnyxModel(config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Onyx pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "OnyxConfig",
    "OnyxPreTrainedModel",
    "OnyxModel",
    "OnyxForCausalLM",
]
