# Copyright 2026 The Sapient AI Authors and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from contextlib import nullcontext

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_masks_for_generate
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.generic import TransformersKwargs, is_flash_attention_requested, split_attention_implementation
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..nanochat.modeling_nanochat import NanoChatRMSNorm


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="sapientinc/HRM-Text-1B")
@strict
class HrmTextConfig(LlamaConfig):
    r"""
    H_cycles (`int`, *optional*, defaults to 2):
        Number of high-level cycles.
    L_cycles (`int`, *optional*, defaults to 3):
        Number of low-level cycles per H-cycle.
    L_bp_cycles (`list[int]`, *optional*, defaults to `[2]`):
        Training-time gradient-routing list; left-padded with `1`s up to `L_cycles` inside the model.
        Inference-time no-op.
    embedding_scale (`float`, *optional*):
        Token-embedding multiplier. If `None`, defaults to `1 / initializer_range`.
    prefix_lm (`bool`, *optional*, defaults to `True`):
        Instruction tokens attend bidirectionally, response tokens attend causally.
    num_layers_per_stack (`int`, *optional*):
        Real number of transformer blocks inside each
        of the H / L stacks. Set automatically on first construction: the value passed as
        `num_hidden_layers` is remembered here and `num_hidden_layers` is then rewritten to
        `num_layers_per_stack * H_cycles * (L_cycles + 1)` so that
        `DynamicCache(config=...)` pre-allocates one slot per unique attention invocation
        under the recurrent forward. Do not set this directly on first construction — pass
        the real per-stack count as `num_hidden_layers` and let `__post_init__` split it.
    """

    model_type = "hrm_text"

    base_model_tp_plan = {
        **{f"{stack}.layers.*.self_attn.q_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.k_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.v_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.gate_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.o_proj": "rowwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.mlp.gate_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.mlp.up_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.mlp.down_proj": "rowwise" for stack in ("L_module", "H_module")},
    }

    vocab_size: int = 151808
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    head_dim: int = 128
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    pretraining_tp = AttributeError()
    num_key_value_heads = AttributeError()

    H_cycles: int = 2
    L_cycles: int = 3
    L_bp_cycles: list[int] | None = None
    embedding_scale: float | None = None
    prefix_lm: bool = True
    num_layers_per_stack: int | None = None  # Usually inferred in post init

    def __post_init__(self, **kwargs):
        if self.L_bp_cycles is None:
            # Default `[2]` matches upstream `hrm_nocarry_more_bp_no_x`. Left-padding to length
            # `L_cycles` is performed inside [`HrmTextModel`] since it depends on `L_cycles`.
            self.L_bp_cycles = [2]

        if self.embedding_scale is None:
            self.embedding_scale = 1.0 / self.initializer_range

        if self.num_layers_per_stack is None:
            # Initial construction, or legacy checkpoint where `num_hidden_layers` carries the
            # real per-stack count: remember that value and rewrite `num_hidden_layers` to the
            # inflated total, so standard HF cache allocation gives us one slot per unique
            # attention invocation. Serialised configs round-trip as (inflated, real) pairs.
            self.num_layers_per_stack = self.num_hidden_layers
            self.num_hidden_layers = self.num_layers_per_stack * self.H_cycles * (self.L_cycles + 1)

        PreTrainedConfig.__post_init__(self, **kwargs)

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value: str | dict | None):
        if value is not None and self.prefix_lm:
            _, base_implementation = split_attention_implementation(value)
            if is_flash_attention_requested(requested_attention_implementation=base_implementation):
                raise ValueError(
                    f"`attn_implementation={value!r}` is not supported when "
                    "`config.prefix_lm=True`: FlashAttention cannot represent the PrefixLM 4-D mask "
                    "overlay. Use `'sdpa'` (default) or `'flex_attention'`, or set `config.prefix_lm=False`."
                )
        PreTrainedConfig._attn_implementation.__set__(self, value)


class HrmTextRMSNorm(NanoChatRMSNorm):
    pass


class HrmTextMLP(LlamaMLP):
    pass


class HrmTextAttention(LlamaAttention):
    def __init__(self, config: HrmTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_key_value_groups = 1  # Uses MHA instead of GQA
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        # Additional sigmoid gate applied at the end
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cycle_offset: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        gate_states = self.gate_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # Adjust cache slot by `cycle_offset` which is determined by it's current recurrent step through the stacks
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx + cycle_offset)

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
            **kwargs,
        )

        # Additional sigmoid gating (similar to Qwen3Next)
        attn_output = torch.sigmoid(gate_states) * attn_output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HrmTextDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HrmTextConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = HrmTextRMSNorm(eps=config.rms_norm_eps)
        self.post_attention_layernorm = HrmTextRMSNorm(eps=config.rms_norm_eps)


class HrmTextStack(nn.Module):
    """A single transformer stack — used twice inside, once as H module and once as L module"""

    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [HrmTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers_per_stack)]
        )
        self.final_norm = HrmTextRMSNorm(eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cycle_offset: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                cycle_offset=cycle_offset,
                **kwargs,
            )
        return self.final_norm(hidden_states)


@auto_docstring
class HrmTextPreTrainedModel(LlamaPreTrainedModel):
    config: HrmTextConfig

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        if attn_implementation is not None and self.config.prefix_lm:
            _, base_implementation = split_attention_implementation(attn_implementation)
            if is_flash_attention_requested(requested_attention_implementation=base_implementation):
                raise ValueError(
                    f"`attn_implementation={attn_implementation!r}` is not supported when "
                    "`config.prefix_lm=True`: FlashAttention cannot represent the PrefixLM 4-D mask "
                    "overlay. Use `'sdpa'` (default) or `'flex_attention'`, or set `config.prefix_lm=False`."
                )
        return PreTrainedModel._check_and_adjust_attn_implementation(
            self, attn_implementation, is_init_check, allow_all_kernels
        )

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, HrmTextModel):
            init.zeros_(module.z_L_init)
            # `z_L_init` is the frozen low-cycle initial state and never trains.
            module.z_L_init.requires_grad_(False)  # trf-ignore: TRF012


@auto_docstring
class HrmTextModel(LlamaModel):
    def __init__(self, config: HrmTextConfig):
        super().__init__(config)
        del self.layers
        del self.norm

        self.embedding_scale = config.embedding_scale

        # Recursive module structures
        self.L_module = HrmTextStack(config)
        self.H_module = HrmTextStack(config)
        # Initial state for the low cycle module
        self.z_L_init = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)

        raw_bp = list(config.L_bp_cycles)
        self.L_bp_cycles_padded = [1] * max(0, config.L_cycles - len(raw_bp)) + raw_bp

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        token_type_ids (`torch.LongTensor` of shape `(batch, seq_len)`, *optional*):
            Per-position bidirectional/causal indicator. Tokens with `token_type_ids == 1`
            form a single bidirectional block; all other positions are causal.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Additional scaling on the input embeds
        inputs_embeds = inputs_embeds * self.embedding_scale

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Create mask with optional prefix-based bidirectionality
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        is_first_iteration = past_key_values is None or not past_key_values.is_initialized
        if token_type_ids is not None and is_first_iteration:
            if self.config.prefix_lm:
                mask_kwargs["block_sequence_ids"] = torch.where(token_type_ids == 1, 0, -1)
            else:
                logger.warning_once("`token_type_ids` was provided but `config.prefix_lm=False`; ignoring it.")

        attention_mask = create_causal_mask(**mask_kwargs)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Hierarchical (H/L)-cycle recurrence
        #
        # `z_H` - slow / high-level state
        hidden_states_high_cycle = inputs_embeds
        # `z_L` - fast / low-level state
        hidden_states_low_cycle = (
            self.z_L_init.to(dtype=hidden_states_high_cycle.dtype, device=hidden_states_high_cycle.device)
            .expand_as(hidden_states_high_cycle)
            .contiguous()
        )

        # Cache-slot layout under the recurrent forward:
        #
        #   slot(h, l, layer)   = (h * (L_cycles + 1) + l) * num_layers_per_stack + layer
        #                                                       ^— L-stack invocation at (h, l)
        #   slot(h, H, layer)   = (h * (L_cycles + 1) + L_cycles) * num_layers_per_stack + layer
        #                                                       ^— trailing H-stack invocation
        #
        # That totals `num_layers_per_stack * H_cycles * (L_cycles + 1)` slots, i.e. the `config.num_hidden_layers`.
        num_layers_per_stack = self.config.num_layers_per_stack
        for high_cycle_idx in range(self.config.H_cycles):
            # `L_bp_cycles` k-step grad trick: only the trailing `num_grad_iterations` of the
            # `L_cycles` inner iterations propagate gradients; earlier iterations run under
            # `torch.no_grad()` to bound activation memory.
            num_grad_iterations = (
                self.L_bp_cycles_padded[high_cycle_idx] if high_cycle_idx < len(self.L_bp_cycles_padded) else 1
            )
            grad_threshold = self.config.L_cycles - num_grad_iterations
            for low_cycle_idx in range(self.config.L_cycles):
                cycle_offset = (high_cycle_idx * (self.config.L_cycles + 1) + low_cycle_idx) * num_layers_per_stack
                ctx = nullcontext() if low_cycle_idx >= grad_threshold else torch.no_grad()
                with ctx:
                    hidden_states_low_cycle = self.L_module(
                        hidden_states_low_cycle.to(hidden_states_high_cycle.device) + hidden_states_high_cycle,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        cycle_offset=cycle_offset,
                        **kwargs,
                    )

            cycle_offset = (high_cycle_idx * (self.config.L_cycles + 1) + self.config.L_cycles) * num_layers_per_stack

            hidden_states_high_cycle = self.H_module(
                hidden_states_high_cycle + hidden_states_low_cycle.to(hidden_states_high_cycle.device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cycle_offset=cycle_offset,
                **kwargs,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states_high_cycle,
            past_key_values=past_key_values,
        )


@auto_docstring
class HrmTextForCausalLM(LlamaForCausalLM):
    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ) -> dict:
        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if token_type_ids is not None and is_first_iteration:
            if config.prefix_lm:
                mask_kwargs["block_sequence_ids"] = torch.where(token_type_ids == 1, 0, -1)
            else:
                logger.warning_once("`token_type_ids` was provided but `config.prefix_lm=False`; ignoring it.")

        return create_masks_for_generate(**mask_kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        token_type_ids (`torch.LongTensor` of shape `(batch, seq_len)`, *optional*):
            Per-position bidirectional/causal indicator. Tokens with `token_type_ids == 1`
            form a single bidirectional block; all other positions are causal.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HrmTextConfig",
    "HrmTextForCausalLM",
    "HrmTextModel",
    "HrmTextPreTrainedModel",
]
