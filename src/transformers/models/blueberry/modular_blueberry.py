# Copyright 2026 Dustin Loring and the HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_masks_for_generate
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..nanochat.modeling_nanochat import (
    NanoChatMLP,
    NanoChatRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@auto_docstring(checkpoint="CrowdMind/Blueberry-base")
@strict
class BlueberryConfig(PreTrainedConfig):
    r"""
    num_unique_layers (`int`, *optional*, defaults to 8):
        Number of distinct decoder blocks. `num_hidden_layers` counts their executions, including repeated passes.
    loop_repeats (`int`, *optional*, defaults to 3):
        Total passes through the middle half of the decoder blocks.
    loop_gate (`bool`, *optional*, defaults to `True`):
        Whether to learn a per-token mix of the prelude and previous pass at loop boundaries.
    window_pattern (`str`, *optional*, defaults to `"SSSL"`):
        Pattern tiled across distinct blocks. `S` uses a short sliding window and `L` uses full causal attention.
        The final block always uses full attention.
    sliding_window (`int`, *optional*):
        Short attention window including the current token. Defaults to the quarter-context window rounded up
        to a multiple of 128 predecessors, plus the current token.
    value_gate_channels (`int`, *optional*, defaults to 12):
        Number of hidden channels used by each attention value gate.
    smear_gate_channels (`int`, *optional*, defaults to 24):
        Number of token embedding channels used by the previous-token mixing gate.
    """

    model_type = "blueberry"

    vocab_size: int = 32768
    hidden_size: int = 512
    intermediate_size: int | None = None
    num_unique_layers: int = 8
    num_attention_heads: int = 4
    num_key_value_heads: int | None = None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    loop_repeats: int = 3
    loop_gate: bool = True
    window_pattern: str = "SSSL"
    sliding_window: int | None = None
    value_gate_channels: int = 12
    smear_gate_channels: int = 24
    final_logit_softcapping: float = 15.0
    use_cache: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.intermediate_size = self.intermediate_size if self.intermediate_size is not None else 4 * self.hidden_size
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100_000.0}
        if self.sliding_window is None:
            self.sliding_window = ((self.max_position_embeddings + 511) // 512) * 128 + 1
        self.window_pattern = self.window_pattern.upper()
        super().__post_init__(**kwargs)

    @property
    def loop_start(self):
        return self.num_unique_layers // 4

    @property
    def layer_indices(self):
        indices = tuple(range(self.num_unique_layers))
        loop_end = self.loop_start + self.num_unique_layers // 2
        return (
            indices[: self.loop_start] + indices[self.loop_start : loop_end] * self.loop_repeats + indices[loop_end:]
        )

    @property
    def num_hidden_layers(self):
        return len(self.layer_indices)

    @property
    def layer_types(self):
        if not self.window_pattern:
            return []
        types = [
            "sliding_attention" if self.window_pattern[index % len(self.window_pattern)] == "S" else "full_attention"
            for index in self.layer_indices
        ]
        # The last cache slot holds full attention KV and the previous unsmeared token embedding.
        return types[:-1] + ["hybrid"]

    def has_value_embedding(self, layer_idx):
        return layer_idx % 2 == (self.num_unique_layers - 1) % 2

    def validate_architecture(self):
        if self.num_unique_layers < 1 or self.loop_repeats < 1:
            raise ValueError("Blueberry requires at least one decoder block and one loop pass.")
        if self.loop_repeats > 1 and (self.num_unique_layers < 4 or self.num_unique_layers % 2):
            raise ValueError("Multiple loop passes require an even number of distinct blocks of at least four.")
        if self.loop_gate and self.loop_repeats == 1:
            raise ValueError("The loop gate requires at least two loop passes.")
        if not self.window_pattern or set(self.window_pattern) - {"S", "L"}:
            raise ValueError("`window_pattern` must contain one or more S or L characters.")
        if self.sliding_window < 1:
            raise ValueError("`sliding_window` must be positive.")
        if (
            self.num_attention_heads < 1
            or self.hidden_size % (2 * self.num_attention_heads)
            or self.num_key_value_heads < 1
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Blueberry requires an even head dimension and valid grouped heads.")
        if not (0 < self.value_gate_channels <= self.hidden_size and 0 < self.smear_gate_channels <= self.hidden_size):
            raise ValueError("Gate channel counts must be positive and at most hidden_size.")


def blueberry_rms_norm(hidden_states):
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],))


class BlueberryRotaryEmbedding(NanoChatRotaryEmbedding):
    pass


class BlueberryMLP(NanoChatMLP):
    pass


class BlueberryAttention(nn.Module):
    def __init__(self, config: BlueberryConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_causal = True

        kv_size = config.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value_gate = (
            nn.Linear(config.value_gate_channels, config.num_key_value_heads, bias=False)
            if config.has_value_embedding(layer_idx)
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        value_embeddings: torch.Tensor | None = None,
        cache: Cache | None = None,
        execution_idx: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(head_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(head_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)

        if value_embeddings is not None:
            gate_input = hidden_states[..., : self.config.value_gate_channels]
            gate = 3 * torch.sigmoid(self.value_gate(gate_input)).transpose(1, 2).unsqueeze(-1)
            value = value + gate * value_embeddings.view(head_shape).transpose(1, 2).to(value.dtype)

        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        query = 1.2 * blueberry_rms_norm(query)
        key = 1.2 * blueberry_rms_norm(key)
        if cache is not None:
            key, value = cache.update(key, value, execution_idx, kwargs=kwargs)

        attention = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        output, weights = attention(self, query, key, value, attention_mask, scaling=self.head_dim**-0.5, **kwargs)
        return self.o_proj(output.reshape(*input_shape, -1)), weights


class BlueberryDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BlueberryConfig, layer_idx: int):
        super().__init__()
        self.self_attn = BlueberryAttention(config, layer_idx)
        self.mlp = BlueberryMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        value_embeddings: torch.Tensor | None = None,
        cache: Cache | None = None,
        execution_idx: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attention_output, _ = self.self_attn(
            blueberry_rms_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            value_embeddings=value_embeddings,
            cache=cache,
            execution_idx=execution_idx,
            **kwargs,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states + self.mlp(blueberry_rms_norm(hidden_states))


@auto_docstring
class BlueberryPreTrainedModel(PreTrainedModel):
    config: BlueberryConfig
    base_model_prefix = "model"
    _no_split_modules = ["BlueberryDecoderLayer"]
    _supports_sdpa = True
    _can_compile_fullgraph = True
    supports_gradient_checkpointing = True
    _can_record_outputs = {"hidden_states": BlueberryDecoderLayer, "attentions": BlueberryAttention}

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        value_embeds = self.base_model.value_embeds
        for key, embedding in value_embeds.items():
            value_embeds[key] = self._get_resized_embeddings(
                embedding, new_num_tokens, pad_to_multiple_of, mean_resizing
            )
        return embeddings

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        bound = (3 / self.config.hidden_size) ** 0.5
        if isinstance(module, BlueberryModel):
            init.normal_(module.embed_tokens.weight, std=0.8)
            for embedding in module.value_embeds.values():
                init.uniform_(embedding.weight, -bound, bound)
            init.copy_(module.resid_lambdas, torch.linspace(1.15, 1.05, self.config.num_unique_layers))
            init.copy_(module.x0_lambdas, torch.linspace(0.20, 0.05, self.config.num_unique_layers))
            init.zeros_(module.smear_lambda)
            init.uniform_(module.smear_gate.weight, 0.0, 0.02)
            init.constant_(module.backout_lambda, 0.2)
            if module.loop_gate is not None:
                init.zeros_(module.loop_gate.weight)
                init.constant_(module.loop_gate_bias, -4.0)
        elif isinstance(module, BlueberryAttention):
            for projection in (module.q_proj, module.k_proj, module.v_proj):
                init.uniform_(projection.weight, -bound, bound)
            init.zeros_(module.o_proj.weight)
            if module.value_gate is not None:
                init.uniform_(module.value_gate.weight, 0.0, 0.02)
        elif isinstance(module, BlueberryMLP):
            init.uniform_(module.fc1.weight, -0.4 * bound, 0.4 * bound)
            init.zeros_(module.fc2.weight)
        elif isinstance(module, BlueberryForCausalLM):
            init.normal_(module.lm_head.weight, std=0.001)


@auto_docstring
class BlueberryModel(BlueberryPreTrainedModel):
    def __init__(self, config: BlueberryConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BlueberryDecoderLayer(config, idx) for idx in range(config.num_unique_layers)])
        self.rotary_emb = BlueberryRotaryEmbedding(config)
        self.layer_indices = config.layer_indices
        self.attention_types = tuple(
            "full_attention" if layer_type == "hybrid" else layer_type for layer_type in config.layer_types
        )

        kv_size = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        self.value_embeds = nn.ModuleDict(
            {
                str(idx): nn.Embedding(config.vocab_size, kv_size)
                for idx in range(config.num_unique_layers)
                if config.has_value_embedding(idx)
            }
        )
        self.resid_lambdas = nn.Parameter(torch.empty(config.num_unique_layers))
        self.x0_lambdas = nn.Parameter(torch.empty(config.num_unique_layers))
        self.smear_gate = nn.Linear(config.smear_gate_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.empty(1))
        self.backout_lambda = nn.Parameter(torch.empty(1))
        self.loop_gate = nn.Linear(2 * config.hidden_size, 1, bias=False) if config.loop_gate else None
        self.loop_gate_bias = nn.Parameter(torch.empty(1)) if config.loop_gate else None

        loop_size = config.num_unique_layers // 2
        self.loop_pass_starts = tuple(config.loop_start + loop_size * idx for idx in range(1, config.loop_repeats))
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Token IDs used to look up both the input and per-layer value embeddings.
        """
        if input_ids is None:
            raise ValueError("Blueberry requires input_ids to look up the per-layer value embeddings.")
        inputs_embeds = self.embed_tokens(input_ids)

        if self.gradient_checkpointing and self.training:
            use_cache = False
        if use_cache is False:
            cache = None
        elif cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -input_ids.shape[1] :]
            else:
                past_length = cache.get_seq_length() if cache is not None else 0
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0) + past_length

        masks = attention_mask
        if not isinstance(masks, dict):
            masks = create_masks_for_generate(self.config, inputs_embeds, attention_mask, cache, position_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = blueberry_rms_norm(inputs_embeds)
        history = hidden_states.transpose(1, 2)
        if cache is not None:
            history = cache.update_conv_state(history, len(self.layer_indices) - 1, conv_kernel_size=1)
        previous = F.pad(history, (1, 0))[..., -input_ids.shape[1] - 1 : -1].transpose(1, 2)
        smear_input = hidden_states[..., : self.config.smear_gate_channels]
        smear = self.smear_lambda.to(hidden_states.dtype) * torch.sigmoid(self.smear_gate(smear_input))
        hidden_states = hidden_states + smear * previous * (position_ids > 0).unsqueeze(-1)

        initial_states = hidden_states
        backout_states = prelude = hidden_states
        backout_idx = len(self.layer_indices) // 2
        for execution_idx, (layer_idx, attention_type) in enumerate(zip(self.layer_indices, self.attention_types)):
            if self.loop_gate is not None:
                if execution_idx == self.config.loop_start:
                    prelude = hidden_states
                elif execution_idx in self.loop_pass_starts:
                    gate_input = torch.cat((blueberry_rms_norm(prelude), blueberry_rms_norm(hidden_states)), dim=-1)
                    gate = torch.sigmoid(self.loop_gate(gate_input) + self.loop_gate_bias.to(hidden_states.dtype))
                    hidden_states = torch.lerp(prelude, hidden_states, gate)

            hidden_states = self.resid_lambdas[layer_idx] * hidden_states + self.x0_lambdas[layer_idx] * initial_states
            value_embeddings = (
                self.value_embeds[str(layer_idx)](input_ids) if str(layer_idx) in self.value_embeds else None
            )
            hidden_states = self.layers[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=masks[attention_type],
                value_embeddings=value_embeddings,
                cache=cache,
                execution_idx=execution_idx,
                **kwargs,
            )
            if execution_idx == backout_idx:
                backout_states = hidden_states

        hidden_states = hidden_states - self.backout_lambda.to(hidden_states.dtype) * backout_states
        return BaseModelOutputWithPast(last_hidden_state=blueberry_rms_norm(hidden_states), cache=cache)


@auto_docstring
class BlueberryForCausalLM(BlueberryPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BlueberryConfig):
        super().__init__(config)
        self.model = BlueberryModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # The per-layer value embeddings also require the original token IDs.
        return super().prepare_inputs_for_generation(input_ids, **kwargs)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Token IDs used to look up both the input and per-layer value embeddings.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        softcap = self.config.final_logit_softcapping
        logits = softcap * torch.tanh(self.lm_head(hidden_states).float() / softcap)
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs) if labels is not None else None
        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["BlueberryConfig", "BlueberryForCausalLM", "BlueberryModel", "BlueberryPreTrainedModel"]
