# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Normalized latent recurrent Dory, with shared weights and explicit cache slots."""

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...cache_utils import Cache, DynamicLayer
from ...generation import GenerationMixin
from ...integrations import is_deepspeed_zero3_enabled
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from .configuration_dory import DoryConfig


def l2_normalize(hidden_states, dim=-1):
    normalized_states = hidden_states.float()
    normalized_states = normalized_states * torch.rsqrt(normalized_states.pow(2).sum(dim, keepdim=True) + 1e-6)
    return normalized_states.to(hidden_states.dtype)


def normalized_residual_update(hidden_states, update, alpha, scale):
    """Normalized interpolation from nGPT: https://arxiv.org/abs/2410.01131."""
    normalized_states = l2_normalize(hidden_states).float()
    normalized_update = l2_normalize(update).float()
    return l2_normalize(
        normalized_states + (alpha.float() * scale).abs() * (normalized_update - normalized_states)
    ).to(hidden_states.dtype)


class DoryCacheLayer(DynamicLayer):
    """Overwrite the current chunk on later loops instead of appending it again."""

    def __init__(self, sliding_window=None):
        super().__init__()
        self.sliding_window = sliding_window
        self.is_sliding = sliding_window is not None
        self.cumulative_length = 0
        self.record_past = False

    def update(self, key_states, value_states, *, start_position, **kwargs):
        if self.keys is None:
            self.lazy_initialization(key_states, value_states)
            self.keys, self.values = key_states, value_states
        else:
            offset = self.cumulative_length - self.keys.shape[-2]
            prefix_length = start_position - offset
            if not 0 <= prefix_length <= self.keys.shape[-2]:
                raise ValueError("Recurrent cache update starts outside the retained history")
            self.keys = torch.cat((self.keys[..., :prefix_length, :], key_states), dim=-2)
            self.values = torch.cat((self.values[..., :prefix_length, :], value_states), dim=-2)
        self.cumulative_length = start_position + key_states.shape[-2]
        visible = self.sliding_window - 1 + key_states.shape[-2] if self.is_sliding else self.keys.shape[-2]
        return self.keys[..., -visible:, :], self.values[..., -visible:, :]

    def finish_forward(self):
        # Later loops need the prefix preceding this chunk, so trim only once
        # every loop has finished. Clone to release the full prefill storage.
        if self.keys is not None and self.is_sliding and not self.record_past:
            start = max(0, self.keys.shape[-2] - self.sliding_window + 1)
            self.keys = self.keys[..., start:, :].clone()
            self.values = self.values[..., start:, :].clone()

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self, query_length):
        offset = max(0, self.cumulative_length - self.sliding_window + 1) if self.is_sliding else 0
        return self.cumulative_length - offset + query_length, offset

    def get_max_length(self):
        return self.sliding_window if self.is_sliding else -1

    def activate_past_recording(self):
        self.record_past = True

    def crop(self, tokens_to_remove):
        if tokens_to_remove > 0:
            raise ValueError("Use a negative number of tokens when cropping DoryCache")
        if not self.is_initialized:
            return
        if tokens_to_remove < 0:
            if self.is_sliding and not self.record_past and self.cumulative_length >= self.sliding_window:
                raise RuntimeError("Call activate_past_recording before rolling back a sliding cache")
            super().crop(tokens_to_remove)
            self.cumulative_length = max(0, self.cumulative_length + tokens_to_remove)
        record_past = self.record_past
        self.record_past = False
        self.finish_forward()
        self.record_past = record_past


class DoryCache(Cache):
    """Dynamic cache indexed by virtual attention layer; no tensors for MLP layers."""

    layers: list[DoryCacheLayer]

    def __init__(self, config=None):
        if config is None:
            super().__init__(layer_class_to_replicate=DoryCacheLayer)
        else:
            loops = config.n_recurrent_loops if config.recurrent_kv_cache_mode == "per_loop" else 1
            super().__init__(
                layers=[
                    DoryCacheLayer(config.sliding_window if kind == "sliding_attention" else None)
                    for _ in range(loops)
                    for kind in config.layer_types
                ]
            )

    def get_seq_length(self, layer_idx=None):
        if layer_idx is not None:
            return super().get_seq_length(layer_idx)
        return max((layer.get_seq_length() for layer in self.layers), default=0)


class DoryRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        # Dory currently stores attention and MLP as separate physical blocks.
        self.layer_types = sorted(set(config.layer_types) - {"mlp"})

    @staticmethod
    def compute_default_rope_parameters(
        config: DoryConfig, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        return super().compute_default_rope_parameters(config, device, layer_type, **kwargs)


class DoryAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.sqk = nn.Parameter(torch.full((self.q_size,), config.initializer_range))
        self.s_gate = nn.Parameter(torch.full((self.q_size,), config.initializer_range))
        # nGPT scaled-parameter initialization, section 2.6: https://arxiv.org/abs/2410.01131
        self._sqk_factor = config.head_dim**0.5 / config.initializer_range
        gate_scale = config.hidden_size**0.5
        self._s_gate_factor = gate_scale**0.5 / config.initializer_range
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.is_causal = True
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        start_position,
        loop_idx=0,
        final_loop=True,
        **kwargs,
    ):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_proj(hidden_states).view(shape)
        key = self.k_proj(hidden_states).view(shape)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        gate = self.g_proj(hidden_states)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings, unsqueeze_dim=2)
        sqk = (self.sqk.float() * self._sqk_factor).view(1, 1, -1, self.head_dim)
        query = (sqk.square() * l2_normalize(query).float()).to(query.dtype).transpose(1, 2)
        key = l2_normalize(key).transpose(1, 2)
        cache_idx = self.layer_idx
        if self.config.recurrent_kv_cache_mode == "per_loop":
            cache_idx += loop_idx * self.config.num_hidden_layers
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, cache_idx, start_position=start_position)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if past_key_values is not None and (self.config.recurrent_kv_cache_mode == "per_loop" or final_loop):
            past_key_values.layers[cache_idx].finish_forward()
        output = output.reshape(*hidden_states.shape[:-1], self.q_size)
        s_gate = (self.s_gate.float() * self._s_gate_factor).square()
        output = (output.float() * torch.sigmoid(gate.float() * s_gate)).to(output.dtype)
        return self.o_proj(l2_normalize(output)), weights


class DoryMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.suv_gate = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.ones(config.intermediate_size))
        self.act_fn = nn.SiLU()
        self._suv_factor = config.hidden_size**0.5

    def forward(self, hidden_states):
        gate, up = self.gate_proj(hidden_states), self.up_proj(hidden_states)
        gate = (gate.float() * (self.suv_gate.float() * self._suv_factor)).to(gate.dtype)
        up = (up.float() * (self.suv_up.float() * self._suv_factor)).to(up.dtype)
        output = (self.act_fn(gate.float()) * up.float()).to(gate.dtype)
        return self.down_proj(l2_normalize(output))


class DoryDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.layer_type == "mlp":
            self.mixer = DoryMLP(config)
            self.mlp_alpha = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        else:
            self.mixer = DoryAttention(config, layer_idx)
            self.attn_alpha = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))

    def forward(self, hidden_states, **kwargs):
        if self.layer_type == "mlp":
            return normalized_residual_update(
                hidden_states, self.mixer(hidden_states), self.mlp_alpha, self._alpha_scale
            )
        output, _ = self.mixer(hidden_states, **kwargs)
        return normalized_residual_update(hidden_states, output, self.attn_alpha, self._alpha_scale)


class DoryLMHead(nn.Linear):
    _is_hf_initialized: bool

    def __init__(self, config):
        super().__init__(config.hidden_size, config.vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.full((config.vocab_size,), config.initializer_range))
        self.initializer_range = config.initializer_range

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight).float() * (self.logit_scale.float() / self.initializer_range)


@auto_docstring
class DoryPreTrainedModel(PreTrainedModel):
    config: DoryConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = ["DoryDecoderLayer"]
    _supports_attention_backend = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_record_outputs = {"hidden_states": DoryDecoderLayer, "attentions": DoryAttention}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DoryLMHead):
            init.constant_(module.logit_scale, self.config.initializer_range)
        elif isinstance(module, DoryAttention):
            init.constant_(module.sqk, self.config.initializer_range)
            init.constant_(module.s_gate, self.config.initializer_range)
        elif isinstance(module, DoryMLP):
            init.ones_(module.suv_gate)
            init.ones_(module.suv_up)
        elif isinstance(module, DoryDecoderLayer):
            alpha = module.mlp_alpha if module.layer_type == "mlp" else module.attn_alpha
            init.constant_(alpha, self.config.initializer_range)
        elif isinstance(module, DoryRotaryEmbedding):
            # from_pretrained recreates non-persistent buffers with empty storage.
            for layer_type in module.layer_types:
                inv_freq, _ = module.compute_default_rope_parameters(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


@auto_docstring
class DoryModel(DoryPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # from_pretrained overrides may change fields after config initialization.
        config.validate()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.gradient_checkpointing = False
        self.rotary_emb = DoryRotaryEmbedding(config)
        self.layers = nn.ModuleList([DoryDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DoryCache(self.config)
        if past_key_values is not None and not isinstance(past_key_values, DoryCache):
            raise ValueError("Dory requires DoryCache to preserve recurrent cache updates")
        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        start = past_key_values.get_seq_length() if past_key_values is not None else 0
        length = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(start, start + length, device=hidden_states.device).unsqueeze(0)
        masks = attention_mask
        if not isinstance(masks, dict):
            masks = {}
            for layer_idx, kind in enumerate(self.config.layer_types):
                if kind == "mlp" or kind in masks:
                    continue
                mask_function = (
                    create_sliding_window_causal_mask if kind == "sliding_attention" else create_causal_mask
                )
                masks[kind] = mask_function(
                    config=self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    layer_idx=layer_idx,
                )

        position_embeddings = {
            kind: self.rotary_emb(hidden_states, position_ids, kind) for kind in self.rotary_emb.layer_types
        }

        input_end = self.config.n_input_layers
        recurrent_end = input_end + self.config.n_recurrent_layers
        for layer in self.layers[:input_end]:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings.get(layer.layer_type),
                position_ids=position_ids,
                attention_mask=masks.get(layer.layer_type),
                past_key_values=past_key_values,
                start_position=start,
                loop_idx=0,
                final_loop=True,
                **kwargs,
            )
        input_hidden = hidden_states
        for loop_idx in range(self.config.n_recurrent_loops):
            final_loop = loop_idx == self.config.n_recurrent_loops - 1
            # This residual addition is outside Accelerate's layer hooks when the model is split across devices.
            hidden_states = l2_normalize(hidden_states + input_hidden.to(hidden_states.device))
            for layer in self.layers[input_end:recurrent_end]:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings.get(layer.layer_type),
                    position_ids=position_ids,
                    attention_mask=masks.get(layer.layer_type),
                    past_key_values=past_key_values,
                    start_position=start,
                    loop_idx=loop_idx,
                    final_loop=final_loop,
                    **kwargs,
                )
            hidden_states = l2_normalize(hidden_states)
        for layer in self.layers[recurrent_end : recurrent_end + self.config.n_output_layers]:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings.get(layer.layer_type),
                position_ids=position_ids,
                attention_mask=masks.get(layer.layer_type),
                past_key_values=past_key_values,
                start_position=start,
                loop_idx=0,
                final_loop=True,
                **kwargs,
            )
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class DoryForCausalLM(DoryPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = DoryModel(config)
        self.lm_head = DoryLMHead(config)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        if model_kwargs.get("past_key_values") is not None or generation_config.use_cache is False:
            return super()._prepare_cache_for_generation(
                generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
            )
        if generation_config.cache_implementation not in (None, "dynamic"):
            raise ValueError("Dory only supports its recurrent dynamic cache")
        model_kwargs["past_key_values"] = DoryCache(self.config)
        if generation_config.is_assistant:
            model_kwargs["past_key_values"].activate_past_recording()

    def _get_resized_lm_head(self, old_lm_head, new_num_tokens=None, transposed=False, mean_resizing=True):
        if new_num_tokens is None or new_num_tokens == old_lm_head.weight.shape[0]:
            return old_lm_head
        # Reuse the upstream projection initialization and copying, then restore
        # Dory's head arithmetic and its independently learned vocabulary scale.
        with torch.device("meta"):
            resized = DoryLMHead(self.config)
        projection = super()._get_resized_lm_head(old_lm_head, new_num_tokens, transposed, mean_resizing)
        resized.weight = projection.weight
        resized.out_features = new_num_tokens
        old_scale = old_lm_head.logit_scale
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_scale, modifier_rank=None):
                old_scale = old_scale.detach().clone()
        resized.logit_scale = nn.Parameter(
            old_scale.new_full((new_num_tokens,), old_scale.mean().item()),
            requires_grad=old_lm_head.logit_scale.requires_grad,
        )
        with torch.no_grad():
            count = min(old_scale.numel(), new_num_tokens)
            resized.logit_scale[:count].copy_(old_scale[:count])
        resized._is_hf_initialized = True
        resized.train(old_lm_head.training)
        return resized

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep))
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


__all__ = ["DoryPreTrainedModel", "DoryModel", "DoryForCausalLM"]
