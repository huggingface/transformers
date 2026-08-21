# Copyright 2026 The Complexity-ML team and the HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_tr_hash import TRHashConfig


class TRHashRMSNorm(nn.Module):
    """RMSNorm used by the reference implementation without an fp32 upcast."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = query_states * cos + rotate_half(query_states) * sin
    key_states = key_states * cos + rotate_half(key_states) * sin
    return query_states, key_states


class TRHashRotaryEmbedding(nn.Module):
    """Per-layer RoPE module matching the persisted reference checkpoint keys."""

    def __init__(self, config: TRHashConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.float())
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)


def repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    batch_size, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    if num_key_value_groups == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        num_key_value_groups,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(
        batch_size,
        num_key_value_heads * num_key_value_groups,
        sequence_length,
        head_dim,
    )


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attention_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_weights = attention_weights + attention_mask
    attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout, training=module.training)
    attention_output = torch.matmul(attention_weights, value_states)
    return attention_output.transpose(1, 2).contiguous(), attention_weights


class TRHashAttention(nn.Module):
    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        key_value_size = self.num_key_value_heads * self.head_dim
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if config.use_qk_norm:  # CODEPATH: AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT
            self.q_norm = TRHashRMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = TRHashRMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None
        # trf-ignore: TRF050 (the released checkpoint persists one RoPE buffer per layer)
        self.rotary_emb = TRHashRotaryEmbedding(config)

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

        # K/Q/V order and fused projection match Complexity Framework exactly.
        projection = F.linear(
            hidden_states,
            torch.cat((self.k_proj.weight, self.q_proj.weight, self.v_proj.weight), dim=0),
        )
        key_value_size = self.num_key_value_heads * self.head_dim
        key_states, query_states, value_states = projection.split(
            (key_value_size, self.num_heads * self.head_dim, key_value_size), dim=-1
        )
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), attention_weights


def _mix32(values: torch.Tensor) -> torch.Tensor:
    values = values.bitwise_and(0xFFFFFFFF)
    values = (values ^ (values >> 16)).bitwise_and(0xFFFFFFFF)
    values = (values * 0x7FEB352D).bitwise_and(0xFFFFFFFF)
    values = (values ^ (values >> 15)).bitwise_and(0xFFFFFFFF)
    values = (values * 0x846CA68B).bitwise_and(0xFFFFFFFF)
    return (values ^ (values >> 16)).bitwise_and(0xFFFFFFFF)


def _build_multi_hash_route_table(config: TRHashConfig, layer_idx: int) -> torch.LongTensor:
    token_ids = torch.arange(config.vocab_size, dtype=torch.int64, device="cpu").unsqueeze(1)
    expert_ids = torch.arange(config.num_experts, dtype=torch.int64, device="cpu").unsqueeze(0)
    scores = torch.zeros(config.vocab_size, config.num_experts, dtype=torch.int64, device="cpu")
    layer_salt = (int(config.route_seed) + int(layer_idx) * 0x9E3779B1) & 0xFFFFFFFF
    for hash_index in range(config.route_hash_count):
        channel_salt = (layer_salt + hash_index * 0x85EBCA77) & 0xFFFFFFFF
        expert_salts = _mix32(expert_ids + (channel_salt ^ 0xC2B2AE3D))
        scores.add_(_mix32(token_ids ^ expert_salts ^ channel_salt))
    return scores.topk(config.num_experts_per_tok, dim=1, largest=True, sorted=True).indices.T.long()


def _compile_top2_pair_metadata(
    route_table: torch.LongTensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_pairs = torch.combinations(
        torch.arange(num_experts, dtype=torch.int32, device="cpu"),
        r=2,
    )
    unordered_routes = route_table.sort(dim=0).values
    pair_matches = (unordered_routes[0].unsqueeze(0) == expert_pairs[:, 0].long().unsqueeze(1)) & (
        unordered_routes[1].unsqueeze(0) == expert_pairs[:, 1].long().unsqueeze(1)
    )
    pair_indices = pair_matches.to(torch.int64).argmax(dim=0)
    swap = route_table[0].eq(unordered_routes[1]).to(torch.int64)
    swap_bit = 0x8 if expert_pairs.shape[0] <= 8 else 0x20
    return (pair_indices | (swap * swap_bit)).to(torch.uint8), expert_pairs


class TRHashExpertEngine(nn.Module):
    """Always-on shared SwiGLU plus deterministic top-k routed experts."""

    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.vocab_size = config.vocab_size
        self.routing_strategy = config.routing_strategy
        self.shared_expert = config.shared_expert
        self.shared_output_scale = config.shared_output_scale
        self.routed_output_scale = config.routed_output_scale
        self.act_fn = ACT2FN[config.hidden_act]

        self.intermediate_size = config.intermediate_size
        expert_width = config.expert_width

        if self.num_experts_per_tok == 1:
            route_weights = (1.0,)
        else:
            primary_weight = (
                1.0 / self.num_experts_per_tok if config.top_k_primary_weight is None else config.top_k_primary_weight
            )
            route_weights = (
                primary_weight,
                *(
                    (1.0 - primary_weight) / (self.num_experts_per_tok - 1)
                    for _ in range(self.num_experts_per_tok - 1)
                ),
            )
        self.route_weights = tuple(float(weight) for weight in route_weights)

        route_table = _build_multi_hash_route_table(config, layer_idx)
        fused_route_codes, fused_expert_pairs = _compile_top2_pair_metadata(
            route_table,
            config.num_experts,
        )
        self.register_buffer("route_table", route_table, persistent=True)
        self.register_buffer("fused_route_codes", fused_route_codes, persistent=True)
        self.register_buffer("fused_expert_pairs", fused_expert_pairs, persistent=True)

        self.expert_gate = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, expert_width))
        self.expert_up = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, expert_width))
        self.expert_down = nn.Parameter(torch.empty(config.num_experts, expert_width, config.hidden_size))
        self.shared_gate = nn.Linear(config.hidden_size, config.shared_intermediate_size, bias=False)
        self.shared_up = nn.Linear(config.hidden_size, config.shared_intermediate_size, bias=False)
        self.shared_down = nn.Linear(config.shared_intermediate_size, config.hidden_size, bias=False)

    @property
    def experts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expert_gate, self.expert_up, self.expert_down

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        shared_output = self.shared_down(self.act_fn(self.shared_gate(flat_states)) * self.shared_up(flat_states))

        routes = self.route_table[:, token_ids.clamp(0, self.vocab_size - 1)].reshape(self.num_experts_per_tok, -1)
        route_weights = flat_states.new_tensor(self.route_weights).view(-1, 1)
        routed_output = torch.zeros_like(flat_states)
        for expert_index in range(self.num_experts):
            token_weight = (routes.eq(expert_index).to(flat_states.dtype) * route_weights).sum(dim=0)
            active_states = flat_states * token_weight.ne(0).to(flat_states.dtype).unsqueeze(-1)
            intermediate_states = self.act_fn(active_states @ self.expert_gate[expert_index]) * (
                active_states @ self.expert_up[expert_index]
            )
            expert_output = intermediate_states @ self.expert_down[expert_index]
            routed_output.add_(expert_output * token_weight.unsqueeze(-1))

        output = self.shared_output_scale * shared_output + self.routed_output_scale * routed_output
        return output.view(batch_size, sequence_length, hidden_size)


class TRHashMLP(nn.Module):
    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__()
        self.mlp_type = config.mlp_type
        self.engine = TRHashExpertEngine(config, layer_idx)

    @property
    def experts(self):
        return self.engine.experts

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.engine(hidden_states, token_ids)


class TRHashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__()
        self.norm_type = config.norm_type
        self.self_attn = TRHashAttention(config, layer_idx)
        self.mlp = TRHashMLP(config, layer_idx)
        self.input_layernorm = TRHashRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = TRHashRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        past_key_values: Cache | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=self.self_attn.rotary_emb(position_ids, hidden_states.dtype),
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states), token_ids)
        return residual + hidden_states


@auto_docstring
class TRHashPreTrainedModel(PreTrainedModel):
    config: TRHashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TRHashDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": TRHashDecoderLayer,
        "attentions": TRHashAttention,
    }

    # trf-ignore: TRF018 (TR-HASH initializes stacked expert parameters and persisted routing buffers)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
            if (
                isinstance(module, nn.Embedding)
                and module.padding_idx is not None
                and not getattr(module.weight, "_is_hf_initialized", False)
            ):
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, TRHashRMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, TRHashRotaryEmbedding):
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (torch.arange(0, self.config.head_dim, 2, dtype=torch.float32) / self.config.head_dim)
            )
            init.copy_(module.inv_freq, inv_freq.to(module.inv_freq.device))
        elif isinstance(module, TRHashExpertEngine):
            for parameter in (module.expert_gate, module.expert_up, module.expert_down):
                init.normal_(parameter, mean=0.0, std=self.config.initializer_range)
            route_table = _build_multi_hash_route_table(self.config, module.layer_idx)
            fused_route_codes, fused_expert_pairs = _compile_top2_pair_metadata(
                route_table,
                self.config.num_experts,
            )
            init.copy_(module.route_table, route_table.to(module.route_table.device))
            init.copy_(module.fused_route_codes, fused_route_codes.to(module.fused_route_codes.device))
            init.copy_(module.fused_expert_pairs, fused_expert_pairs.to(module.fused_expert_pairs.device))


@auto_docstring
class TRHashModel(TRHashPreTrainedModel):
    def __init__(self, config: TRHashConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TRHashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TRHashRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is None:
            raise ValueError("TR-HASH requires `input_ids` because expert routing is token-ID deterministic.")

        hidden_states = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_ids.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                token_ids=input_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class TRHashForCausalLM(TRHashPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: TRHashConfig):
        super().__init__(config)
        self.model = TRHashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "TRHashConfig",
    "TRHashForCausalLM",
    "TRHashModel",
    "TRHashPreTrainedModel",
]
