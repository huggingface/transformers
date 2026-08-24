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
"""PyTorch TR-HASH model."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


@strict
@auto_docstring(checkpoint="AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT")
class TRHashConfig(PreTrainedConfig):
    r"""
    shared_intermediate_size (`int`, *optional*, defaults to 3072):
        Intermediate width of the always-on shared SwiGLU branch.
    attention_type (`str`, *optional*, defaults to `"gqa"`):
        Attention backbone. Native support currently covers grouped-query attention only.
    mlp_type (`str`, *optional*, defaults to `"tr_hash_engine"`):
        Name used by Complexity Framework for the deterministic routed MLP.
    norm_type (`str`, *optional*, defaults to `"rmsnorm"`):
        Normalization type used by the checkpoint.
    num_experts_per_tok (`int`, *optional*, defaults to 2):
        Number of deterministic expert routes activated for every token.
    top_k_primary_weight (`float`, *optional*, defaults to 0.5):
        Blend weight assigned to the primary deterministic route. The remaining weight is
        distributed uniformly over the other active routes.
    routing_strategy (`str`, *optional*, defaults to `"token_id_multi_hash"`):
        Deterministic routing strategy represented by the persisted route tables.
    route_hash_count (`int`, *optional*, defaults to 2):
        Number of persisted independent token-ID hash routes.
    route_seed (`int`, *optional*, defaults to 119364119):
        Seed used to construct deterministic per-layer routing tables when a checkpoint does not
        provide persisted tables.
    shared_expert (`bool`, *optional*, defaults to `True`):
        Whether to execute the always-on shared SwiGLU branch.
    shared_output_scale (`float`, *optional*, defaults to 1.0):
        Fixed multiplier applied to the shared branch output.
    routed_output_scale (`float`, *optional*, defaults to 2.0):
        Fixed multiplier applied to the blended routed expert output.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply per-head RMS normalization to queries and keys.
    norm_eps (`float`, *optional*, defaults to 1e-6):
        Epsilon used by RMS normalization.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period used by rotary position embeddings.
    expert_width (`int`, *optional*):
        Intermediate width of each routed expert. When omitted, this is derived as
        `intermediate_size // num_experts`.

    ```python
    >>> from transformers import TRHashConfig, TRHashForCausalLM

    >>> config = TRHashConfig()
    >>> model = TRHashForCausalLM(config)
    ```
    """

    model_type = "tr_hash_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 896
    intermediate_size: int = 256
    shared_intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 0

    attention_type: str = "gqa"
    mlp_type: str = "tr_hash_engine"
    norm_type: str = "rmsnorm"
    num_experts: int = 4
    num_experts_per_tok: int = 2
    top_k_primary_weight: float | None = 0.5
    routing_strategy: str = "token_id_multi_hash"
    route_hash_count: int = 2
    route_seed: int = 0x71D5A17
    shared_expert: bool = True
    shared_output_scale: float = 1.0
    routed_output_scale: float = 2.0
    use_qk_norm: bool = True
    rope_theta: float = 10000.0
    head_dim: int | None = None
    expert_width: int | None = None

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.expert_width is None:
            self.expert_width = self.intermediate_size // self.num_experts
        super().__post_init__(**kwargs)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate_architecture(self):
        if self.attention_type != "gqa":
            raise ValueError("TR-HASH native support currently requires `attention_type='gqa'`.")
        if self.mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
            raise ValueError("TR-HASH requires `mlp_type='tr_hash_engine'` or `mlp_type='tr_hash_moe'`.")
        if self.norm_type != "rmsnorm":
            raise ValueError("TR-HASH native support currently requires RMSNorm.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("`head_dim * num_attention_heads` must equal `hidden_size`.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        if self.intermediate_size % self.num_experts != 0:
            raise ValueError("`intermediate_size` must be divisible by `num_experts`.")
        if self.expert_width * self.num_experts != self.intermediate_size:
            raise ValueError("`expert_width * num_experts` must equal `intermediate_size`.")
        if self.num_experts_per_tok != 2:
            raise ValueError("TR-HASH native support currently covers deterministic top-2 routing only.")
        if not 2 <= self.num_experts <= 8:
            raise ValueError("TR-HASH top-2 pair metadata supports between 2 and 8 experts.")
        if not 2 <= self.route_hash_count <= 8:
            raise ValueError("`route_hash_count` must be between 2 and 8.")
        if self.routing_strategy != "token_id_multi_hash":
            raise ValueError("Native support currently covers persisted token-ID multi-hash routing only.")
        if not self.shared_expert:
            raise ValueError("The released TR-HASH checkpoints require the shared SwiGLU expert.")
        if self.top_k_primary_weight is not None and not 0.0 <= self.top_k_primary_weight <= 1.0:
            raise ValueError("`top_k_primary_weight` must be in [0, 1].")
        if self.shared_output_scale < 0.0 or self.routed_output_scale < 0.0:
            raise ValueError("TR-HASH output scales must be non-negative.")


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


class TRHashAttention(Qwen3Attention):
    """Qwen3-style GQA with checkpoint-compatible K/Q/V order and per-layer RoPE."""

    def __init__(self, config: TRHashConfig, layer_idx: int):
        nn.Module.__init__(self)
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
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
        fused_route_codes, fused_expert_pairs = _compile_top2_pair_metadata(route_table, config.num_experts)
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
class TRHashPreTrainedModel(Qwen3PreTrainedModel):
    config: TRHashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TRHashDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
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
            fused_route_codes, fused_expert_pairs = _compile_top2_pair_metadata(route_table, self.config.num_experts)
            init.copy_(module.route_table, route_table.to(module.route_table.device))
            init.copy_(module.fused_route_codes, fused_route_codes.to(module.fused_route_codes.device))
            init.copy_(module.fused_expert_pairs, fused_expert_pairs.to(module.fused_expert_pairs.device))


@auto_docstring
class TRHashModel(Qwen3Model):
    def __init__(self, config: TRHashConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [TRHashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TRHashRMSNorm(config.hidden_size, eps=config.norm_eps)
        del self.rotary_emb
        del self.has_sliding_layers

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
class TRHashForCausalLM(Qwen3ForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. Tokens set to `-100` are ignored.

        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer

        >>> checkpoint = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
        >>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        >>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
        >>> inputs = tokenizer("Hello", return_tensors="pt")
        >>> generated_ids = model.generate(**inputs, max_new_tokens=8)
        ```
        """
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
