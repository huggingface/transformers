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

from copy import deepcopy

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..mixtral.modeling_mixtral import MixtralExperts
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
)


logger = logging.get_logger(__name__)


@strict
@auto_docstring(checkpoint="AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT")
class TRHashConfig(Qwen3Config):
    r"""
    shared_intermediate_size (`int`, *optional*, defaults to 3072):
        Intermediate width of the always-on shared SwiGLU expert.
    num_experts (`int`, *optional*, defaults to 4):
        Number of stored routed experts.
    num_experts_per_tok (`int`, *optional*, defaults to 2):
        Number of deterministic expert routes activated for every token.
    top_k_primary_weight (`float`, *optional*, defaults to 0.5):
        Fixed weight assigned to the primary route. The remaining weight is distributed over the
        other active routes.
    route_hash_count (`int`, *optional*, defaults to 2):
        Number of independent token-ID hashes used to build the persisted route table.
    route_seed (`int`, *optional*, defaults to 119364119):
        Seed used to construct a route table when one is not loaded from a checkpoint.
    shared_output_scale (`float`, *optional*, defaults to 1.0):
        Fixed multiplier applied to the shared expert output.
    routed_output_scale (`float`, *optional*, defaults to 2.0):
        Fixed multiplier applied to the routed expert output.
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
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.mlp.router": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 32000
    hidden_size: int = 896
    intermediate_size: int = 256
    shared_intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 0

    num_experts: int = 4
    num_experts_per_tok: int = 2
    top_k_primary_weight: float = 0.5
    route_hash_count: int = 2
    route_seed: int = 0x71D5A17
    shared_output_scale: float = 1.0
    routed_output_scale: float = 2.0
    expert_width: int | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Compatibility with the released config, before RoPE and RMSNorm names were standardized.
        if "norm_eps" in self.__dict__:
            self.rms_norm_eps = self.__dict__["norm_eps"]
        if "rope_theta" in self.__dict__:
            self.rope_parameters = {"rope_type": "default", "rope_theta": self.__dict__["rope_theta"]}
        if self.expert_width is None:
            self.expert_width = self.intermediate_size // self.num_experts
        for legacy_key in (
            "attention_type",
            "mlp_type",
            "norm_type",
            "routing_strategy",
            "shared_expert",
            "top_k",
            "norm_eps",
            "rope_theta",
            "rope_type",
            "use_qk_norm",
        ):
            self.__dict__.pop(legacy_key, None)


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


class TRHashRouter(nn.Module):
    """Select experts from token IDs using the persisted deterministic route table."""

    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.route_table = nn.Buffer(_build_multi_hash_route_table(config, layer_idx), persistent=True)

        remaining_weight = (1.0 - config.top_k_primary_weight) / (self.top_k - 1)
        route_weights = (config.top_k_primary_weight, *((remaining_weight,) * (self.top_k - 1)))
        self.route_weights = nn.Buffer(torch.tensor(route_weights), persistent=False)

    def resize_route_table(self, new_vocab_size: int) -> None:
        old_route_table = self.route_table
        self.config.vocab_size = new_vocab_size
        new_route_table = _build_multi_hash_route_table(self.config, self.layer_idx)
        preserved_tokens = min(old_route_table.shape[1], new_vocab_size)
        new_route_table[:, :preserved_tokens] = old_route_table[:, :preserved_tokens].to(new_route_table.device)
        self.route_table = new_route_table.to(old_route_table.device)

    def forward(self, token_ids: torch.LongTensor) -> tuple[None, torch.Tensor, torch.Tensor]:
        selected_experts = self.route_table[:, token_ids].movedim(0, -1).reshape(-1, self.top_k)
        routing_weights = self.route_weights.expand(selected_experts.shape[0], -1)
        return None, routing_weights, selected_experts


class TRHashExperts(MixtralExperts):
    """Standard fused SwiGLU experts driven by the deterministic router."""

    def __init__(self, config: TRHashConfig):
        # The public TR-HASH config stores the per-expert width separately from the aggregate MLP width.
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.expert_width
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]


class TRHashSharedExpert(Qwen3MLP):
    pass


class TRHashMLP(nn.Module):
    """Always-on shared expert plus deterministically routed experts."""

    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__()
        shared_config = deepcopy(config)
        shared_config.intermediate_size = config.shared_intermediate_size
        self.shared_expert = TRHashSharedExpert(shared_config)
        self.router = TRHashRouter(config, layer_idx)
        self.experts = TRHashExperts(config)
        self.shared_output_scale = config.shared_output_scale
        self.routed_output_scale = config.routed_output_scale

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        _, routing_weights, selected_experts = self.router(token_ids)
        routing_weights = routing_weights.to(dtype=flat_states.dtype, device=flat_states.device)
        routed_output = self.experts(flat_states, selected_experts, routing_weights)
        output = self.shared_output_scale * self.shared_expert(flat_states)
        output = output + self.routed_output_scale * routed_output
        return output.view(batch_size, sequence_length, hidden_size)


class TRHashAttention(Qwen3Attention):
    pass


class TRHashRMSNorm(Qwen3RMSNorm):
    pass


class TRHashDecoderLayer(Qwen3DecoderLayer, GradientCheckpointingLayer):
    def __init__(self, config: TRHashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = TRHashMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states), token_ids)
        return residual + hidden_states


@auto_docstring
class TRHashPreTrainedModel(Qwen3PreTrainedModel):
    config: TRHashConfig
    _no_split_modules = ["TRHashDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": TRHashDecoderLayer,
        "attentions": TRHashAttention,
    }
    _keys_to_ignore_on_load_unexpected = [r"mlp\.engine\.fused_(expert_pairs|route_codes)"]

    # trf-ignore: TRF018 (stacked expert parameters are not Linear modules)
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, TRHashRouter):
            route_table = _build_multi_hash_route_table(module.config, module.layer_idx)
            init.copy_(module.route_table, route_table.to(module.route_table.device))
            remaining_weight = (1.0 - module.config.top_k_primary_weight) / (module.top_k - 1)
            route_weights = (module.config.top_k_primary_weight, *((remaining_weight,) * (module.top_k - 1)))
            init.copy_(module.route_weights, torch.tensor(route_weights, device=module.route_weights.device))
        elif isinstance(module, TRHashExperts):
            for parameter in (module.gate_up_proj, module.down_proj):
                init.normal_(parameter, mean=0.0, std=self.config.initializer_range)

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is not None or pad_to_multiple_of is not None:
            for module in self.modules():
                if isinstance(module, TRHashRouter):
                    module.resize_route_table(self.config.vocab_size)
        return embeddings


@auto_docstring
class TRHashModel(TRHashPreTrainedModel, Qwen3Model):
    def __init__(self, config: TRHashConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [TRHashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TRHashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

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
            raise ValueError("TR-HASH routing requires `input_ids`.")

        hidden_states = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # `generate` can pass a mask that was already prepared for the inherited attention implementation.
        if isinstance(attention_mask, dict):
            causal_mask = attention_mask["full_attention"]
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                token_ids=input_ids,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class TRHashForCausalLM(TRHashPreTrainedModel, Qwen3ForCausalLM):
    def __init__(self, config: TRHashConfig):
        super().__init__(config)
        self.model = TRHashModel(config)
        self.post_init()


__all__ = [
    "TRHashConfig",
    "TRHashForCausalLM",
    "TRHashModel",
    "TRHashPreTrainedModel",
]
