# Copyright 2026 Zyphra and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Zaya model."""

import copy
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import init

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..laguna.modeling_laguna import LagunaRotaryEmbedding
from ..llama.modeling_llama import LlamaAttention
from ..qwen3_5_moe.modeling_qwen3_5_moe import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm


@auto_docstring(checkpoint="Zyphra/ZAYA1-8B")
@strict
class ZayaConfig(PreTrainedConfig):
    r"""
    ffn_hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the feed-forward and expert hidden states, translate it to `intermediate_size`.
    num_key_value_heads (`int`, *optional*, defaults to 2):
        Number of key/value groups.
    partial_rotary_factor (`float`, *optional*, defaults to 0.5):
        Fraction of each attention head dimension using rotary embeddings.
    lm_head_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the language modeling head.
    moe_router_topk (`int`, *optional*, defaults to 1):
        Number of selected experts per token. ZAYA checkpoints use top-1 routing.
    zaya_mlp_expansion (`int`, *optional*, defaults to 256):
        Expansion size used by the dense ZAYA blocks.
    cca_time0 (`int`, *optional*, defaults to 2):
        First temporal parameter of the CCA projection.
    cca_time1 (`int`, *optional*, defaults to 2):
        Second temporal parameter of the CCA projection.
    layer_types (`list[str]`, *optional*):
        Per-layer selector for standard RoPE versus SWA RoPE embeddings.
    swa_rotary_base (`float`, *optional*):
        RoPE base used by SWA layers.

    ```python
    >>> from transformers import ZayaConfig, ZayaModel

    >>> configuration = ZayaConfig()
    >>> model = ZayaModel(configuration)

    >>> configuration = model.config
    ```
    """

    model_type = "zaya"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 5000000.0

    vocab_size: int = 262272
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_hidden_layers: int = 80
    num_experts: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    head_dim: int = 128
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    norm_epsilon: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float = 0.5
    attention_bias: bool = False
    lm_head_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_router_topk: int = 1
    zaya_mlp_expansion: int = 256
    cca_time0: int | None = 2
    cca_time1: int | None = 2
    layer_types: list[str] | None = None
    swa_rotary_base: float | int = 10000.0
    output_router_logits: bool = False
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 106

    def __post_init__(self, **kwargs):
        for unused_checkpoint_kwarg in (
            "cca",
            "num_query_groups",
            "activation_func",
            "normalization",
            "add_bias_linear",
            "gated_linear_unit",
            "fused_add_norm",
            "apply_rope_fusion",
            "bias_activation_fusion",
            "activation_func_fp8_input_store",
            "clamp_temp",
            "kv_channels",
            "mamba_cache_dtype",
            "residual_in_fp32",
            "rope_scaling",
            "scale_residual_merge",
            "sliding_window",
            "zaya_high_prec",
            "zaya_use_mod",
            "zaya_use_eda",
        ):
            kwargs.pop(unused_checkpoint_kwarg, None)

        self.intermediate_size = self.ffn_hidden_size
        self.num_experts_per_tok = self.moe_router_topk

        self.num_key_value_heads = (
            self.num_attention_heads if self.num_key_value_heads is None else self.num_key_value_heads
        )

        legacy_swa_layers = kwargs.pop("swa_layers", None)
        if self.layer_types is None:
            if legacy_swa_layers is None:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
            else:
                self.layer_types = [
                    "full_attention" if layer_type == 0 else "sliding_attention" for layer_type in legacy_swa_layers
                ]
        else:
            self.layer_types = list(self.layer_types)

        self.cca_time0 = 2 if self.cca_time0 is None else self.cca_time0
        self.cca_time1 = 2 if self.cca_time1 is None else self.cca_time1

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        default_rope_params: dict[Literal["full_attention", "sliding_attention"], dict[str, Any]] = {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": kwargs.pop("rope_theta", self.default_theta),
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": self.swa_rotary_base,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
        }
        layer_types = set(self.layer_types)

        if self.rope_parameters is None:
            self.rope_parameters = {layer_type: default_rope_params[layer_type] for layer_type in layer_types}
        else:
            self.rope_parameters = {
                layer_type: {**default_rope_params[layer_type], **(self.rope_parameters.get(layer_type) or {})}
                for layer_type in layer_types
            }

        return kwargs

    def validate_architecture(self):
        if self.head_dim is None:
            raise ValueError("`head_dim` must be set for ZAYA.")
        if self.num_experts_per_tok != 1:
            raise ValueError("ZAYA currently supports `moe_router_topk=1` only.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be a multiple of `num_key_value_heads`.")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("`layer_types` must have one entry per hidden layer.")
        if invalid_layer_types := set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError(f"`layer_types` contains unsupported values: {sorted(invalid_layer_types)}.")
        if (self.cca_time0, self.cca_time1) != (2, 2):
            raise ValueError("ZAYA currently supports `cca_time0=2` and `cca_time1=2` only.")


class ZayaRotaryEmbedding(LagunaRotaryEmbedding):
    pass


class ZayaRMSNorm(Qwen3MoeRMSNorm):
    pass


def _make_zaya_cache(config: ZayaConfig) -> DynamicCache:
    cache_config = copy.copy(config)
    # layer_types is used to distinct the rope_type (full or swa)
    # so need to construct a new layer_types to construct cache
    cache_config.layer_types = [
        "hybrid" if layer_idx % 2 == 0 else "moe" for layer_idx in range(config.num_hidden_layers)
    ]
    return DynamicCache(config=cache_config)


class ZayaCCAProjection(nn.Module):
    """
    Projects hidden states into attention q/k/v states with ZAYA's CCA path.

    `linear_q` and `linear_k` produce the residual q/k states and are concatenated into `qk_states`. The causal
    `conv_qk_depthwise` + `conv_qk_grouped` stack mixes the current q/k stream with the cached pre-convolution tail;
    for example, decoding token `t` uses the cached q/k states from previous tokens plus the current `qk_states[:, t]`.
    Values are built from `val_proj1(hidden_states[:, t])` and a delayed `val_proj2`: during prefill token `t` uses
    `val_proj2(hidden_states[:, t - 1])`, while decoding reads the previous `val_proj2` from **the recurrent cache**.

    The final q/k states are L2-normalized. `temp` is the learned per-KV-head scale applied to keys.
    """

    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size

        self.depthwise_kernel_size = config.cca_time0
        self.grouped_kernel_size = config.cca_time1
        self.total_padding = (self.depthwise_kernel_size - 1) + (self.grouped_kernel_size - 1)

        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.key_value_hidden_size = self.num_key_value_heads * self.head_dim
        self.query_hidden_size = self.num_attention_heads * self.head_dim
        self.sqrt_head_dim = self.head_dim**0.5
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.linear_q = nn.Linear(self.hidden_size, self.query_hidden_size, bias=self.config.attention_bias)
        self.linear_k = nn.Linear(self.hidden_size, self.key_value_hidden_size, bias=self.config.attention_bias)
        self.val_proj1 = nn.Linear(self.hidden_size, self.key_value_hidden_size // 2, bias=self.config.attention_bias)
        self.val_proj2 = nn.Linear(self.hidden_size, self.key_value_hidden_size // 2, bias=self.config.attention_bias)

        conv_channels = self.key_value_hidden_size + self.query_hidden_size
        self.conv_qk_depthwise = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=self.depthwise_kernel_size,
            groups=conv_channels,
            padding=0,
            stride=1,
        )
        self.conv_qk_grouped = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=self.grouped_kernel_size,
            groups=(self.num_key_value_heads + self.num_attention_heads),
            padding=0,
            stride=1,
        )

        self.temp = nn.Parameter(torch.zeros(self.num_key_value_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
        attention_mask: torch.Tensor | None = None,
    ):
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        batch_size, seq_length, _ = hidden_states.shape

        projected_queries = self.linear_q(hidden_states)
        projected_keys = self.linear_k(hidden_states)
        qk_states = torch.cat([projected_queries, projected_keys], dim=-1)

        query_residual = projected_queries.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key_residual = projected_keys.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        key_residual = key_residual.unsqueeze(-2).expand(-1, -1, -1, self.num_key_value_groups, -1)
        key_residual = key_residual.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(
            batch_size, seq_length, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        ).mean(dim=-2)

        qk_states = qk_states.transpose(1, 2)
        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            cached_qk_states = past_key_values.layers[self.layer_idx].conv_states
            conv_input = torch.cat([cached_qk_states, qk_states], dim=-1)
        else:
            conv_input = F.pad(qk_states, (self.total_padding, 0))

        if past_key_values is not None:
            new_conv_state = qk_states[..., -self.total_padding :]
            if new_conv_state.shape[-1] < self.total_padding:
                new_conv_state = F.pad(new_conv_state, (self.total_padding - new_conv_state.shape[-1], 0))
            past_key_values.update_conv_state(new_conv_state, self.layer_idx)

        convolved_qk_states = self.conv_qk_depthwise(conv_input)
        convolved_qk_states = self.conv_qk_grouped(convolved_qk_states).transpose(1, 2)

        query = (
            convolved_qk_states[..., : self.query_hidden_size].view(
                batch_size, seq_length, self.num_attention_heads, self.head_dim
            )
            + query_residual
        )

        key = (
            convolved_qk_states[..., self.query_hidden_size :].view(
                batch_size, seq_length, self.num_key_value_heads, self.head_dim
            )
            + key_residual
        )

        value_current = self.val_proj1(hidden_states)
        projected_v2 = self.val_proj2(hidden_states)
        if use_precomputed_states:
            first_v2 = past_key_values.layers[self.layer_idx].recurrent_states.unsqueeze(1)
        else:
            first_v2 = self.val_proj2(hidden_states.new_zeros(batch_size, 1, self.hidden_size))
        value_delayed = torch.cat([first_v2, projected_v2[:, :-1]], dim=1)

        if past_key_values is not None:
            past_key_values.update_recurrent_state(projected_v2[:, -1, :], self.layer_idx)

        value = torch.cat([value_current, value_delayed], dim=-1).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )

        norm_eps = torch.finfo(query.dtype).eps
        query_norm = query.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps)
        key_norm = key.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps)

        key = (key * (self.sqrt_head_dim / key_norm)) * self.temp[None, None].unsqueeze(-1)
        query = query * (self.sqrt_head_dim / query_norm)

        query = query.reshape(batch_size, seq_length, self.query_hidden_size)
        key = key.reshape(batch_size, seq_length, self.key_value_hidden_size)
        value = value.reshape(batch_size, seq_length, self.key_value_hidden_size)
        return query, key, value


class ZayaAttention(LlamaAttention):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_n = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        del self.q_proj
        del self.k_proj
        del self.v_proj
        self.qkv = ZayaCCAProjection(
            config=self.config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_mask_2d: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        batch_size, seq_length, _ = hidden_states.shape
        query_states, key_states, value_states = self.qkv(hidden_states, past_key_values, attention_mask_2d)
        query_states = query_states.view(batch_size, seq_length, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.config.num_key_value_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_n)

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, : query_states.shape[-2], : key_states.shape[-2]]

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            causal_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
        )

        attn_output = attn_output.view(batch_size, seq_length, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class ZayaDecoderATTLayer(GradientCheckpointingLayer):
    def __init__(self, config: ZayaConfig, layer_n: int):
        super().__init__()
        self.config = config
        self.self_attn = ZayaAttention(config, layer_n)

        self.input_norm = ZayaRMSNorm(self.config.hidden_size, eps=self.config.norm_epsilon)
        self.res_scale = ResidualScaling(config, layer_n)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_mask_2d: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_router_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        hidden_states, residual = _apply_residual_scaling(hidden_states, residual, self.res_scale, self.input_norm)

        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attention_mask_2d=attention_mask_2d,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )

        return hidden_states, self_attn_weights if output_attentions else None, residual, prev_router_hidden_states


class ResidualScaling(nn.Module):
    def __init__(self, config, layer_n):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = torch.nn.Parameter(torch.ones(config.hidden_size))
        self.hidden_states_bias = torch.nn.Parameter(torch.zeros(config.hidden_size))

        if self.not_first_layer:
            self.residual_scale = torch.nn.Parameter(torch.ones(config.hidden_size))
            self.residual_bias = torch.nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        if self.not_first_layer:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states


def _apply_residual_scaling(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    residual_scaling,
    rms_norm: ZayaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual, hidden_states = residual_scaling(residual, hidden_states)
    residual = hidden_states.to(torch.float32) if residual is None else hidden_states + residual
    hidden_states = rms_norm(residual.to(dtype=rms_norm.weight.dtype))
    return hidden_states, residual


class ZayaRouter(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        num_moe_experts: int,
        moe_router_topk: int,
        mlp_expansion: int,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = int(hidden_size or getattr(config, "hidden_size"))
        self.layer_idx = layer_idx

        self.num_experts = num_moe_experts + 1
        self.topk = int(moe_router_topk)
        self.mlp_expansion = int(mlp_expansion)

        self.down_proj = nn.Linear(self.hidden_size, self.mlp_expansion, bias=True)

        zaya_first_layer = 1
        self.use_eda = self.layer_idx != zaya_first_layer

        self.rmsnorm_eda = ZayaRMSNorm(self.mlp_expansion, eps=config.norm_epsilon)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))

        self.non_linearity = nn.GELU()
        self.router_mlp = nn.Sequential(
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            self.non_linearity,
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            self.non_linearity,
            nn.Linear(self.mlp_expansion, self.num_experts, bias=False),
        )

        self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
        self.balancing_biases[-1] = -1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_length = hidden_states.shape[1]

        router_hidden_states = self.down_proj(hidden_states)

        if self.use_eda and (router_states is not None):
            router_hidden_states = router_hidden_states + router_states * self.router_states_scale

        router_hidden_states_next = router_hidden_states[:, -seq_length:].clone()
        router_hidden_states = self.rmsnorm_eda(router_hidden_states)
        logits = self.router_mlp(router_hidden_states)
        expert_prob = torch.softmax(logits, dim=-1)

        expert_choice = expert_prob.detach().to(torch.float32) + self.balancing_biases
        _, expert_choice = torch.topk(expert_choice, self.topk, dim=-1)
        route_prob = torch.gather(expert_prob, dim=2, index=expert_choice)

        return (
            route_prob.reshape(-1, self.topk),
            expert_choice.reshape(-1, self.topk),
            router_hidden_states_next,
            logits.reshape(-1, self.num_experts),
        )


@use_experts_implementation
class ZayaExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config, num_experts: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = intermediate_size // 2
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class ZayaBlock(nn.Module):
    def __init__(
        self,
        config,
        num_moe_experts: int,
        mlp_expansion: int,
        intermediate_size: int,
        layer_n: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_moe_experts = num_moe_experts
        self.router = ZayaRouter(
            config=self.config,
            layer_idx=layer_n,
            num_moe_experts=self.num_moe_experts,
            moe_router_topk=getattr(self.config, "moe_router_topk", 1),
            mlp_expansion=mlp_expansion,
            hidden_size=self.hidden_dim,
        )
        self.experts = ZayaExperts(self.config, self.num_moe_experts, intermediate_size=intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        route_prob, expert_choice, prev_router_hidden_states, router_logits = self.router(
            hidden_states, router_states=prev_router_hidden_states
        )
        batch_size, seq_length, emb_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(batch_size * seq_length, emb_dim)
        expert_output = self.experts(hidden_states_flat, expert_choice, route_prob)
        expert_output = expert_output.view(batch_size, seq_length, emb_dim)

        return expert_output, prev_router_hidden_states, router_logits


class ZayaDecoderMLPLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: ZayaConfig,
        num_moe_experts: int,
        mlp_expansion: int,
        intermediate_size: int,
        layer_n: int,
    ):
        super().__init__()
        self.config = config
        self.zaya_block = ZayaBlock(
            config,
            num_moe_experts,
            mlp_expansion,
            intermediate_size,
            layer_n,
        )
        self.input_norm = ZayaRMSNorm(self.config.hidden_size, eps=self.config.norm_epsilon)
        self.res_scale = ResidualScaling(config, layer_n)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prev_router_hidden_states: torch.Tensor | None = None,
        output_router_logits: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        hidden_states, residual = _apply_residual_scaling(hidden_states, residual, self.res_scale, self.input_norm)

        hidden_states, prev_router_hidden_states, router_logits = self.zaya_block(
            hidden_states,
            prev_router_hidden_states,
        )

        return (
            hidden_states,
            router_logits if output_router_logits else None,
            residual,
            prev_router_hidden_states,
        )


class ZayaPreTrainedModel(PreTrainedModel):
    config: ZayaConfig
    config_class = ZayaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ZayaDecoderATTLayer", "ZayaDecoderMLPLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(ZayaRouter, index=3),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ResidualScaling):
            init.ones_(module.hidden_states_scale)
            init.zeros_(module.hidden_states_bias)
            if module.not_first_layer:
                init.ones_(module.residual_scale)
                init.zeros_(module.residual_bias)
        elif isinstance(module, ZayaRouter):
            if module.use_eda:
                init.ones_(module.router_states_scale)
            init.zeros_(module.balancing_biases)
            module.balancing_biases[-1] = -1.0
        elif isinstance(module, ZayaExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)


@auto_docstring
class ZayaModel(ZayaPreTrainedModel):
    def __init__(self, config: ZayaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = []

        for layer_n in range(config.num_hidden_layers):
            if layer_n % 2 == 1:
                self.layers.append(
                    ZayaDecoderMLPLayer(
                        config,
                        config.num_experts,
                        config.zaya_mlp_expansion,
                        config.intermediate_size,
                        layer_n,
                    )
                )
            else:
                self.layers.append(ZayaDecoderATTLayer(config, layer_n))
        self.layers = nn.ModuleList(self.layers)

        self.gradient_checkpointing = False
        self.res_scale = ResidualScaling(config, config.num_hidden_layers)

        self.final_norm = ZayaRMSNorm(self.config.hidden_size, eps=self.config.norm_epsilon)

        self.rotary_emb = ZayaRotaryEmbedding(config=config)

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
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = _make_zaya_cache(self.config)

        residual = None

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            position_ids,
            past_key_values,
        )
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                "ZAYA CCA projection requires a 2D `attention_mask` to mask padding tokens before convolution."
            )
        # ZAYA's hybrid cache is not compileable, so generation keeps `attention_mask` as the original 2D padding mask.
        # CCA projection only needs it during multi-token prefill; single-token decoding uses the cached convolution state.
        attention_mask_2d = attention_mask[:, -inputs_embeds.shape[1] :] if attention_mask is not None else None
        if inputs_embeds.shape[1] == 1:
            attention_mask_2d = None

        hidden_states = inputs_embeds

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        prev_router_hidden_states = None

        for layer_n, decoder_layer in enumerate(self.layers):
            emb_to_use = position_embeddings[self.config.layer_types[layer_n]]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                residual,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                position_embeddings=emb_to_use,
                prev_router_hidden_states=prev_router_hidden_states,
                attention_mask_2d=attention_mask_2d,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            residual = layer_outputs[2]
            prev_router_hidden_states = layer_outputs[3]

            if isinstance(decoder_layer, ZayaDecoderATTLayer):
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states, residual = _apply_residual_scaling(hidden_states, residual, self.res_scale, self.final_norm)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache,
    ):
        return create_causal_mask(
            config=self.config,
            inputs_embeds=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )


@auto_docstring
class ZayaForCausalLM(ZayaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _is_stateful = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = ZayaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=self.config.lm_head_bias)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    @can_return_tuple
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
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
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

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return model_inputs

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs: dict,
        generation_mode,
        batch_size: int,
        max_cache_length: int,
    ):
        if generation_config.use_cache is False:
            return

        if "past_key_values" not in model_kwargs:
            model_kwargs["past_key_values"] = _make_zaya_cache(self.config)
            generation_config.cache_implementation = None
        return super()._prepare_cache_for_generation(
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            generation_mode=generation_mode,
            batch_size=batch_size,
            max_cache_length=max_cache_length,
        )


__all__ = [
    "ZayaConfig",
    "ZayaPreTrainedModel",
    "ZayaModel",
    "ZayaForCausalLM",
]
