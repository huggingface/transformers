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
from ...cache_utils import Cache, DynamicCache, LinearAttentionAndFullAttentionLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..afmoe.modeling_afmoe import AfmoeForCausalLM
from ..laguna.modeling_laguna import LagunaRotaryEmbedding
from ..llama.modeling_llama import LlamaPreTrainedModel
from ..phi3.modeling_phi3 import Phi3Attention
from ..qwen3_5_moe.modeling_qwen3_5_moe import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeRMSNorm


@auto_docstring(checkpoint="Zyphra/ZAYA1-8B")
@strict
class ZayaConfig(PreTrainedConfig):
    r"""
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimension of the feed-forward and expert hidden states.
    num_key_value_heads (`int`, *optional*, defaults to 2):
        Number of key/value groups.
    partial_rotary_factor (`float`, *optional*, defaults to 0.5):
        Fraction of each attention head dimension using rotary embeddings.
    lm_head_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the language modeling head.
    num_experts_per_tok (`int`, *optional*, defaults to 1):
        Number of selected experts per token. ZAYA checkpoints use top-1 routing.
    zaya_mlp_expansion (`int`, *optional*, defaults to 256):
        Expansion size used by the dense ZAYA blocks.
    cca_time0 (`int`, *optional*, defaults to 2):
        First temporal parameter of the CCA projection.
    cca_time1 (`int`, *optional*, defaults to 2):
        Second temporal parameter of the CCA projection.
    layer_types (`list[str]`, *optional*):
        Per-layer selector for standard RoPE versus SWA RoPE embeddings.

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
    default_swa_theta = 10000.0

    vocab_size: int = 262272
    hidden_size: int = 2048
    intermediate_size: int = 4096
    num_hidden_layers: int = 40
    num_experts: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
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
    num_experts_per_tok: int = 1
    zaya_mlp_expansion: int = 256
    cca_time0: int = 2
    cca_time1: int = 2
    sliding_window: int | None = None
    layer_types: list[str] | None = None
    output_router_logits: bool = False
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 106

    def __post_init__(self, **kwargs):
        self.layer_types = (
            ["full_attention"] * self.num_hidden_layers if self.layer_types is None else list(self.layer_types)
        )

        default_rope_params: dict[Literal["full_attention", "sliding_attention"], dict[str, Any]] = {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": self.default_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": self.default_swa_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
        }
        if self.rope_parameters is None:
            self.rope_parameters = {
                layer_type: default_rope_params[layer_type] for layer_type in set(self.layer_types)
            }

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # ZAYA uses nested RoPE parameters keyed by layer type. Keep the base RoPE BC conversion from treating them
        # like a single flat RoPE dict and injecting top-level keys such as `rope_theta`.
        return kwargs

    def validate_architecture(self):
        if self.num_experts_per_tok != 1:
            raise ValueError("ZAYA currently supports `num_experts_per_tok=1` only.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be a multiple of `num_key_value_heads`.")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("`layer_types` must have one entry per hidden layer.")
        if invalid_layer_types := set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError(f"`layer_types` contains unsupported values: {sorted(invalid_layer_types)}.")
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError("`sliding_window` must be set when `layer_types` contains `sliding_attention`.")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("`sliding_window` must be a strictly positive integer.")


class ZayaRotaryEmbedding(LagunaRotaryEmbedding):
    pass


class ZayaRMSNorm(Qwen3MoeRMSNorm):
    pass


def make_zaya_cache(config: ZayaConfig) -> DynamicCache:
    """
    Create ZAYA's native hybrid cache.

    ZAYA uses `config.layer_types` for the attention mask and RoPE variant of each layer (`"full_attention"` or
    `"sliding_attention"`). That is separate from the cache layout: every ZAYA decoder layer needs the native
    `"hybrid"` cache layer because it stores all three states used during decoding:

    - The regular dynamic attention KV cache, updated after the CCA projection and RoPE application.
    - `conv_states`, the pre-convolution q/k tail used by `ZayaCCAProjection` on the next decoding step. Its channel
      dimension is `num_attention_heads * head_dim + num_key_value_heads * head_dim`, and its time dimension is
      `cca_time0 + cca_time1 - 2`.
    - `recurrent_states`, ZAYA's delayed value state. It stores the previous token's `val_proj2` output (the legacy
      `prev_h2`/second value projection state), so the next token can build its value from the current `val_proj1`
      output plus the cached delayed `val_proj2`.

    The copied config only changes `layer_types` to `"hybrid"` so `DynamicCache` instantiates
    `LinearAttentionAndFullAttentionLayer`; it does not alter the model's mask or RoPE layer types.
    """
    cache_config = copy.copy(config)
    cache_config.layer_types = ["hybrid"] * config.num_hidden_layers
    return DynamicCache(config=cache_config)


def _is_zaya_cache(past_key_values: Cache) -> bool:
    return (
        isinstance(past_key_values, DynamicCache)
        and len(past_key_values.layers) > 0
        and isinstance(past_key_values.layers[0], LinearAttentionAndFullAttentionLayer)
    )


class ZayaCCAProjection(nn.Module):
    """
    Projects hidden states into attention q/k/v states with ZAYA's CCA path.

    `linear_q` and `linear_k` produce the residual q/k states and are concatenated into `qk_states`. The causal
    `conv_qk_depthwise` + `conv_qk_grouped` stack mixes the current q/k stream with the cached pre-convolution tail;
    for example, decoding token `t` uses the cached q/k states from previous tokens plus the current `qk_states[:, t]`.
    Values are built from `val_proj1(hidden_states[:, t])` and a delayed `val_proj2`: during prefill token `t` uses
    `val_proj2(hidden_states[:, t - 1])`, while decoding reads the previous `val_proj2` from **the recurrent cache**.

    Final q/k states are L2-normalized to sqrt(head_dim). `temp` is the learned per-KV-head scale applied to keys.
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
        padding_mask: torch.Tensor | None = None,
    ):
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states.dtype)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        projected_queries = self.linear_q(hidden_states)
        projected_keys = self.linear_k(hidden_states)
        qk_states = torch.cat([projected_queries, projected_keys], dim=-1)

        query_residual = projected_queries.view(*hidden_shape)
        key_residual = projected_keys.view(*input_shape, self.num_key_value_heads, self.head_dim)

        key_residual = key_residual.repeat_interleave(self.num_key_value_groups, dim=-2)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(
            *input_shape, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
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
                *input_shape, self.num_attention_heads, self.head_dim
            )
            + query_residual
        )

        key = (
            convolved_qk_states[..., self.query_hidden_size :].view(
                *input_shape, self.num_key_value_heads, self.head_dim
            )
            + key_residual
        )

        value_current = self.val_proj1(hidden_states)
        projected_v2 = self.val_proj2(hidden_states)
        if use_precomputed_states:
            first_v2 = past_key_values.layers[self.layer_idx].recurrent_states.unsqueeze(1)
        else:
            first_v2 = self.val_proj2(hidden_states.new_zeros(input_shape[0], 1, self.hidden_size))
        value_delayed = torch.cat([first_v2, projected_v2[:, :-1]], dim=1)

        if past_key_values is not None:
            past_key_values.update_recurrent_state(projected_v2[:, -1, :], self.layer_idx)

        value = torch.cat([value_current, value_delayed], dim=-1).view(
            *input_shape, self.num_key_value_heads, self.head_dim
        )

        return query, key, value


class ZayaAttention(Phi3Attention):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del op_size  # noqa: F821
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.temp = nn.Parameter(torch.zeros(config.num_key_value_heads))
        self.qkv_proj = ZayaCCAProjection(
            config=self.config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: dict[str, Any] | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        batch_size, seq_length, _ = hidden_states.shape

        mask_mapping = attention_mask or {}
        causal_mask = mask_mapping.get("causal")
        padding_mask = mask_mapping.get("padding")

        query_states, key_states, value_states = self.qkv_proj(hidden_states, past_key_values, padding_mask)

        norm_eps = torch.finfo(query_states.dtype).eps
        head_dim_scale = self.scaling**-1
        query_states = query_states * (
            head_dim_scale / query_states.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps)
        )
        key_states = key_states * (head_dim_scale / key_states.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps))
        key_states = key_states * self.temp[None, None, :, None]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if isinstance(causal_mask, torch.Tensor):
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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.view(batch_size, seq_length, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_values


class ZayaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = ZayaAttention(config, layer_idx)
        self.input_norm = ZayaRMSNorm(self.config.hidden_size, eps=self.config.norm_epsilon)
        self.zaya_block = ZayaSparseMoeBlock(
            config,
            config.num_experts,
            config.zaya_mlp_expansion,
            config.intermediate_size,
            layer_idx,
        )
        self.post_attention_norm = ZayaRMSNorm(self.config.hidden_size, eps=self.config.norm_epsilon)
        self.post_attention_res_scale = ResidualScaling(config.hidden_size)
        self.post_mlp_res_scale = ResidualScaling(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
        attention_mask: dict[str, Any] | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        residual = hidden_states
        # Matches the original ZAYA `residual_in_fp32` path; norm casts back to the parameter dtype below.
        residual = residual.to(torch.float32)
        hidden_states = self.input_norm(residual.to(dtype=self.input_norm.weight.dtype))

        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        residual = self.post_attention_res_scale(hidden_states, residual)
        hidden_states = self.post_attention_norm(residual.to(dtype=self.post_attention_norm.weight.dtype))

        hidden_states, prev_router_hidden_states, _ = self.zaya_block(
            hidden_states,
            prev_router_hidden_states,
        )

        hidden_states = self.post_mlp_res_scale(hidden_states, residual)

        return hidden_states, self_attn_weights, prev_router_hidden_states


class ResidualScaling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))
        self.residual_scale = nn.Parameter(torch.ones(hidden_size))
        self.residual_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        residual = (residual + self.residual_bias) * self.residual_scale
        return hidden_states + residual


class ZayaRouterMLP(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.fc1(hidden_states))
        hidden_states = self.act_fn(self.fc2(hidden_states))
        return self.out_proj(hidden_states)


class ZayaRouter(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        num_moe_experts: int,
        num_experts_per_tok: int,
        mlp_expansion: int,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = int(hidden_size or getattr(config, "hidden_size"))
        self.layer_idx = layer_idx

        self.num_experts = num_moe_experts + 1
        self.topk = int(num_experts_per_tok)
        self.mlp_expansion = int(mlp_expansion)

        self.down_proj = nn.Linear(self.hidden_size, self.mlp_expansion, bias=True)

        self.use_eda = self.layer_idx != 0

        self.rmsnorm_eda = ZayaRMSNorm(self.mlp_expansion, eps=config.norm_epsilon)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))

        self.router_mlp = ZayaRouterMLP(self.mlp_expansion, self.num_experts)

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


class ZayaExperts(Qwen3MoeExperts):
    def __init__(self, config, num_experts: int, intermediate_size: int):
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = intermediate_size // 2
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]


class ZayaSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        num_moe_experts: int,
        mlp_expansion: int,
        intermediate_size: int,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_moe_experts = num_moe_experts
        self.router = ZayaRouter(
            config=self.config,
            layer_idx=layer_idx,
            num_moe_experts=self.num_moe_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
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

        # if the router outputs num_moe_experts, just skip the tokens
        # by masking them with id=0 and prob=0 to reuse the expert code
        skip_expert = expert_choice == self.num_moe_experts
        route_prob = route_prob.masked_fill(skip_expert, 0)
        expert_choice = expert_choice.masked_fill(skip_expert, 0)

        batch_size, seq_length, emb_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(batch_size * seq_length, emb_dim)
        expert_output = self.experts(hidden_states_flat, expert_choice, route_prob)
        expert_output = expert_output.view(batch_size, seq_length, emb_dim)

        return expert_output, prev_router_hidden_states, router_logits


class ZayaPreTrainedModel(LlamaPreTrainedModel):
    config: ZayaConfig
    config_class = ZayaConfig
    _no_split_modules = ["ZayaDecoderLayer"]
    # ZAYA generation uses the native hybrid dynamic cache, which is not a compileable cache.
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(ZayaRouter, index=3),
        "hidden_states": ZayaDecoderLayer,
        "attentions": ZayaAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ResidualScaling):
            init.ones_(module.hidden_states_scale)
            init.zeros_(module.hidden_states_bias)
            init.ones_(module.residual_scale)
            init.zeros_(module.residual_bias)
        elif isinstance(module, ZayaModel):
            init.ones_(module.input_hidden_states_scale)
            init.zeros_(module.input_hidden_states_bias)
        elif isinstance(module, ZayaRouter):
            if module.use_eda:
                init.ones_(module.router_states_scale)
            init.zeros_(module.balancing_biases)
            module.balancing_biases[-1] = -1.0
        elif isinstance(module, ZayaExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, ZayaRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                getattr(module, f"{layer_type}_inv_freq").copy_(curr_inv_freq)
                getattr(module, f"{layer_type}_original_inv_freq").copy_(curr_inv_freq)


@auto_docstring
class ZayaModel(ZayaPreTrainedModel):
    def __init__(self, config: ZayaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ZayaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False

        self.input_hidden_states_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.input_hidden_states_bias = nn.Parameter(torch.zeros(config.hidden_size))
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and (past_key_values is None or not _is_zaya_cache(past_key_values)):
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                raise ValueError("ZAYA requires a native hybrid cache created from `make_zaya_cache`.")
            past_key_values = make_zaya_cache(self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                "ZAYA CCA projection requires a 2D `attention_mask` to mask padding tokens before convolution."
            )

        causal_mask_mapping = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            position_ids,
            past_key_values,
        )
        padding_mask = attention_mask[:, -inputs_embeds.shape[1] :] if attention_mask is not None else None

        # ZAYA's hybrid cache is not compileable, so generation keeps `attention_mask` as the original 2D padding mask.
        # CCA projection only needs it during multi-token prefill; single-token decoding uses the cached convolution state.
        if inputs_embeds.shape[1] == 1:
            padding_mask = None

        hidden_states = inputs_embeds

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        hidden_states = (hidden_states + self.input_hidden_states_bias) * self.input_hidden_states_scale

        prev_router_hidden_states = None

        for layer_n, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[layer_n]
            emb_to_use = position_embeddings[layer_type]
            mask_mapping = {"causal": causal_mask_mapping[layer_type], "padding": padding_mask}
            layer_outputs = decoder_layer(
                hidden_states,
                prev_router_hidden_states,
                attention_mask=mask_mapping,
                past_key_values=past_key_values,
                position_embeddings=emb_to_use,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            prev_router_hidden_states = layer_outputs[2]

        hidden_states = self.final_norm(hidden_states.to(dtype=self.final_norm.weight.dtype))

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache,
    ):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": input_tensor,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Original ZAYA SWA only applies the local causal pattern; padding tokens are zeroed before the CCA projection.
        sliding_mask_kwargs = {**mask_kwargs, "attention_mask": None}
        mask_creation_functions = {
            "full_attention": lambda: create_causal_mask(**mask_kwargs),
            "sliding_attention": lambda: create_sliding_window_causal_mask(**sliding_mask_kwargs),
        }
        causal_mask_mapping = {}
        for layer_type in set(self.config.layer_types):
            causal_mask_mapping[layer_type] = mask_creation_functions[layer_type]()
        return causal_mask_mapping


@auto_docstring(checkpoint="Zyphra/ZAYA1-8B")
class ZayaForCausalLM(ZayaPreTrainedModel, AfmoeForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _is_stateful = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = ZayaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=self.config.lm_head_bias)

        self.post_init()


__all__ = [
    "ZayaConfig",
    "ZayaPreTrainedModel",
    "ZayaModel",
    "ZayaForCausalLM",
]
