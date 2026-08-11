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

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, remap_legacy_layer_types
from ...integrations import use_kernelized_func
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaDecoderLayer
from ..olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
from ..olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
    Qwen3NextRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="allenai/Olmo-Hybrid-7B")
@strict
class OlmoHybridConfig(LlamaConfig):
    r"""
    linear_num_key_heads (`int`, *optional*):
        Number of key heads for the linear attention layers. Defaults to `num_attention_heads`.
    linear_num_value_heads (`int`, *optional*):
        Number of value heads for the linear attention layers. Defaults to `num_attention_heads`.
    linear_key_head_dim (`int`, *optional*):
        Dimension of each key head in linear attention layers. Defaults to `0.75 * hidden_size / linear_num_key_heads`.
    linear_value_head_dim (`int`, *optional*):
        Dimension of each value head in linear attention layers. Defaults to `2 * linear_key_head_dim`.
    linear_a_log_min (`float`, *optional*, defaults to 0.0):
        Minimum value for uniform initialization of A_log in GatedDeltaNet layers.
    linear_a_log_max (`float`, *optional*, defaults to 16.0):
        Maximum value for uniform initialization of A_log in GatedDeltaNet layers.
    linear_dt_min (`float`, *optional*, defaults to 0.001):
        Minimum value for dt initialization in GatedDeltaNet layers.
    linear_dt_max (`float`, *optional*, defaults to 0.1):
        Maximum value for dt initialization in GatedDeltaNet layers.
    linear_dt_init_floor (`float`, *optional*, defaults to 0.0001):
        Floor value for clamping dt during initialization in GatedDeltaNet layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
    linear_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
        Whether to allow negative eigenvalues in the GatedDeltaNet recurrence. When `True`, the beta
        parameter is scaled by 2.0 to allow values in range [0, 2] instead of [0, 1].

    Example:

    ```python
    >>> from transformers import OlmoHybridModel, OlmoHybridConfig

    >>> # Initializing an OlmoHybrid style configuration
    >>> configuration = OlmoHybridConfig()

    >>> # Initializing a model from the OlmoHybrid style configuration
    >>> model = OlmoHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo_hybrid"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 100352
    hidden_size: int = 3840
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 30
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 65536
    pad_token_id: int | None = 100277
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 100257
    rms_norm_eps: float = 1e-06
    layer_types: list[str] | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_a_log_min: float = 0.0
    linear_a_log_max: float = 16.0
    linear_dt_min: float = 0.001
    linear_dt_max: float = 0.1
    linear_dt_init_floor: float = 1e-4
    linear_conv_kernel_dim: int = 4
    linear_allow_neg_eigval: bool = True

    pretraining_tp = AttributeError()
    mlp_bias = AttributeError()
    head_dim = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            # Default: linear attention for most layers, full attention every 4th layer
            self.layer_types = ["linear_attention"] * int(self.num_hidden_layers)
            for i in range(int(self.num_hidden_layers)):
                if i % 4 == 3:
                    self.layer_types[i] = "full_attention"
            # Ensure at least one full attention layer for small num_hidden_layers
            if "full_attention" not in self.layer_types:
                self.layer_types[-1] = "full_attention"
        else:
            self.layer_types = remap_legacy_layer_types(self.layer_types)

        if self.linear_num_key_heads is None:
            self.linear_num_key_heads = self.num_attention_heads
        if self.linear_num_value_heads is None:
            self.linear_num_value_heads = self.num_attention_heads
        if self.linear_key_head_dim is None:
            self.linear_key_head_dim = int(0.75 * self.hidden_size / self.linear_num_key_heads)
        if self.linear_value_head_dim is None:
            self.linear_value_head_dim = 2 * self.linear_key_head_dim
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        PreTrainedConfig.__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if "linear_attention" not in self.layer_types:
            raise ValueError("OLMoHybrid expects at least one 'linear_attention' layer.")
        if all(t == "linear_attention" for t in self.layer_types):
            raise ValueError("OLMoHybrid expects at least one attention layer.")


class OlmoHybridRMSNormGated(Qwen3NextRMSNormGated):
    pass


class OlmoHybridRMSNorm(Olmo3RMSNorm):
    pass


class OlmoHybridAttention(Olmo3Attention):
    """
    Multi-headed attention for OLMo Hybrid that supports optional RoPE (NoPE mode).

    Inherits from Olmo3Attention. The only behavioral difference is that when
    position_embeddings is None, rotary position embeddings are skipped entirely,
    enabling NoPE mode for long context extension.
    """

    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Hybrid model doesn't use sliding window attention
        del self.sliding_window
        del self.attention_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # NoPE mode: skip RoPE when position_embeddings is None
        cos, sin = None, None
        if position_embeddings is not None:
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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OlmoHybridRotaryEmbedding(Olmo2RotaryEmbedding):
    """
    RoPE for OLMo Hybrid that returns float32 cos/sin to match OLMo-core.
    """

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # KEY difference from parent: return float32, don't cast to x.dtype
        return cos, sin


@use_kernelized_func(
    [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update]
)
class OlmoHybridGatedDeltaNet(nn.Module):
    """
    GatedDeltaNet linear attention for OLMo Hybrid.

    Key differences from Qwen3NextGatedDeltaNet:
    - Fully separate q/k/v/a/b projections (vs. fused qkvz + partially split ba)
    - Dedicated g_proj gate (vs. z derived from the fused qkvz projection)
    - Supports allow_neg_eigval: scales beta by 2.0 to allow range [0, 2]
    """

    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        self.eps = config.rms_norm_eps
        self.activation = config.hidden_act

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(
            config.linear_a_log_min, config.linear_a_log_max
        )
        self.A_log = nn.Parameter(torch.log(A))

        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(config.linear_dt_max) - math.log(config.linear_dt_min))
            + math.log(config.linear_dt_min)
        )
        dt = torch.clamp(dt, min=config.linear_dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Output norm - NOTE: FLA's FusedRMSNormGated uses eps=1e-5 by default
        self.o_norm = OlmoHybridRMSNormGated(self.head_v_dim, eps=1e-5)

        self.layer_type = config.layer_types[layer_idx]

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Requires LEFT padding to work correctly
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_cache = cache_params is not None
        # Reads "we have cached conv/recurrent state to continue from". Single-token vs multi-token
        # branching lives inside `ShortConvolution` and in the recurrent-vs-chunk kernel dispatch
        # below, each of which gates on `seq_len == 1` locally.
        use_precomputed_states = use_cache and cache_params.has_previous_state()

        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # Single token decode path
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        # Multi token prefill or simple "full" prefill
        else:
            # Concatenated state for prefill
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            # Cut out any tail
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        q, k, v = torch.split(
            mixed_qkv.transpose(1, 2),
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        q = q.view(batch_size, seq_len, -1, self.head_k_dim)
        k = k.view(batch_size, seq_len, -1, self.head_k_dim)
        v = v.view(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads > self.num_k_heads:
            expand_ratio = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(expand_ratio, dim=2)
            k = k.repeat_interleave(expand_ratio, dim=2)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        if use_precomputed_states and seq_len == 1:
            output, last_recurrent_state = torch_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            output, last_recurrent_state = torch_chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        # Update cache
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        gate = self.g_proj(hidden_states)
        output = output.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        output = self.o_norm(output, gate)
        output = output.reshape(batch_size, seq_len, -1)

        output = self.o_proj(output)

        return output


class OlmoHybridMLP(Olmo3MLP):
    pass


class OlmoHybridAttentionDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = OlmoHybridAttention(config=config, layer_idx=layer_idx)


class OlmoHybridLinearAttentionDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.self_attn
        self.linear_attn = OlmoHybridGatedDeltaNet(config, layer_idx=layer_idx)
        self.input_layernorm = OlmoHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = OlmoHybridMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Main difference to llama - signature (`cache_params`) and linear attention
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OlmoHybridPreTrainedModel(Qwen3NextPreTrainedModel):
    _is_stateful = True
    _no_split_modules = ["OlmoHybridAttentionDecoderLayer", "OlmoHybridLinearAttentionDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": [OlmoHybridAttentionDecoderLayer, OlmoHybridLinearAttentionDecoderLayer],
        "attentions": OlmoHybridAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, OlmoHybridGatedDeltaNet):
            cfg = self.config
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log).uniform_(cfg.linear_a_log_min, cfg.linear_a_log_max).log_(),
            )
            dt = torch.exp(
                torch.rand_like(module.dt_bias) * (math.log(cfg.linear_dt_max) - math.log(cfg.linear_dt_min))
                + math.log(cfg.linear_dt_min)
            )
            dt = torch.clamp(dt, min=cfg.linear_dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)


class OlmoHybridModel(Qwen3NextModel):
    def __init__(self, config: OlmoHybridConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                OlmoHybridLinearAttentionDecoderLayer(config, layer_idx)
                if config.layer_types[layer_idx] == "linear_attention"
                else OlmoHybridAttentionDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Released ckpt don't use any ROPE and have  it set to `None`
        self.rotary_emb = (
            OlmoHybridRotaryEmbedding(config=config)
            if getattr(config, "rope_parameters", None) is not None
            and config.rope_parameters.get("rope_theta") is not None
            else None
        )
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
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

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        # RoPE or NoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else None

        for i, decoder_layer in enumerate(self.layers):
            layer_position_embeddings = position_embeddings if self.config.layer_types[i] == "full_attention" else None

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=layer_position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class OlmoHybridForCausalLM(Olmo3ForCausalLM):
    pass


__all__ = [
    "OlmoHybridConfig",
    "OlmoHybridForCausalLM",
    "OlmoHybridModel",
    "OlmoHybridPreTrainedModel",
]
