# Copyright 2025 InclusionAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BailingHybrid model."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import is_flash_linear_attention_available
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..bamba.modeling_bamba import apply_mask_to_padding_states
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3NaiveMoe,
    DeepseekV3RMSNorm,
    DeepseekV3TopkRouter,
    rotate_half,
)
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..mixtral.modeling_mixtral import MixtralForCausalLM
from .configuration_bailing_hybrid import BailingHybridConfig


if is_flash_linear_attention_available():
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
else:
    chunk_simple_gla, fused_recurrent_simple_gla = None, None


is_fast_path_available = all((chunk_simple_gla, fused_recurrent_simple_gla))

logger = logging.get_logger(__name__)


class BailingHybridRMSNorm(DeepseekV3RMSNorm):
    pass


class BailingHybridRotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE for MLA layers — uses full qk_rope_head_dim, interleaved application."""

    @staticmethod
    def compute_default_rope_parameters(
        config: BailingHybridConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = config.qk_rope_head_dim

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class BailingHybridLinearRotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE for linear attention layers — uses partial rotary factor on linear attention head dim."""

    @staticmethod
    def compute_default_rope_parameters(
        config: BailingHybridConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        linear_head_dim = config.hidden_size // config.num_kv_heads_for_linear_attn
        dim = int(linear_head_dim * config.partial_rotary_factor)

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class BailingHybridMLP(DeepseekV3MLP):
    pass


class BailingHybridTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config):
        super().__init__(config)
        self.n_routed_experts = config.num_experts


class BailingHybridExperts(DeepseekV3NaiveMoe):
    pass


class BailingHybridMoE(DeepseekV3MoE):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.experts = BailingHybridExperts(config)
        self.gate = BailingHybridTopkRouter(config)
        self.shared_experts = BailingHybridMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size * config.num_shared_experts
        )
        self.n_routed_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok


class BailingHybridGroupRMSNorm(nn.Module):
    """Group-wise RMS normalization for linear attention output."""

    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_norm_size = group_norm_size

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        orig_shape = hidden_states.shape
        # Reshape to groups: (..., group_norm_size, hidden_size // group_norm_size)
        hidden_states = hidden_states.view(*orig_shape[:-1], self.group_norm_size, -1)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.view(orig_shape)
        return (self.weight * hidden_states).to(input_dtype)


def _build_slope_tensor(num_heads: int) -> torch.Tensor:
    """Build ALiBi-style slope tensor for lightning linear attention decay."""

    def _get_interleave(n: int) -> list[float]:
        def _get_interleave_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                _get_interleave_power_of_2(closest_power_of_2)
                + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(_get_interleave(num_heads), dtype=torch.float32)
    return slopes


def torch_chunk_simple_gla(
    query,
    key,
    value,
    g,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
):
    """Pure PyTorch fallback for chunk_simple_gla when fla is not available."""
    initial_dtype = query.dtype
    query, key, value = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value)]
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, head_dim = key.shape
    v_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        g = F.pad(g, (0, pad_size))
    total_len = sequence_length + pad_size

    scale = head_dim**-0.5
    query = query * scale

    # Reshape into chunks
    num_chunks = total_len // chunk_size
    query = query.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
    key = key.view(batch_size, num_heads, num_chunks, chunk_size, head_dim)
    value = value.view(batch_size, num_heads, num_chunks, chunk_size, v_dim)
    g = g.view(batch_size, num_heads, num_chunks, chunk_size)

    # Cumulative decay within each chunk
    g_cumsum = g.cumsum(dim=-1)

    state = (
        torch.zeros(batch_size, num_heads, head_dim, v_dim, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    output = torch.zeros(
        batch_size, num_heads, num_chunks, chunk_size, v_dim, device=query.device, dtype=torch.float32
    )

    for c in range(num_chunks):
        q_c = query[:, :, c]  # [B, H, C, D]
        k_c = key[:, :, c]
        v_c = value[:, :, c]
        g_c = g_cumsum[:, :, c]  # [B, H, C]

        # Intra-chunk attention with decay
        decay_matrix = (g_c.unsqueeze(-1) - g_c.unsqueeze(-2)).tril().exp()
        attn = (q_c @ k_c.transpose(-1, -2)) * decay_matrix.tril()
        intra = attn @ v_c

        # Inter-chunk: query attends to state from previous chunks
        inter = (q_c * g_c.unsqueeze(-1).exp()) @ state

        output[:, :, c] = intra + inter

        # Update state with this chunk's contributions
        chunk_end_decay = g_c[:, :, -1].unsqueeze(-1).unsqueeze(-1)
        per_step_decay = (g_c[:, :, -1].unsqueeze(-1) - g_c).exp()  # [B, H, C]
        state = state * chunk_end_decay.exp() + (k_c * per_step_decay.unsqueeze(-1)).transpose(-1, -2) @ v_c

    output = output.view(batch_size, num_heads, total_len, v_dim)[:, :, :sequence_length]
    output = output.transpose(1, 2).contiguous().to(initial_dtype)

    if not output_final_state:
        state = None
    return output, state


def torch_recurrent_simple_gla(
    query,
    key,
    value,
    g,
    initial_state=None,
    output_final_state=False,
):
    """Pure PyTorch fallback for fused_recurrent_simple_gla."""
    initial_dtype = query.dtype
    query, key, value = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value)]
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, head_dim = key.shape
    v_dim = value.shape[-1]
    scale = head_dim**-0.5
    query = query * scale

    state = (
        torch.zeros(batch_size, num_heads, head_dim, v_dim, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    output = torch.zeros(batch_size, num_heads, sequence_length, v_dim, device=query.device, dtype=torch.float32)

    for t in range(sequence_length):
        decay = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        state = state * decay + key[:, :, t].unsqueeze(-1) * value[:, :, t].unsqueeze(-2)
        output[:, :, t] = (query[:, :, t].unsqueeze(-1) * state).sum(dim=-2)

    if not output_final_state:
        state = None
    output = output.transpose(1, 2).contiguous().to(initial_dtype)
    return output, state


def _apply_rotary_pos_emb_linear(q, k, cos, sin, unsqueeze_dim=2):
    """Apply rotary position embedding with partial rotary support for linear attention.
    Q/K are in [bsz, seq_len, n_heads, head_dim] format.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class BailingHybridLightningAttention(nn.Module):
    """Lightning Linear Attention using SimpleGLA (Simple Gated Linear Attention) from the fla library."""

    def __init__(self, config: BailingHybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_kv_heads_for_linear_attn
        self.head_dim = config.hidden_size // self.num_heads

        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.g_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.g_norm = BailingHybridGroupRMSNorm(config.hidden_size, config.group_norm_size, eps=config.rms_norm_eps)

        if config.use_qk_norm:
            self.query_layernorm = BailingHybridRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = BailingHybridRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Build ALiBi-style slopes for decay, scaled by layer position
        slopes = _build_slope_tensor(self.num_heads)
        layer_scale = 1 - (layer_idx - 1) / (config.num_hidden_layers - 1) + 1e-5
        self.register_buffer("slope", (-slopes * layer_scale).to(torch.float32), persistent=False)

        self.chunk_simple_gla = chunk_simple_gla or torch_chunk_simple_gla
        self.recurrent_simple_gla = fused_recurrent_simple_gla or torch_recurrent_simple_gla

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path for BailingHybridLightningAttention is not available because flash-linear-attention "
                "is not installed. Falling back to pure PyTorch implementation. "
                "To install, see https://github.com/fla-org/flash-linear-attention#installation"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        bsz, q_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx) and q_len == 1
        )

        # Fused QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states = qkv[:, :, 0]
        key_states = qkv[:, :, 1]
        value_states = qkv[:, :, 2]

        # Apply QK norm per head before RoPE (matches training-time behaviour)
        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        # Apply partial RoPE
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_linear(query_states, key_states, cos, sin)

        # Gate projection
        g_proj = self.g_proj(hidden_states)

        # Compute decay from slopes
        g = self.slope[None, None, :].expand(bsz, q_len, self.num_heads)

        if use_precomputed_states:
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
            attn_output, last_state = self.recurrent_simple_gla(
                query_states,
                key_states,
                value_states,
                g=g,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
            )
        else:
            attn_output, last_state = self.chunk_simple_gla(
                query_states,
                key_states,
                value_states,
                g=g,
                initial_state=None,
                output_final_state=cache_params is not None,
            )

        if cache_params is not None:
            # For models without conv1d, we need to ensure dtype/device are set
            # on the cache layer before updating recurrent state
            layer = cache_params.layers[self.layer_idx]
            if not hasattr(layer, "dtype") or layer.dtype is None:
                layer.dtype = last_state.dtype
                layer.device = last_state.device
            cache_params.update_recurrent_state(last_state, self.layer_idx)
            # SimpleGLA has no conv1d, so update_conv_state is never called.
            # We manually set has_previous_state so decode uses the recurrent path.
            layer.has_previous_state = True

        # Reshape from [bsz, q_len, num_heads, head_dim] to [bsz*q_len, hidden_size]
        attn_output = attn_output.reshape(bsz * q_len, self.hidden_size)
        attn_output = self.g_norm(attn_output)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        if self.config.linear_silu:
            attn_output = attn_output * F.silu(g_proj)
        else:
            attn_output = attn_output * torch.sigmoid(g_proj)

        attn_output = self.o_proj(attn_output)
        return attn_output


class BailingHybridAttention(DeepseekV3Attention):
    """MLA (Multi-Latent Attention) inherited from DeepSeek V3."""

    pass


class BailingHybridDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BailingHybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Token mixer: MLA or Lightning Linear Attention
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "full_attention":
            self.self_attn = BailingHybridAttention(config=config, layer_idx=layer_idx)
        else:
            self.linear_attn = BailingHybridLightningAttention(config=config, layer_idx=layer_idx)

        # MLP: MoE or Dense
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = BailingHybridMoE(config)
        else:
            self.mlp = BailingHybridMLP(config)

        self.input_layernorm = BailingHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BailingHybridPreTrainedModel(PreTrainedModel):
    config: BailingHybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingHybridDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(BailingHybridTopkRouter, index=0),
        "hidden_states": BailingHybridDecoderLayer,
        "attentions": BailingHybridAttention,
    }
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BailingHybridTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, BailingHybridExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BailingHybridLightningAttention):
            # Reinitialize the slope buffer from config
            slopes = _build_slope_tensor(module.num_heads)
            layer_scale = 1 - (module.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
            module.slope.copy_((-slopes * layer_scale).to(module.slope.dtype))


class BailingHybridModel(BailingHybridPreTrainedModel):
    def __init__(self, config: BailingHybridConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [BailingHybridDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BailingHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BailingHybridRotaryEmbedding(config=config)
        self.rotary_emb_linear = BailingHybridLinearRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
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
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
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

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings_mla = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_linear = self.rotary_emb_linear(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.config.layer_types[i] == "linear_attention":
                layer_mask = linear_attn_mask
                layer_position_embeddings = position_embeddings_linear
            else:
                layer_mask = causal_mask
                layer_position_embeddings = position_embeddings_mla

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=layer_position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        """For linear attention, we only need a simple mask, not the full causal mask."""
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask


class BailingHybridForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts

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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BailingHybridForCausalLM

        >>> model = BailingHybridForCausalLM.from_pretrained("inclusionAI/Ring-2.5-1T")
        >>> tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-2.5-1T")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


class BailingHybridForSequenceClassification(GenericForSequenceClassification, BailingHybridPreTrainedModel):
    pass


class BailingHybridForTokenClassification(GenericForTokenClassification, BailingHybridPreTrainedModel):
    pass


__all__ = [
    "BailingHybridPreTrainedModel",
    "BailingHybridModel",
    "BailingHybridForCausalLM",
    "BailingHybridForSequenceClassification",
    "BailingHybridForTokenClassification",
]
