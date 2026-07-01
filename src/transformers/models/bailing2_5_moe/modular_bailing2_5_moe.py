# Copyright 2026 InclusionAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BailingMoeV2_5 model."""

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
    DeepseekV3Experts,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3RMSNorm,
    DeepseekV3TopkRouter,
    rotate_half,
)
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..mixtral.modeling_mixtral import MixtralForCausalLM
from .configuration_bailing2_5_moe import BailingMoeV2_5Config


if is_flash_linear_attention_available():
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
else:
    chunk_simple_gla, fused_recurrent_simple_gla = None, None


is_fast_path_available = all((chunk_simple_gla, fused_recurrent_simple_gla))

logger = logging.get_logger(__name__)


class BailingMoeV2_5RMSNorm(DeepseekV3RMSNorm):
    pass


class BailingMoeV2_5RotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE for MLA layers — uses full qk_rope_head_dim, interleaved application."""

    @staticmethod
    def compute_default_rope_parameters(
        config: BailingMoeV2_5Config | None = None,
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


class BailingMoeV2_5LinearRotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE for linear attention layers — uses partial rotary factor on linear attention head dim."""

    @staticmethod
    def compute_default_rope_parameters(
        config: BailingMoeV2_5Config | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        linear_head_dim = config.hidden_size // config.linear_key_value_heads
        dim = int(linear_head_dim * config.partial_rotary_factor)

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class BailingMoeV2_5MLP(DeepseekV3MLP):
    pass


class BailingMoeV2_5TopkRouter(DeepseekV3TopkRouter):
    pass


class BailingMoeV2_5Experts(DeepseekV3Experts):
    pass


class BailingMoeV2_5MoE(DeepseekV3MoE):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.experts = BailingMoeV2_5Experts(config)
        self.gate = BailingMoeV2_5TopkRouter(config)
        # BailingMoeV2_5 sizes the shared expert with its own intermediate size and shared-expert count.
        self.shared_experts = BailingMoeV2_5MLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size * config.num_shared_experts
        )


class BailingMoeV2_5GroupRMSNorm(nn.Module):
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
    **kwargs,
):
    """Pure PyTorch fallback for ``chunk_simple_gla``, mirroring the reference FLA ``simple_gla`` kernel."""
    initial_dtype = query.dtype
    query, key, value, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, g)]

    batch_size, num_heads, sequence_length, head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = query.shape[-1] ** -0.5
    query = query * scale

    # Reshape into chunks
    query, key, value = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Cumulative decay within each chunk
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # for each chunk
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        attn_inter = (q_i * decay_mask[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_i
        last_recurrent_state = (
            last_recurrent_state * decay_mask[:, :, i, -1, None, None].exp()
            + (k_i * (decay_mask[:, :, i, -1, None] - decay_mask[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_simple_gla(
    query,
    key,
    value,
    g,
    initial_state=None,
    output_final_state=False,
    **kwargs,
):
    """Pure PyTorch fallback for ``fused_recurrent_simple_gla``."""
    initial_dtype = query.dtype
    query, key, value, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, g)]

    batch_size, num_heads, sequence_length, head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = query.shape[-1] ** -0.5
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(sequence_length):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        decay = g[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
        core_attn_out[:, :, i] = (q_i.unsqueeze(-1) * last_recurrent_state).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_rotary_pos_emb_linear(q, k, cos, sin, unsqueeze_dim=2):
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


class BailingMoeV2_5LightningAttention(nn.Module):
    """Lightning Linear Attention using SimpleGLA (Simple Gated Linear Attention) from the fla library."""

    def __init__(self, config: BailingMoeV2_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.g_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.g_norm = BailingMoeV2_5GroupRMSNorm(config.hidden_size, config.group_norm_size, eps=config.rms_norm_eps)

        if config.use_qk_norm:
            self.query_layernorm = BailingMoeV2_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = BailingMoeV2_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Build ALiBi-style slopes for decay, scaled by layer position
        slopes = _build_slope_tensor(self.num_heads)
        layer_scale = 1 - (layer_idx - 1) / (config.num_hidden_layers - 1) + 1e-5
        self.register_buffer("slope", (-slopes * layer_scale).to(torch.float32), persistent=False)

        self.chunk_simple_gla = chunk_simple_gla or torch_chunk_simple_gla
        self.recurrent_simple_gla = fused_recurrent_simple_gla or torch_recurrent_simple_gla

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path for BailingMoeV2_5LightningAttention is not available because flash-linear-attention "
                "is not installed. Falling back to pure PyTorch implementation. "
                "To install, see https://github.com/fla-org/flash-linear-attention#installation"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        bsz, q_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx) and q_len == 1
        )

        # Fused QKV projection
        query_states, key_states, value_states = (
            self.query_key_value(hidden_states).view(bsz, q_len, 3, self.num_heads, self.head_dim).unbind(dim=2)
        )

        # Apply QK norm per head before RoPE (matches training-time behaviour)
        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        # Apply partial RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_linear(query_states, key_states, cos, sin)

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

        gate_activation = F.silu if self.config.linear_silu else torch.sigmoid
        attn_output = attn_output * gate_activation(g_proj)

        attn_output = self.o_proj(attn_output)
        return attn_output


class BailingMoeV2_5Attention(DeepseekV3Attention):
    """MLA (Multi-Latent Attention) inherited from DeepSeek V3."""

    pass


class BailingMoeV2_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BailingMoeV2_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Token mixer: MLA or Lightning Linear Attention
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "full_attention":
            self.self_attn = BailingMoeV2_5Attention(config=config, layer_idx=layer_idx)
        else:
            self.linear_attn = BailingMoeV2_5LightningAttention(config=config, layer_idx=layer_idx)

        # MLP: MoE (sparse) or Dense
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = BailingMoeV2_5MoE(config)
        else:
            self.mlp = BailingMoeV2_5MLP(config)

        self.input_layernorm = BailingMoeV2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingMoeV2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class BailingMoeV2_5PreTrainedModel(PreTrainedModel):
    config: BailingMoeV2_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV2_5DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(BailingMoeV2_5TopkRouter, index=0),
        "hidden_states": BailingMoeV2_5DecoderLayer,
        "attentions": BailingMoeV2_5Attention,
    }
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BailingMoeV2_5TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, BailingMoeV2_5Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BailingMoeV2_5LightningAttention):
            # Reinitialize the slope buffer from config
            slopes = _build_slope_tensor(module.num_heads)
            layer_scale = 1 - (module.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
            init.copy_(module.slope, (-slopes * layer_scale).to(module.slope.dtype))


class BailingMoeV2_5Model(BailingMoeV2_5PreTrainedModel):
    def __init__(self, config: BailingMoeV2_5Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [BailingMoeV2_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BailingMoeV2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BailingMoeV2_5RotaryEmbedding(config=config)
        self.rotary_emb_linear = BailingMoeV2_5LinearRotaryEmbedding(config=config)
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


class BailingMoeV2_5ForCausalLM(MixtralForCausalLM):
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
        >>> from transformers import AutoTokenizer, BailingMoeV2_5ForCausalLM

        >>> model = BailingMoeV2_5ForCausalLM.from_pretrained("inclusionAI/Ling-2.6-flash-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-2.6-flash-base")

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


class BailingMoeV2_5ForSequenceClassification(GenericForSequenceClassification, BailingMoeV2_5PreTrainedModel):
    pass


class BailingMoeV2_5ForTokenClassification(GenericForTokenClassification, BailingMoeV2_5PreTrainedModel):
    pass


__all__ = [
    "BailingMoeV2_5PreTrainedModel",
    "BailingMoeV2_5Model",
    "BailingMoeV2_5ForCausalLM",
    "BailingMoeV2_5ForSequenceClassification",
    "BailingMoeV2_5ForTokenClassification",
]
