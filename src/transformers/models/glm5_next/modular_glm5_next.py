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

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache, DynamicLayer, LinearAttentionLayer
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import is_flash_attention_requested
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4Model
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLExperts
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from .configuration_glm5_next import Glm5NextConfig


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb_to_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
    interleaved: bool = False,
) -> torch.Tensor:
    if not interleaved:
        return apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=unsqueeze_dim)[0]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = x.shape[-1]
    cos = cos[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


# =============================================================================
# MHC (Manifold-Constrained Hyper-Connection) helpers
# =============================================================================


class Glm5NextHyperConnection(DeepseekV4HyperConnection):
    pass


class Glm5NextHyperHead(nn.Module):
    """Final GLM-5-Next HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


# =============================================================================
# KDA Linear Attention
# =============================================================================


class Glm5NextShortConv(nn.Conv1d):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
            dtype=torch.float32,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
        has_left_context: bool,
    ) -> torch.Tensor:
        if not has_left_context:
            hidden_states = F.pad(hidden_states, (self.kernel_size[0] - 1, 0))
        out = super().forward(hidden_states.to(self.weight.dtype))
        out = F.silu(out)
        if has_left_context:
            out = out[:, :, -seq_len:]
        return out.transpose(1, 2).to(dtype)


class Glm5NextForgetGate(nn.Module):
    def __init__(self, hidden_size: int, config: Glm5NextConfig):
        super().__init__()
        linear_attn_config = config.linear_attn_config
        self.head_dim = linear_attn_config["head_dim"]
        self.num_heads = linear_attn_config["num_heads"]
        self.safe_gate = linear_attn_config.get("safe_gate", False)
        self.safe_gate_lower_bound = linear_attn_config.get("lower_bound", None)
        qk_projection_size = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, qk_projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(qk_projection_size, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        batch_size, seq_len = forget_gate.shape[:2]
        g = forget_gate.float() + self.dt_bias.view(1, 1, -1)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)

        if self.safe_gate and self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(torch.exp(A_log) * g)

        threshold = 20.0
        g_linear = g > threshold
        sp = torch.where(g_linear, g, torch.log(1.0 + torch.exp(g)))
        return -torch.exp(A_log) * sp


class Glm5NextKdaSequential(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, S, H, D = q.shape
        V = v.shape[-1]
        dtype = q.dtype

        if initial_state is None:
            h = torch.zeros(B, H, D, V, device=q.device, dtype=torch.float32)
        else:
            h = initial_state.to(device=q.device, dtype=torch.float32)
        o = torch.empty(B, S, H, V, device=q.device, dtype=dtype)
        scale = D**-0.5

        for t in range(S):
            q_t = q[:, t].float()
            k_t = k[:, t].float()
            v_t = v[:, t].float()

            q_t = q_t / torch.sqrt(torch.sum(q_t * q_t, dim=-1, keepdim=True) + 1e-6)
            k_t = k_t / torch.sqrt(torch.sum(k_t * k_t, dim=-1, keepdim=True) + 1e-6)

            h = h * torch.exp(g[:, t, :, :, None])
            delta_v = v_t - torch.einsum("bhkv,bhk->bhv", h, k_t)
            delta_v = delta_v * beta[:, t, :, None]
            h = h + torch.einsum("bhk,bhv->bhkv", k_t, delta_v)
            o[:, t] = torch.einsum("bhkv,bhk->bhv", h, q_t * scale).to(dtype)

        return o, h if output_final_state else None


class Glm5NextAttentionCacheLayer(DynamicLayer):
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        for attribute_name in ("indexer_keys", "indexer_gate_scores"):
            state = getattr(self, attribute_name, None)
            if state is not None:
                setattr(
                    self,
                    attribute_name,
                    state.index_select(0, beam_idx.to(state.device)),
                )

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        super().crop(max_length)
        for attribute_name in ("indexer_keys", "indexer_gate_scores"):
            state = getattr(self, attribute_name, None)
            if state is not None:
                setattr(self, attribute_name, state[:, :max_length])


class Glm5NextLinearAttentionCacheLayer(LinearAttentionLayer):
    def update_conv_state(
        self, conv_states: torch.Tensor, conv_kernel_size: int | None = None, **kwargs
    ) -> torch.Tensor:
        if not self.has_previous_state and conv_kernel_size is not None:
            if conv_states.shape[-1] >= conv_kernel_size:
                conv_states = conv_states[..., -conv_kernel_size:]
            else:
                conv_states = F.pad(conv_states, (conv_kernel_size - conv_states.shape[-1], 0))
        return super().update_conv_state(conv_states, **kwargs)


class Glm5NextDynamicCache(DynamicCache):
    def __init__(self, *args, config: Glm5NextConfig | None = None, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        if config is not None:
            for layer_idx, layer_type in enumerate(config.layer_types):
                if layer_type == "linear_attention":
                    self.layers[layer_idx] = Glm5NextLinearAttentionCacheLayer(config)
                else:
                    self.layers[layer_idx] = Glm5NextAttentionCacheLayer(config)


class Glm5NextLinearAttention(nn.Module):
    """
    Kimi-style KDA (Kimi Linear Attention) for GLM-5-Next.

    Replaces standard MLA attention on layers listed in `linear_attn_config["kda_layers"]`.

    Architecture (checkpoint naming):
      - Q/K/V: x -> q_proj/k_proj/v_proj -> causal_conv1d via q_conv1d/k_conv1d/v_conv1d
      - Forget gate: x -> f_a_proj -> f_b_proj -> g = -exp(A_log) * softplus(gate + dt_bias)
        or safe lower-bound gate when `linear_attn_config["lower_bound"]` is set
      - Input gate: x -> b_proj -> sigmoid -> beta [B, S, H]
      - Recurrence: o = kda_sequential(q, k, v, g, beta)  [pure PyTorch]
      - Output gate: x -> g_a_proj -> g_b_proj -> sigmoid -> gated RMSNorm via o_norm
      - Output: o_proj
    """

    def __init__(
        self,
        hidden_size: int,
        config: Glm5NextConfig,
        layer_idx: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.layer_idx = layer_idx

        linear_attn_config = config.linear_attn_config
        self.head_dim = linear_attn_config["head_dim"]
        self.v_head_dim = self.head_dim
        self.num_heads = linear_attn_config["num_heads"]
        self.conv_kernel_size = linear_attn_config.get("short_conv_kernel_size", 4)

        qk_projection_size = self.head_dim * self.num_heads
        v_projection_size = self.v_head_dim * self.num_heads

        # Separate Q, K, V projections (checkpoint uses q_proj/k_proj/v_proj, not qkv_proj)
        self.q_proj = nn.Linear(hidden_size, qk_projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, v_projection_size, bias=False)

        self.q_conv1d = Glm5NextShortConv(qk_projection_size, self.conv_kernel_size)
        self.k_conv1d = Glm5NextShortConv(qk_projection_size, self.conv_kernel_size)
        self.v_conv1d = Glm5NextShortConv(v_projection_size, self.conv_kernel_size)
        self.forget_gate = Glm5NextForgetGate(hidden_size, config)

        # Beta (input gate): hidden -> num_heads
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Output norm gate: hidden -> v_head_dim -> v_projection_size
        self.g_a_proj = nn.Linear(hidden_size, self.v_head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.v_head_dim, v_projection_size, bias=False)

        # FusedRMSNormGated equivalent; keep the module wrapper so checkpoint
        # keys match `self_attn.o_norm.weight`.
        self.o_norm = Glm5NextRMSNorm(self.v_head_dim, eps=rms_norm_eps)
        self.kda_sequential = Glm5NextKdaSequential()

        # Output projection
        self.o_proj = nn.Linear(v_projection_size, hidden_size, bias=False)

    def _gated_rms_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """FusedRMSNormGated(..., activation="sigmoid") reference path."""
        return self.o_norm(x) * torch.sigmoid(gate.float()).to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        _ = use_cache
        batch_size, seq_len = hidden_states.shape[:2]
        qk_projection_size = self.head_dim * self.num_heads
        v_projection_size = self.v_head_dim * self.num_heads

        has_cache_state = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if has_cache_state:
            conv_state = past_key_values.layers[self.layer_idx].conv_states.to(hidden_states.device)
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states

        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)], dim=-1
        ).transpose(1, 2)

        q_input, k_input, v_input = torch.split(
            mixed_qkv, [qk_projection_size, qk_projection_size, v_projection_size], dim=1
        )
        if has_cache_state:
            q_state, k_state, v_state = torch.split(
                conv_state, [qk_projection_size, qk_projection_size, v_projection_size], dim=1
            )
            q_input = torch.cat([q_state, q_input], dim=-1)
            k_input = torch.cat([k_state, k_input], dim=-1)
            v_input = torch.cat([v_state, v_input], dim=-1)

        if past_key_values is not None:
            past_key_values.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)

        # Q, K, V projections and causal conv1d
        q = self.q_conv1d(q_input, seq_len, hidden_states.dtype, has_cache_state)
        k = self.k_conv1d(k_input, seq_len, hidden_states.dtype, has_cache_state)
        v = self.v_conv1d(v_input, seq_len, hidden_states.dtype, has_cache_state)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.v_head_dim)

        # Forget gate and input gate
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # KDA sequential recurrence
        core_attn_out, last_recurrent_state = self.kda_sequential(
            q,
            k,
            v,
            g,
            beta,
            initial_state=recurrent_state if has_cache_state else None,
            output_final_state=past_key_values is not None,
        )
        if past_key_values is not None:
            past_key_values.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # Output norm with gating
        g_proj = self.g_b_proj(self.g_a_proj(hidden_states))
        g_proj = g_proj.view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        core_attn_out = self._gated_rms_norm(core_attn_out, g_proj)

        # Flatten and output projection
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.o_proj(core_attn_out)

        return output, None, None


# =============================================================================
# MLA (Multi-head Latent Attention) with optional DSA indexer scaffold
# =============================================================================


class Glm5NextIndexer(nn.Module):
    """
    GLM-5-Next DSA/NSA indexer.

    This mirrors sglang's NSA `Indexer` parameter layout (`wq_b`, `wk`,
    `weights_proj`, `k_norm`). sglang computes the final scores with optimized
    `fp8_index`/NSA kernels; this path computes the same top-k sparse-attention
    indices with regular PyTorch ops.
    """

    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.index_kpool: int = config.index_kpool
        self.index_kpool_compress: bool = config.index_kpool_compress
        self.indexer_rope_interleave: bool = config.indexer_rope_interleave
        self.kpool_enabled = self.index_kpool > 1 and self.index_kpool_compress
        self.index_kpool_always_select_tail: bool = config.index_kpool_always_select_tail
        self.skip_rope: bool = True

        if self.kpool_enabled:
            self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
            self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))
        else:
            self.register_parameter("index_kpool_compress_ape", None)
            self.register_parameter("index_kpool_compress_gate", None)

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if config.index_dsa_use_layernorm else nn.Identity()
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    def _compress_kpool_keys(self, keys: torch.Tensor, gate_scores: torch.Tensor) -> torch.Tensor:
        num_complete_pools = keys.shape[1] // self.index_kpool
        if num_complete_pools == 0:
            return keys.new_empty(keys.shape[0], 0, self.head_dim)

        pooled_length = num_complete_pools * self.index_kpool
        grouped_keys = keys[:, :pooled_length].reshape(
            keys.shape[0], num_complete_pools, self.index_kpool, self.head_dim
        )
        grouped_gate_scores = gate_scores[:, :pooled_length].reshape_as(grouped_keys)
        compression_logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        probabilities = compression_logits.softmax(dim=2).to(grouped_keys.dtype)
        return (probabilities * grouped_keys).sum(dim=2)

    def _update_key_cache(self, k: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        if past_key_values is None:
            return k

        cache_layer = past_key_values.layers[self.layer_idx]
        cached_keys = getattr(cache_layer, "indexer_keys", None)
        if k.shape[1] > 1 or cached_keys is None or cached_keys.shape[0] != k.shape[0]:
            k_cached = k
        else:
            k_cached = torch.cat([cached_keys.to(k.device), k], dim=1)
        cache_layer.indexer_keys = k_cached
        return k_cached

    def _update_kpool_cache(
        self,
        keys: torch.Tensor,
        gate_scores: torch.Tensor,
        past_key_values: Cache | None,
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is None:
            return keys, gate_scores

        cache_layer = past_key_values.layers[self.layer_idx]
        cached_keys = getattr(cache_layer, "indexer_keys", None)
        cached_gate_scores = getattr(cache_layer, "indexer_gate_scores", None)
        prefix_len = total_len - keys.shape[1]
        can_append = (
            cached_keys is not None
            and cached_gate_scores is not None
            and cached_keys.shape[0] == keys.shape[0]
            and cached_keys.shape[1] == prefix_len
            and cached_gate_scores.shape[:2] == cached_keys.shape[:2]
        )
        if can_append:
            assert cached_keys is not None and cached_gate_scores is not None
            keys = torch.cat((cached_keys.to(keys.device), keys), dim=1)
            gate_scores = torch.cat(
                (cached_gate_scores.to(gate_scores.device), gate_scores),
                dim=1,
            )

        cache_layer.indexer_keys = keys
        cache_layer.indexer_gate_scores = gate_scores
        return keys, gate_scores

    def _indexer_score_mask(self, attention_mask: torch.Tensor | None, total_len: int) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask[:, 0, :, :total_len]
        if attention_mask.dim() == 2:
            return attention_mask[:, None, :total_len]
        return attention_mask[..., :total_len]

    def _expand_kpool_topk(
        self,
        pool_scores: torch.Tensor,
        visible_lengths: torch.Tensor,
    ) -> torch.LongTensor:
        batch_size, query_length, num_pools = pool_scores.shape
        complete_pools = torch.div(visible_lengths, self.index_kpool, rounding_mode="floor")
        group_budget = self.index_topk // self.index_kpool
        select_k = min(group_budget, num_pools)

        if select_k > 0:
            pool_ids = torch.arange(num_pools, device=pool_scores.device)
            pool_valid = pool_ids.view(1, 1, -1) < complete_pools.unsqueeze(-1)
            masked_scores = pool_scores.masked_fill(~pool_valid, torch.finfo(pool_scores.dtype).min)
            selected_groups = masked_scores.topk(select_k, dim=-1).indices
            selected_valid = pool_valid.gather(-1, selected_groups)
            offsets = torch.arange(self.index_kpool, device=pool_scores.device)
            expanded = (selected_groups.unsqueeze(-1) * self.index_kpool + offsets).flatten(-2)
            expanded_valid = selected_valid.unsqueeze(-1).expand(
                batch_size,
                query_length,
                select_k,
                self.index_kpool,
            )
            expanded = expanded.masked_fill(~expanded_valid.flatten(-2), -1)
        else:
            expanded = torch.empty(
                batch_size,
                query_length,
                0,
                dtype=torch.long,
                device=pool_scores.device,
            )

        expanded = F.pad(expanded, (0, self.index_topk - expanded.shape[-1]), value=-1)
        if not self.index_kpool_always_select_tail:
            return expanded

        output_width = self.index_topk + self.index_kpool - 1
        output_positions = torch.arange(output_width, device=pool_scores.device).view(1, 1, -1)
        history_length = (complete_pools * self.index_kpool).clamp(max=self.index_topk)
        history_indices = output_positions.clamp(max=self.index_topk - 1).expand(
            batch_size,
            query_length,
            -1,
        )
        history_values = expanded.gather(-1, history_indices)
        is_history = output_positions < history_length.unsqueeze(-1)

        tail_offset = output_positions - history_length.unsqueeze(-1)
        tail_count = visible_lengths.remainder(self.index_kpool)
        is_tail = tail_offset.ge(0) & tail_offset.lt(tail_count.unsqueeze(-1))
        tail_values = complete_pools.unsqueeze(-1) * self.index_kpool + tail_offset

        result = torch.full(
            (batch_size, query_length, output_width),
            -1,
            dtype=torch.long,
            device=pool_scores.device,
        )
        result = torch.where(is_history, history_values, result)
        return torch.where(is_tail, tail_values, result)

    def build_attention_mask(
        self,
        topk_indices: torch.Tensor,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_length: int,
        total_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        valid_indices = topk_indices.ge(0) & topk_indices.lt(total_len)
        safe_indices = topk_indices.clamp(min=0, max=total_len - 1)
        selected_counts = torch.zeros(
            (batch_size, seq_length, total_len),
            dtype=torch.int32,
            device=device,
        )
        selected_counts.scatter_add_(-1, safe_indices, valid_indices.to(torch.int32))
        index_mask = torch.zeros_like(selected_counts, dtype=dtype)
        index_mask = index_mask.masked_fill(selected_counts.eq(0), float("-inf")).unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            return index_mask + attention_mask[..., :total_len]
        if attention_mask is not None:
            return attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
        return index_mask

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        mask_dtype: torch.dtype | None = None,
        total_len: int | None = None,
        topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        total_len = seq_len if total_len is None else total_len
        mask_dtype = hidden_states.dtype if mask_dtype is None else mask_dtype

        if topk_indices is None:
            cos, sin = position_embeddings
            q = self.wq_b(q_resid)
            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
            q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            if not self.skip_rope and self.qk_rope_head_dim > 0:
                q_pe = apply_rotary_pos_emb_to_single(
                    q_pe, cos, sin, unsqueeze_dim=2, interleaved=self.indexer_rope_interleave
                )
            q = torch.cat([q_pe, q_nope], dim=-1)

            k = self.k_norm(self.wk(hidden_states))
            k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            if not self.skip_rope and self.qk_rope_head_dim > 0:
                k_pe = apply_rotary_pos_emb_to_single(
                    k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2, interleaved=self.indexer_rope_interleave
                ).squeeze(2)
            k = torch.cat([k_pe, k_nope], dim=-1)
            if self.kpool_enabled:
                gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
                k_cached, gate_scores = self._update_kpool_cache(
                    k,
                    gate_scores,
                    past_key_values,
                    total_len,
                )
            else:
                k_cached = self._update_key_cache(k, past_key_values)

            weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
            if self.kpool_enabled:
                pooled_keys = self._compress_kpool_keys(k_cached, gate_scores)
                scores = torch.einsum("bshd,bpd->bshp", q.float(), pooled_keys.float()) * self.softmax_scale
                pool_scores = torch.einsum("bshp,bsh->bsp", F.relu(scores), weights)
                prefix_len = total_len - seq_len
                visible_lengths = prefix_len + torch.arange(
                    1,
                    seq_len + 1,
                    device=hidden_states.device,
                )
                visible_lengths = visible_lengths.unsqueeze(0).expand(batch_size, -1)
                topk_indices = self._expand_kpool_topk(pool_scores, visible_lengths)
            else:
                scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
                index_scores = torch.einsum("bsht,bsh->bst", F.relu(scores), weights)
                score_mask = self._indexer_score_mask(attention_mask, index_scores.shape[-1])
                if score_mask is not None:
                    index_scores = index_scores + score_mask
                topk = min(self.index_topk, index_scores.shape[-1])
                topk_indices = index_scores.topk(topk, dim=-1).indices

        attention_mask = self.build_attention_mask(
            topk_indices,
            attention_mask,
            batch_size,
            seq_len,
            total_len,
            mask_dtype,
            hidden_states.device,
        )
        return topk_indices, attention_mask


class Glm5NextAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) for GLM-5-Next full-attention layers.

    GLM-5-Next checkpoints use the no-RoPE MLA key path: kv_b_proj outputs
    full qk_head_dim + v_head_dim.

    **Caching**: fully expanded K/V, compatible with DynamicCache / SDPA / flash attention.
    """

    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.is_causal = True

        # Query projection
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = Glm5NextRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # Key-Value projections (MLA compressed path plus single-head RoPE key stream).
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = Glm5NextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        kv_b_out = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, kv_b_out, bias=False)

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)

        has_indexer_config = config.index_dsa_use_layernorm is not None or config.index_topk_pattern is not None
        self.indexer = Glm5NextIndexer(config, layer_idx) if has_indexer_config else None
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is None:
            freq = config.index_topk_freq
            offset = config.index_skip_topk_offset
            indexer_types = [
                "full" if (max(index_layer_idx - offset, 0) % freq) == 0 else "shared"
                for index_layer_idx in range(config.num_hidden_layers)
            ]
        self.skip_topk = self.indexer is None or indexer_types[layer_idx] == "shared"
        self.next_skip_topk = (
            self.indexer is not None
            and layer_idx < len(indexer_types) - 1
            and indexer_types[layer_idx + 1] == "shared"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]

        # ===== Query path =====
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        # NoPE MLA: GLM-5-Next full-attention layers keep the `qk_rope_head_dim` query/key
        # slice but do NOT rotate it (matches sglang `skip_rope=mla_nope`). Position
        # information is carried by the KDA linear-attention layers and the DSA indexer,
        # which applies RoPE on its own projections.

        # ===== KV path =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        key_states, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if self.qk_rope_head_dim > 0:
            # Single-head (MQA) key-pe stream broadcast to all heads, unrotated (NoPE).
            k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
            k_pe = k_pe.expand(-1, key_states.shape[1], -1, -1)
            key_states = torch.cat([key_states, k_pe], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        topk_indices = None
        if self.indexer is not None:
            if q_resid is None:
                raise ValueError("GLM-5-Next DSA indexer requires q_lora_rank to be set.")
            reused_topk_indices = prev_topk_indices if self.skip_topk else None
            topk_indices, attention_mask = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                mask_dtype=query_states.dtype,
                total_len=key_states.shape[2],
                topk_indices=reused_topk_indices,
            )

        # Flash attention head_dim padding
        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None


# =============================================================================
# RMSNorm, MLP, MoE
# =============================================================================


class Glm5NextRMSNorm(LlamaRMSNorm):
    pass


class Glm5NextMLP(Qwen2MoeMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Optional clamping
        if self.swiglu_limit is not None:
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextExperts(MiniMaxM3VLExperts):
    def __init__(self, config):
        super().__init__(config)
        del self.limit
        del self.swiglu_alpha
        self.intermediate_dim = config.moe_intermediate_size

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        # Optional clamping
        if self.swiglu_limit is not None:
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up


class Glm5NextMoE(DeepseekV3MoE):
    pass


# =============================================================================
# Decoder Layer
# =============================================================================


class Glm5NextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: Glm5NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.layer_types[layer_idx] == "linear_attention":
            # TODO: remove hidden size and rms norm and move to linear directly
            self.self_attn = Glm5NextLinearAttention(
                hidden_size=config.hidden_size,
                config=config,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = Glm5NextAttention(config, layer_idx)

        self.mhc = config.mhc
        self.hc_mult = config.hc_mult
        self.attn_hc = Glm5NextHyperConnection(config) if config.mhc else None
        self.ffn_hc = Glm5NextHyperConnection(config) if config.mhc else None

    def apply_residual(self, post, comb, hidden_states, residual, dtype=None):
        """Either apply normal additive residual stream or MHC residual stream"""
        if post is None and comb is None:
            return hidden_states + residual

        return post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states) if self.mhc else (None, None, hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.apply_residual(post, comb, hidden_states, residual, dtype=dtype)

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states) if self.mhc else (None, None, hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.apply_residual(post, comb, hidden_states, residual, dtype=dtype)

        return hidden_states, hidden_states.mean(dim=2) if hidden_states.ndim == 4 else hidden_states


# =============================================================================
# PreTrainedModel, RotaryEmbedding, Model, CausalLM
# =============================================================================


@auto_docstring
class Glm5NextPreTrainedModel(PreTrainedModel):
    config: Glm5NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm5NextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OutputRecorder(Glm5NextDecoderLayer, index=1),
        "attentions": Glm5NextAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.45\.", r"model\.layers\.\d+\.shared_head\."]
    _keep_in_fp32_modules = []
    _compatible_flash_implementations = ["kernels-community/flash-mla"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        #if isinstance(module, Glm5NextTopkRouter):
        #    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        #    nn.init.zeros_(module.e_score_correction_bias)
        #elif isinstance(module, Glm5NextForgetGate):
        #    nn.init.normal_(module.A_log, mean=0.0, std=0.02)
        #    nn.init.zeros_(module.dt_bias)
        #elif isinstance(module, Glm5NextLinearAttention):
        #    nn.init.ones_(module.o_norm.weight)
        #elif isinstance(module, Glm5NextHyperConnection):
        #    nn.init.normal_(module.fn, mean=0.0, std=0.02)
        #    nn.init.zeros_(module.base)
        #    nn.init.ones_(module.scale)


class Glm5NextRotaryEmbedding(LlamaRotaryEmbedding):
    pass

    @staticmethod
    def compute_default_rope_parameters(
        config: Glm5NextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        head_dim = config.qk_rope_head_dim
        attention_factor = 1.0
        if head_dim == 0:
            return torch.empty(0, device=device), attention_factor

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
        )
        return inv_freq, attention_factor


@auto_docstring
class Glm5NextModel(DeepseekV4Model):
    def __init__(self, config: Glm5NextConfig):
        super().__init__(config)
        self.hc_head = Glm5NextHyperHead() if config.mhc else nn.Identity()

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
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Glm5NextDynamicCache(config=self.config)

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
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, _ = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                #prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Glm5NextForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "Glm5NextPreTrainedModel",
    "Glm5NextModel",
    "Glm5NextForCausalLM",
]
