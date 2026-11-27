# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from ...masking_utils import create_compression_mask, create_sliding_window_causal_mask
from ...modeling_compression_utils import (
    build_selection_mask,
    prepare_compression_groups,
    select_compression_indices,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..glm.modeling_glm import rotate_half
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward
from ..laguna.modeling_laguna import LagunaRotaryEmbedding
from ..llama.modeling_llama import LlamaMLP, LlamaModel
from ..mixtral.modeling_mixtral import MixtralExperts, MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from .configuration_deepseek_v4 import DeepseekV4Config


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Apply interleaved RoPE to the trailing rotary dimensions of x."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_pass, x_rot = x.split([x.shape[-1] - rotary_dim, rotary_dim], dim=-1)
    x_rot = (x_rot.float() * cos + rotate_half(x_rot).float() * sin).to(x.dtype)
    return torch.cat((x_pass, x_rot), dim=-1)


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply the norm weights in fp32.
        return (self.weight * hidden_states).to(input_dtype)


class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4RotaryEmbedding(LagunaRotaryEmbedding):
    """Interleaved RoPE with separate main and compression frequencies."""

    def __init__(self, config: DeepseekV4Config, device=None):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # Ignore flat RoPE parameters left by config conversion.
        self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        return freq.repeat_interleave(2, dim=-1)


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal linear projection over groups of attention heads."""

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


def compress_kv(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    tail: torch.Tensor | None = None,
    overlap_groups: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool projected KV states and return the compressed entries, RoPE positions and new tail."""
    seq_length = kv.shape[1]
    group_size = position_bias.shape[0]
    groups, tail = prepare_compression_groups(
        torch.cat((kv, gate), dim=-1),
        attention_mask,
        group_size=group_size,
        overlap_groups=overlap_groups,
        tail=tail,
    )
    kv, gate = groups.chunk(2, dim=-1)

    # RoPE is applied at the start of each compression group.
    past_groups = attention_mask[:, -(seq_length + 1) : -seq_length] // group_size
    positions = (past_groups + torch.arange(groups.shape[1], device=kv.device)) * group_size

    gate = gate + position_bias.repeat(1 + overlap_groups, 1)
    if overlap_groups:
        # Take Ca from the previous window and Cb from the current window.
        kv_a, kv_b = kv.chunk(2, dim=-1)
        gate_a, gate_b = gate.chunk(2, dim=-1)
        kv = torch.cat((kv_a[:, :, :group_size], kv_b[:, :, group_size:]), dim=2)

        # The first group has no preceding window.
        gate_a = gate_a[:, :, :group_size].masked_fill(positions[..., None, None] == 0, float("-inf"))
        gate = torch.cat((gate_a, gate_b[:, :, group_size:]), dim=2)

    compressed = (kv * gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)).sum(dim=2)
    return compressed, positions, tail


class DeepseekV4IndexerScorer(nn.Module):
    """Score compressed indexer keys against each query.

    The scoring logic computes:
        index_score[b,s,t] = Σ_h (weight[b,s,h] · ReLU(q[b,s,h,:] · k[b,t,:]) * softmax_scale)

    Each head has a learned query-dependent weight, scaled by the inverse square root of the number of heads.
    Scores are reduced over the heads to produce a single score per query and compressed entry.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

    def forward(self, q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
        return (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]


class DeepseekV4Indexer(nn.Module):
    """
    Lightning indexer for Compressed Sparse Attention (CSA).

    The indexer has its own lightweight query and key projections, separate from the main attention compressor.
    Its keys are compressed using the same overlapping windows as CSA, at `index_head_dim`. Queries use their
    token positions for RoPE, while compressed keys use the position of the first token of each current group.

    Example with a compression rate of 4:
        Tokens:             [A, B, C, D, E, F, G, H]
        Compressed entries: [   0   ] [   1   ]
        Query F:             entry 0 is visible; entry 1 is not yet complete.

    The scoring head computes a weighted sum of per-head ReLU dot products. The top-k selection uses compressed
    entry indices rather than physical KV token indices, with -1 marking unavailable selections.

    **Cache strategy**: compressed indexer keys and the remaining overlap tail are stored in the per-layer cache
    under the `"indexer"` stream, independently of the attention compressor.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim, dtype=torch.float32))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps).to(dtype=torch.float32)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.scorer = DeepseekV4IndexerScorer(config)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Cache | None = None,
    ) -> torch.LongTensor:
        batch, seq_length, _ = hidden_states.shape
        cache_layer = cache.layers[self.layer_idx] if cache is not None else None
        tail = cache_layer.compression_tails.get("indexer") if cache_layer is not None else None

        kv, positions, tail = compress_kv(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            self.position_bias,
            attention_mask,
            tail=tail,
            overlap_groups=1,
        )
        kv = self.kv_norm(kv)
        cos, sin = self.rotary_emb(kv, position_ids=positions, layer_type="compress")
        kv = apply_rotary_pos_emb(kv.unsqueeze(1), cos, sin).squeeze(1)

        if cache_layer is not None:
            kv = cache_layer.update_compressed_states(
                kv, attention_mask, seq_length, group_size=self.compress_rate, name="indexer", tail=tail
            )

        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
        q = self.q_b_proj(q_residual).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)

        scores = self.scorer(q, kv, hidden_states)
        return select_compression_indices(
            scores, attention_mask, self.index_topk, group_size=self.compress_rate, return_groups=True
        )


class DeepseekV4Compressor(nn.Module):
    """
    Compress KV states with learned, position-dependent softmax weights. HCA uses non-overlapping
    groups; CSA uses overlapping groups and an indexer to select the most relevant entries.

    Example with a compression rate of 4 (HCA):
        Tokens:         [A, B, C, D, E, F, G, H]
        Groups:         [A B C D] [E F G H]
        Compressed KV:  [   0   ] [   1   ]

    With CSA, each token projects to two series, Ca and Cb. Each group combines Ca from the previous
    window with Cb from the current window:

        Tokens:         [A, B, C, D, E, F, G, H]
        Group 0:        [  -   -   -   -  | A_b B_b C_b D_b]
        Group 1:        [ A_a B_a C_a D_a | E_b F_b G_b H_b]

    The missing preceding window has zero softmax weight. The cache retains the unfinished group
    and, for CSA, the preceding window in the compression tail. Only completed groups are visible.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]
        self.compress_rate = config.compress_rates[layer_type]
        self.overlap_groups = int(layer_type == "compressed_sparse_attention")
        self.head_dim = config.head_dim
        projection_dim = (1 + self.overlap_groups) * self.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, projection_dim, dtype=torch.float32))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps).to(dtype=torch.float32)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config, layer_idx) if self.overlap_groups else None

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Cache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_length = hidden_states.shape[1]
        cache_layer = cache.layers[self.layer_idx] if cache is not None else None
        tail = cache_layer.compression_tails.get("compressor") if cache_layer is not None else None

        kv, positions, tail = compress_kv(
            self.kv_proj(hidden_states),
            self.gate_proj(hidden_states),
            self.position_bias,
            attention_mask,
            tail=tail,
            overlap_groups=self.overlap_groups,
        )
        kv = self.kv_norm(kv)
        cos, sin = self.rotary_emb(kv, position_ids=positions, layer_type="compress")
        kv = apply_rotary_pos_emb(kv.unsqueeze(1), cos, sin).squeeze(1)

        if cache_layer is not None:
            kv = cache_layer.update_compressed_states(
                kv, attention_mask, seq_length, group_size=self.compress_rate, name="compressor", tail=tail
            )

        if self.indexer is not None:
            topk_indices = self.indexer(hidden_states, q_residual, position_ids, attention_mask, cache)
            block_mask = build_selection_mask(topk_indices, kv.shape[1])
        else:
            # HCA attends to every completed group.
            counts = attention_mask[:, -(seq_length + 1) :]
            query_counts = counts[:, 1:].masked_fill(counts.diff(dim=1) == 0, 0)
            candidates = torch.arange(kv.shape[1], device=kv.device)
            block_mask = candidates < query_counts.unsqueeze(-1) // self.compress_rate

        return kv.unsqueeze(1), block_mask


class DeepseekV4Attention(nn.Module):
    """Shared-KV attention with sliding-window and optional compressed KV states, plus attention sinks."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # CSA and HCA share the compressor's RoPE frequencies.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps).to(dtype=torch.float32)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps).to(dtype=torch.float32)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.compressor = (
            DeepseekV4Compressor(config, layer_idx) if self.layer_type != "sliding_attention" else None
        )

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
        cache: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if cache is not None:
            kv = cache.update(kv, kv, self.layer_idx, kwargs=kwargs)[0]

        sliding_mask = attention_mask["sliding_attention"]
        if self.compressor is not None:
            compressed_kv, block_mask = self.compressor(
                hidden_states,
                q_residual,
                position_ids,
                attention_mask["compression"],
                cache,
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

            # Extend the sliding mask to cover compressed entries.
            block_bias = torch.zeros_like(block_mask, dtype=sliding_mask.dtype)
            block_bias = block_bias.masked_fill(~block_mask, float("-inf"))
            sliding_mask = torch.cat([sliding_mask, block_bias.unsqueeze(1)], dim=-1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            sliding_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # Undo the query-position rotation on the output (K and V share RoPE).
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


class DeepseekV4HyperConnection(nn.Module):
    """Mix residual streams using pre/post gates and a doubly stochastic residual transform."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        # Pre/post gates and the stream mixing matrix.
        concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(concatenated_weights_size))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return post gates, residual mixing weights and the collapsed input."""
        batch_size, seq_len = hidden_streams.shape[:2]
        hc = self.hc_mult

        # Flatten and normalize the streams before projecting the mixing weights.
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        flattened = F.linear(flattened, self.fn.float())

        # Split the projected weights, biases and scales into pre, post and comb.
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)
        comb_b = comb_b.view(hc, hc)

        # Pre and post are gates; comb mixes the residual streams.
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps

        # Sinkhorn-Knopp alternates row and column normalization to constrain the
        # residual mixing matrix to be approximately doubly stochastic.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        # Collapse the input streams for the attention or MLP block.
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by `DeepseekV4Model` before the shared RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MLP(LlamaMLP):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(self.act_fn(gate) * up)


class DeepseekV4Experts(MixtralExperts):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            current_hidden_states = self._apply_gate(gate_up)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DeepseekV4TopKRouter(MixtralTopKRouter):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(MixtralTopKRouter):
    """Route tokens using a fixed token-to-expert table and learned expert weights."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tid2eid = nn.Buffer(torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape

        if self.is_hash:
            _, topk_weights, topk_indices = self.gate(hidden_states, input_ids)
        else:
            _, topk_weights, topk_indices = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    """Decoder layer with manifold-constrained hyperconnections around attention and MoE."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=torch.float32)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
            dtype=torch.float32
        )
        self.attn_hc = DeepseekV4HyperConnection(config).to(dtype=torch.float32)
        self.ffn_hc = DeepseekV4HyperConnection(config).to(dtype=torch.float32)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        # Self Attention
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Mix residual streams along the input HC axis (comb.T @ residual).
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        # Feed Forward
        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids=input_ids)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
        return hidden_states


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config: DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks",
        "position_bias",
        "e_score_correction_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]
    # These compressor projections are stored in BF16 in FP8 checkpoints
    _keep_in_fp32_modules = [
        "self_attn.compressor.kv_proj",
        "self_attn.compressor.gate_proj",
        "self_attn.compressor.indexer.kv_proj",
        "self_attn.compressor.indexer.gate_proj",
        "self_attn.compressor.indexer.scorer.weights_proj",
    ]
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                init.zeros_(module.e_score_correction_bias)  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4Compressor, DeepseekV4Indexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV4Model(LlamaModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.hc_head = DeepseekV4HyperHead(config).to(dtype=torch.float32)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=torch.float32)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            mask_functions = {"sliding_attention": create_sliding_window_causal_mask}
            if set(self.config.layer_types) - {"sliding_attention"}:
                mask_functions["compression"] = create_compression_mask

            causal_mask_mapping = {
                mask_type: mask_function(**mask_kwargs) for mask_type, mask_function in mask_functions.items()
            }

        if position_ids is None:
            if "compression" in causal_mask_mapping:
                # Match compressed-key RoPE positions.
                position_ids = (causal_mask_mapping["compression"][:, -inputs_embeds.shape[1] :] - 1).clamp_min(0)
            else:
                past_seen = cache.get_seq_length() if cache is not None else 0
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
                position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask_mapping,
                input_ids=input_ids,
                cache=cache,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    pass


__all__ = [
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
]
