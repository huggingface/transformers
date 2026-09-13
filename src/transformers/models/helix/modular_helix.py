# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""PyTorch HELIX model.

HELIX braids three sequence mixers in every block. Read `HelixBraid` for the fusion, and the three
`Helix*Strand`-ish sections below for each pathway:

* strand L — multi-scale sliding-window softmax attention on a block staircase,
* strand R — a gated delta-rule recurrence with a matrix state (`HelixRecurrentStrand`),
* strand I — a hierarchical landmark index over the whole past (`HelixLandmarkPooler` + the descent
  in `HelixBraid._select_memory_blocks`).
"""

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import (
    Cache,
    DynamicCache,
    LinearAttentionAndFullAttentionLayer,
    LinearAttentionAndSlidingWindowAttentionLayer,
)
from ...integrations import use_kernelized_func
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from .configuration_helix import HelixConfig


logger = logging.get_logger(__name__)

# Finite stand-in for -inf used when ranking landmark nodes. Kept well inside fp32 range so that adding
# learned biases on top can never produce a NaN.
NEG_SCORE = -1e30


def _duplicate_mask(index: torch.Tensor) -> torch.Tensor:
    """
    Mark every repeat of a value along the last axis except its first occurrence.

    The descent can propose the same node twice — a beam that collapsed onto one node, or a frontier node
    that is also a child of the beam. Without this, `topk` happily fills the beam with copies of one node
    and silently narrows the search.
    """
    order = index.argsort(dim=-1, stable=True)
    ordered = index.gather(-1, order)
    repeated = torch.zeros_like(ordered, dtype=torch.bool)
    repeated[..., 1:] = ordered[..., 1:] == ordered[..., :-1]
    return torch.zeros_like(repeated).scatter_(-1, order, repeated)


class HelixRMSNorm(LlamaRMSNorm):
    pass


class HelixRMSNormGated(Qwen3NextRMSNormGated):
    pass


class HelixRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HelixMLP(LlamaMLP):
    pass


class HelixIndexCacheLayer(LinearAttentionAndFullAttentionLayer):
    """
    Cache for a `"helix"` layer, i.e. one that runs all three strands.

    On top of the linear-attention states (strand R) and the full key/value history (read by both the
    local window and the index), it holds the *episodic memory* of strand I:

    * `leaf_landmarks` — one pooled summary per **complete** memory block, appended as blocks close;
    * `landmark_levels` — the landmark tree, leaves first. Rebuilt only when a block closes, so its cost
      amortizes to `O(num_blocks / block_size)` per generated token;
    * `route_hidden` — the hidden state at the last block boundary, which is the (strictly causal) query
      used to route the *next* block. This is a single `hidden_size` vector, i.e. O(1) state.

    Only `route_hidden` and the strand-R state are *hot*: the key/value history and the landmark tree are
    append-only and can be paged out, and a decode step touches only `O(index_topk * block_size)` of them.
    """

    _layer_type = "helix"

    def __init__(self, config: HelixConfig, number_of_states: int = 2, **kwargs):
        super().__init__(number_of_states=number_of_states)
        self.block_size = config.block_size
        self.leaf_landmarks: torch.Tensor | None = None
        self.landmark_levels: list[torch.Tensor] = []
        self.route_hidden: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.leaf_landmarks = None
        self.landmark_levels = []
        self.route_hidden = None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx)
        if self.leaf_landmarks is not None:
            beam_idx = beam_idx.to(self.leaf_landmarks.device)
            self.leaf_landmarks = self.leaf_landmarks.index_select(0, beam_idx)
            self.landmark_levels = [level.index_select(0, beam_idx) for level in self.landmark_levels]
        if self.route_hidden is not None:
            self.route_hidden = self.route_hidden.index_select(0, beam_idx.to(self.route_hidden.device))


class HelixLocalCacheLayer(LinearAttentionAndSlidingWindowAttentionLayer):
    """
    Cache for a `"helix_local"` layer (strands L + R only). Because nothing reads the distant past on
    these layers, the key/value history is truncated to the widest local window: their KV cache is O(1).
    """

    _layer_type = "helix_local"

    def __init__(self, config: HelixConfig, number_of_states: int = 2, **kwargs):
        super().__init__(sliding_window=config.sliding_window, number_of_states=number_of_states)


@use_kernelized_func([torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn])
class HelixRecurrentStrand(nn.Module):
    """
    Strand R: a gated delta-rule recurrence with a matrix-valued state `S ∈ R^{d_k × d_v}` per head.

    `S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`

    The delta rule is what makes this more than a decaying sum: writing `v_t` *removes* whatever the state
    already associates with `k_t` before adding the new value, so re-binding a key does not pile up
    interference. Training uses the chunkwise-parallel (UT transform) form, so the whole sequence is one
    batch of matmuls; decoding uses the one-token recurrence and a state whose size does not depend on how
    much text came before.

    When `config.use_surprise_gating` is set, the write strength `β_t` is additionally driven by how novel
    `k_t` is with respect to a short causal pool of the keys just before it. Tokens that repeat what the
    state has just seen write weakly; genuinely new bindings write hard. The novelty signal is a
    depthwise convolution, so it stays parallel over the sequence.
    """

    def __init__(self, config: HelixConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_recurrent_heads
        self.head_k_dim = config.recurrent_head_dim
        self.head_v_dim = config.recurrent_value_head_dim
        self.chunk_size = config.recurrent_chunk_size
        self.conv_kernel_size = config.conv_kernel_size
        self.use_surprise_gating = config.use_surprise_gating
        self.surprise_kernel_size = config.surprise_kernel_size

        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkvz = nn.Linear(config.hidden_size, self.conv_dim + self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(config.hidden_size, 2 * self.num_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.surprise_gamma = nn.Parameter(torch.zeros(self.num_heads))
        self.surprise_weight = nn.Buffer(self.build_surprise_weight(), persistent=False)

        self.norm = HelixRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def build_surprise_weight(self, device: torch.device | None = None) -> torch.Tensor:
        """
        Fixed (non-learned) causal mean over the `surprise_kernel_size - 1` *previous* positions. The last
        tap is zero so the pool never sees the token it is judging.
        """
        weight = torch.zeros(self.key_dim, self.surprise_kernel_size, device=device)
        weight[:, :-1] = 1.0 / max(1, self.surprise_kernel_size - 1)
        return weight

    def _novelty(self, key: torch.Tensor, past_key_values: Cache | None, seq_len: int) -> torch.Tensor:
        """Per-head `1 - cos(k_t, mean(k_{t-W..t-1}))`, in `[0, 2]`, computed in parallel over the sequence."""
        key_stream = key.transpose(1, 2)  # (batch, key_dim, seq_len)
        if past_key_values is not None:
            key_stream = past_key_values.update_conv_state(
                key_stream, layer_idx=self.layer_idx, state_idx=1, conv_kernel_size=self.surprise_kernel_size
            )
        pooled = causal_conv1d_fn(key_stream, self.surprise_weight, None)[:, :, -seq_len:]
        pooled = pooled.transpose(1, 2).view(-1, seq_len, self.num_heads, self.head_k_dim)
        heads = key.view(-1, seq_len, self.num_heads, self.head_k_dim)
        return 1.0 - F.cosine_similarity(heads, pooled, dim=-1, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if padding_mask is not None:
            padding_mask = padding_mask[:, -seq_len:]
        hidden_states = apply_mask_to_padding_states(hidden_states, padding_mask)

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        mixed, z = torch.split(projected_states_qkvz, [self.conv_dim, self.value_dim], dim=-1)

        # Short depthwise causal convolution: the locality prior that lets the recurrence spend its state on
        # long-range bindings instead of re-deriving n-gram structure.
        conv_input = mixed.transpose(1, 2)
        if past_key_values is not None:
            conv_input = past_key_values.update_conv_state(
                conv_input, layer_idx=self.layer_idx, state_idx=0, conv_kernel_size=self.conv_kernel_size
            )
        mixed = causal_conv1d_fn(conv_input, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation="silu")
        mixed = mixed[:, :, -seq_len:].transpose(1, 2)

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        beta_logits, a = torch.split(projected_states_ba, [self.num_heads, self.num_heads], dim=-1)
        if self.use_surprise_gating:
            beta_logits = beta_logits + self.surprise_gamma * self._novelty(key, past_key_values, seq_len)
        beta = beta_logits.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        recurrent_state = None
        if past_key_values is not None:
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states[0]

        if seq_len == 1 and recurrent_state is not None:
            core_out, recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_out, recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        if past_key_values is not None:
            past_key_values.update_recurrent_state(recurrent_state, layer_idx=self.layer_idx, state_idx=0)

        core_out = self.norm(core_out, z.view(batch_size, seq_len, self.num_heads, self.head_v_dim))
        return self.out_proj(core_out.reshape(batch_size, seq_len, self.value_dim))


class HelixLandmarkPooler(nn.Module):
    """
    Turns a group of vectors into one landmark. Combines an unweighted mean (what is *typically* here) with
    a learned attention pool (what is *salient* here), because a mean alone erases the rare token that a
    later query will be looking for — exactly the failure mode that sinks fixed-state models on recall.
    """

    def __init__(self, input_dim: int, landmark_dim: int, eps: float):
        super().__init__()
        self.query = nn.Parameter(torch.empty(input_dim))
        self.proj = nn.Linear(2 * input_dim, landmark_dim, bias=False)
        self.norm = HelixRMSNorm(landmark_dim, eps=eps)

    def forward(self, states: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """`states` is `(..., group, dim)`; `valid` is a broadcastable `(..., group)` boolean mask."""
        scores = (states * self.query).sum(-1)
        if valid is not None:
            scores = scores.masked_fill(~valid, NEG_SCORE)
            counts = valid.sum(-1, keepdim=True).clamp(min=1)
            mean = (states * valid.unsqueeze(-1)).sum(-2) / counts
        else:
            mean = states.mean(-2)
        salient = (scores.softmax(-1).unsqueeze(-1) * states).sum(-2)
        return self.norm(self.proj(torch.cat([mean, salient], dim=-1)))


class HelixBraid(nn.Module):
    """
    The three-strand mixer.

    Strands L and I share one set of q/k/v projections and differ only in *which* keys they read:

    * **L** reads a contiguous staircase of the last `local_blocks + 1` memory blocks, with a per-head
      window so that different heads see geometrically different spans (the multi-scale prior).
    * **I** reads `index_topk` memory blocks chosen anywhere in the past by a beam descent over a
      landmark tree. Selection is per query *block* and is driven by the hidden state at the end of the
      *previous* block, which keeps it strictly causal while letting `block_size` queries share one gather.

    Their outputs are combined by a per-token, per-head softmax gate; strand R is added on top through its
    own output projection.
    """

    def __init__(self, config: HelixConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.has_index = config.layer_types[layer_idx] == "helix"

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.block_size = config.block_size
        self.local_blocks = config.local_blocks

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = HelixRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HelixRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.local_window_sizes = nn.Buffer(torch.tensor(config.local_window_sizes), persistent=False)

        if self.has_index:
            self.branching = config.index_branching
            self.beam_width = config.index_beam_width
            self.index_topk = config.index_topk
            self.landmark_dim = config.landmark_dim
            self.num_distance_buckets = config.index_num_distance_buckets
            self.leaf_pooler = HelixLandmarkPooler(self.head_dim, self.landmark_dim, config.rms_norm_eps)
            # One pooler shared by every internal level: a memory of memories is summarized the same way at
            # every scale, which both saves parameters and biases the tree towards scale invariance.
            self.node_pooler = HelixLandmarkPooler(self.landmark_dim, self.landmark_dim, config.rms_norm_eps)
            self.route_q_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.landmark_dim, bias=False)
            self.level_bias = nn.Parameter(torch.zeros(self.num_key_value_heads, config.index_max_levels + 1))
            # Bucketed relative *block* distance instead of RoPE: retrieved blocks sit at distances never seen
            # in training, and a saturating bucket table extrapolates where a rotary phase does not.
            self.distance_bias = nn.Parameter(torch.zeros(self.num_key_value_heads, self.num_distance_buckets))
            self.strand_gate = nn.Linear(config.hidden_size, self.num_heads * 2, bias=False)

    def _distance_bucket(self, distance: torch.Tensor) -> torch.Tensor:
        """Logarithmic bucketing of a non-negative block distance, saturating at the last bucket."""
        distance = distance.clamp(min=0)
        bucket = torch.log2(distance.float() + 1.0).floor().long()
        return bucket.clamp(max=self.num_distance_buckets - 1)

    def _build_landmark_tree(self, leaves: torch.Tensor) -> list[torch.Tensor]:
        """Leaves first. Level `l + 1` has `n_leaves // branching**(l+1)` nodes, all of them complete."""
        levels = [leaves]
        num_leaves = leaves.shape[2]
        for level in range(1, self.config.index_max_levels + 1):
            num_nodes = num_leaves // self.branching**level
            if num_nodes < 1:
                break
            children = levels[-1][:, :, : num_nodes * self.branching]
            children = children.reshape(*children.shape[:2], num_nodes, self.branching, self.landmark_dim)
            levels.append(self.node_pooler(children))
        return levels

    def _score_nodes(
        self,
        levels: list[torch.Tensor],
        level: int,
        node_index: torch.Tensor,
        route_query: torch.Tensor,
        query_block: torch.Tensor,
        parent_path: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score `node_index` (`(batch, kv_heads, num_query_blocks, candidates)`) against `route_query`.

        Returns `(path, rank)`. `path` accumulates the scores of the eligible nodes along the descent, so
        the leaf bias that eventually reaches the attention logits carries gradient for *every* level of
        the tree, not just the leaves. `rank` is `path` with ineligible and duplicated candidates knocked
        out; it is what the beam is selected on.

        A node is eligible for query block `J` only if the *whole* span it summarizes ends at or before `J`.
        That is what keeps the index causal: a summary is never consulted by a query it partly describes.
        Eligibility is monotone downwards (an eligible node has only eligible descendants), so a path score
        is exactly the sum over the eligible suffix of the path.
        """
        landmarks = levels[level]
        num_nodes = landmarks.shape[2]
        span = self.branching**level
        eligible = (node_index < num_nodes) & ((node_index + 1) * span <= query_block[:, None])

        safe_index = node_index.clamp(0, max(num_nodes - 1, 0))
        batch, kv_heads, num_query_blocks, candidates = safe_index.shape
        gathered = landmarks.gather(
            2,
            safe_index.reshape(batch, kv_heads, num_query_blocks * candidates, 1).expand(
                -1, -1, -1, self.landmark_dim
            ),
        ).view(batch, kv_heads, num_query_blocks, candidates, self.landmark_dim)

        score = (route_query.unsqueeze(3) * gathered).sum(-1) * (self.landmark_dim**-0.5)
        distance = query_block[:, None] - (safe_index + 1) * span
        head_index = torch.arange(kv_heads, device=score.device).view(1, kv_heads, 1, 1)
        score = score + self.distance_bias[head_index, self._distance_bucket(distance)]
        score = score + self.level_bias[:, level].view(1, kv_heads, 1, 1)

        path = parent_path + score
        rank = path.masked_fill(~eligible | _duplicate_mask(node_index), NEG_SCORE)
        return torch.where(eligible, path, torch.zeros_like(path)), rank

    def _select_memory_blocks(
        self, levels: list[torch.Tensor], route_query: torch.Tensor, query_block: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Beam descent down the landmark tree. Returns `(block_index, path_score)`, both
        `(batch, kv_heads, num_query_blocks, index_topk)`.

        At every level the candidate set is the children of the current beam **plus** the level's
        *frontier* — the eligible nodes whose parent is not itself eligible. There are at most
        `branching - 1` of those per level, and together the frontiers across all levels tile `[0, J)`
        exactly, so no reachable block is ever cut off by a beam that happened to descend elsewhere.
        Cost per query block is `O(beam_width * branching * log_branching(N))`, which is where HELIX's
        `O(N log N)` comes from.
        """
        batch, kv_heads, num_query_blocks = route_query.shape[:3]
        device = route_query.device
        top_level = len(levels) - 1
        num_slots = self.beam_width * self.branching + self.branching - 1

        def broadcast(index: torch.Tensor) -> torch.Tensor:
            return index.reshape(1, 1, *index.shape[-2:]).expand(batch, kv_heads, -1, -1)

        candidates = broadcast(torch.arange(num_slots, device=device).expand(num_query_blocks, num_slots))
        parent_path = route_query.new_zeros(candidates.shape)
        beam, beam_path = None, None
        for level in range(top_level, -1, -1):
            if beam is not None:
                children = beam.unsqueeze(-1) * self.branching + torch.arange(self.branching, device=device)
                frontier_start = (query_block // self.branching ** (level + 1)) * self.branching
                frontier = frontier_start.view(num_query_blocks, 1) + torch.arange(self.branching - 1, device=device)
                candidates = torch.cat([children.flatten(-2), broadcast(frontier)], dim=-1)
                # A frontier node has no eligible ancestor, so it starts a fresh path.
                parent_path = torch.cat(
                    [
                        beam_path.unsqueeze(-1).expand(*beam.shape, self.branching).flatten(-2),
                        beam_path.new_zeros(*beam.shape[:-1], self.branching - 1),
                    ],
                    dim=-1,
                )
            path, rank = self._score_nodes(levels, level, candidates, route_query, query_block, parent_path)
            width = self.index_topk if level == 0 else self.beam_width
            top = rank.topk(min(width, rank.shape[-1]), dim=-1)
            beam, beam_path = candidates.gather(-1, top.indices), path.gather(-1, top.indices)
        return beam, top.values

    def _local_attention(
        self,
        query_blocks: torch.Tensor,
        key_blocks: torch.Tensor,
        value_blocks: torch.Tensor,
        token_valid: torch.Tensor,
        query_block: torch.Tensor,
        kv_block_offset: int,
    ) -> torch.Tensor:
        """Strand L: per-head multi-scale window attention over the last `local_blocks + 1` memory blocks."""
        batch, _, num_query_blocks, block_size, head_dim = query_blocks.shape
        num_blocks = key_blocks.shape[2]
        span = self.local_blocks + 1

        offsets = torch.arange(-self.local_blocks, 1, device=query_blocks.device)
        gather_index = query_block[:, None] - kv_block_offset + offsets[None, :]
        in_range = (gather_index >= 0) & (gather_index < num_blocks)
        gather_index = gather_index.clamp(0, max(num_blocks - 1, 0))

        keys = key_blocks[:, :, gather_index].flatten(3, 4)
        values = value_blocks[:, :, gather_index].flatten(3, 4)
        in_range = in_range.view(1, 1, num_query_blocks, span, 1).expand(-1, -1, -1, -1, block_size).flatten(3, 4)
        valid = token_valid[:, :, gather_index].flatten(3, 4) & in_range

        # Relative distance inside the gathered window is the same for every query block, so the per-head
        # window mask is built once and broadcast.
        within = torch.arange(block_size, device=query_blocks.device)
        positions = torch.arange(span * block_size, device=query_blocks.device)
        distance = self.local_blocks * block_size + within[:, None] - positions[None, :]
        window = self.local_window_sizes.to(query_blocks.device).view(-1, 1, 1)
        head_mask = (distance >= 0) & (distance < window)
        return self._blocked_attention(query_blocks, keys, values, valid, head_mask.unsqueeze(0), None)

    def _blocked_attention(
        self,
        query_blocks: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        head_mask: torch.Tensor | None,
        logit_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Attention of `(batch, heads, num_query_blocks, block_size, head_dim)` queries against per-query-block
        gathered keys/values of `(batch, kv_heads, num_query_blocks, num_keys, head_dim)`.

        The query-block axis is folded into the batch, so peak activation memory is
        `O(N * num_keys)` rather than `O(N^2)`.
        """
        batch, num_heads, num_query_blocks, block_size, head_dim = query_blocks.shape
        num_keys = keys.shape[3]
        folded = batch * num_query_blocks
        dtype = query_blocks.dtype
        min_value = torch.finfo(dtype).min

        queries = query_blocks.permute(0, 2, 1, 3, 4).reshape(folded, num_heads, block_size, head_dim)
        keys = repeat_kv(
            keys.permute(0, 2, 1, 3, 4).reshape(folded, -1, num_keys, head_dim), self.num_key_value_groups
        )
        values = repeat_kv(
            values.permute(0, 2, 1, 3, 4).reshape(folded, -1, num_keys, head_dim), self.num_key_value_groups
        )

        # `valid` is per key/value group; it is constant along the query axis of the folded batch.
        mask = valid.permute(0, 2, 1, 3).reshape(folded, -1, 1, num_keys)
        mask = mask.repeat_interleave(self.num_key_value_groups, dim=1)
        if head_mask is not None:
            mask = mask & head_mask
        # Rows with nothing to attend to would make softmax produce NaNs, so they are left uniform here and
        # zeroed by the caller instead. Strand L carries no additive bias, so it hands SDPA the boolean mask
        # directly rather than materializing a float one -- that tensor is `O(N * num_keys)` and is the
        # largest single allocation in the block.
        if logit_bias is None:
            attn_mask = mask
        else:
            bias = logit_bias.permute(0, 2, 1, 3).reshape(folded, -1, 1, num_keys)
            bias = bias.repeat_interleave(self.num_key_value_groups, dim=1).to(dtype)
            attn_mask = torch.where(mask, bias, torch.full_like(bias, min_value).expand_as(mask))
        attn = F.scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask, scale=self.scaling)
        return attn.view(batch, num_query_blocks, num_heads, block_size, head_dim).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        block_size = self.block_size
        device = hidden_states.device
        # The memory-block grid is anchored to absolute positions, so the braid needs to know how many
        # tokens came before this chunk. That is exactly what the cache has already counted.
        past_len = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        total_len = past_len + seq_len

        query = self.q_norm(self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        kv_len = key.shape[2]
        kv_start = total_len - kv_len

        # --- align queries and keys to the memory-block grid -------------------------------------------
        query_lo = (past_len // block_size) * block_size
        left_pad = past_len - query_lo
        right_pad = -(left_pad + seq_len) % block_size
        num_query_blocks = (left_pad + seq_len + right_pad) // block_size
        query_blocks = F.pad(query, (0, 0, left_pad, right_pad)).view(
            batch_size, -1, num_query_blocks, block_size, self.head_dim
        )
        query_block = torch.arange(num_query_blocks, device=device) + query_lo // block_size

        kv_lo = (kv_start // block_size) * block_size
        kv_left = kv_start - kv_lo
        kv_right = -(kv_left + kv_len) % block_size
        num_blocks = (kv_left + kv_len + kv_right) // block_size
        kv_block_offset = kv_lo // block_size
        key_blocks = F.pad(key, (0, 0, kv_left, kv_right)).view(batch_size, -1, num_blocks, block_size, self.head_dim)
        value_blocks = F.pad(value, (0, 0, kv_left, kv_right)).view(
            batch_size, -1, num_blocks, block_size, self.head_dim
        )

        token_valid = torch.zeros(batch_size, num_blocks * block_size, dtype=torch.bool, device=device)
        token_valid[:, kv_left : kv_left + kv_len] = True
        if padding_mask is not None:
            token_valid[:, kv_left : kv_left + kv_len] &= padding_mask[:, -kv_len:].bool()
        token_valid = token_valid.view(batch_size, 1, num_blocks, block_size).expand(-1, key_blocks.shape[1], -1, -1)

        local_out = self._local_attention(
            query_blocks, key_blocks, value_blocks, token_valid, query_block, kv_block_offset
        )

        if self.has_index:
            index_out, index_any = self._index_attention(
                hidden_states,
                query_blocks,
                key_blocks,
                value_blocks,
                token_valid,
                query_block,
                past_key_values,
                past_len,
                total_len,
            )
        else:
            index_out, index_any = None, None

        def unpad(blocked: torch.Tensor) -> torch.Tensor:
            flat = blocked.reshape(batch_size, blocked.shape[1], num_query_blocks * block_size, blocked.shape[-1])
            return flat[:, :, left_pad : left_pad + seq_len].transpose(1, 2)

        attn_out = unpad(local_out)
        if index_out is not None:
            gate = self.strand_gate(hidden_states).view(batch_size, seq_len, self.num_heads, 2).softmax(-1)
            index_out = unpad(index_out) * unpad(index_any).to(attn_out.dtype)
            attn_out = gate[..., :1] * attn_out + gate[..., 1:] * index_out

        return self.o_proj(attn_out.reshape(batch_size, seq_len, -1))

    def _index_attention(
        self,
        hidden_states: torch.Tensor,
        query_blocks: torch.Tensor,
        key_blocks: torch.Tensor,
        value_blocks: torch.Tensor,
        token_valid: torch.Tensor,
        query_block: torch.Tensor,
        past_key_values: Cache | None,
        past_len: int,
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strand I: landmark descent, then exact attention over the selected memory blocks."""
        batch_size, seq_len, _ = hidden_states.shape
        block_size = self.block_size
        num_query_blocks = query_block.shape[0]
        kv_heads = key_blocks.shape[1]
        num_leaves = total_len // block_size
        layer_cache = past_key_values.layers[self.layer_idx] if past_key_values is not None else None

        if num_leaves == 0:
            # Not one memory block has closed yet, so nothing is eligible and strand L covers everything.
            zeros = query_blocks.new_zeros(query_blocks.shape)
            return zeros, zeros.new_zeros((*zeros.shape[:-1], 1), dtype=torch.bool)

        cached_leaves = (
            0 if layer_cache is None or layer_cache.leaf_landmarks is None else (layer_cache.leaf_landmarks.shape[2])
        )
        if layer_cache is not None and num_leaves == cached_leaves and layer_cache.landmark_levels:
            levels = layer_cache.landmark_levels
        else:
            new_leaves = self.leaf_pooler(
                key_blocks[:, :, cached_leaves:num_leaves], token_valid[:, :, cached_leaves:num_leaves]
            )
            leaves = new_leaves if cached_leaves == 0 else torch.cat([layer_cache.leaf_landmarks, new_leaves], dim=2)
            levels = self._build_landmark_tree(leaves)
            if layer_cache is not None:
                layer_cache.leaf_landmarks = leaves
                layer_cache.landmark_levels = levels

        # Routing query for block J is the hidden state at the end of block J - 1: strictly in the past for
        # every token of block J, so one gather serves the whole block without leaking anything.
        route_source = hidden_states.new_zeros(batch_size, num_query_blocks, hidden_states.shape[-1])
        route_positions = query_block * block_size - 1
        in_chunk = route_positions - past_len
        available = in_chunk >= 0
        if available.any():
            route_source[:, available] = hidden_states[:, in_chunk[available]]
        if (~available).any() and layer_cache is not None and layer_cache.route_hidden is not None:
            route_source[:, ~available] = layer_cache.route_hidden.unsqueeze(1)

        route_query = (
            self.route_q_proj(route_source)
            .view(batch_size, num_query_blocks, kv_heads, self.landmark_dim)
            .transpose(1, 2)
        )
        selected, scores = self._select_memory_blocks(levels, route_query, query_block)

        # `topk` still returns candidates even when every one of them was masked out, so clamp before
        # gathering; `usable` below is what actually decides whether a selection counts.
        usable = scores > NEG_SCORE / 2
        flat = selected.clamp(0, key_blocks.shape[2] - 1).reshape(batch_size, kv_heads, -1)
        selected_keys = key_blocks.gather(2, flat[..., None, None].expand(-1, -1, -1, block_size, self.head_dim)).view(
            batch_size, kv_heads, num_query_blocks, -1, self.head_dim
        )
        selected_values = value_blocks.gather(
            2, flat[..., None, None].expand(-1, -1, -1, block_size, self.head_dim)
        ).view(batch_size, kv_heads, num_query_blocks, -1, self.head_dim)
        selected_valid = token_valid.gather(2, flat[..., None].expand(-1, -1, -1, block_size)).view(
            batch_size, kv_heads, num_query_blocks, -1
        )

        selected_valid = selected_valid & usable.repeat_interleave(block_size, dim=-1)
        # Feeding the routing score back in as an additive logit is what makes the discrete top-k
        # differentiable: gradients reach the landmark poolers through the blocks that were chosen.
        logit_bias = F.logsigmoid(scores.float()).to(query_blocks.dtype).repeat_interleave(block_size, dim=-1)

        index_out = self._blocked_attention(
            query_blocks, selected_keys, selected_values, selected_valid, None, logit_bias
        )
        index_any = selected_valid.any(-1).view(batch_size, kv_heads, num_query_blocks, 1, 1)
        index_any = index_any.repeat_interleave(self.num_key_value_groups, dim=1).expand(-1, -1, -1, block_size, 1)

        if layer_cache is not None:
            boundary = (total_len // block_size) * block_size - 1
            if boundary >= past_len:
                layer_cache.route_hidden = hidden_states[:, boundary - past_len]
        return index_out, index_any


class HelixDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: HelixConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = HelixBraid(config, layer_idx)
        self.recurrent = HelixRecurrentStrand(config, layer_idx)
        self.post_attention_layernorm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HelixMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # The two halves of the braid read the same normalized input and are summed back into the residual
        # stream, so the block stays a single parallel branch rather than two stacked sub-layers.
        mixed = self.mixer(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            padding_mask=padding_mask,
            **kwargs,
        )
        mixed = mixed + self.recurrent(
            hidden_states, past_key_values=past_key_values, padding_mask=padding_mask, **kwargs
        )
        hidden_states = residual + mixed

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@auto_docstring
class HelixPreTrainedModel(LlamaPreTrainedModel):
    config: HelixConfig
    _no_split_modules = ["HelixDecoderLayer"]
    _can_record_outputs = {"hidden_states": HelixDecoderLayer}
    _is_stateful = True
    _can_compile_fullgraph = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, HelixRecurrentStrand):
            init.copy_(
                module.A_log,
                torch.empty(module.num_heads, device=module.A_log.device).uniform_(0.01, 16).log_(),
            )
            init.ones_(module.dt_bias)
            init.zeros_(module.surprise_gamma)
            init.copy_(module.surprise_weight, module.build_surprise_weight(module.surprise_weight.device))
        elif isinstance(module, HelixLandmarkPooler):
            init.normal_(module.query, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, HelixBraid):
            init.copy_(
                module.local_window_sizes,
                torch.tensor(self.config.local_window_sizes, device=module.local_window_sizes.device),
            )
            if module.has_index:
                init.zeros_(module.level_bias)
                init.zeros_(module.distance_bias)


@auto_docstring
class HelixModel(LlamaModel):
    def __init__(self, config: HelixConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HelixDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HelixRotaryEmbedding(config=config)
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
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        # HELIX builds its own block-structured masks (a staircase for strand L, a gathered mask for strand
        # I), so it consumes the raw 2D padding mask instead of a materialized 4D causal mask.
        padding_mask = attention_mask
        if padding_mask is not None and padding_mask.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects a 2D padding mask (batch, sequence), got shape "
                f"{tuple(padding_mask.shape)}. Causality is enforced by the block geometry, not by the mask."
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                padding_mask=padding_mask,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HelixForCausalLM(LlamaForCausalLM):
    pass


__all__ = ["HelixForCausalLM", "HelixModel", "HelixPreTrainedModel"]
