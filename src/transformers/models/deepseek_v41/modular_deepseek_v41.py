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
"""DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, StaticSlidingWindowLayer
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention, DeepseekV4TopKRouter
from ..deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer
from ..glm5_next.modeling_glm5_next import Glm5NextTextExperts
from ..laguna.modeling_laguna import LagunaRotaryEmbedding, apply_rotary_pos_emb


logger = logging.get_logger(__name__)


def get_pooled_indices(
    valid_keys: torch.Tensor, pool_size: int, partial_pools_are_valid: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a 2D boolean tensor of valid keys (shape [batch_size, full_cache_len]), returns the grouped indices for
    each pool. This strips out the padding for each sequences and adds dummy groups to make sure all sequences have the
    same number of pools. Alsio returns a boolean tensor indicating which pool are valid, ie. with real indices in them.
    Whether partially filled pools are valid is determined by `partial_pools_are_valid`.

    Pooling starts at the first real token, not raw slot 0. This is the part that makes:
        [P, P, A, B, C, D, ...]
    behave like:
        [A, B, C, D, ...]
    for k-pool grouping. For instance, for pool_size = 4, with ✗ as the padding token:

        input                       pool_indices                          valid_pools
        [✗, ✗, A, B, C, D, E, F] -> [ 2,  3,  4,  5], [ 6,  7, -1, -1]    [True, partial_pools_are_valid]
        [A, B, C, D, E, F, G, H] -> [ 0,  1,  2,  3], [ 4,  5,  6,  7]    [True, True]
        [✗, ✗, ✗, ✗, ✗, A, B, C] -> [ 5,  6,  7, -1], [-1, -1, -1, -1]    [True, False]
    """
    batch_size, seq_len = valid_keys.shape
    number_of_pools = (seq_len + pool_size - 1) // pool_size
    device = valid_keys.device

    # Determine the first valid key, accouting for the fact that some sequences may have none (eg. static cache)
    first_valid_index = valid_keys.long().argmax(-1)
    first_valid_index = torch.where(condition=valid_keys.any(-1), input=first_valid_index, other=seq_len)
    # The pool indices are the first valid index + the indices accross all pools
    pool_offsets = torch.arange(number_of_pools * pool_size, device=device)
    pool_offsets = pool_offsets.view(1, number_of_pools, pool_size)
    pool_indices = first_valid_index[:, None, None] + pool_offsets  # [batch_size, num_pools, pool_size]

    # For all indices in the pools, determine if they are valid
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]
    clamped_pool_indices = pool_indices.clamp(0, seq_len - 1)  # avoid index errors
    valid_pool_indices = valid_keys[batch_idx, clamped_pool_indices]
    # ... and within range
    valid_pool_indices = valid_pool_indices & (pool_indices < seq_len)
    # Use this to mask the invalid indices
    pool_indices = pool_indices.masked_fill(~valid_pool_indices, -1)

    # Also a boolean mask indicating which pools are valid. Partial pools validity depend on `partial_pools_are_valid`
    valid_pools = valid_pool_indices.any(-1) if partial_pools_are_valid else valid_pool_indices.all(-1)
    return pool_indices, valid_pools

@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41Config(DeepseekV4Config):
    pass


class DeepseekV41CSACache:  # TODO: inherit from DynamicSlidingWindowLayer?
    def __init__(self, config: DeepseekV41Config):
        self.compression_ratio = config.compression_ratio
        self.compressor_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.indexer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def has_previous_state(self, layer_idx: int) -> bool:
        return layer_idx in self.compressor_cache

    def init_compressor_cache(self, layer_idx: int, tail: torch.Tensor, num_in_buffer: torch.Tensor) -> None:
        """Initializes the compression buffer with the tail of the prefill."""
        # Because the tail is going to init the buffer, it needs to have "compression_ratio" tokens
        padding = self.compression_ratio - tail.shape[1]
        buffer = F.pad(tail, (0, 0, 0, padding))
        # The current index always start at 0 because the tail is right aligned (and buffer is permutation-invariant)
        current_index = torch.zeros(0, dtype=torch.long, device=tail.device)
        self.compressor_cache[layer_idx] = (buffer, num_in_buffer, current_index)

    def update_compressor_cache(self, layer_idx: int, kv_and_score: torch.Tensor) -> torch.Tensor:
        """Updates and returns the compression buffer with the new token (one per sequence)."""
        buffer, num_in_buffer, current_index = self.compressor_cache[layer_idx]
        buffer[:, current_index] = kv_and_score
        num_in_buffer = (num_in_buffer + 1) % self.compression_ratio
        current_index = (current_index + 1) % self.compression_ratio
        return buffer

    def update_indexer_cache(self, layer_idx: int, k: torch.Tensor) -> torch.Tensor:
        # TODO: just overide the normal cache with the indexing mechanism + regular new tensor
        batch_size, seq_len, num_heads, head_dim = k.shape
        num_in_buffer = self.compressor_cache[layer_idx][1]

        # Lazy initialization (prefill): cache is the new keys
        if layer_idx not in self.indexer_cache:
            cache = k.clone()
            indices = seq_len + (num_in_buffer == 0).long()
            self.indexer_cache[layer_idx] = (cache, indices)
            return cache

        # Update (decode): grow the cache if needed and then write in it
        cache, indices = self.indexer_cache[layer_idx]
        if (indices == cache.shape[1]).any().item():
            cache = torch.cat([cache, k], dim=1)

        cache[indices] = k
        indices = indices + (num_in_buffer == 0).long()
        return cache

class DeepseekV41RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV41RotaryEmbedding(LagunaRotaryEmbedding):
    pass


class DeepseekV41Experts(Glm5NextTextExperts):
    pass


class DeepseekV41TopkRouter(DeepseekV4TopKRouter):
    """Deepseek V4 router with a different router bias for image tokens."""

    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.e_score_correction_bias_vl = nn.Buffer(torch.zeros(self.num_experts))

    def forward(
        self, hidden_states: torch.Tensor, image_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routs tokens to experts, with different bias for text and image tokens.
        Args:
            - hidden_states: tensor with shape [batch_size, seq_len, hidden_dim]
            - image_mask: a boolean tensor indicating if the token is a image token with shape [batch_size, seq_len]
        """
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        bias = torch.where(image_mask.unsqueeze(-1), self.e_score_correction_bias_vl, self.e_score_correction_bias)
        indices = torch.topk(scores + bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV41ProjNorm(nn.Module):
    def __init__(self, config: DeepseekV41Config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_dim, config.qk_head_dim, bias=False)
        self.norm = DeepseekV41RMSNorm(config.qk_head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self._proj(x))


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config: DeepseekV41Config, layer_idx: int):
        super().__init__()
        self.proj = nn.Linear(config.hidden_dim, 2 * config.qk_head_dim, bias=False)  # kv and score
        self.layer_idx = layer_idx

        self.compression_ratio = config.compression_ratio
        self.norm = DeepseekV41RMSNorm(config.qk_head_dim)
        self.wgate = nn.Linear(config.hidden_dim, config.qk_head_dim, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, q_resid: torch.Tensor, valid_tokens: torch.Tensor, past_key_values: DeepseekV41CSACache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compresses the KV states and selects the states to use for attention."""
        batch_size, seq_length, _, hidden_dim = hidden_states.shape  # always one head
        original_dtype = hidden_states.dtype

        # Compute new KV and score states
        kv_and_score = self.proj(hidden_states.float())

        # If this is decoding, just update the compression buffer with the new token
        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if use_precomputed_states and seq_length == 1:
            kv_and_score = past_key_values.update_compressor_cache(self.layer_idx, kv_and_score)
            rolled_compressed_indices = None
        # If this is chunked prefill, error out (not supported)
        elif use_precomputed_states:
            raise RuntimeError("Chunked prefill is not supported for Deepseek V4.1")
        # Otherwise, this is a prefill, so we adjust the tokens to have full compression
        else:
            # First use the tail to initialize the compression buffer
            num_valid_tokens = valid_tokens.sum(1)
            num_in_buffer = num_valid_tokens % self.compression_ratio
            tail = kv_and_score[:, -self.compression_ratio:]
            past_key_values.init_compressor_cache(self.layer_idx, tail, num_in_buffer)
            # Get the compressible part of the states by moving the padding to the right and truncating the tail
            first_valid_index = seq_length - num_valid_tokens
            indices = torch.arange(seq_length, device=hidden_states.device)
            rolled_indices = (indices[None, :] + first_valid_index[:, None]) % seq_length
            compressed_length = seq_length // self.compression_ratio
            compressible_length = compressed_length * self.compression_ratio
            kv_and_score = kv_and_score.gather(1, rolled_indices.unsqueeze(-1))[:, :compressible_length]
            # Get rolled (in the other direction) indices to return the padding of compressed tokens to the left
            start_of_compressed_padding = num_valid_tokens // self.compression_ratio
            indices = indices[:compressed_length]
            rolled_compressed_indices = (indices[None, :] + start_of_compressed_padding[:, None]) % compressed_length

        # At this point, whether in prefill or decode, kv_and_score has a compressible shape
        kv_and_score = kv_and_score.reshape(batch_size, -1, self.compression_ratio, hidden_dim)
        kv, score = torch.split(kv_and_score, 2, dim=-1)
        compressed_kv = (kv * score.softmax(dim=-1)).sum(dim=-1)
        compressed_kv = self.norm(compressed_kv.to(original_dtype))

        # If this is prefill, the compressed_kv is right padded: return to left padding
        if rolled_compressed_indices is not None:
            compressed_kv = compressed_kv.gather(1, rolled_compressed_indices.unsqueeze(-1))
        return compressed_kv



class DeepseekV41Indexer(DeepseekV32Indexer):
    """Similar to Deepseek V3.2 Indexer, but instead of indexing tokens directly, indexing is done on groups of tokens""" # TODO no

    def forward( # TODO : reorder
        self,
        hidden_states: torch.Tensor,  # [B, S  , hidden_dim ]
        q_resid: torch.Tensor,        # [B, S  , q_lora_rank]
        compressed_kv: torch.Tensor,  # [B, S_c, hidden_dim ]  where S_c ~= S // compression_ratio
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        compressed_positions_embeddings: torch.Tensor,
        past_key_values: DeepseekV41CSACache,
        compressed_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        compressed_cos, compressed_sin = compressed_positions_embeddings

        # Get indexer queries
        q = self.wq_b(q_resid)
        q= q.view(batch_size, seq_len, -1, self.qk_head_dim)
        q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)

        # Same for indexer keys, which are derived from the already compressed KV states
        k = self.k_norm(self.wk(compressed_kv))
        k = k.view(batch_size, seq_len, 1, self.qk_head_dim)
        k_pass, k_rot = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_rot = apply_rotary_pos_emb(k_rot, compressed_cos, compressed_sin, unsqueeze_dim=2)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # Update the indexer K cache
        if past_key_values is not None:
            k = past_key_values.update_indexer_cache(self.layer_idx, k)

        # Compute the indexer scores
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)
        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if compressed_attn_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~compressed_attn_mask, float("-inf"))
        else:
            index_scores = index_scores + compressed_attn_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class DeepseekV41Attention(DeepseekV4Attention):
    """Compressed Sparse Attention 2 (CSA2) from DeepSeek V4.1 paper. Based on DeepSeek V4's CSA, with these changes:
    - no absolute positional embedding
    - no compression path for the indexer, the indexer_keys are derived directly from the hidden states
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        kv_shape = (batch_size, seq_length, 1, self.kv_head_dim)  # there is always only one KV head

        # Retrieve query states
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Retrieve the sliding-window part of the KV cache
        kv_states = self.kv_norm(self.kv_proj(hidden_states))
        kv_states = kv_states.view(kv_shape).transpose(1, 2)
        kv_nope, k_rot = torch.split(kv_states, [self.kv_nope_head_dim, self.kv_rope_head_dim], dim=-1)

        # Rotate and reconstruct
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        q_states = torch.cat([q_pass, q_rot], dim=-1)
        kv_states = torch.cat([kv_nope, k_rot], dim=-1)

        # Compress new KV states and run the indexer on them
        compressed_kv, compressed_mask = self.compressor(hidden_states, q_resid, position_ids, past_key_values)
        compressed_kv = self.indexer(compressed_kv, compressed_mask)


        kv_nope = self.kv_a_layernorm(kv_nope)


####################################################################################################################################




        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # position_embeddings is a {"main", "compress"} dict from the model; pick the
        # one that matches this layer's rope type (sliding → main, CSA/HCA → compress).
        cos, sin = position_embeddings[self.layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compressor is not None:  # Compressed KV (CSA or HCA)
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressor path concatenates extra entries onto the KV axis after the
        # standard sliding-window cache update, so a tensor `attention_mask` (built
        # for the pre-concat KV length) needs to be extended to cover them. The
        # compressor returns a `block_bias` carrying per-query causality + indexer
        # validity over those new slots — cat it in instead of zero-padding (which
        # would let every query see every compressed slot).
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
        # rotation (`-sin`) at the query position to undo it on the rope slice of the
        # output before the grouped output projection mixes heads. The transpose pair is
        # just a layout fix-up: apply_rotary_pos_emb expects `[B, S, H, D]` (its
        # `unsqueeze_dim=1` adds a head-broadcast dim to cos/sin); attention gave us
        # `[B, H, S, D]`.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


__all__ = [
    "DeepseekV41Config",
]
