# Copyright 2025 the HuggingFace Team. All rights reserved.
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
import math

import torch
from torch import nn

from ...cache_utils import Cache
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2DecoderLayer,
    DeepseekV2Experts,
    DeepseekV2ForCausalLM,
    DeepseekV2ForSequenceClassification,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
)
from ..llama.modeling_llama import apply_rotary_pos_emb


class DeepseekV32Config(DeepseekV2Config):
    def __init__(self, index_n_heads=64, index_head_dim=128, index_topk=2048, **super_kwargs):
        super().__init__(**super_kwargs)
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_top_k = index_topk


class DeepseekV32MoEGate(DeepseekV2Experts):
    pass


class DeepseekV32Experts(DeepseekV2Experts):
    pass


class DeepseekV32MLP(DeepseekV2MLP):
    pass


class DeepseekV32RMSNorm(DeepseekV2RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV2RotaryEmbedding):
    pass


class DeepseekV32Indexer(nn.Module):
    def __init__(self, config: "DeepseekV32Config", index_layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = index_layer_idx

        self.hidden_size: int = config.dim
        self.num_heads: int = config.index_n_heads
        self.num_local_heads: int = config.index_n_heads  # world_size handling can be added as needed
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.weight_proj = nn.Linear(self.hidden_size, self.num_heads, dtype=torch.get_default_dtype(), bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        q_resid: torch.Tensor,  # [B, S, q_lora_rank]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values_index: "Cache",
        cache_position: torch.LongTensor | None,
    ) -> torch.LongTensor:
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Queries
        q_states = self.q_b_proj(q_resid)  # [B, S, H*D]
        q_states = q_states.view(B, S, self.num_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q_states, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin)  # [B, S, H, rope_D]
        q_states = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]

        # Keys
        k = self.k_layernorm(self.k_proj(hidden_states))  # [B, S, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        # MLA uses single-head rope stream, then expands later; keep [B, 1, S, rope_D] here
        k_rot = k_rot.unsqueeze(1)  # [B, 1, S, rope_D]
        k_rot = apply_rotary_pos_emb(k_rot, cos, sin)  # [B, 1, S, rope_D]
        k_states = torch.cat(
            [
                k_rot.expand(B, self.num_heads, S, -1),  # expand rope
                k_pass.view(B, 1, S, -1).expand(B, self.num_heads, S, -1),
            ],
            dim=-1,
        )  # [B, H, S, D]

        # Quantize (per provided utilities)
        # Update indexer cache (layer idx belongs to the attention layer using this indexer)
        # We store as: keys = k_fp8 (as [B, 1, S, D] or [B, H, S, D]? We keep [B, 1, S, D] like original)
        # For compactness, collapse heads to 1 for the indexer (you can keep H if your fp8_index expects it).
        k_1h = k_states.mean(dim=1, keepdim=True)  # [B, 1, S, D]  (cheap head merge; adjust if needed)
        k_cache = past_key_values_index.update(k_1h, self.layer_idx, cache_kwargs={"cache_position": cache_position})

        # Weights per head
        head_weights = self.weight_proj(hidden_states) * (self.num_heads**-0.5)  # [B, S, H]
        head_weights = head_weights.unsqueeze(-1) * self.softmax_scale  # [B, S, H, *]
        logits = torch.matmul(k_cache.unsqueeze(1), q_states.transpose(-1, -2))  # [B, M, N, H]

        # ReLU and sum over heads -> [B, M, N]
        logits.clamp_min_(0)
        index_scores = logits.sum(dim=-1)  # [B, M, N]

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        T = index_scores.shape[-1]
        topk = min(self.index_topk, T)
        topk_indices = index_scores.topk(topk, dim=-1).indices  # [..., topk]
        return topk_indices


class DeepseekV32Attention(DeepseekV2Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.softmax_scale = self.qk_head_dim**-0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,  # must be Cache with MlaLayer at `layer_idx`
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        # ----- Q path -----
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
        q_states = self.q_b_proj(q_resid).view(B, S, self.num_heads, self.qk_head_dim)  # [B, S, H, D]
        # Split into pass/rot then apply RoPE on q_rot
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin)  # [B, S, H, rope_D]
        q_states = torch.cat([q_pass, q_rot], dim=-1)  # [B, S, H, D]

        # Layout for matmul: [B, H, S, D]
        q_states = q_states.transpose(1, 2).contiguous()  # [B, H, S, D]

        # ----- KV path (compressed + rope stream) -----
        kv_all = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
        kv_compressed, k_rot = torch.split(kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_compressed = self.kv_a_layernorm(kv_compressed)  # [B, S, kv_rank]
        # Pre-project to K_pass and V
        kv_proj = self.kv_b_proj(kv_compressed)  # [B, S, H*(qk_nope + v)]
        kv_proj = kv_proj.view(B, S, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v_states = torch.split(
            kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )  # [B,S,H,nope], [B,S,H,V]

        # Rope on K side: keep a single-head rope stream like MLA, then expand
        k_rot = k_rot.view(B, 1, S, self.qk_rope_head_dim)  # [B, 1, S, rope_D]
        k_rot = apply_rotary_pos_emb(k_rot, cos, sin)  # [B, 1, S, rope_D]

        # Concatenate K = [K_pass, K_rot(expanded)]
        k_states = torch.cat(
            (
                k_pass.transpose(1, 2),  # [B, H, S, nope_D]
                k_rot.expand(B, self.num_heads, S, -1),
            ),  # [B, H, S, rope_D]
            dim=-1,
        )  # [B, H, S, D]
        v_states = v_states.transpose(1, 2).contiguous()  # [B, H, S, V]

        # ----- Cache update/usage -----
        if past_key_values is not None:
            # Store compressed stream & rope stream (as in original MLA path)
            # We cache `kv_compressed` under `keys` and `k_rot` under `values` in MlaLayer.
            # Shapes must be [B, H, t, *] and [B, 1, t, rope_D].
            kv_comp_cache = kv_compressed.view(B, 1, S, self.kv_lora_rank).expand(B, self.num_heads, S, -1)
            k_rot_cache = k_rot  # [B, 1, S, rope_D]
            cached_kv, cached_pe = past_key_values.update(
                kv_comp_cache, k_rot_cache, layer_idx=self.layer_idx, cache_kwargs={"cache_position": cache_position}
            )
            # Decode path makes use of cached projections; Prefill can use full K/V directly.

        # ----- Two paths (prefill vs decode) -----
        if attention_mask is not None:
            # Prefill (full attention over local window): standard scaled dot-product with top-k pruning from indexer

            # Build scores: [B, H, S, S_total]
            # K layout already [B, H, T, D]
            scores = (q_states.float() @ k_states.float().transpose(-1, -2)) * self.scaling  # [B, H, S, T]

            # Indexer top-k
            if past_key_values is not None:
                topk_idx = self.indexer(
                    hidden_states,
                    q_resid,
                    position_embeddings,
                    attention_mask,
                    past_key_values_index=past_key_values,  # we reuse same Cache with IndexerLayer? (separate cache recommended)
                    cache_position=cache_position,
                )
                # Build mask to keep only top-k per (B,S,head?)
                # Expect topk_idx shape to broadcast to [B, H, S, T]. We scatter along last dim.
                keep_mask = torch.full_like(scores, float("-inf"))
                # If topk_idx is [B,S,topk], expand for heads:
                if topk_idx.dim() == 3:
                    topk_idx = topk_idx.unsqueeze(1).expand(B, self.num_heads, S, -1)
                keep_mask.scatter_(-1, topk_idx, 0.0)
                scores = scores + keep_mask

            probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(hidden_states)  # [B, H, S, T]
            attn_output = probs @ v_states  # [B, H, S, V]

        elif past_key_values is not None:
            # Decode: use cached compressed KV & rope stream to recompose attention scores efficiently
            # Compose q_pass and q_rot pieces as in MLA math, but via matmul
            # 1) Rebuild "nope" term via kv_b weights (dequant on the fly)
            wkv_b = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            w_k_nope = wkv_b[:, : self.qk_nope_head_dim, :]  # [H, nope_D, kv_rank]
            w_v = wkv_b[:, self.qk_nope_head_dim :, :]  # [H, V,     kv_rank]

            # q_pass: [B,H,S,nope_D]; cached_kv: [B,H,T,kv_rank]
            q_pass = q_states[..., : self.qk_nope_head_dim]  # [B,H,S,nope_D]
            kv_comp = past_key_values[self.layer_idx][0]  # keys -> [B,H,T,kv_rank]
            pe_full = past_key_values[self.layer_idx][1]  # values -> [B,1,T,rope_D]
            # Project q_pass with w_k_nope: [B,H,S,kv_rank]
            qk_nope = torch.matmul(q_pass, w_k_nope.transpose(-1, -2))  # [B,H,S,kv_rank]
            # Scores_nope = qk_nope @ kv_comp^T
            scores_nope = torch.matmul(qk_nope.float(), kv_comp.float().transpose(-1, -2))  # [B,H,S,T]

            # 2) Rope term: q_rot @ k_rot^T
            q_rot_only = q_states[..., -self.qk_rope_head_dim :]  # [B,H,S,rope_D]
            k_rot_only = pe_full.expand(B, self.num_heads, -1, -1)  # [B,H,T,rope_D]
            scores_rot = torch.matmul(q_rot_only.float(), k_rot_only.float().transpose(-1, -2))  # [B,H,S,T]

            scores = (scores_nope + scores_rot) * self.scaling

            # Indexer top-k (decode)
            topk_idx = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask,
                past_key_values_index=past_key_values,
                cache_position=cache_position,
            )
            # For decode single-step S==1 typically; build a [B,H,1,T] mask
            keep_mask = torch.full_like(scores, float("-inf"))
            if topk_idx.dim() == 3:
                topk_idx = topk_idx.unsqueeze(1).expand(B, self.num_heads, S, -1)
            keep_mask.scatter_(-1, topk_idx, 0.0)
            scores = scores + keep_mask

            probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(hidden_states)  # [B,H,S,T]

            # Rebuild V for decode fast-path: v = (kv_comp @ w_v^T)
            # kv_comp: [B,H,T,kv_rank], w_v: [H, V, kv_rank]
            v_from_comp = torch.matmul(kv_comp, w_v.transpose(-1, -2))  # [B,H,T,V]
            attn_output = torch.matmul(probs, v_from_comp)  # [B,H,S,V]

        # Output projection
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1).contiguous()  # [B,S,H*V]
        attn_output = self.o_proj(attn_output)  # [B,S,hidden]
        return attn_output, None, None


class DeepseekV32DecoderLayer(DeepseekV2DecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV2PreTrainedModel):
    pass


class DeepseekV32Model(DeepseekV2Model):
    pass


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV32ForSequenceClassification(DeepseekV2ForSequenceClassification):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
    "DeepseekV32ForSequenceClassification",
]
