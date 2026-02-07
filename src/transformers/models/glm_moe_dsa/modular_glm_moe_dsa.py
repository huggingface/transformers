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


import math

import torch
from torch import nn

from ...cache_utils import Cache
from ...models.llama.modeling_llama import (
    apply_rotary_pos_emb,
)
from ...utils import logging
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig


logger = logging.get_logger(__name__)


class GlmMoeDsaConfig(Glm4MoeLiteConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmMoeDsaModel`]. It is used to instantiate a
    GLM-5 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GLM-5.
    e.g. [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4MoeLiteModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 78):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 64):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 2048):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimension of the query/key heads that don't use rotary position embeddings.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts, None means dense model.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        mlp_layer_types (`list`, *optional*):
            MLP (Moe vs Dense) pattern for each layer.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top tokens selected by the indexer for retrieval/attention in each step.
        index_head_dim (`int`, *optional*, defaults to 128):
            Hidden size (per-head dimension) of each indexer attention head.
        index_n_heads (`int`, *optional*, defaults to 32):
            Number of attention heads used by the indexer module.

    ```python
    >>> from transformers import Glm4MoeLiteModel, Glm4MoeLiteConfig

    >>> # Initializing a GLM-MOE-DSA style configuration
    >>> configuration = GlmMoeDsaConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        hidden_size: int | None = 6144,
        intermediate_size: int | None = 12288,
        moe_intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 78,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = 64,
        first_k_dense_replace: int | None = 3,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 256,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 2048,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 256,
        qk_nope_head_dim: int | None = 192,
        num_experts_per_tok: int | None = 8,
        initializer_range: float | None = 0.02,
        index_topk: int | None = 2048,
        index_head_dim: int | None = 128,
        index_n_heads: int | None = 32,
        **super_kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.first_k_dense_replace = first_k_dense_replace
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads

        super().__init__(**super_kwargs)


class GlmMoeDsaRMSNorm(Glm4MoeRMSNorm):
    pass


class GLmMoeDsaIndexer(nn.Module):
    def __init__(self, config: "GlmMoeDsaConfig", index_layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = index_layer_idx

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.index_n_heads
        self.num_local_heads: int = config.index_n_heads  # world_size handling can be added as needed
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, dtype=torch.get_default_dtype(), bias=False)
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
        q_rot = apply_rotary_pos_emb_interleave(q_rot, cos, sin)  # [B, S, H, rope_D]
        q_states = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]

        # Keys
        k = self.k_layernorm(self.k_proj(hidden_states))  # [B, S, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        # MLA uses single-head rope stream, then expands later; keep [B, 1, S, rope_D] here
        k_rot = k_rot.unsqueeze(1)  # [B, 1, S, rope_D]
        k_rot = apply_rotary_pos_emb_interleave(k_rot, cos, sin)  # [B, 1, S, rope_D]
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
        head_weights = self.weights_proj(hidden_states) * (self.num_heads**-0.5)  # [B, S, H]
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


class GlmMoeDsaAttention(DeepseekV2Attention):
    """
    DeepSeek V3.2 sparse attention mechanism with indexer.

    This implements the native sparse attention from [DeepSeek V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) which uses
    an indexer to select top-k tokens for attention computation, making it more efficient for long sequences.

    Switch to the implementation from this [PR](https://github.com/huggingface/transformers/pull/41251) as soon as it’s merged.
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.softmax_scale = self.qk_head_dim**-0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = GLmMoeDsaIndexer(config, layer_idx)

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


class GlmMoeDsaDecoderLayer(Glm4MoeDecoderLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = GlmMoeDsaAttention(config=config, layer_idx=layer_idx)


class GlmMoeDsaPreTrainedModel(Glm4MoePreTrainedModel):
    pass


class GlmMoeDsaModel(Glm4MoeModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]


class GlmMoeDsaForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaPreTrainedModel",
    "GlmMoeDsaModel",
    "GlmMoeDsaForCausalLM",
]
