# Copyright 2026 HuggingFace Inc. team. All rights reserved.
#
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
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ..deepseek_v32.modeling_deepseek_v32 import eager_attention_forward
from ..glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from ..glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaIndexer,
    GlmMoeDsaModel,
    GlmMoeDsaPreTrainedModel,
)
from ..llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from ..mistral3.configuration_mistral3 import Mistral3Config
from ..mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration, Mistral3Model
from ..mistral4.modeling_mistral4 import Mistral4Experts, Mistral4MoE, Mistral4TopkRouter
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


def apply_rotary_pos_emb_interleave(
    q: torch.Tensor, k: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Applies interleaved Rotary Position Embedding to the query and (optionally) the key tensors.

    DeepSeek lays the rotary dimensions out in interleaved pairs `(x0, x1), (x2, x3), ...`, each rotated by a
    single frequency. We compute that rotation directly on the even/odd slices instead of de-interleaving with a
    `view`/`transpose`/`reshape`; the output is bit-identical to the de-interleaved `rotate_half` formulation while
    avoiding the extra contiguous copy.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`, *optional*): The key tensor if there is one, otherwise `None`.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and (optionally) the key tensors rotated using the Rotary Position
        Embedding.
    """
    # `cos`/`sin` are `cat(freqs, freqs)`; the first half holds the per-pair angle.
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)

    q1, q2 = q[..., 0::2], q[..., 1::2]
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)

    if k is not None:
        k1, k2 = k[..., 0::2], k[..., 1::2]
        k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    else:
        k_embed = None

    return q_embed, k_embed


class MistralLarge4TextConfig(GlmMoeDsaConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of leading layers that use a dense MLP; the rest use the MoE block.
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode (`"full"` runs the indexer, `"shared"` reuses the previous full
        layer's top-k). Defaults to the pattern derived from `index_topk_freq` /
        `index_skip_topk_offset` (or `index_topk_pattern`).
    no_rope_layer_interval (`int`, *optional*, defaults to 4):
        Every `no_rope_layer_interval`-th layer is a NoPE layer (no rotary embedding in the attention nor in the
        indexer). The other layers use RoPE.

    ```python
    >>> from transformers import MistralLarge4TextConfig, MistralLarge4TextModel

    >>> # Initializing a Mistral Large 4 text configuration
    >>> configuration = MistralLarge4TextConfig()

    >>> # Initializing a model from the configuration
    >>> model = MistralLarge4TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mistral_large4_text"

    no_rope_layer_interval: int = 4
    q_lora_rank = AttributeError()


class MistralLarge4Config(Mistral3Config):
    model_type = "mistral_large4"


class MistralLarge4TextRMSNorm(LlamaRMSNorm):
    pass


class MistralLarge4TextRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding with a learned tensor for scaling the inverse frequencies."""

    def __init__(self, config: MistralLarge4TextConfig) -> None:
        super().__init__(config)
        self.alpha = nn.Parameter(torch.empty_like(self.inv_freq))

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        scaled_inv_freq = (
            self.inv_freq.to(device=x.device, dtype=torch.float)
            * self.alpha.to(device=x.device, dtype=torch.float).exp()
        )
        freqs = position_ids[..., None].float() * scaled_inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MistralLarge4TextSoftmaxScaling(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(num_heads))
        self.bias = nn.Parameter(torch.empty(num_heads))

    def forward(self, query_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        log_pos = (position_ids + 1).float().log()
        scale = self.scale.to(query_states.dtype) * log_pos.to(query_states.dtype)
        scale = scale + self.bias.to(query_states.dtype)
        scale = scale.unsqueeze(-1)
        return query_states * scale


class MistralLarge4TextIndexer(GlmMoeDsaIndexer):  # TODO: inherit from DeepseekV32Indexer instead.
    """Indexer from glm_moe_dsa, but since there is no Q-LoRA in the attention, the indexer queries are derived from
    hidden states, using norm -> proj -> softmax scaling. Also, for layers with no RoPE, the indexer does not use RoPE
    as well."""

    def __init__(self, config: MistralLarge4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_lora_rank
        self.q_input_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.wq_b = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.softmax_scaling = MistralLarge4TextSoftmaxScaling(self.n_heads)
        self.use_rope = config.no_rope_layers[layer_idx]

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        Same as [`DeepseekV32Indexer.forward`], but the indexer applies **interleaved** RoPE
        rather than the non-interleaved half-split RoPE used by DeepSeek-V3.2.

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                turn these into an additive sparse mask; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # The indexer queries are derived directly from the hidden states
        q = self.q_input_norm(hidden_states)
        q = self.wq_b(q)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q = self.softmax_scaling(q, position_ids)

        # Same for the keys
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]

        # If there is RoPE, apply it (interleaved)
        if self.use_rope:
            q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)

            q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
            k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class MistralLarge4TextAttention(GlmMoeDsaAttention):
    """DSA attention from GlmMoeDsa, with no Q-LoRA and KV sharing."""

    def __init__(self, config: MistralLarge4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        del self.q_lora_rank
        del self.q_a_proj
        del self.q_a_layernorm
        del self.q_b_proj

        self.softmax_scaling = MistralLarge4TextSoftmaxScaling(self.num_heads)

        self.use_rope = config.no_rope_layers[layer_idx]
        # TODO: dims...

        self.is_kv_shared_layer = config.kv_sharing_layers[layer_idx]
        self.store_full_length_kv = config.store_full_length_kv[layer_idx]
        self.kv_a_proj_with_mqa = (
            nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)
            if not config.kv_sharing_layers[layer_idx]
            else None
        )
        self.kv_a_layernorm = (
            MistralLarge4TextRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
            if not config.kv_sharing_layers[layer_idx]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Either the layer computes it's own cache or it loads it from shared_kv_states
        if self.is_kv_shared_layer:
            k_pass, k_rot = shared_kv_states[self.layer_type]
            k_pass = k_pass.to(q_states.device)  # device of past layer may be different from current one
            k_rot = k_rot.to(q_states.device)
        else:
            compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
            kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            # Both latents are viewed as single-head, 4D tensors, as expected by `expand_kv`
            k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
            k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Apply LeRoPE to the queries and the keys, except if this is a no_rope layer
        if self.use_rope:
            cos, sin = position_embeddings
            # For layers with cache sharing, only rotate the queries: the keys were already rotated by the source layer
            if self.is_kv_shared_layer:
                q_rot, _ = apply_rotary_pos_emb_interleave(q_rot, None, cos, sin)
            else:
                q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # Update the cache while latent KV is still compressed
        if past_key_values is not None and not self.is_kv_shared_layer:
            k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)
        # If the layer takes part in cache sharing and computes its own cache, store a copy in shared_kv_states
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (k_pass, k_rot)

        # Recompose and apply softmax-scaling to the queries
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        query_states = self.softmax_scaling(query_states, position_ids)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        # DSA: select this layer's top-k tokens, or reuse the previous full layer's on `"shared"` layers.
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                position_embeddings,
                attention_mask[:, 0, :, :],
                position_ids,  # Kept for BC
                past_key_values=past_key_values,
            )  # [B, S, topk]
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )

            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & ~index_mask
            else:
                attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        else:
            sparse_indices = topk_indices

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
            indices=sparse_indices,  # consumed by flash_mla_with_kvcache; ignored by eager / SDPA
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices


class MistralLarge4TextMLP(Qwen2MoeMLP):
    pass


class MistralLarge4TextTopkRouter(Mistral4TopkRouter):
    pass


class MistralLarge4TextExperts(Mistral4Experts):
    pass


class MistralLarge4TextMoE(Mistral4MoE):
    pass


class MistralLarge4TextDecoderLayer(GlmMoeDsaDecoderLayer):
    pass


class MistralLarge4PreTrainedModel(GlmMoeDsaPreTrainedModel):
    config: MistralLarge4Config
    input_modalities = ("image", "text")
    _no_split_modules = ["MistralLarge4TextDecoderLayer"]
    _keep_in_fp32_modules_strict = ["rotary_emb"]
    _keys_to_ignore_on_load_unexpected = []

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MistralLarge4TextTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MistralLarge4TextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MistralLarge4TextRotaryEmbedding):
            init.zeros_(module.alpha)  # exp(0) = 1: LeRoPE has no effect until `alpha` is loaded
        elif isinstance(module, MistralLarge4TextSoftmaxScaling):
            # 0 * log(pos + 1) + 1 = 1: no query scaling until `scale` / `bias` are loaded
            init.zeros_(module.scale)
            init.ones_(module.bias)


class MistralLarge4TextModel(MistralLarge4PreTrainedModel, GlmMoeDsaModel):
    pass


class MistralLarge4Model(Mistral3Model):
    pass


class MistralLarge4ForConditionalGeneration(Mistral3ForConditionalGeneration):
    pass


__all__ = [
    "MistralLarge4TextConfig",
    "MistralLarge4Config",
    "MistralLarge4PreTrainedModel",
    "MistralLarge4TextModel",
    "MistralLarge4Model",
    "MistralLarge4ForConditionalGeneration",
]


# TODO: re-organize modules
