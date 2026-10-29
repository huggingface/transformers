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

from collections import UserDict
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPooling, MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer, eager_attention_forward
from ..glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from ..glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaModel,
    GlmMoeDsaPreTrainedModel,
)
from ..llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from ..mistral3.configuration_mistral3 import Mistral3Config
from ..mistral3.modeling_mistral3 import (
    Mistral3CausalLMOutputWithPast,
    Mistral3ForConditionalGeneration,
    Mistral3Model,
    Mistral3ModelOutputWithPast,
)
from ..mistral4.modeling_mistral4 import Mistral4Experts, Mistral4MoE, Mistral4TopkRouter
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


class MistralLarge4TextConfig(GlmMoeDsaConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 1 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of leading layers that use a dense MLP; the rest use the MoE block.
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode (`"full"` runs the indexer, `"shared"` reuses the previous full
        layer's top-k). Defaults to the pattern derived from `index_topk_freq` /
        `index_skip_topk_offset` (or `index_topk_pattern`).
    no_rope_layers (`list[int]`, *optional*):
        List with at least the same length as the number of layers in the model.
        A `1` at an index position indicates that the corresponding layer will use RoPE,
        while a `0` indicates that it's a NoPE layer.
    no_rope_layer_interval (`int`, *optional*, defaults to 4):
        Every `no_rope_layer_interval`-th layer is a NoPE layer (no rotary embedding in the attention nor in the
        indexer). The other layers use RoPE.
    kv_source_layers (`list[int]`, *optional*):
        For each layer, the index of the layer whose latent KV cache it attends to. A layer whose entry is its own
        index computes and caches its own latent KV; a layer pointing at an earlier layer has no `kv_a_proj_with_mqa`
        and reuses that layer's cache (the source layer must be its own source). Defaults to no sharing.

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

    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 204800
    hidden_size: int = 6144
    intermediate_size: int = 24576
    moe_intermediate_size: int = 6144
    num_hidden_layers: int = 72
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 64
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    first_k_dense_replace: int = 1
    no_rope_layers: list[int] | None = None
    no_rope_layer_interval: int = 4
    kv_source_layers: list[int] | None = None
    final_logit_softcapping: float = 30.0

    q_lora_rank = AttributeError()
    mlp_bias = AttributeError()
    norm_topk_prob = AttributeError()

    def __post_init__(self, **kwargs):
        # YaRN over a 512-token pretraining window, no magnitude scaling (attention_factor = 1.0) + LeRoPE
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 2048.0,
                "original_max_position_embeddings": 512,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.0,
            }
        # NoPE every `no_rope_layer_interval`-th layer (smollm3 convention: 1 == uses RoPE)
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((layer_idx + 1) % self.no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
            ]
        # KV sharing: by default every layer owns its latent KV cache
        if self.kv_source_layers is None:
            self.kv_source_layers = list(range(self.num_hidden_layers))
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        # Validate KV sharing patterns
        if len(self.kv_source_layers) != self.num_hidden_layers:
            raise ValueError(
                f"`kv_source_layers` must have one entry per layer ({self.num_hidden_layers}), got {len(self.kv_source_layers)}."
            )
        for layer_idx, source in enumerate(self.kv_source_layers):
            if source > layer_idx or self.kv_source_layers[source] != source:
                raise ValueError(
                    f"Layer {layer_idx} must share KV with an earlier layer that owns its cache, got {source}."
                )
        PreTrainedConfig.validate_architecture(self)


class MistralLarge4Config(Mistral3Config):
    model_type = "mistral_large4"
    sub_configs_defaults = {
        "text_config": MistralLarge4TextConfig,
        # Mistral Large 4 vision tower (same as Mistral Small 3.1: gated SiLU MLP, 1540 px images)
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="pixtral",
            init_kwargs={
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "patch_size": 14,
                "image_size": 1540,
                "hidden_act": "silu",
            },
        ),
    }

    tie_word_embeddings: bool = False


class MistralLarge4RMSNorm(LlamaRMSNorm):
    pass


class MistralLarge4TextRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding with a learned tensor for scaling the inverse frequencies."""

    def __init__(self, config: MistralLarge4TextConfig) -> None:
        super().__init__(config)
        self.alpha = nn.Buffer(torch.zeros_like(self.inv_freq))  # init to 0 to avoid NaNs in alpha.exp()

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

    @staticmethod
    def compute_default_rope_parameters(config: MistralLarge4TextConfig, **kwargs) -> tuple[torch.Tensor, float]:
        return super().compute_default_rope_parameters(config, **kwargs)


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


class MistralLarge4TextSoftmaxScaling(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(num_heads))
        self.bias = nn.Parameter(torch.empty(num_heads))

    def forward(self, query_states: torch.Tensor, position_ids: torch.Tensor, head_dim_first: bool) -> torch.Tensor:
        log_pos = (position_ids + 1).float().log()  # convert to float to avoid overflow
        log_pos, scale, bias = (x.to(query_states.dtype) for x in (log_pos, self.scale, self.bias))
        if head_dim_first:
            scale = scale[None, :, None] * log_pos[:, None, :] + bias[None, :, None]
        else:
            scale = scale[None, None, :] * log_pos[:, :, None] + bias[None, None, :]
        return query_states * scale.unsqueeze(-1)


class MistralLarge4TextIndexer(DeepseekV32Indexer):
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
        q = self.softmax_scaling(q, position_ids, head_dim_first=False)

        # Same for the keys
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]

        # If there is RoPE, apply it (interleaved)
        if self.use_rope:
            q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
            q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
            k = torch.cat([k_rot, k_pass], dim=-1)  # [B, S, 1, D]
        k = k.squeeze(2)

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
        self.use_rope = config.no_rope_layers[layer_idx]
        self.is_kv_shared_layer = config.kv_source_layers[layer_idx] != layer_idx
        super().__init__(config, layer_idx)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        del self.q_lora_rank
        del self.q_a_proj
        del self.q_a_layernorm
        del self.q_b_proj

        self.softmax_scaling = MistralLarge4TextSoftmaxScaling(self.num_heads)

        self.qk_rope_head_dim = config.qk_rope_head_dim if self.use_rope else 0
        self.qk_nope_head_dim = config.qk_nope_head_dim + (0 if self.use_rope else config.qk_rope_head_dim)

        self.store_full_length_kv = any(src == layer_idx for src in config.kv_source_layers[layer_idx + 1 :])

        self.kv_a_proj_with_mqa = (
            nn.Linear(self.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias)
            if not self.is_kv_shared_layer
            else None
        )
        self.kv_a_layernorm = (
            MistralLarge4RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps) if not self.is_kv_shared_layer else None
        )

    # trf-ignore: TRF043
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Either the layer computes it's own cache or it loads it from shared_kv_states
        if self.is_kv_shared_layer:
            k_pass, k_rot = shared_kv_states["rope" if self.use_rope else "nope"]
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
            shared_kv_states["rope" if self.use_rope else "nope"] = (k_pass, k_rot)

        # Recompose and apply softmax-scaling to the queries
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        query_states = self.softmax_scaling(query_states, position_ids, head_dim_first=True)

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
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices


class MistralLarge4TextMLP(Qwen2MoeMLP):
    pass


class MistralLarge4TextTopkRouter(Mistral4TopkRouter):
    """Same as Mistral4TopkRouter, with fp32 precision for routing logits and norming of topk weights is always on."""

    def __init__(self, config):
        super().__init__(config)
        del self.norm_topk_prob

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # weight.float() should be a no-op because the weight is in _keep_in_fp32_modules_strict, but if a user changed
        # the dtype of the model after loading, it will prevent an error
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.softmax(-1)
        group_scores = (
            scores.view(-1, self.num_group, self.num_experts // self.num_group).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class MistralLarge4TextExperts(Mistral4Experts):
    pass


class MistralLarge4TextMoE(Mistral4MoE):
    pass


class MistralLarge4TextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: MistralLarge4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,  # MAIN DIFF with DSV3.2
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,  # the forward is overloaded to pass this args
            position_ids=position_ids,
            past_key_values=past_key_values,
            prev_topk_indices=prev_topk_indices,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class MistralLarge4PreTrainedModel(GlmMoeDsaPreTrainedModel):
    config: MistralLarge4Config
    input_modalities = ("image", "text")
    _no_split_modules = ["MistralLarge4TextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "shared_kv_states"]
    _keep_in_fp32_modules_strict = ["rotary_emb", "mlp.gate.weight"]
    # The checkpoint ships a multi-token-prediction draft head that this model does not implement yet # TODO
    _keys_to_ignore_on_load_unexpected = [r"mtp_dspark\..*"]

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
    def __init__(self, config: MistralLarge4TextConfig):
        super().__init__(config)
        self.norm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if isinstance(attention_mask, dict):
            causal_mask = attention_mask["indexed_attention"]
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                allow_is_causal_skip=False,  # Always force creation to account for causality in the indexer
            )

        # Initialize as empty dict, or reuse past shared states. We use a UserDict instead of built-in dict (it behaves
        # the same) for fsdp2 support (otherwise, `_apply_to_tensors` rebuilds every dict it recurses into, and `shared_kv_states`
        # is not correctly shared, see https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/utils.py#L223-L255)
        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        topk_indices = None  # MAIN DIFF with DSV3.2
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                shared_kv_states=shared_kv_states,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,  # MAIN DIFF with DSV3.2
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MistralLarge4ModelOutputWithPast(Mistral3ModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed):
        Tuple of `torch.FloatTensor` (one for each MoE layer) of shape `(batch_size * sequence_length, num_experts)`.
    """

    router_logits: tuple[torch.FloatTensor] | None = None


class MistralLarge4Model(Mistral3Model):
    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | list[int] | None = None,
        use_cache: bool | None = None,
        image_sizes: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MistralLarge4ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                image_sizes=image_sizes,
                return_dict=True,
            )

        if mm_encoder_outputs.get("image") is not None:
            image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return MistralLarge4ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,  # only diff with Mistral3Model: we also return the router logits
            image_hidden_states=image_features if mm_encoder_outputs.get("image") is not None else None,
        )


class MistralLarge4CausalLMOutputWithPast(Mistral3CausalLMOutputWithPast):
    router_logits: tuple[torch.FloatTensor] | None = None


class MistralLarge4ForConditionalGeneration(Mistral3ForConditionalGeneration):
    def __init__(self, config: MistralLarge4Config):
        super().__init__(config)
        self.final_logit_softcapping = self.config.get_text_config().final_logit_softcapping

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MistralLarge4CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration

        >>> model = MistralLarge4ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        >>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

        >>> prompt = "<s>[INST][IMG]What is the image?[/INST]"
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is the image?The image depicts two cats lying on a pink blanket."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_sizes=image_sizes,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

        # Logits softcapping
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MistralLarge4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "MistralLarge4TextConfig",
    "MistralLarge4Config",
    "MistralLarge4PreTrainedModel",
    "MistralLarge4TextModel",
    "MistralLarge4Model",
    "MistralLarge4ForConditionalGeneration",
]
