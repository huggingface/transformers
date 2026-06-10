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
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM,
    DeepseekV32Indexer,
    DeepseekV32Model,
    DeepseekV32PreTrainedModel,
    DeepseekV32RotaryEmbedding,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-5")
@strict
class GlmMoeDsaConfig(DeepseekV32Config):
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
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode (`"full"` runs the indexer, `"shared"` reuses the previous full
        layer's top-k). Defaults to the pattern derived from `index_topk_freq` /
        `index_skip_topk_offset` (or `index_topk_pattern`).

    ```python
    >>> from transformers import GlmMoeDsaConfig, GlmMoeDsaModel

    >>> # Initializing a GLM-MoE-DSA configuration
    >>> configuration = GlmMoeDsaConfig()

    >>> # Initializing a model from the configuration
    >>> model = GlmMoeDsaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]
        super().__post_init__(**kwargs)


class GlmMoeDsaRMSNorm(DeepseekV3RMSNorm):
    pass


class GlmMoeDsaRotaryEmbedding(DeepseekV32RotaryEmbedding):
    pass


class GlmMoeDsaIndexer(DeepseekV32Indexer):
    pass


class GlmMoeDsaAttention(DeepseekV3Attention):
    """
    DeepSeek-V3 MLA + a DSA indexer, extended with **cross-layer top-k sharing**.

    `config.indexer_types[layer_idx]` decides whether this layer runs its own indexer (`"full"`) or
    reuses the previous full layer's top-k selection (`"shared"`).
    `next_skip_topk` signals that the *next* layer will reuse this
    layer's top-k, so it is propagated upward via `prev_topk_indices`.
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # DSA: select this layer's top-k tokens, or reuse the previous full layer's on `"shared"` layers.
        if not self.skip_topk or prev_topk_indices is None:
            if self.indexer is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                position_ids,
                past_key_values=past_key_values,
            )  # [B, S, topk]
        else:
            topk_indices = prev_topk_indices

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = topk_indices.new_ones(
                (batch_size, seq_length, key_states.shape[2]), dtype=torch.bool
            ).scatter(-1, topk_indices.long(), False).unsqueeze(1)
            if attention_mask is None:
                key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
                index_mask = index_mask | (key_positions[None, None, None, :] > position_ids[:, None, :, None])
                attention_mask = hidden_states.new_zeros((batch_size, 1, seq_length, key_states.shape[2]))
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


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None, # MAIN DIFF with DSV3.2
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices, # MAIN DIFF with DSV3.2
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class GlmMoeDsaPreTrainedModel(DeepseekV32PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]


class GlmMoeDsaModel(DeepseekV32Model):
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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        topk_indices = None # MAIN DIFF with DSV3.2
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices, # MAIN DIFF with DSV3.2
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM):
    pass


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaPreTrainedModel",
    "GlmMoeDsaModel",
    "GlmMoeDsaForCausalLM",
]
