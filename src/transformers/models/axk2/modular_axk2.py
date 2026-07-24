# Copyright 2026 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch A.X-K2 model."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GenericForSequenceClassification, GenericForTokenClassification
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..clip.modeling_clip import CLIPMLP
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Experts,
    DeepseekV32ForCausalLM,
    DeepseekV32Indexer,
    DeepseekV32Model,
    DeepseekV32MoE,
    DeepseekV32PreTrainedModel,
    DeepseekV32RotaryEmbedding,
    DeepseekV32TopkRouter,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="skt/A.X-K2")
@strict
class AXK2Config(DeepseekV32Config):
    r"""
    n_group (`int`, *optional*):
        Number of expert groups for grouped routing, used by the larger A.X-K2 releases. `None` (the
        A.X-K2-Light default) routes over all experts without group restriction.
    topk_group (`int`, *optional*):
        Number of expert groups the top-k selection is restricted to when `n_group` is set.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Derived from the (legacy) kwargs
        `first_k_dense_replace` / `moe_layer_freq` when not provided.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 16):
        Number of heads for the indexer projections (DSA).
    gated_norm_rank (`int`, *optional*, defaults to 16):
        Bottleneck rank for the low-rank input-dependent gate used by `AXK2GatedRMSNorm`. The gate wraps
        `input_layernorm` on every layer and `post_attention_layernorm` on MoE layers.

    ```python
    >>> from transformers import AXK2Config, AXK2Model

    >>> # Initializing an A.X-K2 style configuration
    >>> configuration = AXK2Config()

    >>> # Initializing a model from the configuration
    >>> model = AXK2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "axk2"

    vocab_size: int = 163840
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 128
    q_lora_rank: int = 384
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    qk_nope_head_dim: int = 64
    num_experts_per_tok: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    bos_token_id: int | None = 163691
    eos_token_id: int | list[int] | None = 163691
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    gated_norm_rank: int = 16
    # A.X-K2-Light routes without expert groups; the larger A.X-K2 releases set `n_group`/`topk_group`
    # for DeepSeek-V3-style grouped routing, so both modes are supported (`None` = non-grouped).
    n_group: int | None = None
    topk_group: int | None = None

    first_k_dense_replace = AttributeError()
    mlp_bias = AttributeError()

    base_model_tp_plan = {
        "layers.*.self_attn.q_gate_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so `head_dim` points at it (the inherited rotary embedding
        # reads `config.head_dim`).
        self.head_dim = self.qk_rope_head_dim

        # Convert from legacy args to mlp layer types
        if self.mlp_layer_types is None:
            first_k_dense_replace = kwargs.pop("first_k_dense_replace", 1)
            moe_layer_freq = kwargs.pop("moe_layer_freq", 1)
            self.mlp_layer_types = [
                "sparse" if i >= first_k_dense_replace and i % moe_layer_freq == 0 else "dense"
                for i in range(self.num_hidden_layers)
            ]

        # Indexer cache needed so DSA to indicate correct cache
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers

        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        PreTrainedConfig.validate_architecture(self)
        if self.q_lora_rank is None or self.q_lora_rank <= 0:
            raise ValueError(
                "A.X-K2 requires a positive `q_lora_rank` (the indexer and output gate read the query LoRA "
                f"bottleneck), got {self.q_lora_rank}."
            )
        if (self.n_group is None) != (self.topk_group is None):
            raise ValueError(
                f"`n_group` and `topk_group` must be set together (both `None` for non-grouped routing), got "
                f"n_group={self.n_group}, topk_group={self.topk_group}."
            )
        if self.n_group is not None:
            if self.n_routed_experts % self.n_group != 0:
                raise ValueError(
                    f"`n_routed_experts` ({self.n_routed_experts}) must be divisible by `n_group` ({self.n_group})."
                )
            if self.topk_group > self.n_group:
                raise ValueError(f"`topk_group` ({self.topk_group}) cannot exceed `n_group` ({self.n_group}).")


class AXK2RMSNorm(DeepseekV3RMSNorm):
    pass


class AXK2GateMLP(CLIPMLP):
    def __init__(self, config: AXK2Config):
        super().__init__(config)
        self.activation_fn = nn.SiLU()
        self.fc1 = nn.Linear(config.hidden_size, config.gated_norm_rank, bias=False)
        self.fc2 = nn.Linear(config.gated_norm_rank, config.hidden_size, bias=False)


class AXK2GatedRMSNorm(nn.Module):
    """RMSNorm followed by a low-rank input-dependent sigmoid gate (Megatron `GatedNormWrapper`):

    y = RMSNorm(x)
    return y * sigmoid(gate_mlp(y))
    """

    def __init__(self, config: AXK2Config, eps: float):
        super().__init__()
        self.norm = AXK2RMSNorm(config.hidden_size, eps=eps)
        self.mlp = AXK2GateMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        return (y * torch.sigmoid(self.mlp(y).float())).to(y.dtype)


class AXK2RotaryEmbedding(DeepseekV32RotaryEmbedding):
    pass


class AXK2Indexer(DeepseekV32Indexer):
    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)


class AXK2TopkRouter(DeepseekV32TopkRouter):
    def apply_group_scoring(self, scores_for_choice):
        """Apply DeepSeek style grouped scoring"""
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        # NOTE: fill with `0.0` intentionally to match reference implementation
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        return scores_for_choice

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias

        # Key difference: light checkpoints do not use group based routing and skip this path
        if self.num_group is not None:
            scores_for_choice = self.apply_group_scoring(scores_for_choice)

        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class AXK2Experts(DeepseekV32Experts):
    pass


class AXK2MoE(DeepseekV32MoE):
    pass


class AXK2Attention(DeepseekV32Attention):
    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__()
        # Fused projection for q and gate, needs to be kept fused as the FP8 scales won't match otherwise in split variation
        del self.q_proj
        del self.q_b_proj

        self.q_gate_proj = nn.Linear(
            2 * config.q_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_gate_shape = (batch_size, seq_length, -1, self.qk_head_dim + self.v_head_dim)

        # Key change: Additional gate and fused projection (blocked to split due to FP8 scale mismatches)
        q_compressed = self.q_a_proj(hidden_states)
        q_resid = self.q_a_layernorm(q_compressed)

        query_gate_states = self.q_gate_proj(torch.cat([q_resid, q_compressed], dim=-1)).view(query_gate_shape)
        query_states, gate_states = torch.split(query_gate_states, [self.qk_head_dim, self.v_head_dim], dim=-1)
        query_states = query_states.transpose(1, 2)
        gate_states = gate_states.reshape(batch_size, seq_length, -1)

        q_pass, q_rot = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(kv_pass)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        query_states = torch.cat((q_pass, q_rot), dim=-1)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # The indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`.
        indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
        topk_indices = self.indexer(
            hidden_states,
            q_compressed,
            position_embeddings,
            indexer_mask,
            position_ids,
            past_key_values=past_key_values,
        )

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )
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
        # Input-dependent sigmoid gate on the attention output (the gate half of the fused projection).
        attn_output = (attn_output * torch.sigmoid(gate_states.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AXK2DecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = AXK2GatedRMSNorm(config, eps=config.rms_norm_eps)
        self.post_attention_layernorm = (
            AXK2GatedRMSNorm(config, eps=config.rms_norm_eps)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else AXK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )


class AXK2PreTrainedModel(DeepseekV32PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["inv_freq"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, AXK2TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, AXK2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class AXK2Model(DeepseekV32Model):
    pass


class AXK2ForCausalLM(DeepseekV32ForCausalLM):
    pass


class AXK2ForSequenceClassification(GenericForSequenceClassification, AXK2PreTrainedModel):
    pass


class AXK2ForTokenClassification(GenericForTokenClassification, AXK2PreTrainedModel):
    pass


__all__ = [
    "AXK2Config",
    "AXK2PreTrainedModel",
    "AXK2Model",
    "AXK2ForCausalLM",
    "AXK2ForSequenceClassification",
    "AXK2ForTokenClassification",
]
