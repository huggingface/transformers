# Copyright 2025 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch A.X-K2 model (modular).

A.X-K2 is SK Telecom's flagship LLM. Architecturally it is DeepSeek-V3.2 (Multi-head Latent Attention +
DeepSeek Sparse Attention) with three SK Telecom modifications:

  * non-grouped sigmoid top-k expert routing (no expert groups),
  * a low-rank input-dependent gate on the RMSNorms (`AXK2GatedRMSNorm`), and
  * an input-dependent sigmoid gate on the attention output.

Everything else — MLA, the SGA lightning indexer and its cache, the MoE experts, and RoPE — is
inherited from DeepSeek-V3 / DeepSeek-V3.2.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MLP,
    apply_rotary_pos_emb_interleave,
)
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Experts,
    DeepseekV32Indexer,
    DeepseekV32MoE,
    DeepseekV32TopkRouter,
)
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from .configuration_axk2 import AXK2Config


logger = logging.get_logger(__name__)


class AXK2RMSNorm(LlamaRMSNorm):
    pass


class AXK2GatedRMSNorm(nn.Module):
    """RMSNorm wrapped with a low-rank input-dependent gate (Megatron `GatedNormWrapper`).

    forward(x):
        y = RMSNorm(x)
        gate = W_up(silu(W_down(y)))
        return y * sigmoid(gate.float()).to(y.dtype)
    """

    def __init__(self, hidden_size: int, rank: int, eps: float):
        super().__init__()
        self.norm = AXK2RMSNorm(hidden_size, eps=eps)
        self.W_down = nn.Linear(hidden_size, rank, bias=False)
        self.W_up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        raw_gate = self.W_up(F.silu(self.W_down(y)))
        return (y * torch.sigmoid(raw_gate.float())).to(y.dtype)


class AXK2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class AXK2MLP(DeepseekV3MLP):
    pass


class AXK2Indexer(DeepseekV32Indexer):
    """SGA (Sparse Gated Attention) lightning indexer.

    Identical to DeepSeek-V3.2's DSA indexer except the key `LayerNorm` keeps A.X-K2's training eps
    (the default `1e-5`, versus DeepSeek's `1e-6`). The indexer key cache lives on the shared
    `DynamicIndexedLayer` (accessed via `past_key_values.update_indexer`), not on the module.
    """

    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.k_norm = nn.LayerNorm(self.head_dim)


class AXK2TopkRouter(DeepseekV32TopkRouter):
    """DeepSeek-V3 style sigmoid top-k router, without expert grouping (A.X-K2 uses `n_group=None`)."""

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
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


class AXK2Attention(DeepseekV3Attention):
    """DeepSeek-V3 MLA + the SGA indexer + an input-dependent sigmoid gate on the attention output.

    The output gate (`linear_gate`) is stored fused into `q_b_proj` in the vLLM-style released
    checkpoint (`q_b_proj` takes `[q_post | q_pre]` and emits `[q | gate]` per head); the weight
    converter splits that block-diagonal matrix back into `q_b_proj` (q) + `linear_gate` (gate) at load.
    """

    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = AXK2Indexer(config, layer_idx)
        self.linear_gate = nn.Linear(config.q_lora_rank, self.num_heads * self.v_head_dim, bias=False)

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
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # `q_compressed` (pre-norm) feeds the indexer and the output gate; `q_resid` (post-norm) feeds
        # the main query projection.
        q_compressed = self.q_a_proj(hidden_states)
        q_resid = self.q_a_layernorm(q_compressed)
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

        # SGA: the indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`.
        indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
        topk_indices = self.indexer(
            hidden_states,
            q_compressed,
            position_embeddings,
            indexer_mask,
            position_ids,
            past_key_values=past_key_values,
        )

        # Fold the indexer top-k into an additive sparse mask; eager and SDPA both consume it.
        index_mask = (
            topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
            .scatter(-1, topk_indices.long(), False)
            .unsqueeze(1)
        )  # [B, 1, S, T]; True == masked out
        if attention_mask is None:
            key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
            index_mask = index_mask | (key_positions[None, None, None, :] > position_ids[:, None, :, None])
            attention_mask = hidden_states.new_zeros((batch_size, 1, seq_length, key_states.shape[2]))
        attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)

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

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        # Input-dependent sigmoid gate on the attention output.
        gate = self.linear_gate(q_compressed)
        attn_output = (attn_output * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AXK2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = AXK2Attention(config=config, layer_idx=layer_idx)

        is_moe_layer = config.mlp_layer_types[layer_idx] == "sparse"
        self.mlp = AXK2MoE(config) if is_moe_layer else AXK2MLP(config)

        # A.X-K2 gates the norms: `input_layernorm` on every layer, `post_attention_layernorm` on MoE layers.
        self.input_layernorm = AXK2GatedRMSNorm(config.hidden_size, rank=config.gated_norm_rank, eps=config.rms_norm_eps)
        if is_moe_layer:
            self.post_attention_layernorm = AXK2GatedRMSNorm(
                config.hidden_size, rank=config.gated_norm_rank, eps=config.rms_norm_eps
            )
        else:
            self.post_attention_layernorm = AXK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AXK2PreTrainedModel(LlamaPreTrainedModel):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq"]
    # The SGA sparse mask is an explicit additive bias, supported by eager and SDPA (not flash / flex).
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, AXK2TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, AXK2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class AXK2Model(LlamaModel):
    pass


class AXK2ForCausalLM(LlamaForCausalLM):
    pass


class AXK2ForSequenceClassification(GenericForSequenceClassification, AXK2PreTrainedModel):
    pass


class AXK2ForTokenClassification(GenericForTokenClassification, AXK2PreTrainedModel):
    pass


__all__ = [
    "AXK2PreTrainedModel",
    "AXK2Model",
    "AXK2ForCausalLM",
    "AXK2ForSequenceClassification",
    "AXK2ForTokenClassification",
]
