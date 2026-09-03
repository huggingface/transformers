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
"""PyTorch A.X-K1 model (modular).

A.X-K1 is SK Telecom's Mixture-of-Experts LLM. Architecturally it is DeepSeek-V3 (Multi-head Latent
Attention + grouped sigmoid top-k MoE with a shared expert) with a single SK Telecom modification: an
extra `post_mlp_layernorm` applied to the MoE block output. Everything else — MLA, RoPE, the router,
the experts, and the model / causal-LM scaffolding — is inherited unchanged from DeepSeek-V3.
"""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GenericForSequenceClassification, GenericForTokenClassification
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="skt/A.X-K1")
@strict
class AXK1Config(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 8):
        Number of groups for routed experts.
    topk_group (`int`, *optional*, defaults to 4):
        Number of selected groups per token (each token's experts are drawn from these groups).
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of leading layers that use a dense MLP; the rest use the MoE block.
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to use the interleaved rotary position embedding layout.

    ```python
    >>> from transformers import AXK1Config, AXK1Model

    >>> # Initializing an A.X-K1 style configuration
    >>> configuration = AXK1Config()

    >>> # Initializing a model from the configuration
    >>> model = AXK1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "axk1"
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 192
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    bos_token_id: int | None = 163691
    eos_token_id: int | list[int] | None = 163691

    num_mtp_layers = AttributeError()


class AXK1RMSNorm(DeepseekV3RMSNorm):
    pass


class AXK1RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class AXK1MLP(DeepseekV3MLP):
    pass


class AXK1TopkRouter(DeepseekV3TopkRouter):
    pass


class AXK1Experts(DeepseekV3Experts):
    pass


class AXK1MoE(DeepseekV3MoE):
    """DeepSeek-V3 MoE with an extra `post_mlp_layernorm` on the block output (A.X-K1's single delta).

    The released checkpoints store the norm at the decoder-layer level (`post_mlp_layernorm.*`); the
    checkpoint conversion mapping renames it into `mlp.post_mlp_layernorm.*` at load time.
    """

    def __init__(self, config: AXK1Config):
        super().__init__(config)
        self.post_mlp_layernorm = AXK1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return self.post_mlp_layernorm(hidden_states)


class AXK1Attention(DeepseekV3Attention):
    """Multi-headed Latent Attention (MLA) from Deepseek V3, always with a Q-LoRA rank."""

    def __init__(self, config: AXK1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_proj
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = AXK1RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        # No branching on q_lora_rank being None: unlike Deepseek V3, it is always an int, so we always use Q-LoRA
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

        query_states = torch.cat((q_pass, q_rot), dim=-1)

        key_states, value_states = self.expand_kv(kv_nope, k_rot)

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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AXK1DecoderLayer(DeepseekV3DecoderLayer):
    pass


class AXK1PreTrainedModel(DeepseekV3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["inv_freq"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, AXK1TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, AXK1Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class AXK1Model(DeepseekV3Model):
    pass


class AXK1ForCausalLM(DeepseekV3ForCausalLM):
    pass


class AXK1ForSequenceClassification(GenericForSequenceClassification, AXK1PreTrainedModel):
    pass


class AXK1ForTokenClassification(GenericForTokenClassification, AXK1PreTrainedModel):
    pass


__all__ = [
    "AXK1Config",
    "AXK1PreTrainedModel",
    "AXK1Model",
    "AXK1ForCausalLM",
    "AXK1ForSequenceClassification",
    "AXK1ForTokenClassification",
]
