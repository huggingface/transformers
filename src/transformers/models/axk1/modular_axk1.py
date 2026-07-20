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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_layers import GenericForSequenceClassification, GenericForTokenClassification
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
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
    pass


class AXK1Attention(DeepseekV3Attention):
    pass


class AXK1DecoderLayer(DeepseekV3DecoderLayer):
    """DeepSeek-V3 decoder layer with an extra `post_mlp_layernorm` on the MoE block output."""

    def __init__(self, config: AXK1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # A.X-K1 normalizes the MoE block output before the residual add (dense layers pass through).
        self.post_mlp_layernorm = (
            AXK1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if layer_idx >= config.first_k_dense_replace
            else nn.Identity()
        )

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
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
