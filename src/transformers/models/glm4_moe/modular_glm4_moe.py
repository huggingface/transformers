# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GLM-4-MOE model."""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache, DynamicLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..cohere.modeling_cohere import CohereAttention
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3TopkRouter,
)
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb  # noqa


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-4.5")
@strict
class Glm4MoeConfig(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/
    num_nextn_predict_layers (`int`, *optional*, defaults to 0):
        Number of Multi-Token Prediction (MTP) modules appended after the base
        transformer. When `0`, the model behaves as a standard decoder. When `>0`,
        each extra module predicts one additional future token at inference time
        (speculative decoding via `generate(..., use_mtp=True)`).

    Example:

    ```python
    >>> from transformers import Glm4MoeModel, Glm4MoeConfig

    >>> # Initializing a Glm4Moe style configuration
    >>> configuration = Glm4MoeConfig()

    >>> # Initializing a model from the GLM-4-MOE-100B-A10B style configuration
    >>> model = Glm4MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Glm4Moe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",  # NOTE(3outeille): This needs to be right after down_proj in the dict. Otherwise, the pattern model.layers.*.mlp.experts will have priority over model.layers.*.mlp.experts.down_proj which will assign a wrong TP plan.
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 10944
    num_hidden_layers: int = 46
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 1408
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    num_nextn_predict_layers: int = 0
    use_qk_norm: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


class Glm4MoeRotaryEmbedding(GlmRotaryEmbedding):
    pass


class Glm4MoeAttention(CohereAttention):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_parameters = config.rope_parameters
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class Glm4MoeMLP(DeepseekV3MLP):
    pass


class Glm4MoeTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config: Glm4MoeConfig):
        nn.Module.__init__(self)
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts), dtype=torch.float32))


class Glm4MoeRMSNorm(DeepseekV3RMSNorm):
    pass


class Glm4MoeDecoderLayer(DeepseekV3DecoderLayer):
    pass


class Glm4MoeMTPSharedHead(nn.Module):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.norm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden_states))


class Glm4MoeMTPLayer(nn.Module):
    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.enorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_block = Glm4MoeDecoderLayer(config, layer_idx)
        self.shared_head = Glm4MoeMTPSharedHead(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.eh_proj(torch.cat([self.enorm(inputs_embeds), self.hnorm(previous_hidden_state)], dim=-1))
        hidden_states = self.mtp_block(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.shared_head(hidden_states)
        return hidden_states, logits


class Glm4MoePreTrainedModel(DeepseekV3PreTrainedModel):
    # GLM-4 MoE ships MTP weights at layer index `num_hidden_layers` — 46 for
    # GLM-4.5-Air, 92 for the larger GLM-4.5 variant. Both are ignored when
    # `num_nextn_predict_layers == 0` (the default).
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.92.*", r"model\.layers\.46.*"]


class Glm4MoeModel(DeepseekV3Model):
    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config)
        for k in range(getattr(config, "num_nextn_predict_layers", 0)):
            self.layers.append(Glm4MoeMTPLayer(config, config.num_hidden_layers + k))
        self.post_init()

    def forward_mtp(
        self,
        input_ids: torch.LongTensor,
        previous_hidden_state: torch.Tensor,
        past_key_values: Cache,
        position_ids: torch.LongTensor | None = None,
        mtp_depth: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MTP depth. Returns `(hidden_state, logits)` for position t+depth+2."""
        layer_idx = self.config.num_hidden_layers + mtp_depth
        mtp_layer = self.layers[layer_idx]
        if hasattr(past_key_values, "layers"):
            while len(past_key_values.layers) <= layer_idx:
                past_key_values.layers.append(DynamicLayer())
        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length(layer_idx)
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            ).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        return mtp_layer(
            inputs_embeds=inputs_embeds,
            previous_hidden_state=previous_hidden_state,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs,
        )


class Glm4MoeForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "Glm4MoeConfig",
    "Glm4MoePreTrainedModel",
    "Glm4MoeModel",
    "Glm4MoeForCausalLM",
]
