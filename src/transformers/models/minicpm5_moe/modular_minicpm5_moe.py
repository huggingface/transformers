# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import field

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import maybe_autocast
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
)
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer
from ..llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward


@auto_docstring(checkpoint="openbmb/MiniCPM5-15B-A2.5B")
@strict
class MiniCPM5MoeConfig(Glm4MoeConfig):
    r"""
    n_shared_experts (`int`, *optional*, defaults to 1):
        Number of expert widths in the always-active shared MLP.
    n_routed_experts (`int`, *optional*, defaults to 160):
        Number of routed experts in each sparse layer.
    routed_scaling_factor (`float`, *optional*, defaults to 3.66):
        Scale applied to the selected expert weights after normalization.
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups used for top-k selection.
    topk_group (`int`, *optional*, defaults to 1):
        Number of expert groups retained during routing.
    mlp_layer_types (`list[str]`, *optional*):
        Feed-forward type for each layer, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by sparse layers.
    """

    model_type = "minicpm5_moe"
    default_theta = 5000000.0

    # NOTE: We intentionally don't shard attention as it only has 2 KV heads (== TP size 2) which is unrealistic in practice
    base_model_tp_plan = {
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 130560
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    head_dim: int | None = 128
    max_position_embeddings: int = 524288
    rms_norm_eps: float = 1e-6

    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = field(default_factory=lambda: [1, 130073])

    mlp_layer_types: list[str] | None = None
    n_routed_experts: int = 160
    routed_scaling_factor: float = 3.66
    num_experts_per_tok: int = 16
    moe_intermediate_size: int = 512

    first_k_dense_replace = AttributeError()
    use_qk_norm = AttributeError()
    num_mtp_layers = AttributeError()

    def __post_init__(self, **kwargs):
        # BC: for old naming convention
        if self.architectures == ["MiniCPM5MoEForCausalLM"]:
            self.architectures = ["MiniCPM5MoeForCausalLM"]

        dense_layers = kwargs.pop("first_k_dense_replace", 1)
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense" if i < dense_layers else "sparse" for i in range(self.num_hidden_layers)]
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        PreTrainedConfig.validate_architecture(self)

        if getattr(self, "use_gated_attention", True) is False:
            raise ValueError(
                "MiniCPM5Moe requires gated attention; checkpoints with `use_gated_attention=False` are unsupported."
            )


class MiniCPM5MoeMLP(DeepseekV3MLP):
    pass


class MiniCPM5MoeExperts(DeepseekV3Experts):
    pass


class MiniCPM5MoeTopkRouter(DeepseekV3TopkRouter):
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        # Key difference of forcing fp32 in all cases
        device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
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
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class MiniCPM5MoeMoE(DeepseekV3MoE):
    pass


class MiniCPM5MoeAttention(LlamaAttention):
    def __init__(self, config: MiniCPM5MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_proj
        self.q_gate_proj = nn.Linear(
            config.hidden_size, 2 * config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_gate_hidden_shape = (*input_shape, -1, 2 * self.head_dim)

        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_gate_states = self.q_gate_proj(hidden_states).view(query_gate_hidden_shape)

        query_states, gate_states = query_gate_states.chunk(chunks=2, dim=-1)
        query_states = query_states.transpose(1, 2)
        gate_states = gate_states.reshape(*input_shape, -1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * gate_states.sigmoid()  # key difference
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniCPM5MoeDecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class MiniCPM5MoeRMSNorm(DeepseekV3RMSNorm):
    pass


class MiniCPM5MoePreTrainedModel(DeepseekV3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = AttributeError()


class MiniCPM5MoeRotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class MiniCPM5MoeModel(DeepseekV3Model):
    pass


class MiniCPM5MoeForCausalLM(DeepseekV3ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer

        >>> checkpoint = "openbmb/MiniCPM5-15B-A2.5B"
        >>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        >>> model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")
        >>> inputs = tokenizer("What is 7 + 8?", return_tensors="pt").to(model.device)
        >>> outputs = model(**inputs, logits_to_keep=1)
        ```
        """
        return super().forward(**super_kwargs)


__all__ = [
    "MiniCPM5MoeConfig",
    "MiniCPM5MoePreTrainedModel",
    "MiniCPM5MoeModel",
    "MiniCPM5MoeForCausalLM",
]
