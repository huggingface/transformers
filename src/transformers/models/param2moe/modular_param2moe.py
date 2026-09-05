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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
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
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="bharatgenai/Param2-17B-A2.4B-Thinking")
@strict
class Param2MoeConfig(PreTrainedConfig):
    r"""
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in the shallow layers before switching to MoE layers.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    partial_rotary_factor (`float`, *optional*, defaults to 1.0):
        Fraction of each attention head's dimension to apply rotary position embeddings
        to. A value of 1.0 applies RoPE to the full head dimension.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        Base period (theta) for rotary position embeddings. Larger values extend
        the effective context length.

    Example:

    ```python
    >>> from transformers import Param2MoeModel, Param2MoeConfig
    >>> # Initializing a Param2Moe style configuration
    >>> configuration = Param2MoeConfig()
    >>> # Accessing the model configuration
    >>> model = Param2MoeModel(configuration)
    >>> print(model.config)
    ```
    """

    model_type = "param2moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
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
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_experts": "n_routed_experts",
    }

    vocab_size: int = 128008
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 21
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 3
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    head_dim: int | None = 64
    first_k_dense_replace: int = 1
    n_group: int | None = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    routed_scaling_factor: float = 2.5
    topk_group: int | None = 1
    norm_topk_prob: bool | None = True
    num_experts_per_tok: int | None = 6
    moe_intermediate_size: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float = 1.0
    rope_theta: float = 1000000.0

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


class Param2MoeMLP(DeepseekV3MLP):
    pass


class Param2MoeRMSNorm(DeepseekV3RMSNorm):
    pass


class Param2MoeRotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class Param2MoeExperts(DeepseekV3Experts):
    pass


class Param2MoeTopkRouter(DeepseekV3TopkRouter):
    pass


class Param2MoeMoE(DeepseekV3MoE):
    pass


class Param2MoeAttention(Qwen3MoeAttention):
    pass


class Param2MoeDecoderLayer(DeepseekV3DecoderLayer):
    # `first_k_dense_replace=1` for bharatgenai/Param2-17B-A2.4B-Thinking: layer 0 is dense, the rest are MoE.
    pass


class Param2MoePreTrainedModel(DeepseekV3PreTrainedModel):
    # DeepseekV3 ignores its own `layers.61`; Param2Moe has 21 layers.
    _keys_to_ignore_on_load_unexpected = None


class Param2MoeModel(DeepseekV3Model):
    pass


class Param2MoeForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "Param2MoeConfig",
    "Param2MoePreTrainedModel",
    "Param2MoeModel",
    "Param2MoeForCausalLM",
]
