# Copyright 2025 the HuggingFace Team. All rights reserved.
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


import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeMLP,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoeNaiveMoe,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding,
    Glm4MoeTopkRouter,
)


@auto_docstring(checkpoint="zai-org/GLM-4.5")
@strict
class Glm4MoeLiteConfig(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to interleave the rotary position embeddings.
    mlp_layer_types (`list`, *optional*):
        MLP (Moe vs Dense) pattern for each layer.

    Example:

    ```python
    >>> from transformers import Glm4MoeLiteModel, Glm4MoeLiteConfig

    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = Glm4MoeLiteConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4_moe_lite"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
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
        "head_dim": "qk_rope_head_dim",
    }

    vocab_size: int = 154880
    hidden_size: int = 2048
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    n_shared_experts: int = 1
    n_routed_experts: int = 64
    routed_scaling_factor: float = 1.8
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 768
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 202752
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    rope_interleave: bool = True
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        # Default to MoE from the second layer and on
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        super().__post_init__(**kwargs)


class Glm4MoeLiteRotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class Glm4MoeLiteAttention(DeepseekV3Attention):
    pass


class Glm4MoeLiteMLP(Glm4MoeMLP):
    pass


class Glm4MoeLiteTopkRouter(Glm4MoeTopkRouter):
    pass


class Glm4MoeLiteRMSNorm(Glm4MoeRMSNorm):
    pass


class Glm4MoeLiteNaiveMoe(Glm4MoeNaiveMoe):
    pass


class Glm4MoeLiteMoE(Glm4MoeMoE):
    pass


class Glm4MoeLiteDecoderLayer(Glm4MoeDecoderLayer, nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4MoeLiteAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm4MoeLiteMoE(config)
        else:
            self.mlp = Glm4MoeLiteMLP(config)

        self.input_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, config.rms_norm_eps)


class Glm4MoeLitePreTrainedModel(Glm4MoePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.47.*"]


class Glm4MoeLiteModel(Glm4MoeModel):
    pass


class Glm4MoeLiteForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "Glm4MoeLiteConfig",
    "Glm4MoeLitePreTrainedModel",
    "Glm4MoeLiteModel",
    "Glm4MoeLiteForCausalLM",
]
