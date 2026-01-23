# Copyright 2026 The LG AI Research and HuggingFace Inc. team. All rights reserved.
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
"""LG AI Research EXAONE Lab"""

import torch
import torch.nn as nn

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...utils import is_grouped_mm_available
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE,
    DeepseekV3NaiveMoe,
    DeepseekV3TopkRouter,
)
from ..exaone4.configuration_exaone4 import Exaone4Config
from ..exaone4.modeling_exaone4 import (
    Exaone4Attention,
    Exaone4ForCausalLM,
    Exaone4Model,
    Exaone4PreTrainedModel,
)
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


class ExaoneMoeConfig(Exaone4Config):
    model_type = "exaone_moe"

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_dropout=0.0,
        sliding_window=4096,
        sliding_window_pattern=4,
        layer_types=None,
        mlp_layer_types=None,
        first_k_dense_replace=1,
        moe_intermediate_size=1024,
        num_experts=64,
        num_experts_per_tok=8,
        num_shared_experts=1,
        norm_topk_prob=False,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group

        self.layer_types = layer_types
        if self.sliding_window is None:
            sliding_window_pattern = 0
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if ((i + 1) % (sliding_window_pattern) != 0 and i < self.num_hidden_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)

        PreTrainedConfig.__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


class ExaoneMoeAttention(Exaone4Attention):
    pass


class ExaoneMoeMLP(Qwen2MoeMLP):
    pass


class ExaoneMoeTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config):
        super().__init__()
        del self.n_routed_experts
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_size)))


class ExaoneMoeExperts(DeepseekV3NaiveMoe):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts


class ExaoneMoeSparseMoEBlock(DeepseekV3MoE):
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.num_experts


class ExaoneMoeDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = (
            ExaoneMoeSparseMoEBlock(config) if config.mlp_layer_types[layer_idx] == "sparse" else ExaoneMoeMLP(config)
        )


class ExaoneMoePreTrainedModel(Exaone4PreTrainedModel):
    config: ExaoneMoeConfig

    _can_record_outputs = {
        "hidden_states": ExaoneMoeDecoderLayer,
        "attentions": ExaoneMoeAttention,
        "router_logits": ExaoneMoeSparseMoEBlock,
    }
    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]


class ExaoneMoeModel(Exaone4Model, ExaoneMoePreTrainedModel):
    pass


class ExaoneMoeForCausalLM(Exaone4ForCausalLM):
    pass


__all__ = [
    "ExaoneMoeConfig",
    "ExaoneMoePreTrainedModel",
    "ExaoneMoeModel",
    "ExaoneMoeForCausalLM",
]
