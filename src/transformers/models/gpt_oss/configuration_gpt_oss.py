# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""openai model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters


class GptOssConfig(PreTrainedConfig):
    r"""
    This will yield a configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    """

    model_type = "gpt_oss"
    default_theta = 150000.0
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.router": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
    }

    def __init__(
        self,
        num_hidden_layers: Optional[int] = 36,
        num_local_experts: Optional[int] = 128,
        vocab_size: Optional[int] = 201088,
        hidden_size: Optional[int] = 2880,
        intermediate_size: Optional[int] = 2880,
        head_dim: Optional[int] = 64,
        num_attention_heads: Optional[int] = 64,
        num_key_value_heads: Optional[int] = 8,
        sliding_window: Optional[int] = 128,
        tie_word_embeddings: Optional[bool] = False,
        hidden_act: Optional[str] = "silu",
        initializer_range: Optional[float] = 0.02,
        max_position_embeddings: Optional[int] = 131072,
        rms_norm_eps: Optional[float] = 1e-5,
        rope_parameters: Optional[RopeParameters] = {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        attention_dropout: Optional[float] = 0.0,
        num_experts_per_tok: Optional[int] = 4,
        router_aux_loss_coef: Optional[float] = 0.9,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_local_experts = num_local_experts
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.attention_bias = True
        self.max_position_embeddings = max_position_embeddings
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def __setattr__(self, key, value):
        """
        Overwritten to allow checking for the proper attention implementation to be used.

        Due to `set_attn_implementation` which internally assigns `_attn_implementation_internal = "..."`, simply overwriting
        the specific attention setter is not enough. Using a property/setter for `_attn_implementation_internal` would result in
        a recursive dependency (as `_attn_implementation` acts as a wrapper around `_attn_implementation_internal`) - hence, this
        workaround.
        """
        if key in ("_attn_implementation", "_attn_implementation_internal"):
            if value and "flash" in value and value.removeprefix("paged|") != "kernels-community/vllm-flash-attn3":
                raise ValueError(
                    f"GPT-OSS model does not support the specified flash attention implementation: {value}. "
                    "Only `kernels-community/vllm-flash-attn3` is supported."
                )
        super().__setattr__(key, value)


__all__ = ["GptOssConfig"]
