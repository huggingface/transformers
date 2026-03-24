# Copyright 2025 bzantium and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeekV3 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V3)

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
"""DeepSeekV3 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="bzantium/tiny-deepseek-v3")
@strict
class DeepseekV3Config(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 8):
        Number of groups for routed experts.
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to interleave the rotary position embeddings.

    Example:

    ```python
    >>> from transformers import DeepseekV3Model, DeepseekV3Config

    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = DeepseekV3Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
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
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int | None = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = 128
    qk_nope_head_dim: int = 128
    n_group: int | None = 8
    topk_group: int | None = 4
    num_experts_per_tok: int | None = 8
    first_k_dense_replace: int | None = 3
    norm_topk_prob: bool | None = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    rope_interleave: bool | None = True
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.standardize_rope_params()

        # Convert to float because RoPE fn expect a float. Models on the hub were saved as int
        for key in ["beta_fast", "beta_slow", "factor"]:
            if key in self.rope_parameters:
                self.rope_parameters[key] = float(self.rope_parameters[key])
        return kwargs


__all__ = ["DeepseekV3Config"]
