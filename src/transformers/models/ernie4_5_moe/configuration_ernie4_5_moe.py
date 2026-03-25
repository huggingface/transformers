# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""Ernie 4.5 MoE model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="baidu/ERNIE-4.5-21B-A3B-PT")
@strict
class Ernie4_5_MoeConfig(PreTrainedConfig):
    r"""
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in any of the projections including mlp and attention for example.
    moe_k (`int`, *optional*, defaults to 6):
        Number of selected experts.
    moe_num_experts (`int`, *optional*, defaults to 64):
        Number of routed experts.
    moe_num_shared_experts (`int`, *optional*, defaults to 2):
        The number of experts that are shared for all MoE forwards.
    moe_layer_start_index (`int`, *optional*, defaults to 1):
        The first index at which MoE layers start to appear.
    moe_layer_end_index (`int`, *optional*, defaults to -1):
        The last possible index for a MoE layer.
    moe_layer_interval (`int`, *optional*, defaults to 1):
        The intervals between MoE layers to appear.
    moe_norm_min (`float`, *optional*, defaults to 1e-12):
        Minimum division value during routing normalization.

    Example:

    ```python
    >>> from transformers import Ernie4_5_MoeModel, Ernie4_5_MoEConfig

    >>> # Initializing a Ernie4_5_MoE style configuration
    >>> configuration = Ernie4_5_MoEConfig()

    >>> # Initializing a model from the ERNIE-4.5-21B-A3B style configuration
    >>> model = Ernie4_5_MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie4_5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_experts": "moe_num_experts", "num_experts_per_tok": "moe_k"}
    default_theta = 500000.0

    # Default tensor parallel plan for base model `Ernie4_5_MoE`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
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

    vocab_size: int = 103424
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    hidden_size: int = 2560
    intermediate_size: int = 12288
    num_hidden_layers: int = 28
    num_attention_heads: int = 20
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    use_bias: int | None = False
    moe_intermediate_size: int = 1536
    moe_k: int | None = 6
    moe_num_experts: int | None = 64
    moe_num_shared_experts: int | None = 2
    moe_layer_start_index: int | None = 1
    moe_layer_end_index: int | None = -1
    moe_layer_interval: int | None = 1
    moe_norm_min: float | None = 1e-12
    output_router_logits: bool | None = False
    router_aux_loss_coef: float | None = 0.001

    def __post_init__(self, **kwargs):
        self.moe_layer_end_index = (
            self.num_hidden_layers - 1 if self.moe_layer_end_index == -1 else self.moe_layer_end_index
        )
        super().__post_init__(**kwargs)


__all__ = ["Ernie4_5_MoeConfig"]
