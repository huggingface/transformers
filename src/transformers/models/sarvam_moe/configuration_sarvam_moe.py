# Copyright 2025 Sarvam AI and the HuggingFace Inc. team. All rights reserved.
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
"""SarvamMoe model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="sarvamai/sarvam-30b-fp8")
@strict
class SarvamMoeConfig(PreTrainedConfig):
    r"""
    num_experts (`int`, *optional*, defaults to 128):
        Number of routed experts.
    num_shared_experts (`int`, *optional*, defaults to 1):
        Number of shared experts.
    num_experts_per_tok (`int`, *optional*, defaults to 6):
        Number of experts activated per token.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for group-limited expert routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of top groups to select during routing.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of initial layers that use dense MLP instead of MoE.
    norm_topk_prob (`bool`, *optional*, defaults to `True`):
        Whether to normalize the top-k routing probabilities.
    routed_scaling_factor (`float`, *optional*, defaults to 2.5):
        Scaling factor applied to routed expert outputs.
    moe_intermediate_size (`int`, *optional*, defaults to 1024):
        Intermediate size of MoE expert MLPs.
    output_router_logits (`bool`, *optional*, defaults to `False`):
        Whether to output router logits.
    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
        Coefficient for the auxiliary router loss.

    Example:

    ```python
    >>> from transformers import SarvamMoeModel, SarvamMoeConfig

    >>> # Initializing a SarvamMoe style configuration
    >>> configuration = SarvamMoeConfig()

    >>> # Initializing a model from the configuration
    >>> model = SarvamMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "sarvam_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_local_experts": "num_experts",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "num_shared_experts",
    }

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

    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 8192
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 19
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    sliding_window: int | None = None
    num_experts: int = 128
    num_shared_experts: int = 1
    num_experts_per_tok: int = 6
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 1

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["SarvamMoeConfig"]
