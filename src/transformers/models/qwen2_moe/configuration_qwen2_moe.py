# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen2MoE model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen1.5-MoE-A2.7B")
@strict
class Qwen2MoeConfig(PreTrainedConfig):
    r"""
    decoder_sparse_step (`int`, *optional*, defaults to 1):
        The frequency of the MoE layer.
    shared_expert_intermediate_size (`int`, *optional*, defaults to 5632):
        Intermediate size of the shared expert.
    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
        The aux loss factor for the total loss.
    mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
        Indicate which layers use Qwen2MoeMLP rather than Qwen2MoeSparseMoeBlock
        The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
        If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.

    ```python
    >>> from transformers import Qwen2MoeModel, Qwen2MoeConfig

    >>> # Initializing a Qwen2MoE style configuration
    >>> configuration = Qwen2MoeConfig()

    >>> # Initializing a model from the Qwen1.5-MoE-A2.7B" style configuration
    >>> model = Qwen2MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "qwen2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2Moe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 1408
    shared_expert_intermediate_size: int = 5632
    num_experts_per_tok: int = 4
    num_experts: int = 60
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    qkv_bias: bool = True
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        self.sliding_window = self.sliding_window if self.use_sliding_window else 0
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if bool((i + 1) % 2) and i < self.max_window_layers and self.use_sliding_window
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


__all__ = ["Qwen2MoeConfig"]
