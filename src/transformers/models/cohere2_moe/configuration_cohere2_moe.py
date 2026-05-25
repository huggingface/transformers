# Copyright 2026 Cohere Inc. HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="CohereLabs/command-a-plus-05-2026")
@strict
class Cohere2MoeConfig(PreTrainedConfig):
    r"""
    logit_scale (`float`, *optional*, defaults to 0.0625):
        The scaling factor for the output logits.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings.
    num_experts_per_tok (`int`, *optional*, defaults to 2):
        Number of selected experts per token.
    num_experts (`int`, *optional*, defaults to 8):
        Number of routed experts.
    num_shared_experts (`int`, *optional*, defaults to 0):
        The number of shared experts.
    shared_expert_combination_strategy (`str`, *optional*, defaults to `"average"`):
        The combination strategy of shared expert, must be one of ['average', 'sum'].
    expert_selection_fn (`str`, *optional*, defaults to `"softmax"`):
        Expert selection function of router.
    layer_types (`list`, *optional*):
        Attention pattern for each layer.
    first_k_dense_replace (`int`, *optional*, defaults to 0):
        Number of dense layers before MoE layers.
    prefix_dense_sliding_window_pattern (`int`, *optional*, defaults to 1):
        Sliding window pattern for the prefix dense layers.
    norm_topk_prob (`bool`, *optional*, defaults to `True`):
        Whether to normalize the top-k expert probabilities when sigmoid is used.
    prefix_dense_intermediate_size (`int`, *optional*):
        Intermediate dimension of the dense prefix layers.
    rms_norm_eps (`float`, *optional*):
        The epsilon used by the RMS normalization layers.
    sliding_window_pattern (`int`, *optional*, defaults to 4):
        Sliding window pattern for the layers.

    ```python
    >>> from transformers import Cohere2MoeModel, Cohere2MoeConfig

    >>> # Initializing a Cohere2Moe model configuration
    >>> configuration = Cohere2MoeConfig()

    >>> # Initializing a model from the Cohere2Moe configuration
    >>> model = Cohere2MoeModel(configuration) # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config # doctest: +SKIP
    ```
    """

    model_type = "cohere2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_allreduce",
        # Depending on layers, `mlp` can be a MoE or an MLP layer
        "layers.*.mlp.experts": "moe_experts_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_allreduce",
    }
    # TP + Sequence Parallelism plan (for training).
    base_model_sp_plan = {
        "embed_tokens": "vocab_reduce_scatter",
        "layers.*.input_layernorm": "activation",
        "layers.*.self_attn": "module_allgather_hidden_states",
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_reduce_scatter",
        # Depending on layers, `mlp` can be a MoE or an MLP layer - so they are updated by `_update_sp_plan` to avoid collisions
        "norm": "activation",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    # FSDP2 plan (see Qwen3Config.base_model_fsdp_plan for shape rationale).
    base_model_fsdp_plan = {
        "embed_tokens": "free_full_weight",
        "layers.*": "free_full_weight",
        "norm": "keep_full_weight",
    }

    vocab_size: int = 256000
    hidden_size: int = 8192
    intermediate_size: int = 22528
    logit_scale: float = 0.0625
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 5
    eos_token_id: int | list[int] | None = 255001
    tie_word_embeddings: bool = True
    rope_theta: float | int = 10000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int | None = 4096
    num_experts_per_tok: int = 2
    num_experts: int = 8
    num_shared_experts: int = 0
    shared_expert_combination_strategy: str = "average"
    expert_selection_fn: str = "softmax"
    layer_types: list[str] | None = None
    first_k_dense_replace: int = 0
    prefix_dense_sliding_window_pattern: int = 1
    norm_topk_prob: bool = True
    prefix_dense_intermediate_size: int | None = None
    rms_norm_eps: float | None = None
    sliding_window_pattern: int = 4

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        self.validate_rope()

        if self.layer_types is None:
            # The first k dense layers (MLP instead of MoE) do not use the same sliding window pattern as the MoE layers
            prefix_layers = [
                "sliding_attention" if ((i + 1) % self.prefix_dense_sliding_window_pattern) != 0 else "full_attention"
                for i in range(self.first_k_dense_replace)
            ]
            rest_layers = [
                "sliding_attention" if ((i + 1) % self.sliding_window_pattern) != 0 else "full_attention"
                for i in range(self.num_hidden_layers - self.first_k_dense_replace)
            ]
            self.layer_types = prefix_layers + rest_layers

        self.validate_layer_type()

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"The length of layer_types ({len(self.layer_types)}) does not match "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )

        super().__post_init__(**kwargs)
        self._update_sp_plan()

    def _update_sp_plan(self):
        self.base_model_sp_plan = self.base_model_sp_plan.copy()
        for i in range(self.num_hidden_layers):
            if i < self.first_k_dense_replace:
                self.base_model_sp_plan.update(
                    {
                        f"layers.{i}.mlp": "module_allgather_split",
                        f"layers.{i}.mlp.experts": "moe_experts_allreduce",
                    }
                )
            else:
                self.base_model_sp_plan.update(
                    {
                        f"layers.{i}.mlp": "module_allgather",
                        f"layers.{i}.mlp.gate_proj": "colwise",
                        f"layers.{i}.mlp.up_proj": "colwise",
                        f"layers.{i}.mlp.down_proj": "rowwise_reduce_scatter",
                    }
                )


__all__ = ["Cohere2MoeConfig"]
