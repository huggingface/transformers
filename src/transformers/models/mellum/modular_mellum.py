# Copyright 2026 JetBrains and the HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    RopeParameters,
)
from ...utils import auto_docstring, logging
from ..laguna.modeling_laguna import LagunaDecoderLayer, LagunaModel, LagunaRotaryEmbedding
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeForCausalLM,
    Qwen3MoePreTrainedModel,
    Qwen3MoeSparseMoeBlock,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="JetBrains/Mellum2-12B-A2.5B-Base")
@strict
class MellumConfig(Qwen3MoeConfig):
    r"""
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type — `"dense"` or `"sparse"`. Length must equal
        `num_hidden_layers`. Defaults to all sparse.

    ```python
    >>> from transformers import MellumModel, MellumConfig

    >>> configuration = MellumConfig()
    >>> model = MellumModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "mellum"

    # Per-layer MLP entries are filled in by `_update_sp_plan` (dense vs sparse).
    base_model_sp_plan = {
        "embed_tokens": "vocab_reduce_scatter",
        "layers.*.input_layernorm": "activation",
        "layers.*.self_attn": "module_allgather_hidden_states",
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_reduce_scatter",
        "layers.*.self_attn.q_norm": "activation_seq_dim_2",
        "layers.*.self_attn.k_norm": "activation_seq_dim_2",
        "layers.*.post_attention_layernorm": "activation",
        "norm": "activation",
    }
    # Inference TP + EP (per-layer overrides applied in `_update_parallel_plans`).
    base_model_tp_ep_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_allreduce",
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_experts_allreduce",
    }
    # Training SP + EP (per-layer overrides applied in `_update_parallel_plans`).
    base_model_sp_ep_plan = {
        "embed_tokens": "vocab_reduce_scatter",
        "layers.*.input_layernorm": "activation",
        "layers.*.self_attn": "module_allgather_hidden_states",
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_reduce_scatter",
        "layers.*.self_attn.q_norm": "activation_seq_dim_2",
        "layers.*.self_attn.k_norm": "activation_seq_dim_2",
        "layers.*.post_attention_layernorm": "activation",
        "layers.*.mlp": "module_allgather_split",
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_experts_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_reduce_scatter",
        "norm": "activation",
    }

    vocab_size: int = 98304
    hidden_size: int = 2304
    intermediate_size: int = 7168
    num_hidden_layers: int = 28
    head_dim: int = 128
    max_position_embeddings: int = 131072
    sliding_window: int | None = 1024
    num_experts: int = 64
    moe_intermediate_size: int = 896
    norm_topk_prob: bool = True
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    rope_parameters: dict | RopeParameters | None = None

    use_sliding_window = AttributeError()
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers

        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            }

        self._update_parallel_plans()

        PreTrainedConfig.__post_init__(
            self,
            **kwargs,
            ignore_keys_at_rope_validation={"sliding_attention", "full_attention"},
        )

    def _is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    def _update_parallel_plans(self):
        self._update_sp_plan()
        self._update_sp_ep_plan()
        self._update_tp_ep_plan()

    def _update_sp_plan(self):
        """Set per-layer SP entries depending on whether the MLP is dense or sparse."""
        self.base_model_sp_plan = self.base_model_sp_plan.copy()
        for i, mlp_type in enumerate(self.mlp_layer_types):
            if mlp_type == "dense":
                self.base_model_sp_plan.update(
                    {
                        f"layers.{i}.mlp": "module_allgather",
                        f"layers.{i}.mlp.gate_proj": "colwise",
                        f"layers.{i}.mlp.up_proj": "colwise",
                        f"layers.{i}.mlp.down_proj": "rowwise_reduce_scatter",
                    }
                )
            else:
                self.base_model_sp_plan.update(
                    {
                        f"layers.{i}.mlp": "module_allgather_split",
                        f"layers.{i}.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",
                        f"layers.{i}.mlp.experts.down_proj": "moe_tp_down_rowwise",
                        f"layers.{i}.mlp.experts": "moe_experts_allreduce",
                    }
                )

    def _update_sp_ep_plan(self):
        self.base_model_sp_ep_plan = self.base_model_sp_ep_plan.copy()
        for i, mlp_type in enumerate(self.mlp_layer_types):
            if mlp_type == "dense":
                self.base_model_sp_ep_plan.update(
                    {
                        f"layers.{i}.mlp": "module_allgather",
                        f"layers.{i}.mlp.gate_proj": "colwise",
                        f"layers.{i}.mlp.up_proj": "colwise",
                        f"layers.{i}.mlp.down_proj": "rowwise_reduce_scatter",
                    }
                )
            else:
                self.base_model_sp_ep_plan.update(
                    {
                        f"layers.{i}.mlp": "module_allgather_split",
                        f"layers.{i}.mlp.gate": "ep_router",
                        f"layers.{i}.mlp.experts.gate_up_proj": "grouped_gemm",
                        f"layers.{i}.mlp.experts.down_proj": "grouped_gemm",
                        f"layers.{i}.mlp.experts": "moe_experts_allreduce",
                    }
                )

    def _update_tp_ep_plan(self):
        self.base_model_tp_ep_plan = self.base_model_tp_ep_plan.copy()
        for i, mlp_type in enumerate(self.mlp_layer_types):
            if mlp_type == "dense":
                self.base_model_tp_ep_plan.update(
                    {
                        f"layers.{i}.mlp.gate_proj": "colwise",
                        f"layers.{i}.mlp.up_proj": "colwise",
                        f"layers.{i}.mlp.down_proj": "rowwise_allreduce",
                    }
                )
            else:
                self.base_model_tp_ep_plan.update(
                    {
                        f"layers.{i}.mlp.gate": "ep_router",
                        f"layers.{i}.mlp.experts.gate_up_proj": "grouped_gemm",
                        f"layers.{i}.mlp.experts.down_proj": "grouped_gemm",
                        f"layers.{i}.mlp.experts": "moe_experts_allreduce",
                    }
                )

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs


class MellumRotaryEmbedding(LagunaRotaryEmbedding):
    pass


class MellumAttention(Qwen3MoeAttention):
    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None


class MellumSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    pass


class MellumDecoderLayer(LagunaDecoderLayer):
    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MellumAttention(config, layer_idx)


class MellumPreTrainedModel(Qwen3MoePreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MellumRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


class MellumModel(LagunaModel):
    pass


class MellumForCausalLM(Qwen3MoeForCausalLM):
    pass


__all__ = [
    "MellumConfig",
    "MellumForCausalLM",
    "MellumModel",
    "MellumPreTrainedModel",
]
