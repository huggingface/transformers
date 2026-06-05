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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2Experts,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2PreTrainedModel,
    MiniMaxM2RMSNorm,
    MiniMaxM2RotaryEmbedding,
    MiniMaxM2SparseMoeBlock,
    apply_rotary_pos_emb,  # noqa: F401
    eager_attention_forward,  # noqa: F401
    load_balancing_loss_func,  # noqa: F401
    repeat_kv,  # noqa: F401
    rotate_half,  # noqa: F401
)


@auto_docstring
@strict
class MiniMaxM27Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxM27Model`]. It is used to instantiate an
    MiniMaxM27 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of MiniMaxM27-8x7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ```python
    >>> from transformers import MiniMaxM27Model, MiniMaxM27Config

    >>> # Initializing a MiniMaxM27 style configuration
    >>> configuration = MiniMaxM27Config()

    >>> # Initializing a model from the configuration
    >>> model = MiniMaxM27Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minimax_m2_7"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",
        "layers.*.block_sparse_moe.experts.*.w1": "colwise",
        "layers.*.block_sparse_moe.experts.*.w2": "rowwise",
        "layers.*.block_sparse_moe.experts.*.w3": "colwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    default_theta = 5000000.0

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_theta: float | int = 1e6
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    use_qk_norm: bool = False
    rope_parameters: RopeParameters | dict | None = None


class MiniMaxM27MLP(nn.Module):
    def __init__(self, config: MiniMaxM27Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MiniMaxM27Experts(MiniMaxM2Experts):
    def __init__(self, config: MiniMaxM27Config):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        for i in range(self.num_experts):
            self.add_module(str(i), MiniMaxM27MLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_module = self._modules[str(int(expert_idx))]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = expert_module(current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class MiniMaxM27MoEGate(nn.Linear):
    pass


class MiniMaxM27SparseMoeBlock(MiniMaxM2SparseMoeBlock):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = MiniMaxM27MoEGate(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM27Experts(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states.to(self.gate.weight.dtype))
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


class MiniMaxM27RMSNorm(MiniMaxM2RMSNorm):
    pass


class MiniMaxM27RotaryEmbedding(MiniMaxM2RotaryEmbedding):
    pass


class MiniMaxM27Attention(MiniMaxM2Attention):
    def __init__(self, config: MiniMaxM27Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


class MiniMaxM27DecoderLayer(MiniMaxM2DecoderLayer):
    pass


@auto_docstring
class MiniMaxM27PreTrainedModel(MiniMaxM2PreTrainedModel):
    config: MiniMaxM27Config
    _no_split_modules = ["MiniMaxM27DecoderLayer"]
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(MiniMaxM27MoEGate, index=0),
        "hidden_states": MiniMaxM27DecoderLayer,
        "attentions": MiniMaxM27Attention,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)


@auto_docstring
class MiniMaxM27Model(MiniMaxM2Model):
    def __init__(self, config: MiniMaxM27Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniMaxM27DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MiniMaxM27RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiniMaxM27RotaryEmbedding(config=config)


@auto_docstring
class MiniMaxM27ForCausalLM(MiniMaxM2ForCausalLM, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxM27Model(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


class MiniMaxM27ForSequenceClassification(GenericForSequenceClassification, MiniMaxM27PreTrainedModel):
    pass


class MiniMaxM27ForTokenClassification(GenericForTokenClassification, MiniMaxM27PreTrainedModel):
    pass


class MiniMaxM27ForQuestionAnswering(GenericForQuestionAnswering, MiniMaxM27PreTrainedModel):
    pass


__all__ = [
    "MiniMaxM27Config",
    "MiniMaxM27ForCausalLM",
    "MiniMaxM27ForQuestionAnswering",
    "MiniMaxM27ForSequenceClassification",
    "MiniMaxM27ForTokenClassification",
    "MiniMaxM27Model",
    "MiniMaxM27PreTrainedModel",
]
