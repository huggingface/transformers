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

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2Experts,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2RMSNorm,
    MiniMaxM2RotaryEmbedding,
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
        "layers.*.mlp.gate": "colwise_rep",
        "layers.*.mlp.experts.*.w1": "colwise",
        "layers.*.mlp.experts.*.w2": "rowwise",
        "layers.*.mlp.experts.*.w3": "colwise",
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
    rotary_dim: int | None = None
    rope_parameters: RopeParameters | dict | None = None

    def convert_rope_params_to_dict(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rotary_dim is not None and "partial_rotary_factor" not in kwargs:
            kwargs["partial_rotary_factor"] = self.rotary_dim / self.head_dim
        return super().convert_rope_params_to_dict(**kwargs)


class MiniMaxM27Experts(MiniMaxM2Experts):
    pass


class MiniMaxM27SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM27Experts(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states.to(self.gate.weight.dtype))
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(
            hidden_states.to(self.experts.gate_up_proj.dtype),
            top_k_index,
            top_k_weights.to(self.experts.gate_up_proj.dtype),
        )
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)


class MiniMaxM27RMSNorm(MiniMaxM2RMSNorm):
    pass


class MiniMaxM27RotaryEmbedding(MiniMaxM2RotaryEmbedding):
    pass


class MiniMaxM27Attention(MiniMaxM2Attention):
    pass


class MiniMaxM27DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiniMaxM27Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MiniMaxM27Attention(config, layer_idx)

        self.mlp = MiniMaxM27SparseMoeBlock(config)
        self.input_layernorm = MiniMaxM27RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM27RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class MiniMaxM27PreTrainedModel(PreTrainedModel):
    config: MiniMaxM27Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM27DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(MiniMaxM27SparseMoeBlock, index=1),
        "hidden_states": MiniMaxM27DecoderLayer,
        "attentions": MiniMaxM27Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, MiniMaxM27Experts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM27SparseMoeBlock):
            nn.init.zeros_(module.e_score_correction_bias)


class MiniMaxM27Model(MiniMaxM2Model):
    pass


class MiniMaxM27ForCausalLM(MiniMaxM2ForCausalLM):
    _tp_plan = {"lm_head": "colwise_rep"}


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
