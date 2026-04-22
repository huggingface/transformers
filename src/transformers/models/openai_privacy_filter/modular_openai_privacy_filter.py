# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""PyTorch Privacy Filter model."""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_sliding_window_mask
from ...modeling_layers import GenericForTokenClassification
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..gpt_oss.configuration_gpt_oss import GptOssConfig
from ..gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssModel,
    GptOssPreTrainedModel,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssTopKRouter,
    apply_rotary_pos_emb,
    repeat_kv,
)


logger = logging.get_logger(__name__)


OPENAI_PRIVACY_FILTER_SPAN_LABELS = (
    "O",
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)

OPENAI_PRIVACY_FILTER_NER_LABELS = ("O",) + tuple(
    f"{prefix}-{label}"
    for label in OPENAI_PRIVACY_FILTER_SPAN_LABELS
    if label != "O"
    for prefix in ("B", "I", "E", "S")
)


@auto_docstring(checkpoint="openai/privacy-filter")
@strict
class OpenAIPrivacyFilterConfig(GptOssConfig):
    model_type = "openai_privacy_filter"
    vocab_size: int = 200064
    hidden_size: int = 640
    intermediate_size: int = 640
    num_hidden_layers: int = 8
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    sliding_window: int = 128
    classifier_dropout: float = 0.0
    pad_token_id: int | None = 199999
    eos_token_id: int | list[int] | None = 199999
    layer_types = AttributeError()  # SWA only
    hidden_act = AttributeError()  # Not used as it's expert MLPs only

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": 4096,
            }

        requested_num_labels = kwargs.pop("num_labels", len(OPENAI_PRIVACY_FILTER_NER_LABELS))
        if self.id2label is None and requested_num_labels == len(OPENAI_PRIVACY_FILTER_NER_LABELS):
            self.id2label = dict(enumerate(OPENAI_PRIVACY_FILTER_NER_LABELS))
        elif self.id2label is None:
            self.num_labels = requested_num_labels
        if self.label2id is None and self.id2label is not None:
            self.label2id = {label: idx for idx, label in self.id2label.items()}

        PreTrainedConfig.__post_init__(self, **kwargs)


class OpenAIPrivacyFilterRMSNorm(GptOssRMSNorm):
    pass


class OpenAIPrivacyFilterRotaryEmbedding(GptOssRotaryEmbedding):
    pass


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # Interleaving layout instead of concatenated
    first_half, second_half = x[..., ::2], x[..., 1::2]
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.stack((first_, second_), dim=-1).flatten(-2)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = nn.functional.softmax(combined_logits, dim=-1, dtype=torch.float32)  # Softmax in fp32
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class OpenAIPrivacyFilterAttention(GptOssAttention):
    def __init__(self, config: OpenAIPrivacyFilterConfig):
        super().__init__(config)
        del self.layer_idx  # Only for caching
        del self.layer_type  # SWA only
        self.is_causal = False
        self.sliding_window = config.sliding_window + 1  # Account for FA symmetry using -1
        self.scaling = config.head_dim**-0.25

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Unique: applying scale individually to each Q and K
        query_states = query_states * self.scaling
        key_states = key_states * self.scaling

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=1.0,  # scaling applied before
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OpenAIPrivacyFilterExperts(GptOssExperts):
    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Concatenated layout instead of interleaving
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        original_dtype = hidden_states.dtype

        # Accumulate over fp32
        next_states = torch.zeros_like(hidden_states, dtype=torch.float32, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=self.num_experts
            )  # masking is also a class
            expert_mask = expert_mask.permute(2, 1, 0)
            # we sum on the top_k and on the sequence length to get which experts
            # are hit this time around
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Key change to original gpt oss is to stay in fp32 precision for all linear projections / muls
        for expert_idx in expert_hit:
            # expert_idx only have 1 element, so we can use scale for fast indexing
            expert_idx = expert_idx[0]
            # skip masking index
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = (
                current_state.float() @ self.gate_up_proj[expert_idx].float()
                + self.gate_up_proj_bias[expert_idx].float()
            )
            gated_output = self._apply_gate(gate_up).float()
            out = gated_output.float() @ self.down_proj[expert_idx].float() + self.down_proj_bias[expert_idx].float()
            weighted_output = out * routing_weights[token_idx, top_k_pos, None].float()
            next_states.index_add_(0, token_idx, weighted_output)

        return next_states.to(original_dtype)


class OpenAIPrivacyFilterTopKRouter(GptOssTopKRouter):
    def forward(self, hidden_states):
        # Force fp32
        router_logits = F.linear(
            hidden_states.float(), self.weight.float(), self.bias.float()
        )  # (num_tokens, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (num_tokens, top_k)
        router_scores = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        # Additional scaling
        router_scores = router_scores / self.top_k
        return router_logits, router_scores, router_indices


class OpenAIPrivacyFilterMLP(nn.Module):
    """Similar to GPT Oss but with FP32 focus + added experts scaling"""

    def __init__(self, config):
        super().__init__()
        self.router = OpenAIPrivacyFilterTopKRouter(config)
        self.num_experts = config.num_experts_per_tok
        self.experts = OpenAIPrivacyFilterExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        _, router_scores, router_indices = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_indices, router_scores)
        # Additional scaling
        hidden_states = hidden_states * self.num_experts
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores


class OpenAIPrivacyFilterEncoderLayer(GptOssDecoderLayer):
    def __init__(self, config: OpenAIPrivacyFilterConfig):
        super().__init__(config)
        self.self_attn = OpenAIPrivacyFilterAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OpenAIPrivacyFilterPreTrainedModel(GptOssPreTrainedModel):
    config: OpenAIPrivacyFilterConfig
    _no_split_modules = ["OpenAIPrivacyFilterEncoderLayer"]
    _skip_keys_device_placement = None  # No cache
    _keep_in_fp32_modules = []
    _keep_in_fp32_modules_strict = ["sinks"]

    _can_record_outputs = {
        "router_logits": OutputRecorder(OpenAIPrivacyFilterTopKRouter, index=0),
        "hidden_states": OpenAIPrivacyFilterEncoderLayer,
        "attentions": OpenAIPrivacyFilterAttention,
    }

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        """The model is very sensitive to accumulation orders, hence we default to `eager` instead"""
        requested_experts = "eager" if requested_experts is None else requested_experts
        return PreTrainedModel.get_correct_experts_implementation(self, requested_experts)


@auto_docstring
class OpenAIPrivacyFilterModel(GptOssModel):
    def __init__(self, config: OpenAIPrivacyFilterConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([OpenAIPrivacyFilterEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attention_mask = create_bidirectional_sliding_window_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class OpenAIPrivacyFilterForTokenClassification(GenericForTokenClassification, OpenAIPrivacyFilterPreTrainedModel): ...


__all__ = [
    "OpenAIPrivacyFilterForTokenClassification",
    "OpenAIPrivacyFilterModel",
    "OpenAIPrivacyFilterPreTrainedModel",
    "OpenAIPrivacyFilterConfig",
]
