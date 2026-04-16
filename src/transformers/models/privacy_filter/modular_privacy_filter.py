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

import math
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...masking_utils import create_bidirectional_sliding_window_mask
from ...modeling_outputs import BaseModelOutputWithPast, TokenClassifierOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.output_capturing import OutputRecorder
from ..gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssMLP,
    GptOssPreTrainedModel,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssTopKRouter,
    apply_rotary_pos_emb,
)
from .configuration_privacy_filter import PrivacyFilterConfig


logger = logging.get_logger(__name__)


class PrivacyFilterRMSNorm(GptOssRMSNorm):
    pass


class PrivacyFilterTopKRouter(GptOssTopKRouter):
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.float()
        return super().forward(hidden_states)


class PrivacyFilterRotaryEmbedding(GptOssRotaryEmbedding):
    pass


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = x[..., ::2], x[..., 1::2]
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.stack((first_, second_), dim=-1).flatten(-2)


def _batched_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, num_experts, input_dim = x.shape
    output_dim = weight.shape[-1]
    out = torch.bmm(x.reshape(batch_size * num_experts, 1, input_dim), weight.reshape(-1, input_dim, output_dim))
    out = out.reshape(batch_size, num_experts, output_dim)
    if bias is not None:
        out = out + bias
    return out


def _local_bidirectional_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    window_radius: int,
    device: torch.device,
) -> torch.Tensor:
    window = 2 * window_radius + 1
    relative_positions = torch.arange(window, device=device) - window_radius
    key_positions = torch.arange(sequence_length, device=device)[:, None] + relative_positions[None, :]
    valid_positions = (key_positions >= 0) & (key_positions < sequence_length)

    if attention_mask is None:
        return valid_positions.unsqueeze(0).expand(batch_size, -1, -1)

    if attention_mask.dim() == 3:
        return attention_mask.to(device=device, dtype=torch.bool)
    if attention_mask.dim() != 4:
        raise ValueError("Privacy Filter attention expects a 4D additive mask or a 3D local attention mask.")

    if attention_mask.dtype == torch.bool:
        full_attention_mask = attention_mask[:, 0].to(device=device)
    else:
        full_attention_mask = attention_mask[:, 0].to(device=device) == 0

    padded_attention_mask = F.pad(full_attention_mask, (window_radius, window_radius), value=False)
    window_attention_mask = padded_attention_mask.unfold(-1, window, 1)
    token_positions = torch.arange(sequence_length, device=device)
    return window_attention_mask[:, token_positions, token_positions, :] & valid_positions.unsqueeze(0)


def _local_bidirectional_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    window_radius: int,
) -> torch.Tensor:
    batch_size, num_tokens, num_key_value_heads, num_query_groups, head_dim = query.shape
    window = 2 * window_radius + 1

    attention_mask = _local_bidirectional_attention_mask(
        attention_mask,
        batch_size=batch_size,
        sequence_length=num_tokens,
        window_radius=window_radius,
        device=query.device,
    )
    padded_key = F.pad(key, (0, 0, 0, 0, window_radius, window_radius))
    padded_value = F.pad(value, (0, 0, 0, 0, window_radius, window_radius))
    key_window = padded_key.unfold(1, window, 1).permute(0, 1, 4, 2, 3)
    value_window = padded_value.unfold(1, window, 1).permute(0, 1, 4, 2, 3)

    scores = torch.einsum("bthqd,btwhd->bthqw", query, key_window)
    scores = scores.float()
    scores = scores.masked_fill(~attention_mask[:, :, None, None, :], -float("inf"))

    sink_scores = (sinks * math.log(2.0)).reshape(num_key_value_heads, num_query_groups)
    sink_scores = sink_scores[None, None, :, :, None].expand(batch_size, num_tokens, -1, -1, 1)
    scores = torch.cat([scores, sink_scores], dim=-1)

    weights = torch.softmax(scores, dim=-1)[..., :-1].to(value.dtype)
    attn_output = torch.einsum("bthqw,btwhd->bthqd", weights, value_window)
    return attn_output.reshape(batch_size, num_tokens, num_key_value_heads * num_query_groups * head_dim)


class PrivacyFilterAttention(GptOssAttention):
    def __init__(self, config: PrivacyFilterConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_query_groups = config.num_attention_heads // config.num_key_value_heads
        self.window_radius = int(config.bidirectional_left_context)
        self.scaling = config.head_dim**-0.25
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        original_dtype = hidden_states.dtype
        if hidden_states.dtype != self.q_proj.weight.dtype:
            hidden_states = hidden_states.to(self.q_proj.weight.dtype)

        batch_size, sequence_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        query_states = (query_states * self.scaling).view(
            batch_size, sequence_length, self.num_key_value_heads, self.num_query_groups, self.head_dim
        )
        key_states = key_states * self.scaling

        attn_output = _local_bidirectional_attention(
            query_states,
            key_states,
            value_states,
            self.sinks,
            attention_mask,
            window_radius=self.window_radius,
        )

        if attn_output.dtype != self.o_proj.weight.dtype:
            attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = F.linear(attn_output, self.o_proj.weight, self.o_proj.bias)
        return attn_output.to(original_dtype), None


class PrivacyFilterExperts(GptOssExperts):
    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return (gate * torch.sigmoid(self.alpha * gate)) * (up + 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        *,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        outputs = []
        effective_chunk_size = chunk_size if chunk_size > 0 else hidden_states.shape[0]
        for start in range(0, hidden_states.shape[0], effective_chunk_size):
            end = start + effective_chunk_size
            hidden_chunk = hidden_states[start:end]
            indices_chunk = expert_indices[start:end]
            weights_chunk = expert_weights[start:end]

            gate_up_weight = self.gate_up_proj[indices_chunk, ...].float()
            gate_up_bias = self.gate_up_proj_bias[indices_chunk, ...].float()
            hidden_expanded = hidden_chunk.float().unsqueeze(1).expand(-1, indices_chunk.shape[1], -1)
            hidden_chunk = _batched_linear(hidden_expanded, gate_up_weight, gate_up_bias)
            hidden_chunk = self._apply_gate(hidden_chunk)

            down_weight = self.down_proj[indices_chunk, ...].float()
            down_bias = self.down_proj_bias[indices_chunk, ...].float()
            hidden_chunk = _batched_linear(hidden_chunk.float(), down_weight, down_bias)

            if hidden_chunk.dtype != weights_chunk.dtype:
                hidden_chunk = hidden_chunk.to(weights_chunk.dtype)
            hidden_chunk = torch.einsum("bec,be->bc", hidden_chunk, weights_chunk)
            hidden_chunk = hidden_chunk * expert_indices.shape[1]
            outputs.append(hidden_chunk.to(hidden_states.dtype))
        return torch.cat(outputs, dim=0)


class PrivacyFilterMLP(GptOssMLP):
    def __init__(self, config: PrivacyFilterConfig):
        super().__init__(config)
        self.router = PrivacyFilterTopKRouter(config)
        self.experts = PrivacyFilterExperts(config)


class PrivacyFilterEncoderLayer(GptOssDecoderLayer):
    def __init__(self, config: PrivacyFilterConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = PrivacyFilterAttention(config, layer_idx)
        self.mlp = PrivacyFilterMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        return residual + hidden_states


class PrivacyFilterPreTrainedModel(GptOssPreTrainedModel):
    config: PrivacyFilterConfig
    _no_split_modules = ["PrivacyFilterEncoderLayer"]
    _skip_keys_device_placement = None  # No cache
    _keep_in_fp32_modules = ["norm", "embedding_norm", "input_layernorm", "post_attention_layernorm"]
    _keep_in_fp32_modules_strict: list[str] = ["sinks", "router"]
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(PrivacyFilterTopKRouter, index=0),
        "hidden_states": PrivacyFilterEncoderLayer,
        "attentions": PrivacyFilterAttention,
    }
    _compatible_flash_implementations = None

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # Modular conversion inherits GptOssExperts' decorator, but Privacy Filter's expert forward is checkpoint-specific.
        return False

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        if requested_experts not in (None, "eager"):
            raise ValueError("Privacy Filter only supports the eager experts implementation.")
        return "eager"

    def set_use_kernels(self, use_kernels, kernel_config=None):
        if use_kernels:
            raise ValueError("Privacy Filter does not support kernelized layers.")
        PreTrainedModel.set_use_kernels(self, use_kernels, kernel_config)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, PrivacyFilterTopKRouter):
            init.zeros_(module.bias)
        elif isinstance(module, PrivacyFilterRotaryEmbedding):
            rope_init_fn = module.compute_default_rope_parameters
            if module.rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type]
            inv_freq, module.attention_scaling = rope_init_fn(module.config, module.inv_freq.device)
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)


@auto_docstring
class PrivacyFilterModel(PrivacyFilterPreTrainedModel):
    def __init__(self, config: PrivacyFilterConfig):
        super().__init__(config)
        self.padding_idx = (
            config.pad_token_id
            if config.pad_token_id is not None and config.pad_token_id < config.vocab_size
            else None
        )
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embedding_norm = PrivacyFilterRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [PrivacyFilterEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PrivacyFilterRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PrivacyFilterRotaryEmbedding(config=config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None or use_cache:
            raise ValueError("Privacy Filter is a bidirectional encoder and does not support key/value caching.")
        if output_attentions:
            logger.warning_once("Privacy Filter does not return attention weights.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.embedding_norm(inputs_embeds)
        batch_size, sequence_length, _ = hidden_states.shape

        if not isinstance(attention_mask_mapping := attention_mask, dict):
            window_radius = self.config.bidirectional_left_context
            if window_radius != self.config.bidirectional_right_context:
                raise ValueError(
                    "Privacy Filter only supports symmetric bidirectional context with the shared mask API."
                )
            if self.config.sliding_window != 2 * window_radius + 1:
                raise ValueError(
                    "`sliding_window` must equal `2 * bidirectional_left_context + 1` for Privacy Filter checkpoints."
                )
            mask_config = copy(self.config)
            mask_config.sliding_window = window_radius
            mask_kwargs = {
                "config": mask_config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {"sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs)}

        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask_mapping[self.config.layer_types[i]],
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


@auto_docstring
class PrivacyFilterForTokenClassification(PrivacyFilterPreTrainedModel):
    def __init__(self, config: PrivacyFilterConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PrivacyFilterModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["PrivacyFilterForTokenClassification", "PrivacyFilterModel", "PrivacyFilterPreTrainedModel"]
