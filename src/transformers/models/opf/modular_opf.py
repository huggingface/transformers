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
"""PyTorch OPF model."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...modeling_outputs import BaseModelOutputWithPast, TokenClassifierOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssMLP,
    GptOssPreTrainedModel,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssTopKRouter,
)
from .configuration_opf import OpfConfig


logger = logging.get_logger(__name__)


class OpfRMSNorm(GptOssRMSNorm):
    pass


class OpfTopKRouter(GptOssTopKRouter):
    pass


class OpfRotaryEmbedding(GptOssRotaryEmbedding):
    pass


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return torch.stack((out_even, out_odd), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(2).to(q.dtype)
    sin = sin.unsqueeze(2).to(q.dtype)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


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


def _local_bidirectional_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    left_context: int,
    right_context: int,
) -> torch.Tensor:
    batch_size, num_tokens, num_key_value_heads, num_query_groups, head_dim = query.shape
    window = left_context + right_context + 1

    padded_key = F.pad(key, (0, 0, 0, 0, left_context, right_context))
    padded_value = F.pad(value, (0, 0, 0, 0, left_context, right_context))
    key_window = padded_key.unfold(1, window, 1).permute(0, 1, 4, 2, 3)
    value_window = padded_value.unfold(1, window, 1).permute(0, 1, 4, 2, 3)

    relative_positions = torch.arange(window, device=query.device) - left_context
    key_positions = torch.arange(num_tokens, device=query.device)[:, None] + relative_positions[None, :]
    valid = (key_positions >= 0) & (key_positions < num_tokens)

    if attention_mask is not None:
        key_mask = attention_mask.to(device=query.device, dtype=torch.bool)
        padded_key_mask = F.pad(key_mask, (left_context, right_context))
        window_key_mask = padded_key_mask.unfold(1, window, 1)
        valid = valid[None, :, :] & window_key_mask
    else:
        valid = valid[None, :, :]

    scores = torch.einsum("bthqd,btwhd->bthqw", query, key_window)
    scores = scores.float()
    scores = scores.masked_fill(~valid[:, :, None, None, :], -float("inf"))

    sink_scores = (sinks * math.log(2.0)).reshape(num_key_value_heads, num_query_groups)
    sink_scores = sink_scores[None, None, :, :, None].expand(batch_size, num_tokens, -1, -1, 1)
    scores = torch.cat([scores, sink_scores], dim=-1)

    weights = torch.softmax(scores, dim=-1)[..., :-1].to(value.dtype)
    attn_output = torch.einsum("bthqw,btwhd->bthqd", weights, value_window)
    return attn_output.reshape(batch_size, num_tokens, num_key_value_heads * num_query_groups * head_dim)


class OpfAttention(GptOssAttention):
    def __init__(self, config: OpfConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_query_groups = config.num_attention_heads // config.num_key_value_heads
        self.left_context = int(config.bidirectional_left_context)
        self.right_context = int(config.bidirectional_right_context)
        self.qk_scale = 1 / math.sqrt(math.sqrt(config.head_dim))
        self.norm = OpfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = (query_states * self.qk_scale).view(
            batch_size, sequence_length, self.num_key_value_heads, self.num_query_groups, self.head_dim
        )
        key_states = key_states * self.qk_scale

        attn_output = _local_bidirectional_attention(
            query_states,
            key_states,
            value_states,
            self.sinks,
            attention_mask,
            left_context=self.left_context,
            right_context=self.right_context,
        )

        if attn_output.dtype != self.o_proj.weight.dtype:
            attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = F.linear(attn_output, self.o_proj.weight, self.o_proj.bias)
        return residual + attn_output.to(residual.dtype)


class OpfExperts(GptOssExperts):
    def __init__(self, config: OpfConfig):
        super().__init__(config)

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
        chunk_size: int,
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


class OpfMLP(GptOssMLP):
    def __init__(self, config: OpfConfig):
        super().__init__(config)
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.norm = OpfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router = OpfTopKRouter(config)
        self.experts = OpfExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch_shape = hidden_states.shape[:-1]
        hidden_states = self.norm(hidden_states).reshape(-1, hidden_states.shape[-1])

        router_logits = F.linear(
            hidden_states.float(),
            self.router.weight.float(),
            self.router.bias.float() if self.router.bias is not None else None,
        )
        top_values, top_indices = torch.topk(router_logits, k=self.top_k, dim=-1, sorted=True)
        top_weights = torch.softmax(top_values, dim=1)

        hidden_states = self.experts(
            hidden_states,
            top_indices,
            top_weights,
            chunk_size=32,
        )
        hidden_states = hidden_states.reshape(*batch_shape, -1)
        return residual + hidden_states.to(residual.dtype)


class OpfDecoderLayer(GptOssDecoderLayer):
    def __init__(self, config: OpfConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.self_attn
        del self.input_layernorm
        del self.post_attention_layernorm
        self.attn = OpfAttention(config, layer_idx)
        self.mlp = OpfMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class OpfPreTrainedModel(GptOssPreTrainedModel):
    config: OpfConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OpfDecoderLayer"]
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = ["norm", "embedding_norm"]
    _keep_in_fp32_modules_strict: list[str] = ["sinks"]
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = None
    _compatible_flash_implementations = None

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # Modular conversion inherits GptOssExperts' decorator, but OPF's expert forward is checkpoint-specific.
        return False

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        if requested_experts not in (None, "eager"):
            raise ValueError("OPF only supports the eager experts implementation.")
        return "eager"

    def set_use_kernels(self, use_kernels, kernel_config=None):
        if use_kernels:
            raise ValueError("OPF does not support kernelized layers.")
        PreTrainedModel.set_use_kernels(self, use_kernels, kernel_config)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, OpfTopKRouter):
            init.zeros_(module.bias)
        elif isinstance(module, OpfRotaryEmbedding):
            rope_init_fn = module.compute_default_rope_parameters
            if module.rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type]
            inv_freq, module.attention_scaling = rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq.copy_(inv_freq.clone())


@auto_docstring
class OpfModel(OpfPreTrainedModel):
    def __init__(self, config: OpfConfig):
        super().__init__(config)
        self.padding_idx = (
            config.pad_token_id
            if config.pad_token_id is not None and config.pad_token_id < config.vocab_size
            else None
        )
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embedding_norm = OpfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [OpfDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OpfRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OpfRotaryEmbedding(config=config)
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
            raise ValueError("OPF is a bidirectional encoder and does not support key/value caching.")
        if output_attentions:
            logger.warning_once("OPF does not return attention weights.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.embedding_norm(inputs_embeds)
        batch_size, sequence_length, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
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
class OpfForTokenClassification(OpfPreTrainedModel):
    def __init__(self, config: OpfConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OpfModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        sequence_output = outputs[0]
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["OpfForTokenClassification", "OpfModel", "OpfPreTrainedModel"]
