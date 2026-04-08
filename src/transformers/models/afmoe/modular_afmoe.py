# Copyright 2025 Arcee AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch AFMoE model."""

from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_grouped_mm_available, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..gpt_oss.modeling_gpt_oss import GptOssRMSNorm
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts, Qwen2MoeMLP
from .configuration_afmoe import AfmoeConfig


logger = logging.get_logger(__name__)


class AfmoeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class AfmoeRMSNorm(GptOssRMSNorm):
    pass


class AfmoeMLP(Qwen2MoeMLP):
    pass


class AfmoeTokenChoiceRouter(nn.Module):
    """
    Token-choice top-K router for MoE routing.

    This router assigns each token to the top-K experts based on sigmoid scores, matching the released checkpoints.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.route_scale = config.route_scale
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor, expert_bias: torch.Tensor):
        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states).to(torch.float32)
        scores = torch.sigmoid(router_logits)

        _, selected_experts = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
        top_scores = scores.gather(dim=1, index=selected_experts)
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale
        return router_logits, top_scores, selected_experts


class AfmoeExperts(Qwen2MoeExperts):
    pass


class AfmoeSparseMoeBlock(nn.Module):
    """
    Mixture of Experts (MoE) module for AFMoE.

    This module implements a sparse MoE layer with both shared experts (always active) and
    routed experts (activated based on token-choice routing).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = AfmoeTokenChoiceRouter(config)
        self.shared_experts = AfmoeMLP(config, config.moe_intermediate_size * config.num_shared_experts)
        self.experts = AfmoeExperts(config)
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Get routing decisions (returns flattened top-k)
        router_logits, top_scores, selected_experts = self.router(hidden_states, self.expert_bias)

        # Process through shared experts
        shared_output = self.shared_experts(hidden_states_flat).view(batch_size, seq_len, hidden_dim)
        routed_output = self.experts(hidden_states_flat, selected_experts, top_scores).view(
            batch_size, seq_len, hidden_dim
        )
        return shared_output + routed_output


class AfmoeAttention(LlamaAttention):
    """
    Multi-headed attention module with optional sliding window and gating.

    This attention mechanism supports both full attention and sliding window attention,
    and includes Q/K normalization and gating of the output. It inherits from [`LlamaAttention`] to minimize the amount
    of custom logic we need to maintain.
    """

    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Parent LlamaAttention already sets: layer_idx, num_heads, num_key_value_heads, num_key_value_groups, head_dim
        # We only add AFMoE-specific attributes
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None

        self.q_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        output = output.view(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate_states)
        attn_output = self.o_proj(output)
        return attn_output, attn_weights


class AfmoeDecoderLayer(GradientCheckpointingLayer):
    """
    AFMoE decoder layer with dual normalization.

    This layer applies self-attention followed by either a dense MLP or MoE block,
    with dual normalization (pre and post) around each component.
    """

    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = AfmoeAttention(config=config, layer_idx=layer_idx)

        # Dual normalization for attention
        self.input_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Dual normalization for FFN
        self.pre_mlp_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MoE or dense FFN
        self.moe_enabled = layer_idx >= config.num_dense_layers
        if self.moe_enabled:
            self.mlp = AfmoeSparseMoeBlock(config)
        else:
            self.mlp = AfmoeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        # Self Attention with dual normalization
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN with dual normalization
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class AfmoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config: AfmoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["AfmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(AfmoeTokenChoiceRouter, index=0),
        "hidden_states": AfmoeDecoderLayer,
        "attentions": AfmoeAttention,
    }
    _keep_in_fp32_modules = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_mlp_layernorm",
        "q_norm",
        "k_norm",
        "norm",
        "expert_bias",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _supports_attention_backend = True
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, AfmoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, AfmoeTokenChoiceRouter):
            init.zeros_(module.gate.weight)
        elif isinstance(module, AfmoeSparseMoeBlock):
            init.zeros_(module.expert_bias)


@auto_docstring
class AfmoeModel(AfmoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AfmoeDecoderLayer`]

    Args:
        config: AfmoeConfig
    """

    def __init__(self, config: AfmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [AfmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AfmoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        # Apply muP input scaling if enabled
        if self.config.mup_enabled:
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class AfmoeForCausalLM(LlamaForCausalLM, AfmoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        AfmoePreTrainedModel.__init__(self, config)
        self.model = AfmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
]
