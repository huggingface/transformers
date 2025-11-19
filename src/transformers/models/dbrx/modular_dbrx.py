# coding=utf-8
# Copyright 2024 Databricks Mosaic Research and The HuggingFace Inc. team. All rights reserved.
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
"""Modular components for DBRX model."""

from collections.abc import Callable
from typing import Any, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mixtral.modeling_mixtral import load_balancing_loss_func
from .configuration_dbrx import DbrxConfig


class DbrxRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DbrxAttention(nn.Module):
    """Modular DBRX attention component that can be reused across different model architectures."""

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_seq_len
        self.layer_idx = layer_idx

        attn_config = config.attn_config
        self.attention_dropout = attn_config.attn_pdrop
        self.clip_qkv = attn_config.clip_qkv
        self.num_key_value_heads = attn_config.kv_n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = attn_config.rope_theta
        self.is_causal = True

        self.Wqkv = nn.Linear(
            self.hidden_size, self.hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv_states = self.Wqkv(hidden_states)
        min_val = -self.clip_qkv if self.clip_qkv is not None else None
        qkv_states = qkv_states.clamp(min=min_val, max=self.clip_qkv)

        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class DbrxExpertGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.moe_num_experts = config.moe_num_experts

        self.w1 = nn.Parameter(torch.empty(self.moe_num_experts * self.ffn_hidden_size, self.hidden_size))
        self.v1 = nn.Parameter(torch.empty(self.moe_num_experts * self.ffn_hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.moe_num_experts * self.ffn_hidden_size, self.hidden_size))

        act_fn_name = config.ffn_act_fn.get("name", "silu")
        self.activation_fn = ACT2FN[act_fn_name]

    def forward(
        self, x: torch.Tensor, expert_w1: torch.Tensor, expert_v1: torch.Tensor, expert_w2: torch.Tensor
    ) -> torch.Tensor:
        gate_proj = x.matmul(expert_w1)
        up_proj = x.matmul(expert_v1)
        gate_proj = self.activation_fn(gate_proj)
        intermediate_states = gate_proj * up_proj
        down_proj = intermediate_states.matmul(expert_w2.t())
        return down_proj


class DbrxExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = DbrxExpertGLU(config)
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.num_experts = config.moe_num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.ffn_hidden_size)

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        split_expert_shape = (-1, self.ffn_hidden_size, self.hidden_size)
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            with torch.no_grad():
                idx, token_idx = torch.where(expert_mask[expert_idx])
            v1 = self.mlp.v1.view(split_expert_shape)[expert_idx]
            w1 = self.mlp.w1.view(split_expert_shape)[expert_idx]
            w2 = self.mlp.w2.view(split_expert_shape)[expert_idx]
            states = self.mlp(hidden_states[token_idx], w1, v1, w2)
            states = states.view(-1, self.ffn_hidden_size) * top_k_weights[token_idx, idx, None]
            next_states.index_add_(0, token_idx, states)

        next_states = next_states.view(batch_size, -1, self.ffn_hidden_size)
        return next_states


class DbrxRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.ffn_hidden_size
        self.moe_jitter_eps = config.moe_jitter_eps
        self.layer = nn.Linear(self.hidden_size, config.moe_num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.training and self.moe_jitter_eps is not None:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.moe_jitter_eps, 1.0 + self.moe_jitter_eps
            )
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.layer(hidden_states)
        return router_logits


class DbrxFFN(nn.Module):
    """Modular DBRX MLP/FFN component with MoE support."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.router = DbrxRouter(config.ffn_config)
        self.experts = DbrxExperts(config.ffn_config)

        self.moe_normalize_expert_weights = config.ffn_config.moe_normalize_expert_weights
        self.top_k = config.ffn_config.moe_top_k

    def route_tokens_to_experts(self, router_logits):
        router_logits = torch.nn.functional.softmax(router_logits, dim=1, dtype=router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.moe_normalize_expert_weights is not None:
            router_top_value = router_top_value / torch.norm(
                router_top_value, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True
            )
        return router_top_value, router_indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(hidden_states)
        top_k_weights, top_k_index = self.route_tokens_to_experts(router_logits)
        output = self.experts(hidden_states, top_k_index, top_k_weights)
        return output


class DbrxNormAttentionNorm(nn.Module):
    def __init__(self, config: DbrxConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.resid_pdrop = config.resid_pdrop
        self.norm_1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = DbrxAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.norm_2 = nn.LayerNorm(config.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states).to(hidden_states.dtype)

        hidden_states, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states).to(hidden_states.dtype)

        return residual_states, hidden_states


class DbrxBlock(GradientCheckpointingLayer):
    def __init__(self, config: DbrxConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.layer_idx = layer_idx
        self.norm_attn_norm = DbrxNormAttentionNorm(
            config=config,
            layer_idx=layer_idx,
        )
        self.ffn = DbrxFFN(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        resid_states, hidden_states = self.norm_attn_norm(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = self.ffn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = resid_states + hidden_states
        return hidden_states


class DbrxPreTrainedModel(PreTrainedModel):
    config: DbrxConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DbrxBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flex_attn = True
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _can_record_outputs = {
        "hidden_states": DbrxBlock,
        "attentions": DbrxAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, DbrxExpertGLU):
            init.normal_(module.w1, mean=0.0, std=std)
            init.normal_(module.v1, mean=0.0, std=std)
            init.normal_(module.w2, mean=0.0, std=std)


@auto_docstring
class DbrxModel(DbrxPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers*. Each layer is a [`DbrxBlock`] layer.

    Args:
        config ([`DbrxConfig`]): Model configuration class with all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.emb_pdrop = config.emb_pdrop
        self.rotary_emb = DbrxRotaryEmbedding(config)
        self.wte = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.blocks = nn.ModuleList([DbrxBlock(config, layer_idx) for layer_idx in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, bias=False)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding):
        self.wte = value

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.blocks[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm_f(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class DbrxForCausalLM(DbrxPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.transformer = DbrxModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.ffn_config.moe_loss_weight
        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: DbrxModel):
        self.transformer = decoder

    def get_decoder(self) -> DbrxModel:
        return self.transformer

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >> from transformers import AutoTokenizer, DbrxForCausalLM

        >> model = DbrxForCausalLM.from_pretrained("databricks/dbrx-instruct")
        >> tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct")

        >> prompt = "Hey, are you conscious? Can you talk to me?"
        >> inputs = tokenizer(prompt, return_tensors="pt")

        >> # Generate
        >> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["DbrxForCausalLM", "DbrxModel", "DbrxPreTrainedModel"]
