# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""PyTorch Ernie 4.5 MoE model."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, logging
from ..llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding
from ..mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from .configuration_ernie4_5_moe import Ernie4_5_MoEConfig


logger = logging.get_logger(__name__)


class Ernie4_5_MoERMSNorm(LlamaRMSNorm):
    pass


class Ernie4_5_MoEMLP(Qwen3MoeMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config, intermediate_size)

        del self.gate_proj
        del self.up_proj
        del self.down_proj

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)


# TODO: can become majority wise llama copy (except the fp32 consistency)
class Ernie4_5_MoERotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # key difference to llama rope happens here to force an even/odd pattern instead
            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(1, 2)
            emb = torch.stack((freqs, freqs), dim=-1).reshape(*freqs.shape[:2], -1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos, sin


def rotate_half(x):
    """Rotates half (in even/odd pattern) the hidden dims of the input."""
    input_shape = x.shape[:-1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(*input_shape, -1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # key difference to llama is the forward in fp32
    original_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q.float() * cos) + (rotate_half(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half(k).float() * sin)
    return q_embed.to(original_dtype), k_embed.to(original_dtype)


class Ernie4_5_MoEAttention(LlamaAttention):
    def __init__(self, config: Ernie4_5_MoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.o_proj
        del self.attention_dropout

        self.attention_dropout = 0.0

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias)


class Ernie4_5_MoEStatics(nn.Module):
    """
    Stores MoE (Mixture of Experts) statistics
        - Bias for the gating
        - Additionally, usage per expert in the original codebase
    """

    def __init__(self, config):
        super().__init__()

        num_experts_groups = 1
        num_experts = config.moe_num_experts

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts_groups, num_experts, dtype=torch.float32),
            # TODO: it has non-zero values...
            # requires_grad=False,
        )


class Ernie4_5_MoESparseMoEBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.

    Ernie 4.5 MoE's original formula is based on case (2) with
    (optional) shared experts and a corrections bias during gating.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k

        # correction bias (yes it seems to be a typo with statics <> statistics)
        self.moe_statics = Ernie4_5_MoEStatics(config)

        # gating
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Ernie4_5_MoEMLP(config, config.moe_intermediate_size) for _ in range(config.moe_num_experts)]
        )
        self.norm_min = config.moe_norm_min

        # (optional) shared experts for all forwards
        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_MoEMLP(config, config.moe_intermediate_size * config.moe_num_shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states.float())

        # temporarily forward in fp32 and then cast back to the input dtype
        # TODO: check below
        # See https://github.com/PaddlePaddle/ERNIE/blob/d4e1c371dfd089ef618ef378e8996049bd54da00/ernie/moe/moe_layer.py#L607 in combination with
        # https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/incubate/nn/functional/moe_gate_dispatch.py#L104 == the "correction bias"
        # correction_bias = self.moe_statics.e_score_correction_bias[0]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights, selected_experts = torch.topk(routing_weights + correction_bias, self.top_k, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / torch.clamp(routing_weights.sum(dim=-1, keepdim=True), min=self.norm_min)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # Add (optional) shared experts to the result
        if self.shared_experts is not None:
            final_hidden_states += self.shared_experts(hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Ernie4_5_MoEDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5_MoEAttention(config, layer_idx)

        if (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= config.moe_layer_start_index
            and layer_idx <= config.moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoESparseMoEBlock(config)
        else:
            self.mlp = Ernie4_5_MoEMLP(config)

        self.input_layernorm = Ernie4_5_MoERMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_MoERMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            attention_mask (`torch.FloatTensor`, *optional*):
                Attention mask of size `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Cache`, *optional*):
                Cached past key and value projection states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(self.mlp, Ernie4_5_MoESparseMoEBlock):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


@auto_docstring
class Ernie4_5_MoEPreTrainedModel(MixtralPreTrainedModel):
    _keep_in_fp32_modules_strict = ["gate", "moe_statics"]


# TODO: add mtp option? - a lot of unclear details
#    - Do we act as if we were in pos 0, no past?
#    - Which variables would need to be cut then?
#    - ...
@auto_docstring
class Ernie4_5_MoEModel(MixtralModel):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


@auto_docstring
class Ernie4_5_MoEForCausalLM(MixtralForCausalLM, Ernie4_5_MoEPreTrainedModel):
    def __init__(self, config):
        Ernie4_5_MoEPreTrainedModel().__init__(config)
        self.model = Ernie4_5_MoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.moe_num_experts
        self.num_experts_per_tok = config.moe_k

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        super().forward(**super_kwargs)


__all__ = [
    "Ernie4_5_MoEForCausalLM",
    "Ernie4_5_MoEModel",
    "Ernie4_5_MoEPreTrainedModel",
]
