# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""Ernie VL model"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import logging
from .configuration_ernie4_5_vl import (
    Ernie4_5_VLConfig,
    Ernie4_5_VLTextConfig,
    Ernie4_5_VLVisionConfig,
)


logger = logging.get_logger(__name__)


class TokenType:
    """token type definition"""

    text = 0
    image = 1
    video = 2


class Ernie4_5_VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        if self.rope_type != "ernie_3d":
            raise ValueError(
                f"Ernie 4.5 VL requires the `ernie_3d` rope type, but found {self.rope_type} instead."
            )

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        # for 3d recomposition
        t_dim = config.rope_scaling["freq_allocation"]  # time dimension
        hw_dim = inv_freq.shape[-1] - t_dim  # height and width dimension
        self.split_sizes = (hw_dim // 2, hw_dim // 2, t_dim)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids.permute(2, 0, 1)[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        sin = self.recomposition_to_3d(sin)
        cos = self.recomposition_to_3d(cos)

        return cos, sin

    def recomposition_to_3d(self, freq):
        freq_h, freq_w, freq_t = (m[(i+1) % 3] for i, m in enumerate(freq.split([*self.split_sizes], dim=-1)))
        # TODO: can we avoid this stack somehow?
        freq_hw = torch.stack([freq_h, freq_w], dim=-1).flatten(-2)
        freq_hwt = torch.cat([freq_hw, freq_t], dim=-1)
        return freq_hwt.repeat_interleave(2, dim=-1)


# copy glm rotate
def rotate_half_text(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# closest are the qwen vl models (vision)
def apply_rotary_pos_emb_text(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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
    original_dtype = q.dtype

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q.float() * cos) + (rotate_half_text(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half_text(k).float() * sin)

    return q_embed.to(original_dtype), k_embed.to(original_dtype)


# copy Llama
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copy Llama
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs#: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# copy Llama after moving rope etc out
class Ernie4_5_VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias
        )

        # TODO: rope to be moved outside
        self.rotary_emb = Ernie4_5_VLTextRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        assert position_ids is not None, "rope3d requires pos-id"

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # rope
        if past_key_value is not None:
            position_ids = position_ids[:, -1:, :]

        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb_text(query_states, key_states, cos, sin)

        # cache
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = [key_states, value_states] if use_cache else None

        # core attention
        #attention_interface: Callable = eager_attention_forward
        #if self.config._attn_implementation != "eager":
        #    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]  # forcing sdpa for now

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
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


# Copy LlamaRMSNorm
class Ernie4_5_VLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Ernie4_5_MoERMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copy Ernie4_5_MoE
class Ernie4_5_VLMoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Copy ernie 4.5 moe
class Ernie4_5_VLMoeStatics(nn.Module):
    """
    Stores MoE (Mixture of Experts) statistics
        - Bias for the gating
        - Additionally, usage per expert in the original codebase
    """

    def __init__(self, num_experts_groups, num_experts):
        super().__init__()

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts_groups, num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        # NOTE: This is a workaround to enable TP with a module that only has parameters
        #
        # Otherwise, it stays as `DTensor` when called in the "super" forward
        #   1. All other tensors are local (`torch.Tensor`)
        #   2. Isolate does not work on `nn.Module` which only has parameters
        return hidden_states + self.e_score_correction_bias.squeeze()


# Copy ernie 4.5 moe (except forward + del shared experts)
class Ernie4_5_VLSparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = config.moe_k

        # correction bias (yes it seems to be a typo with statics <> statistics)
        self.moe_statics = Ernie4_5_VLMoeStatics(num_experts_groups=1, num_experts=self.num_experts)

        # gating
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False, dtype=torch.float32)
        self.experts = nn.ModuleList(
            [Ernie4_5_VLMoeMLP(config, intermediate_size) for _ in range(self.num_experts)]
        )
        self.norm_min = config.moe_norm_min

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states.float())

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(self.moe_statics(routing_weights), self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, dim=-1, index=selected_experts)
            routing_weights = routing_weights / torch.clamp(
                routing_weights.sum(dim=-1, keepdim=True), min=self.norm_min
            )
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

        # moe results are changed to a flattened shape to ease the modality isolated assigning of results
        return final_hidden_states.flatten(), router_logits.flatten()


class Ernie4_5_VLMoeBlock(nn.Module):
    """
    Similar to `Ernie4_5_Moe` where we have modality isolated experts:
        - A set of text experts that are only run on text tokens
        - A set of vision experts that are only run on vision (image/video) tokens

    This modality isolation is unique to the Ernie 4.5 VL models.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts

        self.text_moe = Ernie4_5_VLSparseMoeBlock(
            config,
            num_experts=self.num_experts,
            intermediate_size=config.moe_intermediate_size[0]
        )
        self.vision_moe = Ernie4_5_VLSparseMoeBlock(
            config,
            num_experts=self.num_experts,
            intermediate_size=config.moe_intermediate_size[1]
        )

        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_VLMoeMLP(config, config.moe_intermediate_size[0] * config.moe_num_shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # (Optional) shared experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        if token_type_ids is not None and token_type_ids.any():
            final_hidden_states = torch.zeros_like(hidden_states)
            router_logits = torch.zeros(
                size=(batch_size * sequence_length, self.num_experts),
                device=final_hidden_states.device, dtype=torch.float
            )

            # True (1) == vision, False (0) == text tokens
            token_type_ids = token_type_ids[:, :-1].bool()
            token_type_ids_router = token_type_ids.reshape(-1)[:, None].expand(-1, self.num_experts)
            token_type_ids_states = token_type_ids[..., None].expand(-1, -1, hidden_dim)

            # Extract and separate tokens into their modalities
            text_hidden_states = hidden_states[~token_type_ids_states].reshape(batch_size, -1, hidden_dim)
            vision_hidden_states = hidden_states[token_type_ids_states].reshape(batch_size, -1, hidden_dim)

            # Run moe on each modality and assign their results to the original token positions
            final_hidden_states[~token_type_ids_states], router_logits[~token_type_ids_router] = self.text_moe(text_hidden_states)
            final_hidden_states[token_type_ids_states], router_logits[token_type_ids_router] = self.vision_moe(vision_hidden_states)
        else:
            final_hidden_states, router_logits = self.text_moe(hidden_states)
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            router_logits = router_logits.reshape(-1, self.num_experts)

        # Add (optional) shared experts to the result
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, None, 1, router_logits


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Ernie4_5_VLTextAttention(config, layer_idx)

        moe_layer_start_index = config.moe_layer_start_index
        moe_layer_end_index = config.moe_layer_end_index

        if (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.mlp = Ernie4_5_VLMoeBlock(config)
        else:
            self.mlp = Ernie4_5_VLMoeMLP(config)

        self.input_layernorm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_start_row_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_gate_logits=True,  # PP model should not output gate logits,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[torch.Tensor]): Indices for variable length attention
            position_ids (Optional[torch.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states
            output_gate_logits (bool): Whether to return MoE gate logits

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
                - With MoE: May include gate logits in output tuple
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (hidden_states, self_attn_weights, present_key_value, *router_loss_attn) = (
            self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, Ernie4_5_VLMoeBlock):
            hidden_states, _, router_loss, gate_logits = self.mlp(
                hidden_states, token_type_ids
            )
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits, router_loss = None, None

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # Non-empty only if `use_moe`
        if router_loss_attn:
            router_loss_attn = router_loss_attn[0]
            router_loss = router_loss + router_loss_attn

        if output_gate_logits:
            outputs += (gate_logits,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Ernie4_5_PretrainedModel(PreTrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = Ernie4_5_VLTextConfig
    base_model_prefix = "ernie"
    _no_split_modules = ["Ernie4_5_DecoderLayer"]

    _keep_in_fp32_modules_strict = ["gate", "moe_statics"]


class Ernie4_5_Model(Ernie4_5_PretrainedModel):
    """The core ERNIE transformer model with MoE (Mixture of Experts) support."""

    def __init__(self, config: Ernie4_5_VLTextConfig):
        """Initialize the ERNIE model architecture.

        Args:
            config (Ernie4_5_MoEConfig): Model configuration.
        """
        super().__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList(
            [Ernie4_5_DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs
            position_ids (Optional[torch.Tensor]): Position indices
            attention_mask (Optional[torch.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length attention indices
            inputs_embeds (Optional[torch.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Tuple[Tuple[torch.Tensor]]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
                - router_loss: MoE router loss if use_moe=True
                - gate_logits: MoE gate logits if use_moe=True
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.dtype)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        if getattr(self.config, "use_moe", False):
            all_router_loss = torch.tensor(0.0).to(device=inputs_embeds.device)
        else:
            all_router_loss = None
        all_gate_logits = ()

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                token_type_ids,
                output_attentions,
                past_key_value,
                use_cache,
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
            all_gate_logits = all_gate_logits + (gate_logits,)

            if past_key_value is not None:
                hidden_states = hidden_states[:, -1:, :]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                ]
                if v is not None
            )

        # assert all_router_loss is None, f'moe not support `return-dict`'
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
        )


class Ernie4_5_MoeForCausalLM(Ernie4_5_PretrainedModel, GenerationMixin):
    """ERNIE Mixture of Experts (MoE) model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the ERNIE MoE model for causal language modeling.

        Args:
            config (dict): Model configuration.
        """
        super().__init__(config)

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        self.model = Ernie4_5_Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)

        self.post_init()  # maybe weight share

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @staticmethod
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        """
        Updates model kwargs for generation.

        Args:
            outputs (Any): Model outputs.
            model_kwargs (dict): Current model kwargs.
            is_encoder_decoder (bool): Whether using encoder-decoder architecture.

        Returns:
            dict: Updated model kwargs.
        """
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], torch.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1:]], dim=-1)

        if not is_encoder_decoder and model_kwargs.get("attention_mask", None) is not None:
            # update attention mask
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=torch.int64, device=attention_mask.device),
                ],
                dim=-1,
            )

        assert "position_ids" in model_kwargs, "position_ids must be provided if rope_3d is on"
        position_ids = model_kwargs["position_ids"]

        max_position = position_ids.max(dim=1, keepdim=True)[0]  # [batch_size, 1, hidden_dim]
        new_positions = max_position + 1

        model_kwargs["position_ids"] = torch.cat(
            [position_ids, new_positions],
            dim=1
        )

        return model_kwargs


class VisionMlp(nn.Module):
    """VisionMLP"""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: VisionMLP output tensor
        """
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """PatchEmbed"""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        """
        Args:
            patch_size (int, optional): patch size. Defaults to 14.
            in_channels (int, optional): number of channels. Defaults to 3.
            embed_dim (int, optional): embedding dimension. Defaults to 1152.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): hidden states

        Returns:
            torch.Tensor: output tensor
        """
        target_dtype = self.proj.weight.dtype

        hidden_states = self.proj(hidden_states.to(target_dtype))

        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    """VisionRotaryEmbedding"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Args:
            dim (int): the dimension of each token.
            theta (float, optional): the frequency factor. Defaults to 10000.0.
        """
        super().__init__()
        self.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Args:
            seqlen (int): length of sequence.

        Returns:
            torch.Tensor: rotary position embedding
        """
        seq = torch.arange(seqlen).to(self.inv_freq.dtype)
        freqs = torch.outer(input=seq, vec2=self.inv_freq)
        return freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  # shape is the same as x


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Applies Rotary Position Embedding to the input tensors.

    Args:
        tensor (torch.Tensor): The input tensor.
        freqs (torch.Tensor): The frequencies used for the rotation.
    Returns:
        output (torch.Tensor): the tensor rotated using the Rotary Position Embedding.
    """
    orig_dtype = tensor.dtype

    tensor = tensor.type(dtype=torch.float32)
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).tile(1, 1, 2).unsqueeze(0).type(dtype=torch.float32)
    sin = sin.unsqueeze(1).tile(1, 1, 2).unsqueeze(0).type(dtype=torch.float32)
    output = tensor * cos + rotate_half(tensor) * sin
    output = output.to(orig_dtype)
    return output


class VisionAttention(nn.Module):
    """VisionAttention"""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """forward function for vision attention"""
        seq_length = hidden_states.shape[0]
        qkv = (
            self.qkv(hidden_states)
            .reshape([seq_length, 3, self.num_heads, -1])
            .permute(1, 0, 2, 3)
        )
        q, k, v = qkv.unbind(axis=0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(dim=0), rotary_pos_emb).squeeze(
            dim=0
        )
        k = apply_rotary_pos_emb_vision(k.unsqueeze(dim=0), rotary_pos_emb).squeeze(
            dim=0
        )

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=1) for tensor in (q, k, v)
        ]

        attn_output = []
        for q, k, v in zip(*splits):
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_output_splited = torch.matmul(attn_weights, v)
            attn_output_splited = attn_output_splited.transpose(0, 1)
            attn_output.append(attn_output_splited)
        attn_output = torch.cat(attn_output, dim=0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class DFNRopeVisionBlock(nn.Module):
    """DFNRopeVisionBlock"""

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        """
        Args:
            config (dict): model configuration.
            attn_implementation (str, optional): attention implementation. Defaults to "sdpa".
        """
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.hidden_size, config.vision_rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.vision_rms_norm_eps)

        self.attn = VisionAttention(config.hidden_size, num_heads=config.num_heads)
        self.mlp = VisionMlp(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        """
        Args:
            hidden_states(torch.Tensor): hidden states
            cu_seqlens (torch.Tensor): cumulative sequence lengths
            rotary_pos_emb: rotary position embedding

        Returns:
            torch.Tensor: output tensor
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DFNRopeVisionTransformerPreTrainedModel(PreTrainedModel):
    """DFNRopeVisionTransformerPreTrainedModel"""

    config_class = Ernie4_5_VLVisionConfig
    _tp_plan = {}

    def __init__(self, config) -> None:
        """
        Args:
            config (dict): model configuration
        """
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [DFNRopeVisionBlock(config) for _ in range(config.depth)]
        )

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.vision_rms_norm_eps)

    def rot_pos_emb(self, grid_thw, num_pad=0):
        """rot_pos_emb

        Args:
            grid_thw (torch.Tensor): grid thw of input

        Returns:
            torch.Tensor: rotary position embedding
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw.cpu(), dtype=np.int64)
        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape([-1, 1])
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape([1, -1])
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate(
                [pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)]
            )
        max_grid_size = np.amax(grid_hw_array[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_dim=1)
        return rotary_pos_emb

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_pad=0
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input tensor
            grid_thw (torch.Tensor): grid thw of input
            num_pad (int): number of padding tokens

        Returns:
            torch.Tensor: output tensor
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw, num_pad=num_pad)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        ret = self.ln(hidden_states)  # add norm
        return ret


class VariableResolutionResamplerModel(nn.Module):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.in_dim = config.hidden_size
        self.out_dim = config.text_hidden_size
        self.spatial_conv_size = config.spatial_conv_size
        self.temporal_conv_size = config.temporal_conv_size

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress 3d conv(video) to 1d
        self.temporal_dim = (
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        self.spatial_linear = nn.Sequential(
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.GELU(),
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.LayerNorm(self.spatial_dim, eps=config.vision_rms_norm_eps),
        )

        self.temporal_linear = nn.Sequential(
            nn.Linear(self.temporal_dim, self.spatial_dim),
            nn.GELU(),
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.LayerNorm(self.spatial_dim, eps=config.vision_rms_norm_eps),
        )

        self.mlp = nn.Linear(self.spatial_dim, self.out_dim)

        self.after_norm = Ernie4_5_VLRMSNorm(self.out_dim, config.rms_norm_eps)

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        reshape before linear to imitation conv
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x = self.spatial_linear(x)

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                the second dimension: [t, h, w]
            """

            grid_thw_cpu = grid_thw.cpu().numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = torch.tensor(np.concatenate(slice_offsets, axis=-1)).to(
                x.device
            )

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = torch.tensor(np.concatenate(slice_offsets2, axis=-1)).to(
                x.device
            )

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets)
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2)
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1)
            return x

        def fwd_temporal(x):
            x = self.temporal_linear(x)
            return x

        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            return x

        x = fwd_spatial(x)
        x = fwd_placeholder(x, grid_thw)
        x = fwd_temporal(x)
        x = fwd_mlp(x)
        return x


class Ernie4_5_VLMoeForConditionalGeneration(Ernie4_5_MoeForCausalLM):
    """Ernie4_5_VLMoeForConditionalGeneration"""

    config_class = Ernie4_5_VLConfig
    main_input_name = "pixel_values"
    _keep_in_fp16_modules = ["vision_model"]
    _tp_plan = {}

    def __init__(
        self, config: Ernie4_5_VLConfig, vision_model=None, resampler_model=None
    ):
        """
        initialize Ernie4_5_VLMoeForConditionalGeneration

        Args:
            config(Ernie4_5_VLMoEConfig): Model configuration.
            vision_model(nn.Module): vision model
            resampler_model(nn.Module): resampler model
        """
        super().__init__(config.text_config)

        self.config = config

        self.vision_model = DFNRopeVisionTransformerPreTrainedModel(
            config.vision_config
        )

        # TODO: move to vision
        self.model.resampler_model = VariableResolutionResamplerModel(
            config.vision_config
        )

        self.image_preprocess = None
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def add_image_preprocess(self, processor):
        """add image preprocess"""
        logger.info("image preprocess is set")

        image_preprocess = processor.image_processor
        image_preprocess.image_mean_tensor = torch.tensor(
            image_preprocess.image_mean, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_preprocess.image_std_tensor = torch.tensor(
            image_preprocess.image_std, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_preprocess.rescale_factor = torch.tensor(
            image_preprocess.rescale_factor, dtype=torch.float32
        )
        image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze(
            [-2, -1]
        ).repeat_interleave(self.config.vision_config.patch_size**2 * 1, -1)
        image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze(
            [-2, -1]
        ).repeat_interleave(self.config.vision_config.patch_size**2 * 1, -1)

        self.image_preprocess = image_preprocess

    def vision_forward(
        self,
        images,
        image_position_ids,
        image_attention_mask,
        grid_thw,
    ):
        """vision_forward"""
        if self.image_preprocess is not None:
            assert images.dtype == torch.uint8, images.dtype
            current_device = images.device
            self.image_preprocess.image_mean_tensor = (
                self.image_preprocess.image_mean_tensor.to(current_device)
            )
            self.image_preprocess.image_std_tensor = (
                self.image_preprocess.image_std_tensor.to(current_device)
            )
            images = self.image_preprocess.rescale_factor * images.to(torch.float32)
            images = (
                images - self.image_preprocess.image_mean_tensor
            ) / self.image_preprocess.image_std_tensor
            images = images.to(torch.bfloat16)
        else:
            assert images.dtype == torch.bfloat16, images.dtype
        # logger.info(f"extract feature input - {images}--{grid_thw}")
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_features = self.vision_model(images, grid_thw)
        return image_features

    def vision_mapping_forward(
        self,
        token_type_ids,
        token_type_ids_w_video,
        input_ids,
        mm_input_ids,
        image_features,
        inputs_embeds,
        image_type_ids,
        grid_thw,
    ):
        """vision_mapping_forward"""
        image_mask = input_ids == self.config.image_token_id
        image_features = self.model.resampler_model(
            image_features,
            image_mask,
            token_type_ids_w_video,
            image_type_ids,
            grid_thw,
        )

        if image_features.dim == 2:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C]).to(inputs_embeds.dtype)
        # Will overwrite the part of `ids==image_token_id` in `mm_ids_features`
        inputs_embeds[image_mask.to(inputs_embeds.device)] = image_features.to(
            inputs_embeds.device
        )
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        image_position_ids=None,
        image_attention_mask=None,
        token_type_ids=None,
        image_type_ids=None,
        grid_thw=None,
        **kwargs,
    ):
        """
        Prepare inputs for the decoder that can be used for generation.

        Args:
            input_ids (torch.Tensor): Input ids.
            images (torch.Tensor): Images. Default to None.
            use_cache (bool): Whether to use cache. Default to False.
            past_key_values (list): Past key values. Default to None.
            inputs_embeds (torch.Tensor): Input embeddings. Default to None.
            image_position_ids (torch.Tensor): Image position ids. Default to None.
            image_attention_mask (torch.Tensor): Image attention mask. Default to None.
            token_type_ids (torch.Tensor): Token type ids. Default to None.
            image_type_ids (torch.Tensor): Image type ids. Default to None.
            grid_thw (torch.Tensor): Grid thw. Default to None.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            image_type_ids = (
                image_type_ids[:, -1:] if image_type_ids is not None else None
            )

        #attention_mask = kwargs.get("attention_mask")  # non-fa usage
        attention_mask = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "images": images,
                "image_position_ids": image_position_ids,
                "image_attention_mask": image_attention_mask,
                "image_type_ids": image_type_ids,
                "token_type_ids": torch.cat(
                    [
                        token_type_ids,
                        torch.zeros(
                            [len(token_type_ids), 1], dtype=token_type_ids.dtype
                        ).to(token_type_ids.device),
                    ],
                    dim=-1,
                ),
                "grid_thw": grid_thw,
            }
        )
        model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    def _post_init(self, original_init, *args, **kwargs):
        """
        Label all multimodal parameters in the model, only head and Embedding
        Experts parameters are already labeled
        """
        super()._post_init(self, original_init, *args, **kwargs)
        if self.lm_head.mm_head is not None:
            self.lm_head.mm_head.weight.expert_type = "expert_type_1"
        if getattr(self.lm_head.mm_head, "bias", None) is not None:
            self.lm_head.mm_head.bias.expert_type = "expert_type_1"

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        ignored_index: Optional[int] = 0,
        return_dict: Optional[bool] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        image_type_ids: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward for Ernie4_5_VLMoeForConditionalGeneration

        Args:
            input_ids (torch.Tensor): Input ids.
            position_ids (Optional[torch.Tensor], optional): Position ids. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            past_key_values (Optional[List[torch.Tensor]], optional): Past key values. Defaults to None.
            use_cache (Optional[bool], optional): Use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Output hidden states. Defaults to None.
            labels (Optional[torch.Tensor], optional): Labels. Defaults to None.
            images (Optional[torch.Tensor]): Images. Defaults to None.
            ignored_index (Optional[int], optional): Ignored index. Defaults to 0.
            return_dict (Optional[bool], optional): Return dict. Defaults to None.
            image_position_ids (Optional[torch.Tensor], optional): Image position ids. Defaults to None.
            image_attention_mask (Optional[torch.Tensor], optional): Image attention mask. Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids. Defaults to None.
            image_type_ids (Optional[torch.Tensor], optional): Image type ids. Defaults to None.
            grid_thw (Optional[torch.Tensor], optional): Grid thw. Defaults to None.
        """
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_mask = input_ids == self.config.image_token_id

        if past_key_values is None:
            if images is not None:
                assert (image_mask).any().item(), (
                    image_mask.detach().cpu().numpy().tolist(),
                    input_ids.detach().cpu().numpy().tolist(),
                    self.config.image_token_id,
                    images.shape,
                )
                image_features = self.vision_forward(
                    images,
                    image_position_ids,
                    image_attention_mask,
                    grid_thw,
                )
            else:
                image_features = None  # no more faking
        else:
            image_features = None
        if token_type_ids is None:
            token_type_ids = image_mask.to(torch.int64)
            token_type_ids_labels = torch.cat(
                [token_type_ids[:, 1:], token_type_ids[:, -1:]], 1
            )
        else:
            assert (
                token_type_ids.shape[1] == input_ids.shape[1] + 1
            ), f"token_type:{token_type_ids.shape}, ids:{input_ids.shape}"
            token_type_ids_labels = token_type_ids[..., 1:]

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()

        inputs_embeds = self.model.embed_tokens(lm_input_ids)
        token_type_ids_w_video = token_type_ids[..., :-1].clone()
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image

        if images is not None and image_features is not None:
            inputs_embeds = self.vision_mapping_forward(
                token_type_ids[..., :-1],
                token_type_ids_w_video,
                input_ids,
                mm_input_ids,
                image_features,
                inputs_embeds,
                image_type_ids,
                grid_thw,
            )
        else:
            pass  # do nothing, should not hang under DygraphShardingOptimizerV2

        outputs = self.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not use_cache:
            assert outputs.last_hidden_state.shape[:2] == token_type_ids_labels.shape, (
                outputs.last_hidden_state.shape,
                token_type_ids_labels.shape,
            )
            if self.config.use_recompute_loss_fn:
                logits = outputs.last_hidden_state
            else:
                logits = self.lm_head(outputs.last_hidden_state)
        else:
            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])

        # aka Generate Decoding
        loss = None
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_loss=outputs.router_loss,
        )


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model outputs with past key values and cross attention layers,
    with additional support for router components in mixture-of-experts models.

    This extends the base model output to include:
    1. Router-related outputs for expert selection
    2. Maintains all existing functionality from the parent class
    """

    last_hidden_state: Optional[tuple[torch.Tensor]] = None
    past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None
    hidden_states: Optional[tuple[torch.Tensor]] = None
    attentions: Optional[tuple[torch.Tensor]] = None
    cross_attentions: Optional[tuple[torch.Tensor]] = None
    router_loss: Optional[torch.Tensor] = None
    gate_logits: Optional[tuple[torch.Tensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_loss (Optional[torch.Tensor]):
            The routing loss computed by the gating network in mixture-of-experts models.
            This is typically the load balancing loss that encourages equal expert utilization.
            None when not using mixture-of-experts routing.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None
    hidden_states: Optional[tuple[torch.Tensor]] = None
    attentions: Optional[tuple[torch.Tensor]] = None
    router_loss: Optional[tuple[torch.Tensor]] = None


__all__ = [
    "Ernie4_5_VLMoeForConditionalGeneration",
    "DFNRopeVisionTransformerPreTrainedModel",
    "VariableResolutionResamplerModel",
]
