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
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from .configuration_ernie4_5_vl import (
    Ernie4_5_VLConfig,
    Ernie4_5_VLTextConfig,
    Ernie4_5_VLVisionConfig,
)


logger = logging.get_logger(__name__)


class TokenType:
    text = 0
    image = 1
    video = 2


# no copy
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
    **kwargs: Unpack[TransformersKwargs],
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


# copy Llama after making it cache compatible
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_text(query_states, key_states, cos, sin)

        # cache
        if past_key_values is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)
        past_key_values = [key_states, value_states] if use_cache else None

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

        return attn_output, attn_weights, past_key_values


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
            torch.zeros(num_experts_groups, num_experts),
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
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
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
            token_type_ids = token_type_ids.bool()
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

        return final_hidden_states, router_logits


# Copy Ernie 4.5 Moe
class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5_VLTextAttention(config, layer_idx)

        if (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= config.moe_layer_start_index
            and layer_idx <= config.moe_layer_end_index
        ):
            self.mlp = Ernie4_5_VLMoeBlock(config)
        else:
            self.mlp = Ernie4_5_VLMoeMLP(config)

        self.input_layernorm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, Ernie4_5_VLMoeBlock):
            hidden_states, gate_logits = self.mlp(
                hidden_states, token_type_ids
            )
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits = None
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_values,)
        if output_router_logits:
            outputs += (gate_logits,)

        return outputs


class Ernie4_5_PretrainedModel(PreTrainedModel):
    config: Ernie4_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ernie4_5_DecoderLayer", "Ernie4_5VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keep_in_fp32_modules_strict = [r"gate", r"e_score_correction_bias"]

    # TODO: set correct supports flags etc


class Ernie4_5VLTextModel(Ernie4_5_PretrainedModel):
    config: Ernie4_5_VLTextConfig

    def __init__(self, config: Ernie4_5_VLTextConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

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

        self.rotary_emb = Ernie4_5_VLTextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = output_router_logits# if output_router_logits is not None else self.config.output_router_logits
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        cache_length = 0
        seq_length_with_past = inputs_embeds.shape[1]
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                token_type_ids,
                output_attentions,
                output_router_logits,
                past_key_value,  # NOTE: without the s, as we have tuples extracted per layer
                cache_position,
                use_cache,
                **kwargs,
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

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
                    all_router_logits,
                ]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


# copy qwen2 vl (change init to fp16)
class Ernie4_5VLVisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# copy qwen2 vl (without temporal and different forward)?
class Ernie4_5VLPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        return self.proj(hidden_states.to(target_dtype))


# copy qwen 2.5 vl
class VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# copy qwen 2.5 vl
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# copy qwen 2.5 vl
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


# copy qwen 2.5 vl (init changes to fp16)
class Ernie4_5VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        """attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]"""
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]  # forcing sdpa for now

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


# copy qwen 2.5 vl (change init)
class Ernie4_5VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, config.vision_rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.vision_rms_norm_eps)
        self.attn = Ernie4_5VLVisionAttention(config)
        self.mlp = Ernie4_5VLVisionMLP(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# similar to qwen 2.5 vl - hard to copy since no window attn, merger
class Ernie4_5VLVisionTransformerPreTrainedModel(PreTrainedModel):
    config_class = Ernie4_5_VLVisionConfig
    _no_split_modules = ["Ernie4_5VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Ernie4_5VLPatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Ernie4_5VLVisionBlock(config) for _ in range(config.depth)]
        )

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.vision_rms_norm_eps)

    # copy qwen 2.5 vl
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    # qwen 2.5 vl without windowed attention and merger at the end
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.ln(hidden_states)
        return hidden_states


# no copy
class Ernie4_5VLVariableResolutionResamplerModel(nn.Module):
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

    def _temporal_slicing(self, x, grid_thw):
        """
        Creates slices along the temporal dimension (usually if we have a video input).

        If a "real" (video) slicing happens, then we change [1,2,1,2,1,2] to [1,1,1,2,2,2] patterns.
        Otherwise, we repeat along the axis, i.e. [1,1,1] to [1,1,1,1,1,1]. NOTE: It is hard-coded
        for `temporal_conv_size == 2`.
        """
        # Calculating offsets (based on flattened tensors)
        grid_t, grid_hw = grid_thw[:, 0], grid_thw[:, 1:]
        grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

        tokens_per_img_or_vid = (grid_thw.prod(-1) // (self.spatial_conv_size**2)).flatten()
        batch_offsets = torch.empty(
            tokens_per_img_or_vid.size(), dtype=tokens_per_img_or_vid.dtype
        )
        batch_offsets[0] = 0
        batch_offsets[1:] = tokens_per_img_or_vid.cumsum(dim=0)[:-1]

        first_slice_offsets = []
        second_slice_offsets = []
        for temporal_size, spatial_size, batch_offset in zip(
            grid_t, grid_hw_after_conv, batch_offsets
        ):
            # Depending on temporal, we may interleave
            first_offset_range = range(0, temporal_size, 2)
            second_offset_range = range(1 if temporal_size > 1 else 0, temporal_size, 2)

            is_same_offset_range = first_offset_range == second_offset_range
            for temporal_offset in first_offset_range:
                first_slice_offsets.append(
                    torch.arange(
                        batch_offset + (temporal_offset) * spatial_size,
                        batch_offset + (temporal_offset + 1) * spatial_size,
                    )
                )

                # We can avoid looping another time if the ranges are the same
                if is_same_offset_range:
                    second_slice_offsets.append(
                        torch.arange(
                            batch_offset + (temporal_offset) * spatial_size,
                            batch_offset + (temporal_offset + 1) * spatial_size,
                        )
                    )

            if not is_same_offset_range:
                for temporal_offset in second_offset_range:
                    second_slice_offsets.append(
                        torch.arange(
                            batch_offset + (temporal_offset) * spatial_size,
                            batch_offset + (temporal_offset + 1) * spatial_size,
                        )
                    )

        first_slice_offsets = torch.cat(first_slice_offsets, dim=-1).to(x.device)
        second_slice_offsets = torch.cat(second_slice_offsets, dim=-1).to(x.device)

        return torch.concat(
            [
                torch.index_select(x, dim=0, index=first_slice_offsets),
                torch.index_select(x, dim=0, index=second_slice_offsets)
            ],
            dim=-1
        )

    def forward(self, x, grid_thw):
        # image spatial
        x = x.reshape([-1, x.shape[-1] * (self.spatial_conv_size**2)])
        x = self.spatial_linear(x.to(self.mlp.weight.dtype))

        # video temporal
        x = self._temporal_slicing(x, grid_thw)
        x = self.temporal_linear(x)

        # final mlp
        x = self.mlp(x)
        x = self.after_norm(x)

        return x


class Ernie4_5VLModel(Ernie4_5_PretrainedModel):
    def __init__(self, config: Ernie4_5_VLConfig):
        super().__init__(config)

        self.language_model = Ernie4_5VLTextModel(config.text_config)

        self.vision_tower = Ernie4_5VLVisionTransformerPreTrainedModel(config.vision_config)
        self.resampler_model = Ernie4_5VLVariableResolutionResamplerModel(config.vision_config)
        self.image_preprocess = None  # TODO: move to preprocessor

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    # TODO: move to processor
    def add_image_preprocess(self, processor):
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

    # TODO: move to processor
    def forward_image_preprocess(self, images):
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

        return images

    # TODO: same with videos
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        if image_grid_thw is not None:
            grid_thw = image_grid_thw[image_grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_embeds = self.vision_tower(pixel_values, grid_thw)
        image_embeds = self.resampler_model(image_embeds, grid_thw)
        return image_embeds

    # TODO: fixup with videos, iirc this is not handled with a token atm
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor = None,
        video_features: torch.FloatTensor = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            #special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        """n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )"""

        #return special_image_mask, special_video_mask
        return special_image_mask, None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        #past_key_values: Optional[Cache] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        #pixel_values: Optional[torch.Tensor] = None,
        #pixel_values_videos: Optional[torch.FloatTensor] = None,
        #image_grid_thw: Optional[torch.LongTensor] = None,
        #video_grid_thw: Optional[torch.LongTensor] = None,
        #rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        #second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = output_router_logits# if output_router_logits is not None else self.config.output_router_logits
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO: logic change for input embeds and videos
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if past_key_values is None and images is not None:
            # TODO: change logic to preprocessor
            pixel_values = self.forward_image_preprocess(images)

            image_embeds = self.get_image_features(pixel_values, image_grid_thw=grid_thw)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # TODO: add and check logic with videos ("or" mask?)
        if token_type_ids is None:
            token_type_ids = image_mask.to(torch.int64)
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image

        outputs = self.language_model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )

        output = MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
        return output if return_dict else output.to_tuple()


class Ernie4_5VLForConditionalGeneration(Ernie4_5_PretrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Ernie4_5_VLConfig):
        super().__init__(config)
        self.model = Ernie4_5VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        position_ids = model_kwargs.pop("position_ids")
        position_ids = torch.cat(
            [position_ids, position_ids.max(dim=1, keepdim=True)[0] + 1],
            dim=1
        )

        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
        model_kwargs["position_ids"] = position_ids

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        token_type_ids=None,
        grid_thw=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]

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
                "token_type_ids": token_type_ids,
                "grid_thw": grid_thw,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        #past_key_values: Optional[Cache] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        #pixel_values: Optional[torch.Tensor] = None,
        #pixel_values_videos: Optional[torch.FloatTensor] = None,
        #image_grid_thw: Optional[torch.LongTensor] = None,
        #video_grid_thw: Optional[torch.LongTensor] = None,
        #rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        #second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
            images=images,
            grid_thw=grid_thw,
            #pixel_values=pixel_values,
            #pixel_values_videos=pixel_values_videos,
            #image_grid_thw=image_grid_thw,
            #video_grid_thw=video_grid_thw,
            #rope_deltas=rope_deltas,
            cache_position=cache_position,
            #second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        if not use_cache:
            logits = self.lm_head(outputs.last_hidden_state)
        else:
            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])

        # aka Generate Decoding
        loss = None  # TODO
        aux_loss = None  # TODO: load balancing loss

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Ernie4_5VLForConditionalGeneration",
    "Ernie4_5VLVisionTransformerPreTrainedModel",
    "Ernie4_5VLVariableResolutionResamplerModel",
]
