# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mixtral.configuration_mixtral import MixtralConfig
from ..phi3.modeling_phi3 import Phi3MLP


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "meta-ai/Llama-4-17B"


class Llama4Config(MixtralConfig):
    model_type = "llama4"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",  # we need to replicate here to correctly route experts
        "layers.*.block_sparse_moe.experts.*.gate_up_proj": "colwise",
        "layers.*.block_sparse_moe.experts.*.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=14336,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_local_experts=16,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_key_value_heads

        self.attention_bias = False
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        del self.sliding_window


class Llama4Experts(nn.Module):
    def __init__(self, config: Llama4Config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor: _description_
        """

        # To support all arch, and EP, we are gonna scatter the input hidden states
        # based on the selected experts
        gate, up = torch.mm(self.gate_up_proj, hidden_states).chunk(2, dim=-1)
        next_states = self.down_proj * (up * self.activation_fn(gate))

        num_local_experts = self.num_local_experts

        self.gate_up_proj = self.gate_up_proj.view(num_local_experts, 2 * self.intermediate_dim, -1)

        hidden_states = hidden_states.view(num_local_experts, -1, self.intermediate_dim)

        middle_egF, swiglu_hidden_egF = torch.bmm(hidden_states, self.gate_up_proj).chunk(2, 0)
        middle_out_egF = torch.nn.functional.silu(middle_egF) * swiglu_hidden_egF

        out_egD = torch.bmm(middle_out_egF, self.down_proj)
        out_egD = out_egD.view(-1, self.intermediate_dim)
        return next_states


class Llama4MLP(Phi3MLP):
    pass


class Llama4L2Norm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)


class Llama4RMSNorm(LlamaRMSNorm):
    pass


class Llama4Moe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = Llama4Experts(config)
        self.router = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # win, wout, w_swiglu
        # col, row, col
        self.shared_expert = Llama4MLP(config)

        # Token choice is used
        self.global_gate_stats = torch.zeros(3, config.num_local_experts, dtype=torch.float32)

    def forward(self, hidden_states):
        router_scores = self.router(hidden_states)
        tokens_per_expert = hidden_states.shape[1]
        D = hidden_states.shape[2]

        router_scores, router_indices = torch.topk(router_scores.transpose(0, 1), self.config.top_k, dim=1)
        router_scores = (
            torch.full_like(router_scores.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices, router_scores)
            .transpose(0, 1)
        )  # We do this to make sure we have -inf for non topK tokens!
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device).view(1, -1).expand(router_scores.size(0), -1)
        )
        router_indices = router_indices.reshape(-1, 1).expand(-1, D)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        )  # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)  # routed in is "sorted" / ready for EP

        out = self.shared_expert(hidden_states)

        # now that we finished expert computation -> we scatter add because we gathered previously
        out.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, D))
        return out


class Llama4Attention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(self, config, layer_idx)
        self.qk_norm = Llama4L2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)

        query_states = self.qk_norm(query_states).view(hidden_shape).transpose(1, 2)
        key_states = self.qk_norm(key_states).view(hidden_shape).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Llama4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Llama4Attention(config, layer_idx)
        self.feed_forward = Llama4Moe(config)

        self.input_layernorm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_attention_heads = config.num_attention_heads
        self.dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class Llama4ForCausalLM(LlamaForCausalLM):
    pass


class Llama4ForSequenceClassification(LlamaForSequenceClassification):
    pass


class Llama4ForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "Llama4Config",
    "Llama4PreTrainedModel",  # noqa: F822
    "Llama4Model",  # noqa: F822
    "Llama4ForCausalLM",
    "Llama4ForSequenceClassification",
    "Llama4ForTokenClassification",
]
