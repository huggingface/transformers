# coding=utf-8
# Copyright 2023 NllbMoe Authors and HuggingFace Inc. team.
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

import math
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import OutputRecorder, can_return_tuple, check_model_inputs
from .configuration_nllb_moe import NllbMoeConfig


logger = logging.get_logger(__name__)


class NllbMoeScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding with M2M100->NllbMoe
class NllbMoeSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length
            ).to(input_ids.device)
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(
                inputs_embeds, past_key_values_length, self.padding_idx
            )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    @staticmethod
    def create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length, padding_idx):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length

    @staticmethod
    # Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings.create_position_ids_from_input_ids
    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


class NllbMoeTop2Router(nn.Module):
    """
    Router using tokens choose top-2 experts assignment.

    This router uses the same mechanism as in NLLB-MoE from the fairseq repository. Items are sorted by router_probs
    and then routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee
    that each token is processed by an expert**, or that each expert receives at least one token.

    The router combining weights are also returned to make sure that the states that are not updated will be masked.

    """

    def __init__(self, config: NllbMoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.router_ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

        self.second_expert_policy = config.second_expert_policy
        self.normalize_router_prob_before_dropping = config.normalize_router_prob_before_dropping
        self.batch_prioritized_routing = config.batch_prioritized_routing
        self.moe_eval_capacity_token_fraction = config.moe_eval_capacity_token_fraction

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def normalize_router_probabilities(self, router_probs, top_1_mask, top_2_mask):
        top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
        top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
        denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
        top_1_max_probs = top_1_max_probs / denom_s
        top_2_max_probs = top_2_max_probs / denom_s
        return top_1_max_probs, top_2_max_probs

    def route_tokens(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype = torch.float32,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> tuple:
        """
        Computes the `dispatch_mask` and the `dispatch_weights` for each experts. The masks are adapted to the expert
        capacity.
        """
        nb_tokens = router_logits.shape[0]
        # Apply Softmax and cast back to the original `dtype`
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(input_dtype)
        top_1_expert_index = torch.argmax(router_probs, dim=-1)
        top_1_mask = torch.nn.functional.one_hot(top_1_expert_index, num_classes=self.num_experts)

        if self.second_expert_policy == "sampling":
            gumbel = torch.distributions.gumbel.Gumbel(0, 1).rsample
            router_logits += gumbel(router_logits.shape).to(router_logits.device)

        # replace top_1_expert_index with min values
        logits_except_top_1 = router_logits.masked_fill(top_1_mask.bool(), float("-inf"))
        top_2_expert_index = torch.argmax(logits_except_top_1, dim=-1)
        top_2_mask = torch.nn.functional.one_hot(top_2_expert_index, num_classes=self.num_experts)

        if self.normalize_router_prob_before_dropping:
            top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(
                router_probs, top_1_mask, top_2_mask
            )

        if self.second_expert_policy == "random":
            top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
            sampled = (2 * top_2_max_probs) > torch.rand_like(top_2_max_probs.float())
            top_2_mask = top_2_mask * sampled.repeat(self.num_experts, 1).transpose(1, 0)

        if padding_mask is not None and not self.router_ignore_padding_tokens:
            if len(padding_mask.shape) == 4:
                # only get the last causal mask
                padding_mask = padding_mask[:, :, -1, :].reshape(-1)[-nb_tokens:]
            non_padding = ~padding_mask.bool()
            top_1_mask = top_1_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)
            top_2_mask = top_2_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)

        if self.batch_prioritized_routing:
            # sort tokens based on their routing probability
            # to make sure important tokens are routed, first
            importance_scores = -1 * router_probs.max(dim=1)[0]
            sorted_top_1_mask = top_1_mask[importance_scores.argsort(dim=0)]
            sorted_cumsum1 = (torch.cumsum(sorted_top_1_mask, dim=0) - 1) * sorted_top_1_mask
            locations1 = sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]

            sorted_top_2_mask = top_2_mask[importance_scores.argsort(dim=0)]
            sorted_cumsum2 = (torch.cumsum(sorted_top_2_mask, dim=0) - 1) * sorted_top_2_mask
            locations2 = sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]
            # Update 2nd's location by accounting for locations of 1st
            locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)

        else:
            locations1 = torch.cumsum(top_1_mask, dim=0) - 1
            locations2 = torch.cumsum(top_2_mask, dim=0) - 1
            # Update 2nd's location by accounting for locations of 1st
            locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)

        if not self.training and self.moe_eval_capacity_token_fraction > 0:
            self.expert_capacity = math.ceil(self.moe_eval_capacity_token_fraction * nb_tokens)
        else:
            capacity = 2 * math.ceil(nb_tokens / self.num_experts)
            self.expert_capacity = capacity if self.expert_capacity is None else self.expert_capacity

        # Remove locations outside capacity from ( cumsum < capacity = False will not be routed)
        top_1_mask = top_1_mask * torch.lt(locations1, self.expert_capacity)
        top_2_mask = top_2_mask * torch.lt(locations2, self.expert_capacity)

        if not self.normalize_router_prob_before_dropping:
            top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(
                router_probs, top_1_mask, top_2_mask
            )

        # Calculate combine_weights and dispatch_mask
        gates1 = top_1_max_probs[:, None] * top_1_mask
        gates2 = top_2_max_probs[:, None] * top_2_mask
        router_probs = gates1 + gates2

        return top_1_mask, router_probs

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor] = None) -> tuple:
        r"""
        The hidden states are reshaped to simplify the computation of the router probabilities (combining weights for
        each experts.)

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            top_1_mask (`torch.Tensor` of shape (batch_size, sequence_length)):
                Index tensor of shape [batch_size, sequence_length] corresponding to the expert selected for each token
                using the top1 probabilities of the router.
            router_probabilities (`torch.Tensor` of shape (batch_size, sequence_length, nump_experts)):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor` of shape (batch_size, sequence_length))):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)
        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        return top_1_mask, router_probs, router_logits


class NllbMoeDenseActDense(nn.Module):
    def __init__(self, config: NllbMoeConfig, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.d_model)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class NllbMoeExperts(nn.ModuleDict):
    def __init__(self, config: NllbMoeConfig, ffn_dim: int):
        super().__init__()
        self.num_experts = config.num_experts
        for idx in range(self.num_experts):
            self[f"expert_{idx}"] = NllbMoeDenseActDense(config, ffn_dim)
        self.moe_token_dropout = config.moe_token_dropout
        self.token_dropout = nn.Dropout(self.moe_token_dropout)

    def forward(self, hidden_states: torch.Tensor, router_mask: torch.Tensor, router_probs: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(router_mask, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[f"expert_{expert_idx[0]}"](current_state) * router_probs[top_x, idx, None]
            if self.moe_token_dropout > 0:
                if self.training:
                    current_hidden_states = self.token_dropout(current_hidden_states)
                else:
                    current_hidden_states *= 1 - self.moe_token_dropout
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class NllbMoeSparseMLP(nn.Module):
    r"""
    Implementation of the NLLB-MoE sparse MLP module.
    """

    def __init__(self, config: NllbMoeConfig, ffn_dim: int):
        super().__init__()
        self.router = NllbMoeTop2Router(config)
        self.num_experts = config.num_experts
        self.experts = NllbMoeExperts(config, ffn_dim)

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_1_mask, router_probs, _ = self.router(hidden_states, padding_mask)
        hidden_states = self.experts(hidden_states, top_1_mask, router_probs)
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class NllbMoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: Optional[float] = 0.0,
        is_decoder: Optional[bool] = False,
        bias: Optional[bool] = True,
        is_causal: Optional[bool] = False,
        config: Optional[NllbMoeConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_cross_attention = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)
        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states).view(*kv_input_shape).transpose(1, 2)
            value_states = self.v_proj(current_states).view(*kv_input_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class NllbMoeEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False, layer_idx: int = 0):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        if not self.is_sparse:
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.encoder_ffn_dim)
        else:
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.encoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        self.ff_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.ff_layer_norm(hidden_states)
        if self.is_sparse:
            hidden_states = self.ffn(hidden_states, attention_mask)
        else:
            hidden_states = self.ffn(hidden_states)
        hidden_states = self.ff_dropout(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class NllbMoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(config.dropout)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = NllbMoeAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)
        if not self.is_sparse:
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.decoder_ffn_dim)
        else:
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.decoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        self.ff_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)

            hidden_states, _ = self.cross_attention(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_values=past_key_values,
                attention_mask=encoder_attention_mask,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ff_layer_norm(hidden_states)
        if self.is_sparse:
            hidden_states = self.ffn(hidden_states, attention_mask)
        else:
            hidden_states = self.ffn(hidden_states)

        hidden_states = self.ff_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


@auto_docstring
class NllbMoePreTrainedModel(PreTrainedModel):
    config: NllbMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NllbMoeEncoderLayer", "NllbMoeDecoderLayer"]
    # TODO: If anyone is up to it to make sure tests pass etc
    # Flash attention has problems due to not preparing masks the same way as eager/sdpa
    # SDPA has more flaky logits which requires more time to look into tests
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False


class NllbMoeEncoder(NllbMoePreTrainedModel):
    _can_record_outputs = {
        "hidden_states": NllbMoeEncoderLayer,
        "router_logits": OutputRecorder(NllbMoeTop2Router, index=2),
        "attentions": NllbMoeAttention,
    }

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = NllbMoeScaledWordEmbedding(
            config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        )

        self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        sparse_step = config.encoder_sparse_step
        self.layers = nn.ModuleList()
        for i in range(config.encoder_layers):
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            self.layers.append(NllbMoeEncoderLayer(config, is_sparse, layer_idx=i))

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input_ids, inputs_embeds)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                continue
            else:
                hidden_states = encoder_layer(hidden_states, attention_mask, **kwargs)

        last_hidden_state = self.layer_norm(hidden_states)
        return MoEModelOutput(last_hidden_state=last_hidden_state)


class NllbMoeDecoder(NllbMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`NllbMoeDecoderLayer`]

    Args:
        config:
            NllbMoeConfig
        embed_tokens (nn.Embedding):
            output embedding
    """

    _can_record_outputs = {
        "hidden_states": NllbMoeDecoderLayer,
        "attentions": OutputRecorder(NllbMoeAttention, layer_name="self_attn", index=1),
        "router_logits": OutputRecorder(NllbMoeTop2Router, index=2),
        "cross_attentions": OutputRecorder(NllbMoeAttention, layer_name="cross_attention", index=1),
    }

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = NllbMoeScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )

        sparse_step = config.decoder_sparse_step
        self.layers = nn.ModuleList()
        for i in range(config.decoder_layers):
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            self.layers.append(NllbMoeDecoderLayer(config, is_sparse, layer_idx=i))

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    @check_model_inputs()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        input_shape = inputs_embeds.size()[:-1]

        # initialize `past_key_values`
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
            )

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        # embed positions
        positions = self.embed_positions(input_ids, inputs_embeds, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])
            skip_the_layer = self.training and dropout_probability < self.layerdrop
            if not skip_the_layer or synced_gpus:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,  # as a positional argument for gradient checkpointing
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            if skip_the_layer:
                continue

        last_hidden_states = self.layer_norm(hidden_states)

        return MoEModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states, past_key_values=past_key_values
        )


@auto_docstring
class NllbMoeModel(NllbMoePreTrainedModel):
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = NllbMoeScaledWordEmbedding(vocab_size, config.d_model, padding_idx, embed_scale=embed_scale)

        self.encoder = NllbMoeEncoder(config)
        self.decoder = NllbMoeDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], Seq2SeqMoEModelOutput]:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqMoEModelOutput(
            past_key_values=decoder_outputs.past_key_values,
            cross_attentions=decoder_outputs.cross_attentions,
            last_hidden_state=decoder_outputs.last_hidden_state,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_attentions=decoder_outputs.attentions,
            encoder_router_logits=encoder_outputs.router_logits,
            decoder_router_logits=decoder_outputs.router_logits,
        )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@auto_docstring(
    custom_intro="""
    The NllbMoe Model with a language modeling head. Can be used for summarization.
    """
)
class NllbMoeForConditionalGeneration(NllbMoePreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = {
        "lm_head.weight": "model.shared.weight",
    }

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)
        self.model = NllbMoeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.num_experts = config.num_experts
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], Seq2SeqMoEOutput]:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )
        lm_logits = self.lm_head(outputs[0])

        loss = None
        encoder_aux_loss = None
        decoder_aux_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # todo check in the config if router loss enables

            if output_router_logits:
                encoder_router_logits = outputs.encoder_router_logits
                decoder_router_logits = outputs.decoder_router_logits
                encoder_aux_loss = load_balancing_loss_func(
                    encoder_router_logits, self.num_experts, top_k=2, attention_mask=attention_mask
                )
                decoder_aux_loss = load_balancing_loss_func(
                    decoder_router_logits, self.num_experts, top_k=2, attention_mask=decoder_attention_mask
                )

            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if output_router_logits and labels is not None:
                aux_loss = self.router_aux_loss_coef * (encoder_aux_loss + decoder_aux_loss)
                loss = loss + aux_loss

        return Seq2SeqMoEOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            cross_attentions=outputs.cross_attentions,
            encoder_aux_loss=encoder_aux_loss,
            decoder_aux_loss=decoder_aux_loss,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            decoder_hidden_states=outputs.decoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            decoder_attentions=outputs.decoder_attentions,
            encoder_router_logits=outputs.encoder_router_logits,
            decoder_router_logits=outputs.decoder_router_logits,
        )


__all__ = [
    "NllbMoeForConditionalGeneration",
    "NllbMoeModel",
    "NllbMoePreTrainedModel",
    "NllbMoeTop2Router",
    "NllbMoeSparseMLP",
]
