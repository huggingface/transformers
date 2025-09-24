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
"""PyTorch NLLB-MoE model."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ..m2m_100.modeling_m2m_100 import (
    M2M100ScaledWordEmbedding,
    M2M100SinusoidalPositionalEmbedding,
    shift_tokens_right,
)
from ..musicgen.modeling_musicgen import MusicgenAttention, MusicgenDecoder
from ..switch_transformers.modeling_switch_transformers import load_balancing_loss_func
from .configuration_nllb_moe import NllbMoeConfig


logger = logging.get_logger(__name__)


class NllbMoeScaledWordEmbedding(M2M100ScaledWordEmbedding):
    pass


class NllbMoeSinusoidalPositionalEmbedding(M2M100SinusoidalPositionalEmbedding):
    pass


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
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        hidden_states = hidden_states.to(self.dtype)
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)
        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        return top_1_mask, router_probs


class NllbMoeDenseActDense(nn.Module):
    def __init__(self, config: NllbMoeConfig, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.d_model)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
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
    def __init__(self, config: NllbMoeConfig, ffn_dim: int, expert_class: nn.Module = NllbMoeDenseActDense):
        super().__init__()
        self.num_experts = config.num_experts
        for idx in range(self.num_experts):
            self[f"expert_{idx}"] = expert_class(config, ffn_dim)

    def forward(self, hidden_states, router_mask, router_probs):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        masked_hidden_states = torch.einsum("bm,be->ebm", hidden_states, router_mask)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, idx]
            combining_weights = router_probs[token_indices, idx]
            expert_output = expert(masked_hidden_states[idx, token_indices])
            if self.moe_token_dropout > 0:
                if self.training:
                    expert_output = self.token_dropout(expert_output)
                else:
                    expert_output *= 1 - self.moe_token_dropout
            masked_hidden_states[idx, token_indices] = torch.einsum("b,be->be", combining_weights, expert_output)
        hidden_states = masked_hidden_states.sum(dim=0).reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


class NllbMoeSparseMLP(nn.Module):
    r"""
    Implementation of the NLLB-MoE sparse MLP module.
    """

    def __init__(self, config: NllbMoeConfig, ffn_dim: int, expert_class: nn.Module = NllbMoeDenseActDense):
        super().__init__()
        self.router = NllbMoeTop2Router(config)
        self.moe_token_dropout = config.moe_token_dropout
        self.token_dropout = nn.Dropout(self.moe_token_dropout)
        self.num_experts = config.num_experts
        self.experts = NllbMoeExperts(config, ffn_dim, expert_class)

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = False):
        r"""
        The goal of this forward pass is to have the same number of operation as the equivalent `NllbMoeDenseActDense`
        (mlp) layer. This means that all of the hidden states should be processed at most twice ( since we are using a
        top_2 gating mechanism). This means that we keep the complexity to O(batch_size x sequence_length x hidden_dim)
        instead of O(num_experts x batch_size x sequence_length x hidden_dim).

        1- Get the `router_probs` from the `router`. The shape of the `router_mask` is `(batch_size X sequence_length,
        num_expert)` and corresponds to the boolean version of the `router_probs`. The inputs are masked using the
        `router_mask`.

        2- Dispatch the hidden_states to its associated experts. The router probabilities are used to weight the
        contribution of each experts when updating the masked hidden states.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                The hidden states
            padding_mask (`torch.Tensor`, *optional*, defaults to `False`):
                Attention mask. Can be in the causal form or not.

        Returns:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Updated hidden states
            router_logits (`torch.Tensor` of shape `(batch_size, sequence_length, num_experts)`):
                Needed for computing the loss

        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        top_1_mask, router_probs = self.router(hidden_states, padding_mask)
        router_mask = router_probs.bool()
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        hidden_states = self.experts(hidden_states, router_mask, router_probs)

        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)
        return hidden_states, (router_probs, top_1_expert_index)


class NllbMoeAttention(MusicgenAttention):
    pass


class NllbMoeEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
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
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ff_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, attention_mask)
        if self.is_sparse:
            hidden_states, _ = hidden_states

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

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)

            hidden_states, cross_attn_weights = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.ff_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, attention_mask)
        if self.is_sparse:
            hidden_states, _ = hidden_states

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

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class NllbMoeEncoder(NllbMoePreTrainedModel):
    def __init__(self, config: NllbMoeConfig, embed_tokens: Optional[nn.Embedding] = None):
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

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        sparse_step = config.encoder_sparse_step
        self.layers = nn.ModuleList()
        for i in range(config.encoder_layers):
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            self.layers.append(NllbMoeEncoderLayer(config, is_sparse))

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input_ids, inputs_embeds)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = self._update_full_mask(
            attention_mask,
            inputs_embeds,
        )

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                continue
            else:
                hidden_states = encoder_layer(hidden_states, attention_mask)

        last_hidden_state = self.layer_norm(hidden_states)

        return MoEModelOutput(last_hidden_state=last_hidden_state)


class NllbMoeDecoder(NllbMoePreTrainedModel, MusicgenDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`NllbMoeDecoderLayer`]

    Args:
        config:
            NllbMoeConfig
        embed_tokens (nn.Embedding):
            output embedding
    """

    def __init__(self, config: NllbMoeConfig, embed_tokens: Optional[nn.Embedding] = None):
        NllbMoePreTrainedModel.__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = NllbMoeScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = True,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        input_shape = inputs_embeds.size()[:-1]

        # initialize `past_key_values`
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        if use_cache and isinstance(past_key_values, tuple):
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        attention_mask = self._update_causal_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            input_shape,
            inputs_embeds,
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
                )

            if skip_the_layer:
                continue

        hidden_states = self.layer_norm(hidden_states)

        return MoEModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class NllbMoeModel(NllbMoePreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = NllbMoeScaledWordEmbedding(vocab_size, config.d_model, padding_idx, embed_scale=embed_scale)

        self.encoder = NllbMoeEncoder(config, self.shared)
        self.decoder = NllbMoeDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], Seq2SeqMoEModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            NllbMoe uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, NllbMoeModel

        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/random-nllb-moe-2-experts")
        >>> model = SwitchTransformersModel.from_pretrained("hf-internal-testing/random-nllb-moe-2-experts")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for NllbMoeModel
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=return_dict,
            )

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqMoEModelOutput(
            past_key_values=decoder_outputs.past_key_values,
            cross_attentions=decoder_outputs.cross_attentions,
            last_hidden_state=decoder_outputs.last_hidden_state,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_attentions=decoder_outputs.attentions,
            encoder_router_logits=encoder_outputs.router_probs,
            decoder_router_logits=decoder_outputs.router_probs,
        )


@auto_docstring(
    custom_intro="""
    The NllbMoe Model with a language modeling head. Can be used for summarization.
    """
)
class NllbMoeForConditionalGeneration(NllbMoePreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: NllbMoeConfig):
        super().__init__(config)
        self.model = NllbMoeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqMoEOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            NllbMoe uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example Translation:

        ```python
        >>> from transformers import AutoTokenizer, NllbMoeForConditionalGeneration

        >>> model = NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe-54b")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")

        >>> text_to_translate = "Life is like a box of chocolates"
        >>> model_inputs = tokenizer(text_to_translate, return_tensors="pt")

        >>> # translate to French
        >>> gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("eng_Latn"))
        >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.lm_head(outputs[0])

        loss = None
        encoder_aux_loss = None
        decoder_aux_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # todo check in the config if router loss enables

            if output_router_logits:
                encoder_router_logits = outputs[-1]
                decoder_router_logits = outputs[3 if output_attentions else 4]

                # Compute the router loss (z_loss + auxiliary loss) for each router in the encoder and decoder
                encoder_router_logits, encoder_expert_indexes = self._unpack_router_logits(encoder_router_logits)
                encoder_aux_loss = load_balancing_loss_func(encoder_router_logits, encoder_expert_indexes)

                decoder_router_logits, decoder_expert_indexes = self._unpack_router_logits(decoder_router_logits)
                decoder_aux_loss = load_balancing_loss_func(decoder_router_logits, decoder_expert_indexes)

            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if output_router_logits and labels is not None:
                aux_loss = self.router_aux_loss_coef * (encoder_aux_loss + decoder_aux_loss)
                loss = loss + aux_loss

        output = (loss,) if loss is not None else ()
        if not return_dict:
            output += (lm_logits,)
            if output_router_logits:  # only return the loss if they are not None
                output += (
                    encoder_aux_loss,
                    decoder_aux_loss,
                    *outputs[1:],
                )
            else:
                output += outputs[1:]

            return output

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

    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if router_output is not None:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)

        total_router_logits = torch.cat(total_router_logits, dim=1) if len(total_router_logits) > 0 else None
        total_expert_indexes = torch.stack(total_expert_indexes, dim=1) if len(total_expert_indexes) > 0 else None
        return total_router_logits, total_expert_indexes


__all__ = [
    "NllbMoeForConditionalGeneration",
    "NllbMoeModel",
    "NllbMoePreTrainedModel",
    "NllbMoeTop2Router",
    "NllbMoeSparseMLP",
]
