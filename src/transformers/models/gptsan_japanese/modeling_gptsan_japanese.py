# coding=utf-8
# Copyright 2023 Toshiyuki Sakamoto(tanreinama) and HuggingFace Inc. team.
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
""" PyTorch GPTSANJapanese model."""


import collections
import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
)
from .configuration_gptsan_japanese import GPTSANJapaneseConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPTSANJapaneseConfig"
_TOKENIZER_FOR_DOC = "GPTSANJapaneseTokenizer"
_CHECKPOINT_FOR_DOC = "Tanrei/GPTSAN-japanese"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Tanrei/GPTSAN-japanese",
    # See all GPTSAN-japanese models at https://huggingface.co/models?filter=gptsan-japanese
]


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.router_z_loss_func
def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.load_balancing_loss_func
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


class GPTSANJapaneseDenseActDense(nn.Module):
    """
    FFN Layer for Switch Transformer and Extra layers

    GPTSAN can mix Switch Transformer layers and normal Transformer layers This class is used as Expert in Switch
    Transformer layers and as FFN in regular Transformer layers. RELU is used in the Switch Transformer layer, and
    Swish is used in the normal Transformer layer, so there is a choice of which is used in the argument.

    """

    def __init__(self, config: GPTSANJapaneseConfig, ext_layer=False):
        super().__init__()
        d_inter = config.d_ext if ext_layer else config.d_ff
        self.wi = nn.Linear(config.d_model, d_inter, bias=ext_layer)
        self.wo = nn.Linear(d_inter, config.d_model, bias=ext_layer)
        self.dropout = nn.Identity() if ext_layer else nn.Dropout(config.dropout_rate)
        self.act = ACT2FN["swish" if ext_layer else "relu"]

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router with SwitchTransformers->GPTSANJapanese
class GPTSANJapaneseTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.jitter_noise > 0:
            # Get the lower and upper bound of the uniform distribution
            # Adapted from: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = torch.rand(hidden_states.shape, device=hidden_states.device, dtype=self.dtype)
            uniform_distrib = uniform_distrib * (distrib_lower_bound - distrib_upper_bound)

            uniform_distrib = uniform_distrib + distrib_upper_bound
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= uniform_distrib

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP with SwitchTransformers->GPTSANJapanese
class GPTSANJapaneseSparseMLP(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: GPTSANJapaneseConfig, expert_class: nn.Module = GPTSANJapaneseDenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = GPTSANJapaneseTop1Router(config)

        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.

        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices])

        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)


class GPTSANJapaneseLayerSparseFF(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`GPTSANJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__()
        self.mlp = GPTSANJapaneseSparseMLP(config)
        self.soft_bypass_mlp = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, output_router_logits):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
            output_router_logits (`bool`) :
                output experts router output.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        forwarded_states, router_tuple = self.mlp(hidden_states)
        forwarded_states += torch.tanh(self.soft_bypass_mlp(hidden_states))
        output = hidden_states + self.norm(forwarded_states)

        if output_router_logits and router_tuple is not None:
            return output, router_tuple
        else:
            return output


class GPTSANJapaneseLayerDenseFF(nn.Module):
    r"""
    Extra Transformers Feed Forward layer module.

    Parameters:
        config : ([`GPTSANJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__()
        # Check if it is a sparse layer, if not then it is a dense layer
        self.mlp = GPTSANJapaneseDenseActDense(config, ext_layer=True)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        forwarded_states = self.mlp(hidden_states)
        output = hidden_states + self.norm(forwarded_states)
        return output


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->GPTSANJapanese
class GPTSANJapaneseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GPTSANJapaneseLayerSelfAttention(nn.Module):
    """
    Self Attention and Normalization Unit
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.self_attn = GPTSANJapaneseAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            is_decoder=True,
            bias=has_relative_attention_bias,
        )
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        r"""
        Self-attention and normalize block.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
            past_key_value (`torch.Tensor`) :
                past status from previous present output with using `use_cache`.
            attention_mask (`torch.Tensor`) :
                apply mask for attention.
            head_mask (`torch.Tensor`) :
                apply mask for heads.
            use_cache (`bool`) :
                output present key values.
            output_attentions (`bool`) :
                output attention probabirities.
        Returns:
            Tuple[torch.Tensor[num_groups, tokens_per_group, hidden_dim],...]
        """
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        atten_out = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attn_weights = (atten_out[1],)
        else:
            attn_weights = ()

        attention_output = atten_out[0]

        hidden = hidden_states + self.norm(attention_output)

        if use_cache:
            outputs = (hidden, atten_out[2])  # hidden, present, (attentions)
        else:
            outputs = (hidden,)  # hidden, (attentions)

        return outputs + attn_weights


class GPTSANJapaneseBlock(nn.Module):
    """
    Self Attention and FFN Unit
    """

    def __init__(self, config, ext_layer=False):
        super().__init__()
        self.self_attn = GPTSANJapaneseLayerSelfAttention(config)
        self.feed_forward = GPTSANJapaneseLayerDenseFF(config) if ext_layer else GPTSANJapaneseLayerSparseFF(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_router_tuple: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        r"""
        GPTSAN transformer block.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
            past_key_value (`torch.Tensor`) :
                past status from previous present output with using `use_cache`.
            attention_mask (`torch.Tensor`) :
                apply mask for attention.
            head_mask (`torch.Tensor`) :
                apply mask for heads.
            use_cache (`bool`) :
                output present key values.
            output_attentions (`bool`) :
                output attention probabirities.
            output_router_tuple:
                output experts router logits and expert id.
        Returns:
            Tuple[torch.Tensor[num_groups, tokens_per_group, hidden_dim],...]
        """
        atten_out = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attention_output = atten_out[0]

        if isinstance(self.feed_forward, GPTSANJapaneseLayerSparseFF):
            sparse_out = self.feed_forward(attention_output, output_router_tuple)
            if output_router_tuple:
                hidden, router_tuple = sparse_out
            else:
                hidden = sparse_out
        else:
            hidden = self.feed_forward(attention_output)

        outputs = (hidden,) + atten_out[1:]

        if isinstance(self.feed_forward, GPTSANJapaneseLayerSparseFF) and output_router_tuple:
            outputs += (router_tuple,)

        return outputs


class GPTSANJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTSANJapaneseConfig
    base_model_prefix = "gptsan_japanese"
    supports_gradient_checkpointing = False
    _no_split_modules = ["GPTSANJapaneseBlock"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(factor * 1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSANJapaneseModel):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.embed_tokens.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "extra_position_embeddings") and module.extra_position_embeddings is not None:
                module.extra_position_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, (GPTSANJapaneseModel, GPTSANJapaneseForConditionalGeneration)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.final_logits_bias.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSANJapaneseDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, GPTSANJapaneseAttention):
            # Multi-headed attention
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.k_proj.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.v_proj.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.q_proj.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
        elif isinstance(module, GPTSANJapaneseSparseMLP):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.router.classifier.weight.data.normal_(mean=0.0, std=factor * 1)
            for idx in range(self.config.num_experts):
                module.experts[f"expert_{idx}"].wi.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
                module.experts[f"expert_{idx}"].wo.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (GPTSANJapaneseAttention,)):
            module.gradient_checkpointing = value

    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


GPTSAN_JAPANESE_START_DOCSTRING = r"""

    The [GPTSAN_JAPANESE](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
    based Japanese language model

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTSANJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPTSAN_JAPANESE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. GPTSAN_JAPANESE is a model that generates sentence
            continuations or predicts tokens at mask positions. Special tokens required for inputs to the model are
            automatically appended.
        squad (`torch.Tensor` of shape `(batch_size, config.d_spout)`):
                This vector is transformed through an 8-layer FFN and can be used instead of `past_key_values`.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size,
"""


# Create a mask to be used when making the prefix Input length of Prefix-LM variable
# When prefix_area_len=0, it is equivalent to regular causal mask
# When prefix_area_len=total_seq, it is equivalent to no mask
# Otherwise, no mask at token positions from 0 to prefix_area_len, and regular causal mask thereafter.
#
# ex, total_seq=6, output_seq=6;
# prefix_area_len=0  prefix_area_len=3  prefix_area_len=5
# | 0 1 1 1 1 1 |    | 0 0 0 1 1 1 |    | 0 0 0 0 0 1 |
# | 0 0 1 1 1 1 |    | 0 0 0 1 1 1 |    | 0 0 0 0 0 1 |
# | 0 0 0 1 1 1 |    | 0 0 0 1 1 1 |    | 0 0 0 0 0 1 |
# | 0 0 0 0 1 1 |    | 0 0 0 0 1 1 |    | 0 0 0 0 0 1 |
# | 0 0 0 0 0 1 |    | 0 0 0 0 0 1 |    | 0 0 0 0 0 1 |
# | 0 0 0 0 0 0 |    | 0 0 0 0 0 0 |    | 0 0 0 0 0 0 |
#
# Normally, the length of the prefix is set according to the position of the separator_token inserted by the tokenizer.
# Since the prefix part refers back and forth, it can also work as an MLM. The part after that can be used for text generation as LM.
#
# ex;
# input tokens are: x_token = tokenizer.encode("ぁぃぅぇ")
# mask is;
# SOT | 0 1 1 1 1 1 |
# SEG | 0 0 1 1 1 1 |
# ぁ  | 0 0 0 1 1 1 |
# ぃ  | 0 0 0 0 1 1 |
# ぅ  | 0 0 0 0 0 1 |
# ぇ  | 0 0 0 0 0 0 |
#
# input tokens are: x_token = tokenizer.encode("", prefix_text="ぁぃぅぇ")
# mask is;
# SOT | 0 0 0 0 0 1 |
# ぁ  | 0 0 0 0 0 1 |
# ぃ  | 0 0 0 0 0 1 |
# ぅ  | 0 0 0 0 0 1 |
# ぇ  | 0 0 0 0 0 1 |
# SEG | 0 0 0 0 0 0 |
#
# input tokens are: x_token = tokenizer.encode("ぅぇ", prefix_text="ぁぃ")
# mask is;
# SOT | 0 0 0 1 1 1 |
# ぁ  | 0 0 0 1 1 1 |
# ぃ  | 0 0 0 1 1 1 |
# SEG | 0 0 0 0 1 1 |
# ぅ  | 0 0 0 0 0 1 |
# ぇ  | 0 0 0 0 0 0 |
def make_variable_prefix_area_mask(total_seq, output_seq, prefix_area_len):
    r"""
    Args:
        total_seq (`int`) :
            inputs to token length.
        output_seq (`int`) :
            inputs to output token length.
        prefix_area_len (`torch.Tensor` of shape [batch_size]) :
            inputs to prefix area length of each batch.
    Returns:
        torch.Tensor[batch_size, 1, total_seq, output_seq]
    """
    device = prefix_area_len.device
    inp_index = torch.arange(total_seq)[:, None].to(device)
    out_index = torch.arange(output_seq).to(device)
    m = inp_index >= out_index - output_seq + total_seq
    lm_mask = m.float()  # language model mask
    lm_mask = torch.reshape(lm_mask, [total_seq, output_seq])
    # lm_mask shuld be [sequence, sequence]
    weight = torch.transpose(torch.arange(output_seq)[:, None].to(device) < prefix_area_len, 1, 0)
    weight = weight.float()  # mask for prefix area
    # weight shuld be [batch, sequence]
    mlm_mask = torch.reshape(weight, [1, -1, output_seq]).float()
    mlm_ones = torch.ones(size=[total_seq, 1, 1], dtype=torch.float32).to(device)
    mlm_mask = mlm_ones * mlm_mask
    mlm_mask = torch.transpose(mlm_mask, 1, 0)
    # mlm_mask shuld be [batch, sequence, sequence]
    mask = ((lm_mask + mlm_mask) > 0).float()  # marge prefix area to causal mask
    # mask shuld be [batch, sequence, sequence]
    return 1 - mask.unsqueeze(dim=1)  # [batch, 1, sequence, sequence]


@add_start_docstrings(
    "The bare GPTSAN_JAPANESE Model transformer outputting raw hidden-states without any specific head on top.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSANJapaneseModel(GPTSANJapanesePreTrainedModel):
    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.config = copy.deepcopy(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.last_project = nn.Linear(config.d_model, config.d_model, bias=True)
        self.act = ACT2FN["swish"]

        self.blocks = torch.nn.ModuleList([])
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSANJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSANJapaneseBlock(config, ext_layer=True))

        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        num_precontext: Optional[torch.LongTensor] = None,
    ) -> Union[MoEModelOutputWithPastAndCrossAttentions, Tuple[torch.FloatTensor]]:
        r"""
        num_precontext (`torch.LongTensor` of shape `(batch_size,1)`):
            length of `hybrid` input tokens in the input. Tokens up to this length refer to both front and back like
            BERT, tokens after that refer only to front like GPT. see also:
            https://github.com/tanreinama/GPTSAN/blob/main/report/model.md

        Returns:
            `MoEModelOutputWithPastAndCrossAttentions` or `tuple` if `return_dict` returns
            MoEModelOutputWithPastAndCrossAttentions insted of tuple
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        device = self.position_embeddings.weight.device
        if input_ids is None:
            input_ids = torch.zeros([1, 1]).int().to(device)
        num_pasts_contexts = 0
        num_batch = input_ids.shape[0]
        pasts_or_squad_value = None
        if past_key_values is not None:
            num_pasts_contexts = past_key_values[0][0].shape[2]
        elif self.config.d_spout and spout is not None:
            num_pasts_contexts = 1

        if num_precontext is not None:
            if not (
                len(num_precontext.shape) == 2 and num_precontext.shape[1] == 1
            ):  # num_precontext Should be [batch,1]
                raise ValueError("num_precontext should be [batch, 1] size.")
            num_precontext = torch.reshape(num_precontext, [-1])
        else:
            num_precontext = torch.zeros([num_batch]).int().to(device)

        num_input_contexts = input_ids.shape[1]
        num_output_contexts = num_input_contexts + num_pasts_contexts

        hidden_states = self.embed_tokens(input_ids)

        if past_key_values is not None:
            pasts_or_squad_value = past_key_values
        elif self.config.d_spout and spout is not None:
            pasts_or_squad_value = self.spout(spout)
            pasts_or_squad_value = torch.reshape(
                pasts_or_squad_value,
                [
                    num_batch,
                    self.config.num_layers,
                    2,
                    self.config.num_heads,
                    num_pasts_contexts,
                    self.config.d_model // self.config.num_heads,
                ],
            )
            pasts_or_squad_value = torch.split(pasts_or_squad_value, [1] * self.config.num_layers, dim=1)
            pasts_or_squad_value = tuple(
                torch.split(a.squeeze(1), [1, 1], dim=1).squeeze(1) for a in pasts_or_squad_value
            )
        else:
            pasts_or_squad_value = [None] * self.config.num_layers

        token_position = torch.arange(num_input_contexts).to(device) + num_pasts_contexts
        token_position = torch.clip(token_position, num_pasts_contexts, self.config.max_position_embeddings - 1)

        gather_position = (
            (torch.zeros((self.config.d_model, num_input_contexts)).to(device) + token_position).transpose(0, 1).long()
        )
        hidden_states += torch.gather(self.position_embeddings.weight, dim=0, index=gather_position)

        attention_mask = (
            make_variable_prefix_area_mask(num_input_contexts, num_output_contexts, num_precontext) * -10000.0
        )

        if self.config.num_ext_layers > 0:
            extra_position_embedding = torch.gather(
                self.extra_position_embeddings.weight, dim=0, index=gather_position
            )
        del gather_position

        # Prepare head mask if needed
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.n_layer)  # n_layer x batch x n_heads x N x N

        present_key_value_states = tuple() if self.config.use_cache or use_cache else None
        out_hidden_states = (hidden_states,) if self.config.output_hidden_states or output_hidden_states else None
        all_attentions = tuple() if self.config.output_attentions or output_attentions else None
        all_router_probs = tuple() if self.config.output_router_logits or output_router_logits else None

        for layer, past in enumerate(pasts_or_squad_value):
            if layer == self.config.num_switch_layers:
                hidden_states = hidden_states + extra_position_embedding

            output_router_tuple = (
                self.config.output_router_logits or output_router_logits
            ) and layer < self.config.num_switch_layers
            block_output = self.blocks[layer](
                hidden_states=hidden_states,
                past_key_value=past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=self.config.use_cache or use_cache,
                output_attentions=self.config.output_attentions or output_attentions,
                output_router_tuple=output_router_tuple,
            )

            outpos = 0
            hidden_states = block_output[outpos]
            if self.config.output_hidden_states or output_hidden_states:
                out_hidden_states += (hidden_states,)
            if self.config.use_cache or use_cache:
                outpos += 1
                present = block_output[outpos]
                present_key_value_states += (present,)
            if self.config.output_attentions or output_attentions:
                outpos += 1
                attention_probs = block_output[outpos]
                all_attentions += (attention_probs,)
            if output_router_tuple:
                outpos += 1
                router_tuple = block_output[outpos]
                all_router_probs.append(router_tuple[0])

        hidden_states = self.last_project(hidden_states)
        hidden_states = self.act(hidden_states)

        ret = collections.OrderedDict()
        ret["last_hidden_state"] = hidden_states
        if self.config.use_cache or use_cache:
            ret["past_key_values"] = present_key_value_states
        if self.config.output_hidden_states or output_hidden_states:
            ret["hidden_states"] = out_hidden_states
        if self.config.output_attentions or output_attentions:
            ret["attentions"] = all_attentions
        if self.config.output_router_logits or output_router_logits:
            ret["router_probs"] = all_router_probs

        if not return_dict:
            return tuple(list(ret.values()))

        return MoEModelOutputWithPastAndCrossAttentions(**ret)


@add_start_docstrings(
    "The bare GPTSAN_JAPANESE Model with a language modeling head.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSANJapaneseForConditionalGeneration(GPTSANJapanesePreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"lm_head.weight",
    ]

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__(config)
        self.model = GPTSANJapaneseModel(config)
        self.register_buffer("final_logits_bias", torch.zeros([1, config.vocab_size]))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if not self.config.torchscript:
            self.lm_head.weight = self.model.embed_tokens.weight

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], MoECausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
            `MoECausalLMOutputWithPast` or `tuple` if `return_dict` returns MoECausalLMOutputWithPast insted of tuple

        Example:

        ```python
        from transformers import AutoModel, AutoTokenizer, trainer_utils

        device = "cuda"
        model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")

        # Text Generation with regular LM Model
        x_token = tokenizer.encode("織田信長は、", return_tensors="pt").to(device)
        trainer_utils.set_seed(30)
        gen_token = model.generate(x_token, max_new_tokens=50)
        tokenizer.decode(gen_token[0])
        "織田信長は、政治・軍事の中枢まで掌握した政治家であり、日本史上類を見ない驚異的な軍事侵攻を続け..."

        # Text Generation with Prefix-LM Model
        x_token = tokenizer.encode("", prefix_text="織田信長は、", return_tensors="pt").to(device)
        trainer_utils.set_seed(30)
        gen_token = model.generate(x_token, max_new_tokens=50)
        tokenizer.decode(gen_token[0])
        "織田信長は、政治・外交で数々の戦果を上げるが、1568年からは、いわゆる本能寺の変で細川晴元に暗殺される..."

        # Simultaneously Text Generation And Masked Language Model
        x_token = tokenizer.encode(
            "", prefix_text="武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。", return_tensors="pt"
        ).to(device)
        trainer_utils.set_seed(30)
        out_lm_token = model.generate(x_token, max_new_tokens=50)
        out_mlm_token = model(x_token).logits.argmax(axis=-1)
        tokenizer.decode(out_mlm_token[0])
        "武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。"
        tokenizer.decode(out_lm_token[0][x_token.shape[1] :])
        "武田氏の三代に渡った武田家のひとり\n甲斐市に住む、日本史上最大の戦国大名。..."
        ```"""
        SEP_TOKEN = self.config.pad_token_id
        NOT_TOKEN = self.config.unk_token_id
        SEG_TOKEN = self.config.separator_token_id
        use_cache = use_cache or self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_return_dict = True
        model_output_hidden_states = True
        num_precontext = None
        if input_ids is not None:
            num_batch = input_ids.shape[0]
            num_precontext = torch.zeros([num_batch]).int().to(input_ids.device)
            where_separators = torch.where(input_ids == SEG_TOKEN)
            num_precontext[where_separators[0]] += where_separators[1]
            num_precontext = num_precontext.unsqueeze(1)

        outputs = self.model(
            input_ids,
            spout,
            past_key_values,
            head_mask,
            use_cache,
            inputs_embeds,
            decoder_inputs_embeds,
            output_attentions,
            model_output_hidden_states,
            model_return_dict,
            output_router_logits,
            num_precontext,
        )

        lm_logits = self.lm_head(outputs[0])
        if lm_logits.shape[-1] == self.final_logits_bias.shape[-1]:
            lm_logits = lm_logits + self.final_logits_bias

        loss = None
        z_loss = None
        router_probs = None
        aux_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            if output_router_logits:
                # Compute the router loss (z_loss + auxiliary loss) for each router in the encoder and decoder
                router_logits, expert_indexes = self._unpack_router_logits(outputs.router_probs)
                z_loss = router_z_loss_func(router_logits)
                router_probs = nn.Softmax(dim=-1)(router_logits)
                aux_loss = load_balancing_loss_func(router_probs, expert_indexes)

            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        # Ignore the two tokens when generating
        if SEP_TOKEN < lm_logits.shape[2]:
            lm_logits[:, :, SEP_TOKEN] = -1e10
        if NOT_TOKEN < lm_logits.shape[2]:
            lm_logits[:, :, NOT_TOKEN] = -1e10

        ret = collections.OrderedDict()
        if loss is not None:
            ret["loss"] = loss
        if lm_logits is not None:
            ret["logits"] = lm_logits
        if self.config.use_cache or use_cache:
            ret["past_key_values"] = outputs.past_key_values
        if output_hidden_states:
            ret["hidden_states"] = outputs.hidden_states
        if output_attentions:
            ret["attentions"] = outputs.attentions
        if self.config.output_router_logits or output_router_logits:
            ret["router_logits"] = outputs.router_probs
        if z_loss is not None:
            ret["z_loss"] = z_loss
        if aux_loss is not None:
            ret["aux_loss"] = aux_loss

        if not return_dict:
            return tuple(list(ret.values()))

        return MoECausalLMOutputWithPast(**ret)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            return {
                "input_ids": input_ids[:, -1:],
                "spout": None,
                "past_key_values": past_key_values,
            }
        return {
            "input_ids": input_ids,
            "spout": spout,
            "past_key_values": None,
        }

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.prepare_decoder_input_ids_from_labels with SwitchTransformers->GPTSANJapanese
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # Copied from transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration.resize_token_embeddings with MBart->GPTSANJapanese
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # Copied from transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration._resize_final_logits_bias with MBart->GPTSANJapanese
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.set_output_embeddings with SwitchTransformers->GPTSANJapanese
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.get_output_embeddings with SwitchTransformers->GPTSANJapanese
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration._unpack_router_logits with SwitchTransformers->GPTSANJapanese
    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if router_output[0] is not None:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)
