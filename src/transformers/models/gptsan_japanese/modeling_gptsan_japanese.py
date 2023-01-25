# coding=utf-8
# Copyright 2022 GPTSANJapanese Authors and HuggingFace Inc. team.
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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...generation import GenerationConfig, LogitsProcessorList, TopKLogitsWarper
from ...modeling_utils import PreTrainedModel
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    ModelOutput,
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
        # Check if it is a sparse layer, if not then it is a dense layer
        self.mlp = GPTSANJapaneseSparseMLP(config)
        self.smlp = nn.Linear(config.d_model, config.d_model, bias=False)
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
        forwarded_states += torch.tanh(self.smlp(hidden_states))
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


class GPTSANJapaneseAttention(nn.Module):
    r"""
    A version of self-attention introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) using
    compatible weights of the model stored in [GPTSAN](https://github.com/tanreinama/GPTSAN/blob/main/modeling.py) to
    split key,value,query.

    Parameters:
        config : ([`GPTSANJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__()
        self.d_kernel = config.d_model // config.num_heads
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.qkv = nn.Parameter(torch.zeros([config.d_model, 3, config.num_heads, self.d_kernel]))
        self.o = nn.Parameter(torch.zeros([config.num_heads, self.d_kernel, config.d_model]))

    def _split(self, hidden_states):
        qh, kh, vh = torch.split(self.qkv, [1, 1, 1], dim=1)
        qh, kh, vh = qh.squeeze(dim=1), kh.squeeze(dim=1), vh.squeeze(dim=1)  # [input_ch, heads, kernel]
        qh, kh, vh = (
            torch.reshape(qh, (vh.shape[0], -1)),
            torch.reshape(kh, (vh.shape[0], -1)),
            torch.reshape(vh, (vh.shape[0], -1)),
        )  # [input_ch, heads×kernel]
        hidden_states_2d = torch.reshape(hidden_states, (-1, hidden_states.shape[-1]))  # [batch×sequence, input_ch]
        q = torch.mm(hidden_states_2d, qh)
        k = torch.mm(hidden_states_2d, kh)
        v = torch.mm(hidden_states_2d, vh)  # [batch×sequence, heads×kernel]
        q = torch.reshape(q, (-1, self.qkv.shape[-2], self.qkv.shape[-1]))
        k = torch.reshape(k, (-1, self.qkv.shape[-2], self.qkv.shape[-1]))
        v = torch.reshape(v, (-1, self.qkv.shape[-2], self.qkv.shape[-1]))  # [batch×sequence, heads, kernel]
        q = torch.reshape(q, (-1, hidden_states.shape[1], self.qkv.shape[-2], self.qkv.shape[-1]))
        k = torch.reshape(k, (-1, hidden_states.shape[1], self.qkv.shape[-2], self.qkv.shape[-1]))
        v = torch.reshape(
            v, (-1, hidden_states.shape[1], self.qkv.shape[-2], self.qkv.shape[-1])
        )  # [batch, sequence, heads, kernel]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # [batch, heads, sequence, kernel]
        return q, k, v

    def _split2(self, hidden_states):
        qkv = torch.einsum("bsc,cqhk->bsqhk", hidden_states, self.qkv)  # [batch, sequence, 3, heads, kernel]
        q, k, v = torch.split(qkv, [1, 1, 1], dim=2)  # [batch, sequence, 1, heads, kernel]
        q = q.squeeze(dim=2).transpose(1, 2)
        k = k.squeeze(dim=2).transpose(1, 2)
        v = v.squeeze(dim=2).transpose(1, 2)  # [batch, heads, sequence, kernel]
        return q, k, v

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        r"""
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
            layer_past (`torch.Tensor`) :
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
        q, k, v = self._split2(hidden_states)  # [batch, sequence, 1, heads, kernel]

        if use_cache is True:
            present = (k, v)
        # present shuld be ([batch, heads, sequence, hidden], [batch, heads, sequence, hidden])

        if layer_past is not None:
            pk, pv = torch.split(layer_past, [1, 1], dim=1)
            pk = pk.squeeze(dim=1)
            pv = pv.squeeze(dim=1)
            if use_cache is True:
                present = (pk, pv)
            # pk, pv shuld be [batch, heads, sequence, hidden]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        umask = attention_mask.unsqueeze(1)
        scores = torch.einsum("bhsk,bhmk->bhsm", q, k)
        scores *= self.d_kernel**-0.5
        scores *= umask
        scores -= (1 - umask) * 10000.0
        probs = torch.exp(nn.functional.log_softmax(scores, dim=-1))  # same as mesh-tensorflow
        if head_mask is not None:
            probs = probs * head_mask
        output = torch.einsum("bhsm,bhmk->bhsk", probs, v)  # [batch, heads, sequence, kernel]
        output = output.transpose(1, 2)  # [batch, sequence, heads, kernel]
        output = torch.einsum("bshk,hkc->bsc", output, self.o)  # [batch, sequence, hidden]

        if use_cache is True:
            outputs = (output, present)
        else:
            outputs = (output,)

        if output_attentions:
            outputs += (probs,)

        return outputs


class GPTSANJapaneseLayerSelfAttention(nn.Module):
    """
    Self Attention and Normalization Unit
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = GPTSANJapaneseAttention(config)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
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
            layer_past (`torch.Tensor`) :
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
        atten_out = self.SelfAttention(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attention_output = atten_out[0]

        hidden = hidden_states + self.norm(attention_output)

        outputs = (hidden,) + atten_out[1:]

        return outputs


class GPTSANJapaneseBlock(nn.Module):
    """
    Self Attention and FFN Unit
    """

    def __init__(self, config, ext_layer=False):
        super().__init__()
        self.SelfAttention = GPTSANJapaneseLayerSelfAttention(config)
        self.FeedForward = GPTSANJapaneseLayerDenseFF(config) if ext_layer else GPTSANJapaneseLayerSparseFF(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
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
            layer_past (`torch.Tensor`) :
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
        atten_out = self.SelfAttention(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attention_output = atten_out[0]

        if isinstance(self.FeedForward, GPTSANJapaneseLayerSparseFF):
            sparse_out = self.FeedForward(attention_output, output_router_tuple)
            if output_router_tuple:
                hidden, router_tuple = sparse_out
            else:
                hidden = sparse_out
        else:
            hidden = self.FeedForward(attention_output)

        outputs = (hidden,) + atten_out[1:]

        if isinstance(self.FeedForward, GPTSANJapaneseLayerSparseFF) and output_router_tuple:
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
        elif isinstance(module, GPTSANJapaneseForConditionalGeneration):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.token_bias.data.normal_(mean=0.0, std=factor * 1.0)
            module.logits.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.logits, "bias") and module.logits.bias is not None:
                module.logits.bias.data.zero_()
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
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.qkv.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.o.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
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

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In GPTSANJapanese it is usually set"
                " to the pad_token_id. See GPTSANJapanese docs for more information"
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

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
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
        num_precontext (`torch.LongTensor` of shape `(batch_size,1)`):
            length of `hybrid` input tokens in the input. Tokens up to this length refer to both front and back like
            BERT, tokens after that refer only to front like GPT. see also:
            https://github.com/tanreinama/GPTSAN/blob/main/report/model.md
        squad (`torch.Tensor` of shape `(batch_size, config.d_spout)`):
                This vector is transformed through an 8-layer FFN and can be used instead of `past_key_values`.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
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
"""


def make_attention_mask_torch(total_seq, output_seq, input_len):
    device = input_len.device
    i = torch.arange(total_seq)[:, None].to(device)
    j = torch.arange(output_seq).to(device)
    m = i >= j - output_seq + total_seq
    lm_mask = m.float()  # language model mask
    lm_mask = torch.reshape(lm_mask, [total_seq, output_seq])
    # lm_mask shuld be [sequence, sequence]
    weight = torch.transpose(torch.arange(output_seq)[:, None].to(device) < input_len, 1, 0)
    weight = weight.float()  # Masked language model mask
    # weight shuld be [batch, sequence]
    mlm_mask = torch.reshape(weight, [1, -1, output_seq]).float()
    mlm_ones = torch.ones(size=[total_seq, 1, 1], dtype=torch.float32).to(device)
    mlm_mask = mlm_ones * mlm_mask
    mlm_mask = torch.transpose(mlm_mask, 1, 0)
    # mlm_mask shuld be [batch, sequence, sequence]
    mask = ((lm_mask + mlm_mask) > 0).float()
    # mask shuld be [batch, sequence, sequence]
    return mask


@add_start_docstrings(
    "The bare GPTSAN_JAPANESE Model transformer outputting logits.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSANJapaneseForConditionalGeneration(GPTSANJapanesePreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"embed_tokens.weight",
    ]

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.num_contexts, config.d_model)
        self.token_bias = nn.Parameter(torch.zeros([config.vocab_size]))
        self.logits = nn.Linear(config.d_model, config.d_model, bias=True)
        self.logact = self.act = ACT2FN["swish"]
        self.config = copy.deepcopy(config)

        self.blocks = torch.nn.ModuleList([])
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSANJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSANJapaneseBlock(config, ext_layer=True))

        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.num_contexts, config.d_model)

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
        num_precontext: Optional[torch.LongTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], ModelOutput]:
        r"""
        Returns:
            `ModelOutput` or `namedtuple` if `return_dict` returns ModelOutput insted of namedtuple

        Example:

        ```python
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese")
        tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        x_tok = tokenizer.encode("武田信玄は、", return_tensors="pt")
        model = model.cuda()
        c = model.generate(x_tok.cuda(), max_new_tokens=50, random_seed=63)
        tokenizer.decode(c[0])
        "武田信玄は、戦国の頃より「智勇兼備」した英雄として織田信長に比されてきた戦国武将であり、..."
        ```"""
        NUM_TOKENS = self.config.vocab_size
        SEP_TOKEN = NUM_TOKENS - 5
        NOT_TOKEN = NUM_TOKENS - 4
        device = self.position_embeddings.weight.device
        if input_ids is None:
            input_ids = torch.zeros([1, 1]).int().to(device)
        num_pasts_contexts = 0
        num_batch = input_ids.shape[0]
        pasts = None
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

        if past_key_values is not None:
            if type(past_key_values) is tuple or type(past_key_values) is list:
                op = []
                for p in past_key_values:
                    if type(p) is tuple or type(p) is list:
                        p = torch.stack(p, dim=1)  # p Shuold be [batch, 2, heads, sequence, kernel]
                    if p.shape != (
                        num_batch,
                        2,
                        self.config.num_heads,
                        num_pasts_contexts,
                        self.config.d_model // self.config.num_heads,
                    ):  # pasts Should be [batch, layer, 2, heads, sqquence, kernel]
                        raise ValueError("past_key_values Tuple[Tuple[FloatTensor]]")
                    op.append(p)
                pasts = torch.stack(op, dim=1)  # pasts Shuold be [batch, layer, 2, heads, sequence, kernel]
            if pasts is None or pasts.shape != (
                num_batch,
                self.config.num_layers,
                2,
                self.config.num_heads,
                num_pasts_contexts,
                self.config.d_model // self.config.num_heads,
            ):  # pasts Should be [batch, layer, 2, heads, sqquence, kernel]
                raise ValueError("past_key_values Tuple[Tuple[FloatTensor]]")
        elif self.config.d_spout and spout is not None:
            pasts = self.spout(spout)
            pasts = torch.reshape(
                pasts,
                [
                    num_batch,
                    self.config.num_layers,
                    2,
                    self.config.num_heads,
                    num_pasts_contexts,
                    self.config.d_model // self.config.num_heads,
                ],
            )

        hidden = self.embed_tokens(input_ids)

        if pasts is None:
            pasts = [None] * self.config.num_layers
            pos = torch.arange(num_input_contexts).to(device)
            pos = torch.clip(pos, 0, self.config.num_contexts - 1)
        else:
            pasts = [p.squeeze(1) for p in torch.split(pasts, [1] * self.config.num_layers, 1)]
            pos = torch.arange(num_input_contexts).to(device) + num_pasts_contexts
            pos = torch.clip(pos, num_pasts_contexts, self.config.num_contexts - 1)

        ppos = (torch.zeros((self.config.d_model, num_input_contexts)).to(device) + pos).transpose(0, 1).long()
        hidden += torch.gather(self.position_embeddings.weight, dim=0, index=ppos)

        atten_mask = make_attention_mask_torch(num_input_contexts, num_output_contexts, num_precontext)

        if self.config.num_ext_layers > 0:
            ete = torch.gather(self.extra_position_embeddings.weight, dim=0, index=ppos)
        del ppos

        # Prepare head mask if needed
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.n_layer)  # n_layer x batch x n_heads x N x N

        out_past_key_values = tuple() if self.config.use_cache or use_cache else None
        out_hidden_states = (hidden,) if self.config.output_hidden_states or output_hidden_states else None
        out_attentions = tuple() if self.config.output_attentions or output_attentions else None
        out_router_logits = tuple() if self.config.output_router_logits or output_router_logits else None

        for layer, past in enumerate(pasts):
            if layer == self.config.num_switch_layers:
                hidden = hidden + ete

            output_router_tuple = (
                self.config.output_router_logits or output_router_logits
            ) and layer < self.config.num_switch_layers
            block_output = self.blocks[layer](
                hidden_states=hidden,
                layer_past=past,
                attention_mask=atten_mask,
                head_mask=head_mask,
                use_cache=self.config.use_cache or use_cache,
                output_attentions=self.config.output_attentions or output_attentions,
                output_router_tuple=output_router_tuple,
            )

            outpos = 0
            hidden = block_output[outpos]
            if self.config.output_hidden_states or output_hidden_states:
                out_hidden_states += (hidden,)
            if self.config.use_cache or use_cache:
                outpos += 1
                present = block_output[outpos]
                out_past_key_values += (present,)
            if self.config.output_attentions or output_attentions:
                outpos += 1
                attention_probs = block_output[outpos]
                out_attentions += (attention_probs,)
            if output_router_tuple:
                outpos += 1
                router_tuple = block_output[outpos]
                out_router_logits.append(router_tuple[0])

        hidden = self.logits(hidden)
        hidden = self.logact(hidden)

        logits = torch.einsum("bsc,vc->bsv", hidden, self.embed_tokens.weight)
        if logits.shape[-1] == self.token_bias.shape[-1]:
            logits = logits + self.token_bias

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        ret = ModelOutput()
        logits[:, :, SEP_TOKEN] = -1e10
        logits[:, :, NOT_TOKEN] = -1e10
        ret["logits"] = logits
        if self.config.output_hidden_states or output_hidden_states:
            ret["hidden_states"] = out_hidden_states
        if self.config.use_cache or use_cache:
            ret["past_key_values"] = out_past_key_values
        if self.config.output_attentions or output_attentions:
            ret["attentions"] = out_attentions
        if self.config.output_router_logits or output_router_logits:
            ret["router_logits"] = out_router_logits
        ret["loss"] = loss

        if return_dict:
            return ret

        outp = collections.namedtuple("GPTSANOutputs", " ".join(ret.keys()))
        return outp(**ret)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        connected_inputs: Optional[torch.LongTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        if past_key_values is not None:
            return {
                "input_ids": input_ids,
                "num_precontext": connected_inputs,
                "spout": None,
                "past_key_values": past_key_values,
            }
        NUM_TOKENS = self.config.vocab_size
        SOT_TOKEN = NUM_TOKENS - 7
        SEG_TOKEN = NUM_TOKENS - 2
        if connected_inputs is None:
            connected_inputs = torch.zeros(input_ids.shape[0]).int().to(input_ids.device)
        pre_input = torch.stack(
            [
                torch.tensor(
                    [SOT_TOKEN]
                    + input_ids[i, : connected_inputs[i]].cpu().numpy().tolist()
                    + [SEG_TOKEN]
                    + input_ids[i, connected_inputs[i] :].cpu().numpy().tolist()
                )
                for i in range(input_ids.shape[0])
            ]
        )
        pre_input = pre_input.to(input_ids.device)
        connected_inputs = connected_inputs + 1
        connected_inputs = torch.unsqueeze(connected_inputs, dim=1)
        if random_seed is not None:
            # PyTorchの内部を決定論的に設定する
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        return {
            "input_ids": pre_input,
            "num_precontext": connected_inputs,
            "spout": spout,
            "past_key_values": past_key_values,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return labels

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        logits_processor = super()._get_logits_processor(
            generation_config, input_ids_seq_length, encoder_input_ids, prefix_allowed_tokens_fn, logits_processor
        )
        if generation_config.top_k is not None:
            logits_processor.append(TopKLogitsWarper(generation_config.top_k))
        return logits_processor
