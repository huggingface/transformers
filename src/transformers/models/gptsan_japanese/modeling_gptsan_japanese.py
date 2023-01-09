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
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...activations import ACT2FN
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
_TOKENIZER_FOR_DOC = "GPTNeoXJapaneseTokenizer"
_CHECKPOINT_FOR_DOC = "Tanrei/GPTSAN-japanese"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Tanrei/GPTSAN-japanese",
    # See all GPTSAN-japanese models at https://huggingface.co/models?filter=gptsan-japanese
]


class GPTSANJapaneseNorm(nn.Module):
    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.d_model))
        self.bias = nn.Parameter(torch.zeros(config.d_model))
        self.epsilon = config.layer_norm_epsilon

    def forward(self, x):
        x -= torch.mean(x, dim=-1, keepdims=True)
        s = torch.mean(x**2, dim=-1, keepdims=True)
        x *= torch.rsqrt(s + self.epsilon)
        return x * self.weight + self.bias


class GPTSANJapaneseDenseActDense(nn.Module):
    def __init__(self, config: GPTSANJapaneseConfig, ext_layer=False):
        super().__init__()
        d_inter = config.d_ext if ext_layer else config.d_ff
        self.wi = nn.Linear(config.d_model, d_inter, bias=ext_layer)
        self.wo = nn.Linear(d_inter, config.d_model, bias=ext_layer)
        self.dropout = nn.Identity() if ext_layer else nn.Dropout(config.dropout_rate)
        self.act = ACT2FN["swish" if ext_layer else "relu"]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


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

        if self.training and self.jitter_noise > 0:
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


class GPTSANJapaneseLayerFF(nn.Module):
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
        self.norm = GPTSANJapaneseNorm(config)

    def forward(self, hidden_states):
        forwarded_states, _ = self.mlp(hidden_states)
        forwarded_states += torch.tanh(self.smlp(hidden_states))
        output = hidden_states + self.norm(forwarded_states)
        return output


class GPTSANJapaneseLayerEFF(nn.Module):
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
        self.norm = GPTSANJapaneseNorm(config)

    def forward(self, hidden_states):
        forwarded_states = self.mlp(hidden_states)
        output = hidden_states + self.norm(forwarded_states)
        return output


class GPTSANJapaneseAttention(nn.Module):
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

    def forward(self, hidden_states, mask, past=None, out_attentions=None):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        q, k, v = self._split2(hidden_states)  # [batch, sequence, 1, heads, kernel]

        present = (k, v)
        # present shuld be ([batch, heads, sequence, hidden], [batch, heads, sequence, hidden])

        if past is not None:
            pk, pv = torch.split(past, [1, 1], dim=1)
            pk = pk.squeeze(dim=1)
            pv = pv.squeeze(dim=1)
            # pk, pv shuld be [batch, heads, sequence, hidden]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        umask = mask.unsqueeze(1)
        scores = torch.einsum("bhsk,bhmk->bhsm", q, k)
        scores *= self.d_kernel**-0.5
        scores *= umask
        scores -= (1 - umask) * 10000.0
        probs = torch.exp(nn.functional.log_softmax(scores, dim=-1))  # same as mesh-tensorflow
        if out_attentions is not None:
            out_attentions.append(probs)
        output = torch.einsum("bhsm,bhmk->bhsk", probs, v)  # [batch, heads, sequence, kernel]
        output = output.transpose(1, 2)  # [batch, sequence, heads, kernel]
        output = torch.einsum("bshk,hkc->bsc", output, self.o)  # [batch, sequence, hidden]
        return output, present


class GPTSANJapaneseLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = GPTSANJapaneseAttention(config)
        self.norm = GPTSANJapaneseNorm(config)

    def forward(self, hidden_states, attention_mask, past=None, out_attentions=None):
        attention_output, present = self.SelfAttention(
            hidden_states,
            mask=attention_mask,
            past=past,
            out_attentions=out_attentions,
        )
        outputs = hidden_states + self.norm(attention_output)
        return outputs, present


class GPTSANJapaneseBlock(nn.Module):
    def __init__(self, config, ext_layer=False):
        super().__init__()
        self.att = GPTSANJapaneseLayerSelfAttention(config)
        self.ff = GPTSANJapaneseLayerEFF(config) if ext_layer else GPTSANJapaneseLayerFF(config)

    def forward(self, hidden_states, attention_mask, past=None, out_attentions=None):
        attention_status, present = self.att(hidden_states, attention_mask, past=past, out_attentions=out_attentions)
        outputs = self.ff(attention_status)
        return outputs, present


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
        if isinstance(module, GPTSANJapaneseNorm):
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
            module.wob.data.normal_(mean=0.0, std=factor * 1.0)
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

    The GPTSAN_JAPANESE model was proposed in General-purpose Swich transformer based Japanese language model

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
            Indices of input sequence tokens in the vocabulary. GPTSAN_JAPANESE is a model that
            generates sentence continuations or predicts tokens at mask positions. Special tokens
            required for inputs to the model are automatically appended.
        num_precontext (`torch.LongTensor` of shape `(batch_size,1)`):
            length of `hybrid` input tokens in the input.
            Tokens up to this length refer to both front and back like BERT, tokens after that refer only to front like GPT.
            see also:
            https://github.com/tanreinama/GPTSAN/blob/main/report/model.md
        squad (`torch.Tensor` of shape `(batch_size, config.d_spout)`):
                This vector is transformed through an 8-layer FFN and can be used instead of `pasts`.
        pasts (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
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


class GPTSANSentenceGenerator:
    """
    An text generator class to find token sequence from Model outputs
    """

    def __init__(self, model: GPTSANJapanesePreTrainedModel, config: GPTSANJapaneseConfig):
        self.model = model
        self.config = config

    def forward(self, x):
        """
        dummy function
        """
        pass

    def _convert_token(self, tokens):
        return [t if t != 35782 else 35593 for t in tokens]  # tokenizer bug for "," token

    def _encode_bytearray(self, tokens, tokenizer):
        if tokenizer is None:
            return tokens
        byte_token_start = 35738
        words = []
        chunk_chunks = []
        byte_tokens = []
        for i in tokens:
            if i >= byte_token_start and i < byte_token_start + 255:
                if len(chunk_chunks) > 0:
                    words.append(tokenizer.decode(chunk_chunks))
                    chunk_chunks = []
                byte_tokens.append(i - byte_token_start)
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                    byte_tokens = []
                if i == 35593:
                    chunk_chunks.append(35782)
                else:
                    chunk_chunks.append(i)
        if len(chunk_chunks) > 0:
            words.append(tokenizer.decode(chunk_chunks))
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text

    def predict_mlm(self, input_tokens, tokenizer):
        r"""
        Run the model in masked language model mode

        Example:
        ```python
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_tok = tokenizer.encode("武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。")
        >>> model = model.cuda()
        >>> res = model.generator.predict_mlm(x_tok, tokenizer)
        >>> res[0]
        '武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。'
        ```

        Args:
            input_tokens (`List[int]`) :
              tokens to input.
            tokenizer (`PreTrainedTokenizer`) :
              tokens to decode
              if passed None, token lists returned instead of string.
        Returns:
            List[str] or List[List[int]]
        """
        input_tokens = self._convert_token(input_tokens)
        NUM_TOKENS = self.config.vocab_size
        SOT_TOKEN = NUM_TOKENS - 7
        MSK_TOKEN = NUM_TOKENS - 6
        pre_input = [SOT_TOKEN] + list(input_tokens)
        connected_inputs = len(input_tokens) + 1
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            x_inp = torch.tensor([pre_input]).to(device)
            n_inp = torch.tensor([[connected_inputs]]).to(device)
            log = self.model(input_ids=x_inp, num_precontext=n_inp).logits
            logits = log.detach().cpu().numpy()[0]
        pred_token, pred_score = [], []
        for i in range(len(pre_input) - 1):
            p = int(logits[i].argmax()) if pre_input[i + 1] == MSK_TOKEN else pre_input[i + 1]
            s = float(logits[i][pre_input[i + 1]])
            pred_token.append(p)
            pred_score.append(s)
        return self._encode_bytearray(pred_token, tokenizer), pred_score

    def generate_lm(self, input_tokens, tokenizer, seed=None, top_k=120, max_generate=200, beam_width=4, batch_size=4):
        r"""
        Run the model in sentence generation mode

        Example:
        ```python
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_tok = tokenizer.encode("武田信玄は、")
        >>> model = model.cuda()
        >>> res = model.generator.generate_lm(x_tok, tokenizer)
        >>> res[0]
        '勝頼の父であり、天正四年(1576)に死去するまで甲府14万石の大名として甲府を治めた戦国大名ですが...'
        ```

        Args:
            input_tokens (`List[int]`) :
              tokens to input.
            tokenizer (`PreTrainedTokenizer`) :
              tokens to decode
              if passed None, token lists returned instead of string.
            seed (`int`, defaults to `None`) :
              random seed.
            top_k (`int`, defaults to `120`) :
              top K parameter for generation.
            max_generate (`int`, defaults to `200`) :
              maximum number of generate tokens.
            beam_width (`int`, defaults to `4`) :
              maximum beams for search token.
            batch_size (`int`, defaults to `4`) :
              batch size for input.
        Returns:
            List[str] or List[List[int]]
        """
        return self.generate_hybrid(input_tokens, tokenizer, 0, seed, top_k, max_generate, beam_width, batch_size)

    def generate_hybrid(
        self,
        input_tokens,
        tokenizer,
        connected_inputs,
        seed=None,
        top_k=120,
        max_generate=200,
        beam_width=4,
        batch_size=4,
    ):
        r"""
        Run the model in sentence generation mode
        You can specify the number of `hybrid` inputs.

        Example:
        ```python
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_tok = tokenizer.encode("武田信玄は、")
        >>> model = model.cuda()
        >>> res = model.generator.generate_hybrid(x_tok, tokenizer, connected_inputs=0)
        >>> res[0]
        '勝頼の父であり、天正四年(1576)に死去するまで甲府14万石の大名として甲府を治めた戦国大名ですが...'
        ```

        Args:
            input_tokens (`List[int]`) :
              tokens to input.
            tokenizer (`PreTrainedTokenizer`) :
              tokens to decode
              if passed None, token lists returned instead of string.
            connected_inputs (`int`) :
              the number of `hybrid` inputs.
            seed (`int`, defaults to `None`) :
              random seed.
            top_k (`int`, defaults to `120`) :
              top K parameter for generation.
            max_generate (`int`, defaults to `200`) :
              maximum number of generate tokens.
            beam_width (`int`, defaults to `4`) :
              maximum beams for search token.
            batch_size (`int`, defaults to `4`) :
              batch size for input.
        Returns:
            List[str] or List[List[int]]
        """
        input_tokens = self._convert_token(input_tokens)
        assert beam_width >= batch_size and beam_width % batch_size == 0
        NUM_TOKENS = self.config.vocab_size
        SOT_TOKEN = NUM_TOKENS - 7
        # MSK_TOKEN = NUM_TOKENS-6
        SEP_TOKEN = NUM_TOKENS - 5
        NOT_TOKEN = NUM_TOKENS - 4
        BAG_TOKEN = NUM_TOKENS - 3
        SEG_TOKEN = NUM_TOKENS - 2
        EOT_TOKEN = NUM_TOKENS - 1
        LAST_TOKEN = 35738  # <|byte0|>
        pre_input = (
            [SOT_TOKEN] + list(input_tokens)[:connected_inputs] + [SEG_TOKEN] + list(input_tokens)[connected_inputs:]
        )
        connected_inputs = connected_inputs + 1
        NUM_CTX = self.config.num_contexts
        device = next(self.model.parameters()).device
        self.model.eval()
        input_size = min(max_generate + len(pre_input), NUM_CTX)  # Transformerへ入力する長さ
        generated_all = [[] for _ in range(beam_width)]
        generated_scores = [[] for _ in range(beam_width)]
        generated_ranks = [[] for _ in range(beam_width)]
        if seed is not None and type(seed) is int:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms = True

        def input_gen():  # モデルへの入力を1ビーム分ずつ返す
            while True:
                endednum = 0  # 全ビームで終了かチェック
                for generated in generated_all[:beam_width]:
                    if len(generated) > 0 and (generated[-1] == EOT_TOKEN or len(generated) >= max_generate):
                        endednum += 1  # EOTなら終了
                if endednum == beam_width:
                    return  # 全ビームで終了なら生成終わり
                gen_x, gen_num = [], []
                for generated in generated_all[:beam_width]:
                    input_tokens = pre_input + generated  # 一つ前までの生成文を入力し、次のトークンを得る
                    input_tokens = input_tokens[-input_size:]  # モデルの最大入力数まで
                    nocon_length = (len(pre_input) - connected_inputs) + len(generated)
                    con_length = connected_inputs - max(
                        nocon_length - (input_size - connected_inputs), 0
                    )  # hybridで入力するトークン列数
                    gen_x.append(input_tokens)  # PyTorch版は可変長の入力に対応しているので後ろは無しで良い
                    gen_num.append([con_length])
                # モデルへの次の入力＝一つ前までの生成文、マルチモーダル用ベクトル入力、Hybridの部分の長さ
                yield {"x": gen_x, "num_precontext": gen_num}

        def predict_one():
            logits = []
            try:
                inp = input_gen().__next__()
            except StopIteration:
                return logits
            for d in range(0, len(inp["x"]), batch_size):
                e = min(d + batch_size, len(inp["x"]))
                x_inp = [i for i in inp["x"][d:e]]
                n_inp = [i for i in inp["num_precontext"][d:e]]
                with torch.no_grad():
                    x_inp = torch.tensor(x_inp).to(device)
                    n_inp = torch.tensor(n_inp).to(device)
                    log = self.model(input_ids=x_inp, num_precontext=n_inp).logits
                    for l in log.detach().cpu().numpy():
                        logits.append(l)
            return logits

        for pos in range(max_generate):
            for batch_dim, result in zip(range(beam_width), predict_one()):
                # select_fn
                input_size = min(max_generate + len(pre_input), NUM_CTX)  # Transformerへ入力する長さ
                # 一つ前までの生成文を入れて、1つ多いトークンが出てくるので、出てきた場所を計算
                output_pos = min(len(pre_input) + len(generated_all[batch_dim]) - 1, input_size - 1)
                logits = result[output_pos]  # 新しく出てきたトークン1つ分
                out = np.argmax(logits)
                # if out == SEP_TOKEN: # SEP_TOKENは文章の区切り文字に変換
                #    logits = [(logits[l] if TOKEN_IS_DOT_NL[l] else -1e10) for l in range(NUM_TOKENS)]
                if out != EOT_TOKEN:  # TOP_Kロジックで選択
                    ind = np.arange(NUM_TOKENS)
                    log = np.array(logits)
                    # log = (log - np.max(log)) / (np.max(log)-np.min(log))
                    log[NOT_TOKEN] = -1e10
                    log[SEP_TOKEN] = -1e10
                    exp = np.exp(log)
                    log = exp / np.sum(exp)  # softmax
                    k = np.sort(log)[-top_k]
                    log[np.where(log < k)] = 1e-10
                    out = np.random.choice(ind, 1, p=log / np.sum(log))[0]
                    rank = np.sum(log > log[out])
                else:  # NOT_TOKENは無視するトークン
                    rank = 0
                generated_all[batch_dim].append(int(out))
                generated_scores[batch_dim].append(logits[int(out)])  # 生成文のスコア
                generated_ranks[batch_dim].append(rank)  # 生成文のスコア
            # バッチ終了時にビームを評価
            beam_scores = [
                (np.mean(generated_scores[s]) if EOT_TOKEN != generated_all[s][-1] else -1e10)
                for s in range(beam_width)
            ]
            best_beam = np.argmax(beam_scores)  # この時点で終了しておらず最も良かった生成文
            if beam_scores[best_beam] != -1e10:
                for batch_dim in range(beam_width):  # 1バッチ生成時のビームの内容
                    if EOT_TOKEN == generated_all[batch_dim][-1]:  # 終了したらFixして保存しておく
                        fixed_beam = copy.copy(generated_all[batch_dim])
                        fixed_score = copy.copy(generated_scores[batch_dim])
                        fixed_rank = copy.copy(generated_ranks[batch_dim])
                        generated_all.append(fixed_beam)  # beam_width以上の次元にあるデータはFixした生成文
                        generated_scores.append(fixed_score)  # beam_width以上の次元にあるデータはFixした生成文
                        generated_ranks.append(fixed_rank)  # beam_width以上の次元にあるデータはFixした生成文
                        generated_all[batch_dim] = copy.copy(generated_all[best_beam])  # 空いたバッチで終了していないのの続きを試す
                        generated_scores[batch_dim] = copy.copy(generated_scores[best_beam])  # 空いたバッチで終了していないのの続きを試す
                        generated_ranks[batch_dim] = copy.copy(generated_ranks[best_beam])  # 空いたバッチで終了していないのの続きを試す
        # 最も良かったビーム内のトークン列を取得
        last_scores = []
        for scores, generated, rank in zip(generated_scores, generated_all, generated_ranks):
            if EOT_TOKEN in generated:
                endpos = generated.index(EOT_TOKEN)
                scores = scores[:endpos]
                generated = generated[:endpos]
                rank = rank[:endpos]
            # 最も良かった場所から取得したトークンのスコアから外れ生成を判定
            cs = [s for s, g, r in zip(scores, generated, rank) if g < LAST_TOKEN and r == 0]
            cs = cs if len(cs) > 0 else [-1e10]
            ss = scores if len(scores) > 0 else [-1e10]
            last_scores.append(-1e10 if np.mean(cs) > 0 else np.median(ss))

        # 生成文を選択
        result_tokens = []
        for generated in generated_all:
            # 特殊トークンの処理
            generated_nobag = []
            for token in generated:
                if token == BAG_TOKEN:  # BAG_TOKENは直前のトークンの繰り返し
                    if len(generated_nobag) > 0:  # 個数の指定は無いのでとりあえず3個
                        bagged = generated_nobag[-1]
                        generated_nobag.append(bagged)
                        generated_nobag.append(bagged)
                elif token < LAST_TOKEN:  # 元NOT_TOKEN等無視するトークンは入れない
                    generated_nobag.append(token)
            # 結果を保存
            result_tokens.append(generated_nobag)
        return [self._encode_bytearray(r, tokenizer) for r in result_tokens]


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
class GPTSANJapaneseModel(GPTSANJapanesePreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"wte.weight",
    ]

    def __init__(self, config: GPTSANJapaneseConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.num_contexts, config.d_model)
        self.wob = nn.Parameter(torch.zeros([config.vocab_size]))
        self.logits = nn.Linear(config.d_model, config.d_model, bias=True)
        self.logact = self.act = ACT2FN["swish"]
        self.config = copy.deepcopy(config)
        self.generator = GPTSANSentenceGenerator(self, config)

        self.blocks = torch.nn.ModuleList([])
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSANJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSANJapaneseBlock(config, ext_layer=True))

        if config.num_ext_layers > 0:
            self.ete = nn.Embedding(config.num_contexts, config.d_model)

        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)

        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        num_precontext: Optional[torch.LongTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        pasts: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], ModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_tok = tokenizer.encode("武田信玄は、")
        >>> model = model.cuda()
        >>> res = model.generator.generate_lm(x_tok, tokenizer)
        res[0]
        '勝頼の父であり、天正四年(1576)に死去するまで甲府14万石の大名として甲府を治めた戦国大名ですが...'
        ```"""
        # PyTorchの内部を決定論的に設定する
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = self.wpe.weight.device
        if input_ids is None:
            input_ids = torch.zeros([1, 1]).int().to(device)
        num_pasts_contexts = 0
        num_batch = input_ids.shape[0]
        if pasts is not None:
            num_pasts_contexts = pasts.shape[4]
        elif self.config.d_spout and spout is not None:
            num_pasts_contexts = 1

        if num_precontext is not None:
            assert (
                len(num_precontext.shape) == 2 and num_precontext.shape[1] == 1
            )  # num_precontext Should be [batch,1]
            num_precontext = torch.reshape(num_precontext, [-1])
        else:
            num_precontext = torch.zeros([num_batch]).int().to(device)

        num_input_contexts = input_ids.shape[1]
        num_output_contexts = num_input_contexts + num_pasts_contexts

        if pasts is not None:
            if type(pasts) is tuple:
                op = []
                for p in pasts:
                    if type(p) is tuple:
                        p = torch.stack(p, dim=1)  # p Shuold be [batch, 2, heads, sequence, kernel]
                    assert p.shape == (
                        num_batch,
                        2,
                        self.config.num_heads,
                        num_pasts_contexts,
                        self.config.d_model // self.config.num_heads,
                    )  # pasts Should be [batch, layer, 2, heads, sqquence, kernel]
                    op.append(p)
                pasts = torch.stack(op, dim=1)  # pasts Shuold be [batch, layer, 2, heads, sequence, kernel]
            assert pasts.shape == (
                num_batch,
                self.config.num_layers,
                2,
                self.config.num_heads,
                num_pasts_contexts,
                self.config.d_model // self.config.num_heads,
            )  # pasts Should be [batch, layer, 2, heads, sqquence, kernel]
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

        hidden = self.wte(input_ids)

        if pasts is None:
            pasts = [None] * self.config.num_layers
            pos = torch.arange(num_input_contexts).to(device)
            pos = torch.clip(pos, 0, self.config.num_contexts - 1)
        else:
            pasts = [p.squeeze(1) for p in torch.split(pasts, [1] * self.config.num_layers, 1)]
            pos = torch.arange(num_input_contexts).to(device) + num_pasts_contexts
            pos = torch.clip(pos, num_pasts_contexts, self.config.num_contexts - 1)

        ppos = (torch.zeros((self.config.d_model, num_input_contexts)).to(device) + pos).transpose(0, 1).long()
        hidden += torch.gather(self.wpe.weight, dim=0, index=ppos)

        atten_mask = make_attention_mask_torch(num_input_contexts, num_output_contexts, num_precontext)

        if self.config.num_ext_layers > 0:
            ete = torch.gather(self.ete.weight, dim=0, index=ppos)
        del ppos

        out_past_key_values = [] if self.config.use_cache else None
        out_hidden_states = [hidden] if self.config.output_hidden_states or output_hidden_states else None
        out_attentions = [] if self.config.output_attentions or output_attentions else None

        for layer, past in enumerate(pasts):
            if layer == self.config.num_switch_layers:
                hidden = hidden + ete

            hidden, present = self.blocks[layer](hidden, atten_mask, past=past, out_attentions=out_attentions)
            if self.config.use_cache:
                out_past_key_values.append(present)
            if self.config.output_hidden_states or output_hidden_states:
                out_hidden_states.append(hidden)

        if self.config.use_cache:
            out_past_key_values = tuple(out_past_key_values)

        if self.config.output_hidden_states or output_hidden_states:
            out_hidden_states = tuple(out_hidden_states)

        if self.config.output_attentions or output_attentions:
            out_attentions = tuple(out_attentions)

        hidden = self.logits(hidden)
        hidden = self.logact(hidden)

        logits = torch.einsum("bsc,vc->bsv", hidden, self.wte.weight)
        if logits.shape[-1] == self.wob.shape[-1]:
            logits = logits + self.wob

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        ret = ModelOutput()
        ret["logits"] = logits
        ret["past_key_values"] = out_past_key_values
        if self.config.output_hidden_states or output_hidden_states:
            ret["hidden_states"] = out_hidden_states
        if self.config.output_attentions or output_attentions:
            ret["attentions"] = out_attentions
        ret["loss"] = loss

        if return_dict:
            return ret

        outp = collections.namedtuple("GPTSANOutputs", " ".join(ret.keys()))
        return outp(**ret)
