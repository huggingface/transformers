# coding=utf-8
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch LLaMa model."""
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_llama import LLaMaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LLaMaConfig"

# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->LLaMa
class LLaMaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Construct a RMSNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # LLaMaLayerNorm uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LLaMaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMaConfig
    base_model_prefix = "llama"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaMaLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LLaMaLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LLaMaModel):
            module.gradient_checkpointing = value


class LLaMaAttention(nn.Module):
    def __init__(self, config: LLaMaConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states,
        attention_mask,
        complex_freq,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.qkv(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3, head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute token offset for rotary embeddings (when decoding)
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
        query = apply_rotary_pos_emb(embedding=query, complex_freq=complex_freq, offset=offset)
        key = apply_rotary_pos_emb(embedding=key, complex_freq=complex_freq, offset=offset)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.o(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.shape
        key_length = key.shape[-2]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)

        # TODO @thomasw21: Use `baddbmm` in order to fuse the kernels together. This comes with a loss of precision compared to original inference code
        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.head_size)
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # Build attention mask
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]
        if attention_mask is not None:
            attention_mask = causal_mask * attention_mask
        else:
            attention_mask = causal_mask

        if attn_scores.dtype == torch.float16:
            attn_scores = attn_scores.to(torch.float)
        attn_scores = torch.masked_fill(attn_scores, ~attention_mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = max_position_embeddings
        self.build_new_freq(length=max_position_embeddings, device=None)

    def build_new_freq(self, length, device):
        assert self.dim % 2 == 0
        assert self.max_seq_len_cached <= length
        self.max_seq_len_cached = length
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device)[: self.dim // 2] / self.dim)
        )
        self.device = self.inv_freq.device

        # Build here to make `torch.jit.trace` work.
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # We don't register as a buffer as this needs to be kept in fp32 at all time.
        self.complex_freq = torch.polar(torch.ones((1,), device=device, dtype=torch.float), freqs)

    def forward(self, device, seq_len=None):
        if seq_len > self.max_seq_len_cached or self.device != device:
            self.build_new_freq(length=max(self.max_seq_len_cached, seq_len), device=device)

        return self.complex_freq[:seq_len, ...]


def apply_rotary_pos_emb(embedding, complex_freq, offset: int = 0):
    complex_freq = complex_freq[..., offset : embedding.shape[-2] + offset, :]
    # q[...,::2] is considered the real part, q[...,1::2] is the imaginary part
    assert complex_freq.dtype == torch.complex64
    assert embedding.shape[-1] % 2 == 0
    complex_embed = torch.view_as_complex(embedding.float().view(*embedding.shape[:-1], embedding.shape[-1] // 2, 2))
    complex_embed_rot = complex_embed * complex_freq
    embed_rot = torch.view_as_real(complex_embed_rot).view(embedding.shape)
    return embed_rot.type_as(embedding)


class LLaMaFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.wo(F.silu(self.wi_0(hidden_states)) * self.wi_1(hidden_states))


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer with GPTNeoX->LLaMa
class LLaMaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = LLaMaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = LLaMaAttention(config)
        self.ff_norm = LLaMaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff = LLaMaFF(config)

    def forward(
        self,
        hidden_states,
        complex_freq,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        attention_layer_outputs = self.attention(
            hidden_states=self.attention_norm(hidden_states),
            complex_freq=complex_freq,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        attn_output = attn_output + hidden_states
        ff_output = self.ff(self.ff_norm(attn_output))
        hidden_states = ff_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~LLaMaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMa Model transformer outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXModel with GPTNeoX->LLaMa,GPT_NEOX->LLAMA
class LLaMaModel(LLaMaPreTrainedModel):
    def __init__(self, config: LLaMaConfig):
        super().__init__(config)
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLaMaLayer(config) for _ in range(config.num_hidden_layers)])
        head_size = config.hidden_size // config.num_attention_heads
        self.rotary_emb = RotaryEmbedding(head_size, config.max_position_embeddings)
        self.final_layer_norm = LLaMaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :].to(torch.bool)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        # Compute token offset for rotary embeddings (when decoding)
        all_seq_length = seq_length
        if past_key_values[0] is not None:
            all_seq_length += past_key_values[0][0].shape[-2]
        complex_freq = self.rotary_emb(device=inputs_embeds.device, seq_len=all_seq_length)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for layer_past
                        return module(*inputs, use_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    complex_freq,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    complex_freq=complex_freq,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    """LLaMa Model with a `language modeling` head on top for CLM fine-tuning.""", LLAMA_START_DOCSTRING
)
# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM with GPTNeoX->LLaMa,GPT_NEOX->LLAMA,gpt_neox->llama
class LLaMaForCausalLM(LLaMaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.llama = LLaMaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LLaMaForCausalLM, LLaMaConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/llama")
        >>> config = LLaMaConfig.from_pretrained("facebook/llama")
        >>> model = LLaMaForCausalLM.from_pretrained("facebook/llama")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.llama(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
