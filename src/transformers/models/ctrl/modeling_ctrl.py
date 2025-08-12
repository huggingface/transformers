# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch CTRL model."""

from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    auto_docstring,
    logging,
)
from .configuration_ctrl import CTRLConfig


logger = logging.get_logger(__name__)


def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size, dtype):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(
        torch.arange(position, dtype=torch.int64).to(dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=torch.int64).to(dtype).unsqueeze(0),
        d_model_size,
    )

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # calculate attention
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        nd, ns = scaled_attention_logits.size(-2), scaled_attention_logits.size(-1)
        scaled_attention_logits += mask[ns - nd : ns, :ns] * -1e4

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    # Mask heads if we want to
    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_size, num_heads, layer_idx=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.layer_idx = layer_idx

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = nn.Linear(d_model_size, d_model_size)
        self.Wk = nn.Linear(d_model_size, d_model_size)
        self.Wv = nn.Linear(d_model_size, d_model_size)

        self.dense = nn.Linear(d_model_size, d_model_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.d_model_size // self.num_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.Wq = prune_linear_layer(self.Wq, index)
        self.Wk = prune_linear_layer(self.Wk, index)
        self.Wv = prune_linear_layer(self.Wv, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params
        self.num_heads = self.num_heads - len(heads)
        self.d_model_size = attention_head_size * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(
        self,
        v,
        k,
        q,
        mask,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        if layer_past is not None:
            k, v = layer_past.update(k, v, self.layer_idx, {"cache_position": cache_position})

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        output = self.dense(original_size_attention)
        return output, attn


def point_wise_feed_forward_network(d_model_size, dff):
    return nn.Sequential(nn.Linear(d_model_size, dff), nn.ReLU(), nn.Linear(dff, d_model_size))


class EncoderLayer(nn.Module):
    def __init__(self, d_model_size, num_heads, dff, rate=0.1, layer_idx=None):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads, layer_idx=layer_idx)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)

        self.layernorm1 = nn.LayerNorm(d_model_size, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model_size, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(
        self,
        x,
        mask,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        outputs = (out2,) + attn_outputs[1:]
        return outputs


@auto_docstring
class CTRLPreTrainedModel(PreTrainedModel):
    config: CTRLConfig
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring
class CTRLModel(CTRLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)

        self.w = nn.Embedding(config.vocab_size, config.n_embd)

        self.dropout = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                EncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop, layer_idx=i)
                for i in range(config.n_layer)
            ]
        )
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].multi_head_attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,  # NOOP kwargs, for now
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPast]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`
            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)

        Example:

        ```python
        >>> from transformers import AutoTokenizer, CTRLModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLModel.from_pretrained("Salesforce/ctrl")

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 5, 1280]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `DynamicCache` instead, e.g. "
                "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
            )
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= np.sqrt(self.d_model_size)
        else:
            token_type_embeds = 0

        if inputs_embeds is None:
            inputs_embeds = self.w(input_ids)
        # inputs_embeds = embedded.unsqueeze(0) if len(input_ids.shape)<2 else embedded
        seq_len = input_shape[-1]
        mask = torch.triu(torch.ones(seq_len + past_length, seq_len + past_length), 1).to(device)

        inputs_embeds *= np.sqrt(self.d_model_size)

        # `self.pos_encoding` won't be sent to the correct device along the model, so we do it manually.
        self.pos_encoding = self.pos_encoding.to(device)
        pos_embeds = self.pos_encoding[position_ids, :]

        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, h in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = h(
                hidden_states,
                mask,
                layer_past=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)

        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@auto_docstring(
    custom_intro="""
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class CTRLLMHeadModel(CTRLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CTRLModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], CausalLMOutputWithPast]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`
            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CTRLLMHeadModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Wikipedia The llama is", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> sequence_ids = model.generate(inputs["input_ids"])
        >>> sequences = tokenizer.batch_decode(sequence_ids)
        >>> sequences
        ['Wikipedia The llama is a member of the family Bovidae. It is native to the Andes of Peru,']

        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> round(outputs.loss.item(), 2)
        9.21

        >>> list(outputs.logits.shape)
        [1, 5, 246534]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        # Overwritten -- inputs_embeds not working properly

        # only last tokens for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache}


@auto_docstring(
    custom_intro="""
    The CTRL Model transformer with a sequence classification head on top (linear layer).
    [`CTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last
    token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in
    each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last
    value in each row of the batch).
    """
)
class CTRLForSequenceClassification(CTRLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = CTRLModel(config)
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`
            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Example of single-label classification:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CTRLForSequenceClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_class_id = logits.argmax().item()
        >>> model.config.id2label[predicted_class_id]
        'LABEL_0'
        ```

        ```python
        >>> import torch

        >>> torch.manual_seed(42)  # doctest: +IGNORE_RESULT
        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        >>> num_labels = len(model.config.id2label)
        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

        >>> labels = torch.tensor(1)
        >>> loss = model(**inputs, labels=labels).loss
        >>> round(loss.item(), 2)
        0.93
        ```

        Example of multi-label classification:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CTRLForSequenceClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLForSequenceClassification.from_pretrained(
        ...     "Salesforce/ctrl", problem_type="multi_label_classification"
        ... )

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_class_id = logits.argmax().item()
        >>> model.config.id2label[predicted_class_id]
        'LABEL_0'
        ```

        ```python
        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        >>> num_labels = len(model.config.id2label)
        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

        >>> num_labels = len(model.config.id2label)
        >>> labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
        ...     torch.float
        ... )
        >>> loss = model(**inputs, labels=labels).loss
        >>> loss.backward()  # doctest: +IGNORE_RESULT
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = ["CTRLForSequenceClassification", "CTRLLMHeadModel", "CTRLModel", "CTRLPreTrainedModel"]
