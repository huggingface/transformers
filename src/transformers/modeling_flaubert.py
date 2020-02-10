# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
""" PyTorch Flaubert model, based on XLM. """


import logging
import random

import torch
from torch.nn import functional as F

from .configuration_flaubert import FlaubertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_xlm import (
    XLMForQuestionAnswering,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMModel,
    XLMWithLMHeadModel,
    get_masks,
)


logger = logging.getLogger(__name__)

FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "flaubert-small-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_small_cased/pytorch_model.bin",
    "flaubert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_uncased/pytorch_model.bin",
    "flaubert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_cased/pytorch_model.bin",
    "flaubert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_large_cased/pytorch_model.bin",
}


FLAUBERT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.FlaubertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

FLAUBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        lengths (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Length of each sentence that can be used to avoid performing attention on padding token indices.
            You can also use `attention_mask` for the same result (see above), kept here for compatbility.
            Indices selected in ``[0, ..., input_ids.size(-1)]``:
        cache (:obj:`Dict[str, torch.FloatTensor]`, `optional`, defaults to :obj:`None`):
            dictionary with ``torch.FloatTensor`` that contains pre-computed
            hidden-states (key and values in the attention blocks) as computed by the model
            (see `cache` output below). Can be used to speed up sequential decoding.
            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
class FlaubertModel(XLMModel):

    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):  # , dico, is_encoder, with_output):
        super(FlaubertModel, self).__init__(config)
        self.layerdrop = getattr(config, "layerdrop", 0.0)
        self.pre_norm = getattr(config, "pre_norm", False)

    @add_start_docstrings_to_callable(FLAUBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.XLMConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
        model = FlaubertModel.from_pretrained('flaubert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Le chat manges une pomme.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        # removed: src_enc=None, src_len=None
        if input_ids is not None:
            bs, slen = input_ids.size()
        else:
            bs, slen = inputs_embeds.size()[:-1]

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = torch.LongTensor([slen] * bs)
        # mask = input_ids != self.pad_index

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # position_ids
        if position_ids is None:
            position_ids = torch.arange(slen, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand((bs, slen))
        else:
            assert position_ids.size() == (bs, slen)  # (slen, bs)
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (bs, slen)  # (slen, bs)
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x qlen x klen]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.n_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layers

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb and self.config.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        hidden_states = ()
        attentions = ()
        for i in range(self.n_layers):
            # LayerDrop
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            if self.output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            if not self.pre_norm:
                attn_outputs = self.attentions[i](tensor, attn_mask, cache=cache, head_mask=head_mask[i])
                attn = attn_outputs[0]
                if self.output_attentions:
                    attentions = attentions + (attn_outputs[1],)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm1[i](tensor)
            else:
                tensor_normalized = self.layer_norm1[i](tensor)
                attn_outputs = self.attentions[i](tensor_normalized, attn_mask, cache=cache, head_mask=head_mask[i])
                attn = attn_outputs[0]
                if self.output_attentions:
                    attentions = attentions + (attn_outputs[1],)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = F.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            if not self.pre_norm:
                tensor = tensor + self.ffns[i](tensor)
                tensor = self.layer_norm2[i](tensor)
            else:
                tensor_normalized = self.layer_norm2[i](tensor)
                tensor = tensor + self.ffns[i](tensor_normalized)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        outputs = (tensor,)
        if self.output_hidden_states:
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (attentions,)
        return outputs  # outputs, (hidden_states), (attentions)


@add_start_docstrings(
    """The Flaubert Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    FLAUBERT_START_DOCSTRING,
)
class FlaubertWithLMHeadModel(XLMWithLMHeadModel):
    """
    This class overrides :class:`~transformers.XLMWithLMHeadModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(FlaubertWithLMHeadModel, self).__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()


@add_start_docstrings(
    """Flaubert Model with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    FLAUBERT_START_DOCSTRING,
)
class FlaubertForSequenceClassification(XLMForSequenceClassification):
    """
    This class overrides :class:`~transformers.XLMForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(FlaubertForSequenceClassification, self).__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()


@add_start_docstrings(
    """Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    FLAUBERT_START_DOCSTRING,
)
class FlaubertForQuestionAnsweringSimple(XLMForQuestionAnsweringSimple):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnsweringSimple`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(FlaubertForQuestionAnsweringSimple, self).__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()


@add_start_docstrings(
    """Flaubert Model with a beam-search span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    FLAUBERT_START_DOCSTRING,
)
class FlaubertForQuestionAnswering(XLMForQuestionAnswering):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(FlaubertForQuestionAnswering, self).__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()
