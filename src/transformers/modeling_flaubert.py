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

import torch
from torch.nn import functional as F

from .configuration_flaubert import FlaubertConfig
from .file_utils import add_start_docstrings
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


FLAUBERT_START_DOCSTRING = r"""    The Flaubert model was proposed in
    `FlauBERT: Unsupervised Language Model Pre-training for French`_
    by Hang Le et al. It's a transformer pre-trained using a masked
    language modeling (MLM) objective (BERT-like).

    Original code can be found `here`_.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    .. _`FlauBERT: Unsupervised Language Model Pre-training for French`:
        https://arxiv.org/abs/1912.05372

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    .. _`here`:
        https://github.com/getalp/Flaubert

    Parameters:
        config (:class:`~transformers.FlaubertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

FLAUBERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.

            Flaubert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.FlaubertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **lengths**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Length of each sentence that can be used to avoid performing attention on padding token indices.
            You can also use `attention_mask` for the same result (see above), kept here for compatbility.
            Indices selected in ``[0, ..., input_ids.size(-1)]``:
        **cache**:
            dictionary with ``torch.FloatTensor`` that contains pre-computed
            hidden-states (key and values in the attention blocks) as computed by the model
            (see `cache` output below). Can be used to speed up sequential decoding.
            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
    FLAUBERT_INPUTS_DOCSTRING,
)
class FlaubertModel(XLMModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
        model = FlaubertModel.from_pretrained('flaubert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Le chat manges une pomme.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    model_type = "flaubert"

    def __init__(self, config):  # , dico, is_encoder, with_output):
        super(FlaubertModel, self).__init__(config)
        self.layerdrop = 0.0 if not hasattr(config, "layerdrop") else config.layerdrop
        self.pre_norm = False if not hasattr(config, "pre_norm") else config.pre_norm

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
    ):  # removed: src_enc=None, src_len=None
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
        if langs is not None and self.use_lang_emb:
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
    FLAUBERT_INPUTS_DOCSTRING,
)
class FlaubertWithLMHeadModel(XLMWithLMHeadModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
        model = FlaubertWithLMHeadModel.from_pretrained('flaubert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Le chat manges une pomme.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

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
    FLAUBERT_INPUTS_DOCSTRING,
)
class FlaubertForSequenceClassification(XLMForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
        model = FlaubertForSequenceClassification.from_pretrained('flaubert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Le chat manges une pomme.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

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
    FLAUBERT_INPUTS_DOCSTRING,
)
class FlaubertForQuestionAnsweringSimple(XLMForQuestionAnsweringSimple):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **is_impossible**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        **cls_index**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the classification token to use as input for computing plausibility of the answer.
        **p_mask**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
        model = FlaubertForQuestionAnsweringSimple.from_pretrained('flaubert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Le chat manges une pomme", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

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
    FLAUBERT_INPUTS_DOCSTRING,
)
class FlaubertForQuestionAnswering(XLMForQuestionAnswering):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **is_impossible**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        **cls_index**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the classification token to use as input for computing plausibility of the answer.
        **p_mask**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = FlaubertTokenizer.from_pretrained('flaubert-base-cased')
        model = FlaubertForQuestionAnswering.from_pretrained('flaubert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Le chat manges une pomme.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """
    config_class = FlaubertConfig
    pretrained_model_archive_map = FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(FlaubertForQuestionAnswering, self).__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()
