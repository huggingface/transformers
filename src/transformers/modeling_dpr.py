# coding=utf-8
# Copyright 2018 DPR Authors
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
""" PyTorch DPR model for Open Domain Question Answering."""


import logging
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .configuration_dpr import DPRConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bert import BertModel
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
]
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
]
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
]


class DPREncoder(PreTrainedModel):

    base_model_prefix = "bert_model"

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.bert_model = BertModel(config)
        assert self.bert_model.config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, ...]:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        sequence_output, pooled_output, hidden_states = outputs[:3]
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        dpr_encoder_outputs = (sequence_output, pooled_output)

        if output_hidden_states:
            dpr_encoder_outputs += (hidden_states,)
        if output_attentions:
            dpr_encoder_outputs += (outputs[-1],)

        return dpr_encoder_outputs

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.hidden_size

    def init_weights(self):
        self.bert_model.init_weights()
        if self.projection_dim > 0:
            self.encode_proj.apply(self.bert_model._init_weights)


class DPRSpanPredictor(PreTrainedModel):

    base_model_prefix = "encoder"

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.encoder = DPREncoder(config)
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        n_passages, sequence_length = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        # feed encoder
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]

        # compute logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])
        # resize and return
        return (
            start_logits.view(n_passages, sequence_length),
            end_logits.view(n_passages, sequence_length),
            relevance_logits.view(n_passages),
        ) + outputs[2:]

    def init_weights(self):
        self.encoder.init_weights()


##################
# PreTrainedModel
##################


class DPRPretrainedContextEncoder(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "ctx_encoder"

    def init_weights(self):
        self.ctx_encoder.init_weights()


class DPRPretrainedQuestionEncoder(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "question_encoder"

    def init_weights(self):
        self.question_encoder.init_weights()


class DPRPretrainedReader(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "span_predictor"

    def init_weights(self):
        self.span_predictor.encoder.init_weights()
        self.span_predictor.qa_classifier.apply(self.span_predictor.encoder.bert_model._init_weights)
        self.span_predictor.qa_outputs.apply(self.span_predictor.encoder.bert_model._init_weights)


###############
# Actual Models
###############


DPR_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.DPRConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DPR_ENCODERS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids: (:obj:``torch.LongTensor`` of shape ``(batch_size, sequence_length)``):
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, DPR input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DPRTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        attention_mask: (:obj:``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        token_type_ids: (:obj:``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states tensors of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
"""

DPR_READER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids: (:obj:``torch.LongTensor`` of shape ``(n_passages, sequence_length)``):
            Indices of input sequence tokens in the vocabulary.
            It has to be a sequence triplet with 1) the question and 2) the passages titles and 3) the passages texts
            To match pre-training, DPR `input_ids` sequence should be formatted with [CLS] and [SEP] with the format:

                [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DPRReaderTokenizer`.
            See :class:`transformers.DPRReaderTokenizer` for more details
        attention_mask: (:obj:torch.FloatTensor``, of shape ``(n_passages, sequence_length)``, `optional`, defaults to :obj:`None):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(n_passages, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states tensors of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
"""


@add_start_docstrings(
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    DPR_START_DOCSTRING,
)
class DPRContextEncoder(DPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        self.init_weights()

    @add_start_docstrings_to_callable(DPR_ENCODERS_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
    ) -> Tensor:
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DPRConfig`) and inputs:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the context representation.
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer. This output is to be used to embed contexts for
            nearest neighbors queries with questions embeddings.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
        embeddings = model(input_ids)  # the embeddings of the given context.

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.ctx_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output, pooled_output = outputs[:2]
        return (pooled_output,) + outputs[2:]


@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    DPR_START_DOCSTRING,
)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = DPREncoder(config)
        self.init_weights()

    @add_start_docstrings_to_callable(DPR_ENCODERS_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
    ) -> Tensor:
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DPRConfig`) and inputs:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the question representation.
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer. This output is to be used to embed questions for
            nearest neighbors queries with context embeddings.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
        embeddings = model(input_ids)  # the embeddings of the given question.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output, pooled_output = outputs[:2]
        return (pooled_output,) + outputs[2:]


@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.", DPR_START_DOCSTRING,
)
class DPRReader(DPRPretrainedReader):
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.span_predictor = DPRSpanPredictor(config)
        self.init_weights()

    @add_start_docstrings_to_callable(DPR_READER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
    ) -> Tuple[Tensor, ...]:
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DPRConfig`) and inputs:
        input_ids: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``)
            They correspond to the combined `input_ids` from `(question + context title + context content`).
        start_logits: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the start index of the span for each passage.
        end_logits: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the end index of the span for each passage.
        relevance_logits: (:obj:`torch.FloatTensor`` of shape ``(n_passages, )``):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage
            to answer the question, compared to all the other passages.

    Examples::

        from transformers import DPRReader, DPRReaderTokenizer
        tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
        encoded_inputs = tokenizer(
                questions=["What is love ?"],
                titles=["Haddaway"],
                texts=["'What Is Love' is a song recorded by the artist Haddaway"],
                return_tensors='pt'
            )
        outputs = model(**encoded_inputs)
        start_logits = outputs[0]  # The logits of the start of the spans
        end_logits = outputs[1]  # The logits of the end of the spans
        relevance_logits = outputs[2]  # The relevance scores of the passages

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        span_outputs = self.span_predictor(
            input_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        start_logits, end_logits, relevance_logits = span_outputs[:3]

        return (start_logits, end_logits, relevance_logits) + span_outputs[3:]
