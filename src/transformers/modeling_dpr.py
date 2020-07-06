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
from .file_utils import add_start_docstrings
from .modeling_bert import BertModel
from .modeling_utils import PreTrainedModel
from .tokenization_dpr import DPRReaderOutput


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
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.bert_model = BertModel(config)
        assert self.bert_model.config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        self.init_weights()

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, token_type_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        sequence_output, pooled_output, hidden_states = outputs
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

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
    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.encoder = DPREncoder(config)
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        self.init_weights()

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        n_passages, sequence_length = input_ids.size()
        # feed encoder
        sequence_output, *_ = self.encoder(input_ids, attention_mask=attention_mask)
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
        )

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
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
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
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

DPR_READER_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(n_passages, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            It has to be a sequence triplet with 1) the question and 2) the passages titles and 3) the passages texts
            To match pre-training, DPR `input_ids` sequence should be formatted with [CLS] and [SEP] with the format:

                [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DPRReaderTokenizer`.
            See :class:`transformers.DPRReaderTokenizer` for more details
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    DPR_START_DOCSTRING,
    DPR_ENCODERS_INPUTS_DOCSTRING,
)
class DPRContextEncoder(DPRPretrainedContextEncoder):
    r"""
    Outputs: The DPR encoder only outputs the `pooler_output` that corresponds to the context representation:
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer. This output is to be used to embed contexts for
            nearest neighbors queries with question embeddings.

    Examples::

        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
        embeddings = model(input_ids)  # the embeddings of the given context.

    """

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        self.init_weights()

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        sequence_output, pooled_output, hidden_states = self.ctx_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return pooled_output


@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    DPR_START_DOCSTRING,
    DPR_ENCODERS_INPUTS_DOCSTRING,
)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    r"""
    Outputs: The DPR encoder only outputs the `pooler_output` that corresponds to the question representation:
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer. This output is to be used to embed questions for
            nearest neighbors queries with context embeddings.

    Examples::

        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
        embeddings = model(input_ids)  # the embeddings of the given question.

    """

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = DPREncoder(config)
        self.init_weights()

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        sequence_output, pooled_output, hidden_states = self.question_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return pooled_output


@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.", DPR_START_DOCSTRING, DPR_READER_INPUTS_DOCSTRING,
)
class DPRReader(DPRPretrainedReader):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **input_ids**: ``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``
            They correspond to the combined `input_ids` from `(question + context title + context content`).
        **start_logits**: ``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``
            Logits of the start index of the span for each passage.
        **end_logits**: ``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``
            Logits of the end index of the span for each passage.
        **relevance_logits**: `torch.FloatTensor`` of shape ``(n_passages, )``
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
        relevance_logits = outputs[2]  # The relrevance score of the passages

    """

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.config = config
        self.span_predictor = DPRSpanPredictor(config)
        self.init_weights()

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """Compute logits from batched inputs of size (n_questions, n_passages, sequence_length)"""
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        start_logits, end_logits, relevance_logits = self.span_predictor(input_ids, attention_mask)
        return DPRReaderOutput(start_logits, end_logits, relevance_logits)
