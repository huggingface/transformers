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


import collections
import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor as T
from torch import nn
from torch.serialization import default_restore_location

from .configuration_dpr import DprConfig
from .file_utils import add_start_docstrings
from .modeling_bert import BertConfig, BertModel
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

DPR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-reader-single-nq-base",
]


################
# DprBertEncoder
################


class DprBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.0, **kwargs) -> "DprBertEncoder":
        cfg = BertConfig.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls(cfg, project_dim=projection_dim)

    def forward(self, input_ids: T, token_type_ids: Optional[T], attention_mask: Optional[T]) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self) -> int:
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


###########
# Reader
###########


class Reader(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: T, attention_mask: T):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(
            input_ids.view(N * M, L), attention_mask.view(N * M, L)
        )
        return start_logits.view(N, M, L), end_logits.view(N, M, L), relevance_logits.view(N, M)

    def _forward(self, input_ids, attention_mask):
        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, None, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return start_logits, end_logits, rank_logits


##############
# Providers
##############


def get_bert_question_encoder_component(config: DprConfig, **kwargs) -> DprBertEncoder:
    question_encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    return question_encoder


def get_bert_ctx_encoder_component(config: DprConfig, **kwargs) -> DprBertEncoder:
    ctx_encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    return ctx_encoder


def get_bert_reader_component(config: DprConfig, **kwargs) -> Reader:
    encoder = DprBertEncoder.init_encoder(config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs)
    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)
    return reader


##################
# PreTrainedModel
##################


CheckpointState = collections.namedtuple(
    "CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)


class DprPretrainedContextEncoder(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DprConfig
    load_tf_weights = None
    base_model_prefix = "dpr"

    def init_weights(self):
        """Load the weights from the official DPR repo's format if specified."""
        if self.config.biencoder_model_file is not None:
            logger.info("Loading DPR biencoder from {}".format(self.config.biencoder_model_file))
            saved_state = load_states_from_checkpoint(self.config.biencoder_model_file)
            encoder, prefix = self.ctx_encoder, "ctx_model."
            prefix_len = len(prefix)
            ctx_state = {
                key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith(prefix)
            }
            encoder.load_state_dict(ctx_state)
        else:
            self.ctx_encoder.init_weights()


class DprPretrainedQuestionEncoder(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DprConfig
    load_tf_weights = None
    base_model_prefix = "dpr"

    def init_weights(self):
        """Load the weights from the official DPR repo's format if specified."""
        if self.config.biencoder_model_file is not None:
            logger.info("Loading DPR biencoder from {}".format(self.config.biencoder_model_file))
            saved_state = load_states_from_checkpoint(self.config.biencoder_model_file)
            encoder, prefix = self.question_encoder, "question_model."
            prefix_len = len(prefix)
            ctx_state = {
                key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith(prefix)
            }
            encoder.load_state_dict(ctx_state)
        else:
            self.question_encoder.init_weights()


class DprPretrainedReader(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DprConfig
    load_tf_weights = None
    base_model_prefix = "dpr"

    def init_weights(self):
        """Load the weights from the official DPR repo's format if specified."""
        if self.config.reader_model_file is not None:
            logger.info("Loading DPR reader from {}".format(self.config.reader_model_file))
            saved_state = load_states_from_checkpoint(self.config.reader_model_file)
            self.reader.load_state_dict(saved_state.model_dict)
        else:
            self.reader.encoder.init_weights()


###############
# Actual Models
###############


DPR_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.DprConfig`): Model configuration class with all the parameters of the model.
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

            Dpr is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DprTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

DPR_READER_INPUTS_DOCSTRING = r"""
    Inputs:
        **question_and_titles_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            It has to be a sequence pair with 1) the question and 2) the context title:
            To match pre-training, DPR `question_and_titles_ids` sequence should be formatted with [CLS] and [SEP] tokens as follows:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] jack ##son ##ville page [SEP]``

            Dpr is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DprTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **text_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            It has to be a single sequence with the context text (content of the passage):
            To match pre-training, DPR `text_ids` sequence should be formatted without [CLS] and [SEP] tokens as follows:

                ``tokens:         this is jack ##son ##ville .``

            This is because the `text_ids` are then concatenated after the `question_and_titles_ids`

            Dpr is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.DprTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
"""


@add_start_docstrings(
    "The bare DprContextEncoder transformer outputting pooler outputs as context representations.",
    DPR_START_DOCSTRING,
    DPR_ENCODERS_INPUTS_DOCSTRING,
)
class DprContextEncoder(DprPretrainedContextEncoder):
    r"""
    Outputs: The DPR encoder only outputs the `pooler_output` that corresponds to the context representation:
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer. This output is to be used to embed contexts for
            nearest neighbors queries with question embeddings.

    Examples::

        tokenizer = DprTokenizer.from_pretrained('dpr-base-uncased')
        model = DprContextEncoder.from_pretrained('dpr-ctx_encoder-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        embeddings = model(input_ids)  # the embeddings of the given context.

    """

    def __init__(self, config: DprConfig):
        super().__init__(config)
        self.config = config
        self.ctx_encoder = get_bert_ctx_encoder_component(config)
        self.init_weights()

    def forward(self, input_ids: T, token_type_ids: Optional[T] = None, attention_mask: Optional[T] = None) -> T:
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
            attention_mask = attention_mask.to(device=input_ids.device)
        sequence_output, pooled_output, hidden_states = self.ctx_encoder(input_ids, token_type_ids, attention_mask)
        return pooled_output


@add_start_docstrings(
    "The bare DprQuestionEncoder transformer outputting pooler outputs as question representations.",
    DPR_START_DOCSTRING,
    DPR_ENCODERS_INPUTS_DOCSTRING,
)
class DprQuestionEncoder(DprPretrainedQuestionEncoder):
    r"""
    Outputs: The DPR encoder only outputs the `pooler_output` that corresponds to the question representation:
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer. This output is to be used to embed questions for
            nearest neighbors queries with context embeddings.

    Examples::

        tokenizer = DprTokenizer.from_pretrained('dpr-base-uncased')
        model = DprQuestionEncoder.from_pretrained('dpr-ctx_encoder-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, is my dog cute ?")).unsqueeze(0)  # Batch size 1
        embeddings = model(input_ids)  # the embeddings of the given question.

    """

    def __init__(self, config: DprConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = get_bert_question_encoder_component(config)
        self.init_weights()

    def forward(self, input_ids: T, token_type_ids: Optional[T] = None, attention_mask: Optional[T] = None) -> T:
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
            attention_mask = attention_mask.to(device=input_ids.device)
        sequence_output, pooled_output, hidden_states = self.question_encoder(
            input_ids, token_type_ids, attention_mask
        )
        return pooled_output


DprSpanPrediction = collections.namedtuple(
    "SpanPrediction", ["span_score", "relevance_score", "doc_id", "start_index", "end_index"]
)


@add_start_docstrings(
    "The bare DprReader transformer outputting span predictions.", DPR_START_DOCSTRING, DPR_READER_INPUTS_DOCSTRING,
)
class DprReader(DprPretrainedReader):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **input_ids**: ``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``
            They correspond to the combined `input_ids` from `(question + context title + context content`).
        **start_logits**: ``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``
            Logits of the start index of the span for each passage.
        **end_logits**: ``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``
            Logits of the end index of the span for each passage.
        **relevance_logits**: `torch.FloatTensor`` of shape ``(n_passages, )``
            Outputs of the QA classifier of the DprReader that corresponds to the scores of each passage
            to answer the question, compared to all the other passages.


    Examples::

        tokenizer = DprReader.from_pretrained('dpr-base-uncased')
        model = DprModel.from_pretrained('dpr-reader-base')
        question_and_titles_ids = [
            torch.tensor(tokenizer.encode("Hello, is my dog cute ?", "Dog cuteness"))
            ]  # One tensor per passage. It corresponds to the concatenation of the question and the context title.
        texts_ids = [
            torch.tensor(tokenizer.encode("Hello, my dog is definitely cute", add_special_tokens=False))
            ]  # One tensor per passage. It corresponds to the context text in which we're looking for the answer.
        outputs = model(question_and_titles_ids, texts_ids)
        start_logits = outputs[1]  # The logits of the start of the span
        end_logits = outputs[2]  # The logits of the end of the span

    """

    def __init__(self, config: DprConfig):
        super().__init__(config)
        self.config = config
        self.reader = get_bert_reader_component(config)
        self.init_weights()

    def forward(self, question_and_titles_ids: List[T], texts_ids: List[T],) -> Tuple[T, ...]:
        assert len(question_and_titles_ids) == len(texts_ids)
        device = question_and_titles_ids[0].device
        n_contexts = len(question_and_titles_ids)
        input_ids = torch.ones((n_contexts, self.config.sequence_length), dtype=torch.int64) * int(
            self.config.pad_token_id
        )
        input_ids = input_ids.to(device=device)
        for i in range(n_contexts):
            question_and_title_ids = question_and_titles_ids[i]
            text_ids = texts_ids[i]
            _, len_qt = question_and_title_ids.size()
            _, len_txt = text_ids.size()
            input_ids[i, 0:len_qt] = question_and_title_ids
            input_ids[i, len_qt : len_qt + len_txt] = text_ids
        input_ids = input_ids.unsqueeze(0)
        attention_mask = input_ids != self.config.pad_token_id
        attention_mask = attention_mask.to(device=device)
        start_logits, end_logits, relevance_logits = self.reader(input_ids, attention_mask)
        return input_ids, start_logits, end_logits, relevance_logits

    def generate(
        self,
        question_and_titles_ids: List[T],
        texts_ids: List[T],
        k: int = 16,
        max_answer_length: int = 64,
        top_spans_per_passage: int = 4,
    ) -> List[DprSpanPrediction]:
        """
        Get the span predictions for the extractive Q&A model.
        Outputs: `List` of `DprSpanPrediction` sorted by descending `(relevance_score, span_score)`.
            Each `DprSpanPrediction` is a `Tuple` with:
            **span_score**: ``float`` that corresponds to the score given by the reader for this span compared to other spans
                in the same passage. It corresponds to the sum of the start and end logits of the span.
            **relevance_score**: ``float`` that corresponds to the score of the each passage to answer the question,
                compared to all the other passages. It corresponds to the output of the QA classifier of the DprReader.
            **doc_id**: ``int``` the id of the passage.
            **start_index**: ``int`` the start index of the span (inclusive).
            **end_index**: ``int`` the end index of the span (inclusive).

        Examples::

            tokenizer = DprTokenizer.from_pretrained('dpr-base-uncased')
            model = DprModel.from_pretrained('dpr-reader-base')
            question_and_titles_ids = [
                torch.tensor(tokenizer.encode("Hello, is my dog cute ?", "Dog cuteness"))
                ]  # One tensor per passage. It corresponds to the concatenation of the question and the context title.
            texts_ids = [
                torch.tensor(tokenizer.encode("Hello, my dog is definitely cute", add_special_tokens=False))
                ]  # One tensor per passage. It corresponds to the context text in which we're looking for the answer.
            predicted_spans = model.generate(question_and_titles_ids, texts_ids)
            # get best answer
            best_span = predicted_spans[0]
            best_span_ids = texts_ids[best_span.doc_id].numpy().flatten()
            best_span_ids = best_span_ids[best_span.start_index:best_span.end_index + 1]
            print(tokenizer.decode(best_span_ids))

        """

        input_ids, start_logits, end_logits, relevance_logits = self.forward(question_and_titles_ids, texts_ids)

        questions_num, docs_per_question = relevance_logits.size()
        assert questions_num == 1
        _, idxs = torch.sort(relevance_logits, dim=1, descending=True,)
        nbest_spans_predictions: List[DprSpanPrediction] = []
        for p in range(docs_per_question):
            doc_id = idxs[0, p].item()
            sequence_ids = input_ids[0, doc_id]
            sequence_len = sequence_ids.size(0)
            # assuming question & title information is at the beginning of the sequence
            passage_offset = question_and_titles_ids[p].size(1)

            p_start_logits = start_logits[0, doc_id].tolist()[passage_offset:sequence_len]
            p_end_logits = end_logits[0, doc_id].tolist()[passage_offset:sequence_len]
            ctx_ids = sequence_ids.tolist()[passage_offset:]
            best_spans = self._get_best_spans(
                p_start_logits,
                p_end_logits,
                ctx_ids,
                max_answer_length,
                doc_id,
                relevance_logits[0, doc_id].item(),
                top_spans=top_spans_per_passage,
            )
            nbest_spans_predictions.extend(best_spans)
            if len(nbest_spans_predictions) > k:
                break
        return nbest_spans_predictions[:k]

    def _get_best_spans(
        self,
        start_logits: List,
        end_logits: List,
        ctx_ids: List,
        max_answer_length: int,
        doc_id: int,
        relevance_score: float,
        top_spans: int,
    ) -> List[DprSpanPrediction]:
        """
        Finds the best answer span for the extractive Q&A model for one passage.
        It returns the best span by descending `span_score` order and keeping max `top_spans` spans.
        Spans longer that `max_answer_length` are ignored.
        """
        scores = []
        for (i, s) in enumerate(start_logits):
            for (j, e) in enumerate(end_logits[i : i + max_answer_length]):
                scores.append(((i, i + j), s + e))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        best_spans = []
        for (start_index, end_index), score in scores:
            assert start_index <= end_index
            length = end_index - start_index + 1
            assert length <= max_answer_length
            if any(
                [
                    start_index <= prev_start_index <= prev_end_index <= end_index
                    or prev_start_index <= start_index <= end_index <= prev_end_index
                    for (prev_start_index, prev_end_index) in chosen_span_intervals
                ]
            ):
                continue
            best_spans.append(DprSpanPrediction(score, relevance_score, doc_id, start_index, end_index))
            chosen_span_intervals.append((start_index, end_index))

            if len(chosen_span_intervals) == top_spans:
                break
        return best_spans
