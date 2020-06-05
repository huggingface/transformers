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
""" PyTorch DPR model. """

####################################################
# In this template, replace all the DPR (various casings) with your model name
####################################################


import collections
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.serialization import default_restore_location

from .configuration_dpr import DprConfig
from .file_utils import add_start_docstrings
from .modeling_bert import BertConfig, BertModel
from .modeling_utils import PreTrainedModel
from .tokenization_bert import BertTokenizer


logger = logging.getLogger(__name__)

####################################################
# This list contrains shortcut names for some of
# the pretrained weights provided with the models
####################################################
DPR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dpr-base-uncased",
]

####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (itself a sub-class of torch.nn.Module)
####################################################


#############
# BertEncoder
#############


class DprBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.0, **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
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

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


###########
# Reader
###########


ReaderBatch = collections.namedtuple("ReaderBatch", ["input_ids", "start_positions", "end_positions", "answers_mask"])


class ReaderPassage(object):
    """
    Container to collect and cache all Q&A passages related attributes before generating the reader input
    """

    def __init__(self, id=None, text: str = None, title: str = None, score=None, has_answer: bool = None):
        self.id = id
        # string passage representations
        self.passage_text = text
        self.title = title
        self.score = score
        self.passage_token_ids = None
        # offset of the actual passage (i.e. not a question or may be title) in the sequence_ids
        self.passage_offset = None
        self.answers_spans = None
        # passage token ids
        self.sequence_ids: T = None


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


#############
# Tensorizer
#############


class DprBertTensorizer:
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True, pad_to_max_length: bool = True
    ):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
            )
        else:
            token_ids = self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens, max_length=self.max_length, pad_to_max_length=False
            )

        seq_len = self.max_length
        if pad_to_max_length and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_type_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


def create_reader_sample_ids(
    sample: ReaderPassage, question: str, tensorizer: DprBertTensorizer, pad_to_max_length=True
):
    def _concat_pair(t1: T, t2: T, middle_sep: T = None, tailing_sep: T = None):
        middle = [middle_sep] if middle_sep else []
        r = [t1] + middle + [t2] + ([tailing_sep] if tailing_sep else [])
        return torch.cat(r, dim=0), t1.size(0) + len(middle)

    def _pad_to_len(seq: T, pad_id: int, max_len: int):
        s_len = seq.size(0)
        if s_len > max_len:
            return seq[0:max_len]
        return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)

    question_and_title = tensorizer.text_to_tensor(
        sample.title, title=question, add_special_tokens=True, pad_to_max_length=False
    )
    sample.passage_token_ids = tensorizer.text_to_tensor(
        sample.passage_text, add_special_tokens=False, pad_to_max_length=False
    )
    all_concatenated, shift = _concat_pair(question_and_title, sample.passage_token_ids, None)
    sample.sequence_ids = all_concatenated
    if pad_to_max_length:
        sample.sequence_ids = _pad_to_len(sample.sequence_ids, tensorizer.get_pad_id(), tensorizer.max_length)
    sample.passage_offset = shift
    assert shift > 1
    return sample


##############
# Providers
##############


def get_bert_question_encoder_components(config: DprConfig, **kwargs):
    question_encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    return question_encoder


def get_bert_ctx_encoder_components(config: DprConfig, **kwargs):
    ctx_encoder = DprBertEncoder.init_encoder(
        config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs
    )
    return ctx_encoder


def get_bert_reader_components(config: DprConfig, **kwargs):
    encoder = DprBertEncoder.init_encoder(config.pretrained_model_cfg, projection_dim=config.projection_dim, **kwargs)
    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)
    return reader


def get_bert_tensorizer(config: DprConfig, tokenizer=None):
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_cfg, do_lower_case=config.do_lower_case)
    return DprBertTensorizer(tokenizer, config.sequence_length)


####################
# Predictions utils
####################


SpanPrediction = collections.namedtuple(
    "SpanPrediction", ["prediction_text", "span_score", "relevance_score", "passage_index", "passage_token_ids"]
)


def get_best_spans(
    tensorizer: DprBertTensorizer,
    start_logits: List,
    end_logits: List,
    ctx_ids: List,
    max_answer_length: int,
    passage_idx: int,
    relevance_score: float,
    top_spans: int = 1,
) -> List[SpanPrediction]:
    """
    Finds the best answer span for the extractive Q&A model
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

        # extend bpe subtokens to full tokens
        start_index, end_index = _extend_span_to_full_words(tensorizer, ctx_ids, (start_index, end_index))

        predicted_answer = tensorizer.to_string(ctx_ids[start_index : end_index + 1])
        best_spans.append(SpanPrediction(predicted_answer, score, relevance_score, passage_idx, ctx_ids))
        chosen_span_intervals.append((start_index, end_index))

        if len(chosen_span_intervals) == top_spans:
            break
    return best_spans


def _extend_span_to_full_words(
    tensorizer: DprBertTensorizer, tokens: List[int], span: Tuple[int, int]
) -> Tuple[int, int]:
    start_index, end_index = span
    max_len = len(tokens)
    while start_index > 0 and tensorizer.is_sub_word_id(tokens[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and tensorizer.is_sub_word_id(tokens[end_index + 1]):
        end_index += 1

    return start_index, end_index


def _get_sorted_prediction(
    start_logits,
    end_logits,
    relevance_logits,
    all_reader_passages: List[List[ReaderPassage]],
    tensorizer: DprBertTensorizer,
    max_answer_length: int,
) -> List[List[SpanPrediction]]:

    questions_num, passages_per_question = relevance_logits.size()

    _, idxs = torch.sort(relevance_logits, dim=1, descending=True,)

    batch_results = []
    for q in range(questions_num):
        reader_passages = all_reader_passages[q]
        non_empty_passages_num = len(reader_passages)
        nbest = []
        for p in range(passages_per_question):
            passage_idx = idxs[q, p].item()
            if passage_idx >= non_empty_passages_num:  # empty passage selected, skip
                continue
            reader_passage = reader_passages[passage_idx]
            sequence_ids = reader_passage.sequence_ids
            sequence_len = sequence_ids.size(0)
            # assuming question & title information is at the beginning of the sequence
            passage_offset = reader_passage.passage_offset

            p_start_logits = start_logits[q, passage_idx].tolist()[passage_offset:sequence_len]
            p_end_logits = end_logits[q, passage_idx].tolist()[passage_offset:sequence_len]

            ctx_ids = sequence_ids.tolist()[passage_offset:]
            best_spans = get_best_spans(
                tensorizer,
                p_start_logits,
                p_end_logits,
                ctx_ids,
                max_answer_length,
                passage_idx,
                relevance_logits[q, passage_idx].item(),
                top_spans=10,
            )
            nbest.extend(best_spans)

        if len(nbest) == 0:
            predictions = [SpanPrediction("", -1, -1, -1, "")]
        else:
            predictions = nbest
        batch_results.append(predictions)
    return batch_results


###########################################################
# EmbedModel to use with the `nlp` library by Hugging Face
###########################################################


class DprEmbedModel:
    def __init__(self, dpr_model):
        self.dpr_model = dpr_model

    def embed_documents(
        self, texts: List[str], titles: Optional[List[str]] = None, tokenizer: Optional[Callable] = None
    ) -> np.array:
        if titles is not None:
            token_tensors = [
                self.dpr_model.tensorizer.text_to_tensor(text, title=title) for text, title in zip(texts, titles)
            ]
        else:
            token_tensors = [self.dpr_model.tensorizer.text_to_tensor(text) for text in texts]
        input_ids = torch.stack(token_tensors, dim=0)
        type_ids = torch.zeros_like(input_ids)
        attention_mask = self.dpr_model.tensorizer.get_attn_mask(input_ids)
        _d_seq, d_pooled_out, _d_hidden = self.dpr_model.ctx_encoder(input_ids, type_ids, attention_mask)
        return d_pooled_out.numpy().astype(np.float32)

    def embed_queries(self, queries: List[str], tokenizer: Optional[Callable] = None) -> np.array:
        token_tensors = [self.dpr_model.tensorizer.text_to_tensor(query) for query in queries]
        input_ids = torch.stack(token_tensors, dim=0)
        attention_mask = self.dpr_model.tensorizer.get_attn_mask(input_ids)
        _q_seq, q_pooled_out, _q_hidden = self.dpr_model.question_encoder(input_ids, None, attention_mask)
        return q_pooled_out.numpy().astype(np.float32)


####################################################
# PreTrainedModel is a sub-class of torch.nn.Module
# which take care of loading and saving pretrained weights
# and various common utilities.
#
# Here you just need to specify a few (self-explanatory)
# pointers for your model and the weights initialization
# method if its not fully covered by PreTrainedModel's default method
####################################################


class DprPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DprConfig
    load_tf_weights = None
    base_model_prefix = "dpr"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


DPR_START_DOCSTRING = r"""    The DPR model was proposed in
    `DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.DprConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DPR_INPUTS_DOCSTRING = r"""
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
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `DPR: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


CheckpointState = collections.namedtuple(
    "CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)


@add_start_docstrings(
    "The bare Dpr Model transformer outputting raw hidden-states without any specific head on top.",
    DPR_START_DOCSTRING,
    DPR_INPUTS_DOCSTRING,
)
class DprModel(DprPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Dpr pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DprTokenizer.from_pretrained('dpr-base-uncased')
        model = DprModel.from_pretrained('dpr-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config: DprConfig):
        super().__init__(config)
        self.config = config
        self.question_encoder = get_bert_question_encoder_components(config)
        self.ctx_encoder = get_bert_ctx_encoder_components(config)
        self.reader = get_bert_reader_components(config)
        self.tensorizer = get_bert_tensorizer(config)
        self.init_weights()

    def init_weights(self):
        if self.config.biencoder_model_file is not None:
            logger.info("Loading DPR biencoder from {}".format(self.config.biencoder_model_file))
            saved_state = load_states_from_checkpoint(self.config.biencoder_model_file)
            for encoder, prefix in zip([self.question_encoder, self.ctx_encoder], ["question_model.", "ctx_model."]):
                prefix_len = len(prefix)
                ctx_state = {
                    key[prefix_len:]: value
                    for (key, value) in saved_state.model_dict.items()
                    if key.startswith(prefix)
                }
                encoder.load_state_dict(ctx_state)
        if self.config.reader_model_file is not None:
            logger.info("Loading DPR reader from {}".format(self.config.reader_model_file))
            saved_state = load_states_from_checkpoint(self.config.reader_model_file)
            self.reader.load_state_dict(saved_state.model_dict)

    def _extract_reader_passages(self, index, scores_and_top_ids, questions) -> List[List[ReaderPassage]]:
        all_reader_passages: List[List[ReaderPassage]] = []
        for scores, indices in zip(*scores_and_top_ids):
            reader_passages: List[ReaderPassage] = []
            ctxs_num = len(scores)
            docs = [index[int(doc_id)] for doc_id in indices]
            for c in range(ctxs_num):
                reader_passages.append(
                    ReaderPassage(id=indices[c], title=docs[c]["title"], text=docs[c]["text"], score=scores[c],)
                )
            all_reader_passages.append(reader_passages)
        for question, reader_passages in zip(questions, all_reader_passages):
            for passage in reader_passages:
                create_reader_sample_ids(passage, question, self.tensorizer)
        return all_reader_passages

    def to_nlp_embed_model(self) -> DprEmbedModel:
        return DprEmbedModel(self)

    def forward(self, index, questions: List[str]):
        # Question Encoder
        token_tensors = [self.tensorizer.text_to_tensor(q) for q in questions]
        input_ids = torch.stack(token_tensors, dim=0)
        attention_mask = self.tensorizer.get_attn_mask(input_ids)
        _q_seq, q_pooled_out, _q_hidden = self.question_encoder(input_ids, None, attention_mask)
        # Dense Retriever
        scores_and_top_ids = index.query_index_batch(
            questions, self.config.k, q_rep=q_pooled_out.numpy().astype(np.float32)
        )
        all_reader_passages = self._extract_reader_passages(index, scores_and_top_ids, questions)
        # Reader
        input_ids = torch.stack(
            [
                torch.stack([passage.sequence_ids for passage in reader_passages], dim=0)
                for reader_passages in all_reader_passages
            ],
            dim=0,
        )
        attention_mask = self.tensorizer.get_attn_mask(input_ids)
        reader_output = self.reader(input_ids, attention_mask)
        start_logits, end_logits, relevance_logits = reader_output
        sorted_results = _get_sorted_prediction(
            start_logits, end_logits, relevance_logits, all_reader_passages, self.tensorizer, self.config.max_length
        )
        best_results = [results[0] for results in sorted_results]
        return best_results, sorted_results, all_reader_passages, reader_output
