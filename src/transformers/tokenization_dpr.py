# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenization classes for DPR."""


import logging
from typing import Optional, Union

from .tokenization_bert import BertTokenizer, BertTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "dpr-model-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dpr-model-base": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "dpr-model-base": {"do_lower_case": True},
}


class DPRTokenizer(BertTokenizer):
    r"""
    Constructs a  DPRBertTokenizer.

    :class:`~transformers.DPRBertTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["attention_mask"]


class DPRTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a  "Fast" DPRBertTokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DPRBertTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["attention_mask"]


class DPRReaderTokenizer(BertTokenizer):
    r"""
    Constructs a  DPRReaderTokenizer.

    :class:`~transformers.DPRReaderTokenizer is almost identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece. The input is different from the standard :class:`~transformers.BertTokenizer`
    as the `__call__` method actually accepts three text inputs: `question`, `titles`, `texts` to fit the use cases of DPR.

    Examples::

        tokenizer = DPRReaderTokenizer.from_pretrained('dpr-base-uncased')
        model = DPRReader.from_pretrained('dpr-reader-single-nq-base')
        question = "Is my dog cute ?"
        titles = ["Things about my dog", "My love for cats"]  # 2 documents to look for answers
        texts = ["My dog is very cute", "I love cats more than dogs, but don't tell anyone !"]
        tokenized_input = tokenizer(question, titles, texts)
        extracted_spans = model.generate(**tokenized_input)

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["attention_mask"]

    def __call__(
        self,
        question,
        titles,
        texts,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        max_length: Optional[int] = 512,
        return_tensors=None,
        **kwargs
    ):
        """
        Return a dictionary with the token ids of the input strings and other information to give to `DPRReader.generate`.
        It converts the strings of a question and different passages (title + text) in a sequence of ids (integer), using the tokenizer and vocabulary.
        The resulting `input_ids` is a matrix of size (n_passages, sequence_length) with the format:

            [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

        Args:
            `question` (:obj:`str`):
                The question to be encoded.
            `titles` (:obj:`str`, :obj:`List[str]`):
                The passagestitles to be encoded. This can be a string, a list of strings if there are several passages.
            `texts` (:obj:`str`, :obj:`List[str]`):
                The passages texts to be encoded. This can be a string, a list of strings if there are several passages.
            `padding` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`True`):
                Activate and control padding. Accepts the following values:

                * `True` or `'longest'`: pad to the longest sequence in the batch (or no padding if only a single sequence if provided),
                * `'max_length'`: pad to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`)
                * `False` or `'do_not_pad'` (default): No padding (i.e. can output batch with sequences of uneven lengths)
            `truncation` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`True`):
                Activate and control truncation. Accepts the following values:

                * `True` or `'only_first'`: truncate to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`).
                * `False` or `'do_not_truncate'` (default): No truncation (i.e. can output batch with sequences length greater than the model max admissible input size)
            `max_length` (:obj:`Union[int, None]`, `optional`, defaults to :obj:`None`):
                Control the length for padding/truncation. Accepts the following values

                * `None` (default): This will use the predefined model max length if required by one of the truncation/padding parameters. If the model has no specific max input length (e.g. XLNet) truncation/padding to max length is deactivated.
                * `any integer value` (e.g. `42`): Use this specific maximum length value if required by one of the truncation/padding parameters.

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[list[int]],
                    passage_offsets: list[int],
                    sequence_lenghts: list[int]
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a the `DPRReader.generate` method
            - ``passage_offsets``: list of indices of the beginning of each passage text inside the `input_ids` matrix
            - ``sequence_lenghts``: list of indices of the whole sequqnce length of row inside the `input_ids` matrix

        """
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        assert len(titles) == len(
            texts
        ), "There should be as many titles than texts but got {} titles and {} texts.".format(len(titles), len(texts))
        encoded_question_and_titles = super().__call__(
            [question] * n_passages, titles, padding=False, truncation=False
        )["input_ids"]
        passage_offsets = [
            len(encoded_question_and_title) for encoded_question_and_title in encoded_question_and_titles
        ]
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        encoded_sequences = [
            (encoded_question_and_title + encoded_text)[:max_length]
            if max_length is not None and truncation
            else encoded_question_and_title + encoded_text
            for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
        ]
        input_ids = self.pad(
            {"input_ids": encoded_sequences}, padding=padding, max_length=max_length, return_tensors=return_tensors
        )["input_ids"]
        sequence_lenghts = [len(encoded_sequence) for encoded_sequence in encoded_sequences]
        return {"input_ids": input_ids, "passage_offsets": passage_offsets, "sequence_lenghts": sequence_lenghts}
