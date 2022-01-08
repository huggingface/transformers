# coding=utf-8
# Copyright 2022 The REALM authors and The HuggingFace Inc. team.
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
"""Tokenization classes for REALM."""

from ...file_utils import PaddingStrategy
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "realm-cc-news-pretrained-embedder": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-embedder/resolve/main/vocab.txt",
        "realm-cc-news-pretrained-encoder": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-encoder/resolve/main/vocab.txt",
        "realm-cc-news-pretrained-scorer": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-scorer/resolve/main/vocab.txt",
        "realm-cc-news-pretrained-openqa": "https://huggingface.co/qqaatw/realm-cc-news-pretrained-openqa/aresolve/main/vocab.txt",
        "realm-orqa-nq-openqa": "https://huggingface.co/qqaatw/realm-orqa-nq-openqa/resolve/main/vocab.txt",
        "realm-orqa-nq-reader": "https://huggingface.co/qqaatw/realm-orqa-nq-reader/resolve/main/vocab.txt",
        "realm-orqa-wq-openqa": "https://huggingface.co/qqaatw/realm-orqa-wq-openqa/resolve/main/vocab.txt",
        "realm-orqa-wq-reader": "https://huggingface.co/qqaatw/realm-orqa-wq-reader/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "realm-cc-news-pretrained-embedder": 512,
    "realm-cc-news-pretrained-encoder": 512,
    "realm-cc-news-pretrained-scorer": 512,
    "realm-cc-news-pretrained-openqa": 512,
    "realm-orqa-nq-openqa": 512,
    "realm-orqa-nq-reader": 512,
    "realm-orqa-wq-openqa": 512,
    "realm-orqa-wq-reader": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "realm-cc-news-pretrained-embedder": {"do_lower_case": True},
    "realm-cc-news-pretrained-encoder": {"do_lower_case": True},
    "realm-cc-news-pretrained-scorer": {"do_lower_case": True},
    "realm-cc-news-pretrained-openqa": {"do_lower_case": True},
    "realm-orqa-nq-openqa": {"do_lower_case": True},
    "realm-orqa-nq-reader": {"do_lower_case": True},
    "realm-orqa-wq-openqa": {"do_lower_case": True},
    "realm-orqa-wq-reader": {"do_lower_case": True},
}


class RealmTokenizer(BertTokenizer):
    r"""
    Construct a REALM tokenizer.

    [`RealmTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation splitting and
    wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def batch_encode_candidates(self, text, **kwargs):
        r"""
        Encode a batch of text or text pair. This method is similar to regular __call__ method but has the following
        differences:

            1. Handle additional num_candidate axis. (batch_size, num_candidates, text)
            2. Always pad the sequences to *max_length*.
            3. Must specify *max_length* in order to stack packs of candidates into a batch.

            - single sequence: `[CLS] X [SEP]`
            - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            text (`List[List[str]]`):
                The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,
                num_candidates, text).
            text_pair (`List[List[str]]`, *optional*):
                The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,
                num_candidates, text).
            **kwargs:
                Keyword arguments of the __call__ method.

        Returns:
            [`BatchEncoding`]: Encoded text or text pair.

        Example:

        ```python
        >>> from transformers import RealmTokenizer

        >>> # batch_size = 2, num_candidates = 2
        >>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

        >>> tokenizer = RealmTokenizer.from_pretrained("qqaatw/realm-cc-news-pretrained-encoder")
        >>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
        ```"""

        # Always using a fixed sequence length to encode in order to stack candidates into a batch.
        kwargs["padding"] = PaddingStrategy.MAX_LENGTH

        batch_text = text
        batch_text_pair = kwargs.pop("text_pair", None)
        return_tensors = kwargs.pop("return_tensors", None)

        output_data = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        for idx, candidate_text in enumerate(batch_text):
            if batch_text_pair is not None:
                candidate_text_pair = batch_text_pair[idx]
            else:
                candidate_text_pair = None

            encoded_candidates = super().__call__(candidate_text, candidate_text_pair, return_tensors=None, **kwargs)

            encoded_input_ids = encoded_candidates.get("input_ids")
            encoded_attention_mask = encoded_candidates.get("attention_mask")
            encoded_token_type_ids = encoded_candidates.get("token_type_ids")

            if encoded_input_ids is not None:
                output_data["input_ids"].append(encoded_input_ids)
            if encoded_attention_mask is not None:
                output_data["attention_mask"].append(encoded_attention_mask)
            if encoded_token_type_ids is not None:
                output_data["token_type_ids"].append(encoded_token_type_ids)

        output_data = dict((key, item) for key, item in output_data.items() if len(item) != 0)

        return BatchEncoding(output_data, tensor_type=return_tensors)
