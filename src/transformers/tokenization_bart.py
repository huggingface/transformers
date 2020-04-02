# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

import logging
from typing import List

from .tokenization_roberta import RobertaTokenizer
from .tokenization_t5 import T5Tokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer


logger = logging.getLogger(__name__)


def _s3_url(suffix):
    return "https://s3.amazonaws.com/models.huggingface.co/bert/{}".format(suffix)


# vocab and merges same as roberta
vocab_url = _s3_url("roberta-large-vocab.json")
merges_url = _s3_url("roberta-large-merges.txt")
_all_bart_models = ["bart-large", "bart-large-mnli", "bart-large-cnn", "bart-large-xsum"]

VOCAB_FILES_NAMES = {"vocab_file": "sentence.bpe.model"}


class BartTokenizer(RobertaTokenizer):
    # merges and vocab same as Roberta
    max_model_input_sizes = {m: 1024 for m in _all_bart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_bart_models},
        "merges_file": {m: merges_url for m in _all_bart_models},
    }


_all_mbart_models = ["mbart-large-en-ro", "mbart-large-cc25"]
lang_codes = {
    "ar_AR": 0,
    "cs_CZ": 1,
    "de_DE": 2,
    "en_XX": 3,
    "es_XX": 4,
    "et_EE": 5,
    "fi_FI": 6,
    "fr_XX": 7,
    "gu_IN": 8,
    "hi_IN": 9,
    "it_IT": 10,
    "ja_XX": 11,
    "kk_KZ": 12,
    "ko_KR": 13,
    "lt_LT": 14,
    "lv_LV": 15,
    "my_MM": 16,
    "ne_NP": 17,
    "nl_XX": 18,
    "ro_RO": 19,
    "ru_RU": 20,
    "si_LK": 21,
    "tr_TR": 22,
    "vi_VN": 23,
    "zh_CN": 24,
}


class MBartTokenizer(XLMRobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {
            "mbart-large-en-ro": _s3_url("facebook/mbart-large-en-ro/sentence.bpe.model"),
            "mbart-large-cc25": _s3_url("facebook/mbart-large-cc25/sentence.bpe.model"),
        }
    }

    # def build_inputs_with_special_tokens(
    #     self, token_ids_0: List[int], token_ids_1= None
    # ) -> List[int]:
    #     """
    #     Build model inputs from a sequence or a pair of sequence for sequence classification tasks
    #     by concatenating and adding special tokens.
    #     A XLM-R sequence has the following format:
    #
    #     - single sequence: ``<s> X </s>``
    #     - pair of sequences: ``<s> A </s></s> B </s>``
    #
    #     Args:
    #         token_ids_0 (:obj:`List[int]`):
    #             List of IDs to which the special tokens will be added
    #         token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
    #             Optional second list of IDs for sequence pairs.
    #
    #     Returns:
    #         :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
    #     """
    #
    #     if token_ids_1 is None:
    #         return token_ids_0 + [self.sep_token_id, self.lang_id]
    #     raise NotImplementedError("MBart does not support text pairs")
