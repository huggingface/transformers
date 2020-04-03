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

from .tokenization_roberta import RobertaTokenizer
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


class MBartTokenizer(XLMRobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {
            "mbart-large-en-ro": _s3_url("facebook/mbart-large-en-ro/sentence.bpe.model"),
            "mbart-large-cc25": _s3_url("facebook/mbart-large-cc25/sentence.bpe.model"),
        }
    }
