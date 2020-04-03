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

import torch

from .tokenization_roberta import RobertaTokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer


logger = logging.getLogger(__name__)


# vocab and merges same as roberta
vocab_url = "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json"
merges_url = "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt"
_all_bart_models = ["bart-large", "bart-large-mnli", "bart-large-cnn", "bart-large-xsum"]

VOCAB_FILES_NAMES = {"vocab_file": "sentence.bpe.model"}


class BartTokenizer(RobertaTokenizer):
    # merges and vocab same as Roberta
    max_model_input_sizes = {m: 1024 for m in _all_bart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_bart_models},
        "merges_file": {m: merges_url for m in _all_bart_models},
    }


_all_mbart_models = ["mbart-large-en-ro"]
SPM_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/mbart-large-en-ro/sentence.bpe.model"


class MBartTokenizer(XLMRobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {"vocab_file": {m: SPM_URL for m in _all_mbart_models}}
    lang_code_to_id = {
        "ar_AR": 250001,
        "cs_CZ": 250002,
        "de_DE": 250003,
        "en_XX": 250004,
        "es_XX": 250005,
        "et_EE": 250006,
        "fi_FI": 250007,
        "fr_XX": 250008,
        "gu_IN": 250009,
        "hi_IN": 250010,
        "it_IT": 250011,
        "ja_XX": 250012,
        "kk_KZ": 250013,
        "ko_KR": 250014,
        "lt_LT": 250015,
        "lv_LV": 250016,
        "my_MM": 250017,
        "ne_NP": 250018,
        "nl_XX": 250019,
        "ro_RO": 250020,
        "ru_RU": 250021,
        "si_LK": 250022,
        "tr_TR": 250023,
        "vi_VN": 250024,
        "zh_CN": 250025,
    }

    def _encode_with_lang_code(self, raw_text: str, lang_code: int) -> list:
        ids = self.convert_tokens_to_ids(self.tokenize(raw_text,))[: self.max_len_single_sentence]
        return ids + [self.eos_token_id, self.lang_code_to_id[lang_code]]

    def prepare_translation_example(self, src_text: str, src_lang="en_XX", tgt_text=None, tgt_lang="ro_RO"):
        id_key = "input_ids"
        src_ids = self._encode_with_lang_code(src_text, src_lang)
        encoder_inputs = self.prepare_for_model(src_ids, return_tensors="pt")

        if tgt_text is not None:
            tgt_ids = self._encode_with_lang_code(tgt_text, tgt_lang)
            decoder_input_ids = self.prepare_for_model(tgt_ids, return_tensors="pt")[id_key]
        else:
            decoder_input_ids = None
        return {
            id_key: encoder_inputs[id_key],
            "attention_mask": encoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }
