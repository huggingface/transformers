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
from typing import Dict, List, Optional

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
    """
    This inherits from XLMRobertaTokenizer. ``prepare_translation_batch`` should be used to encode inputs.
    Other tokenizer methods like encode do not work properly.
    The tokenization method is <tokens> <eos> <language code>. There is no BOS token.

    Examples::
        from transformers import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained('mbart-large-en-ro')
        tok.prepare_translation_batch([
        example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
        batch: dict = tokenizer.prepare_translation_batch(
            example_english_phrase, src_lang="en_XX", tgt_lang="ro_RO", tgt_texts=expected_translation_romanian
        )

    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {"vocab_file": {m: SPM_URL for m in _all_mbart_models}}
    lang_code_to_id = {  # TODO(SS): resize embeddings will break this
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

    def _append_special_tokens_and_truncate(self, raw_text: str, lang_code: str, max_length: int,) -> List[int]:
        tokenized_text: str = self.tokenize(raw_text)
        ids: list = self.convert_tokens_to_ids(tokenized_text)[:max_length]
        lang_id: int = self.lang_code_to_id[lang_code]
        return ids + [self.eos_token_id, lang_id]

    def prepare_translation_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            src_texts: list of src language texts
            src_lang: default en_XX (english)
            tgt_texts: list of tgt language texts
            tgt_lang: default ro_RO (romanian)
            max_length: (None) defer to config (1024 for mbart-large-en-ro)
            pad_to_max_length: (bool)

        Returns:
            dict with keys input_ids, attention_mask, decoder_input_ids, each value is a torch.Tensor.
        """
        if max_length is None:
            max_length = self.max_len
        encoder_ids: list = [self._append_special_tokens_and_truncate(t, src_lang, max_length - 2) for t in src_texts]
        encoder_inputs = self.batch_encode_plus(
            encoder_ids,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )

        if tgt_texts is not None:
            decoder_ids = [self._append_special_tokens_and_truncate(t, tgt_lang, max_length - 2) for t in tgt_texts]
            decoder_inputs = self.batch_encode_plus(
                decoder_ids,
                add_special_tokens=False,
                return_tensors=return_tensors,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
            )
        else:
            decoder_inputs = {}
        return {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs.get("input_ids", None),
        }
