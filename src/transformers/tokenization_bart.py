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

from .tokenization_roberta import RobertaTokenizer
from .tokenization_t5 import T5Tokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer
import logging

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
lang_codes = {'ar_AR': 0, 'cs_CZ': 1, 'de_DE': 2, 'en_XX': 3, 'es_XX': 4, 'et_EE': 5, 'fi_FI': 6, 'fr_XX': 7,
              'gu_IN': 8, 'hi_IN': 9, 'it_IT': 10, 'ja_XX': 11, 'kk_KZ': 12, 'ko_KR': 13, 'lt_LT': 14, 'lv_LV': 15,
              'my_MM': 16, 'ne_NP': 17, 'nl_XX': 18, 'ro_RO': 19, 'ru_RU': 20, 'si_LK': 21, 'tr_TR': 22,
              'vi_VN': 23, 'zh_CN': 24}

class MBartTokenizerV2(XLMRobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {
            "mbart-large-en-ro": _s3_url("facebook/mbart-large-en-ro/sentence.bpe.model"),
            "mbart-large-cc25": _s3_url("facebook/mbart-large-cc25/sentence.bpe.model"),
        }
    }

class MBartTokenizer(T5Tokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}

    pretrained_vocab_files_map = {
        "vocab_file": {
            "mbart-large-en-ro": _s3_url("facebook/mbart-large-en-ro/sentence.bpe.model"),
            "mbart-large-cc25": _s3_url("facebook/mbart-large-cc25/sentence.bpe.model"),
        }
    }

    def pretrained_init(self, max_len=None, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)

        # Padding side is right by default and over-riden in subclasses. If specified in the kwargs, it is changed.
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)

        # Added tokens
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                else:
                    assert isinstance(value, str)
                setattr(self, key, value)

    def __init__(
            self,
            vocab_file,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=0,
            **kwargs
    ):
        # Add extra_ids to the special token list
        additional_special_tokens = list(lang_codes.keys())
        if extra_ids > 0:
            additional_special_tokens.extend(["<extra_id_{}>".format(i) for i in range(extra_ids)])

        self.pretrained_init(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.max_len_single_sentence = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use MBARTTokenizer:"
                "https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.LoadFromSerializedProto(open(vocab_file, 'rb').read())
