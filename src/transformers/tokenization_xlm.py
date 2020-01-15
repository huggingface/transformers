# coding=utf-8
# Copyright 2019 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for XLM."""


import json
import logging
import os
import re
import sys
import unicodedata

import sacremoses as sm

from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlm-mlm-en-2048": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-vocab.json",
        "xlm-mlm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-vocab.json",
        "xlm-mlm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-vocab.json",
        "xlm-mlm-enro-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-vocab.json",
        "xlm-mlm-tlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-vocab.json",
        "xlm-mlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-vocab.json",
        "xlm-clm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-vocab.json",
        "xlm-clm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-vocab.json",
        "xlm-mlm-17-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-vocab.json",
        "xlm-mlm-100-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-vocab.json",
    },
    "merges_file": {
        "xlm-mlm-en-2048": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-merges.txt",
        "xlm-mlm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
        "xlm-mlm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        "xlm-mlm-enro-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-merges.txt",
        "xlm-mlm-tlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-merges.txt",
        "xlm-mlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-merges.txt",
        "xlm-clm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        "xlm-clm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
        "xlm-mlm-17-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-merges.txt",
        "xlm-mlm-100-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-mlm-en-2048": 512,
    "xlm-mlm-ende-1024": 512,
    "xlm-mlm-enfr-1024": 512,
    "xlm-mlm-enro-1024": 512,
    "xlm-mlm-tlm-xnli15-1024": 512,
    "xlm-mlm-xnli15-1024": 512,
    "xlm-clm-enfr-1024": 512,
    "xlm-clm-ende-1024": 512,
    "xlm-mlm-17-1280": 512,
    "xlm-mlm-100-1280": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "xlm-mlm-en-2048": {"do_lowercase_and_remove_accent": True},
    "xlm-mlm-ende-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {"0": "de", "1": "en"},
        "lang2id": {"de": 0, "en": 1},
    },
    "xlm-mlm-enfr-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {"0": "en", "1": "fr"},
        "lang2id": {"en": 0, "fr": 1},
    },
    "xlm-mlm-enro-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {"0": "en", "1": "ro"},
        "lang2id": {"en": 0, "ro": 1},
    },
    "xlm-mlm-tlm-xnli15-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {
            "0": "ar",
            "1": "bg",
            "2": "de",
            "3": "el",
            "4": "en",
            "5": "es",
            "6": "fr",
            "7": "hi",
            "8": "ru",
            "9": "sw",
            "10": "th",
            "11": "tr",
            "12": "ur",
            "13": "vi",
            "14": "zh",
        },
        "lang2id": {
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    },
    "xlm-mlm-xnli15-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {
            "0": "ar",
            "1": "bg",
            "2": "de",
            "3": "el",
            "4": "en",
            "5": "es",
            "6": "fr",
            "7": "hi",
            "8": "ru",
            "9": "sw",
            "10": "th",
            "11": "tr",
            "12": "ur",
            "13": "vi",
            "14": "zh",
        },
        "lang2id": {
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    },
    "xlm-clm-enfr-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {"0": "en", "1": "fr"},
        "lang2id": {"en": 0, "fr": 1},
    },
    "xlm-clm-ende-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {"0": "de", "1": "en"},
        "lang2id": {"de": 0, "en": 1},
    },
    "xlm-mlm-17-1280": {
        "do_lowercase_and_remove_accent": False,
        "id2lang": {
            "0": "ar",
            "1": "de",
            "2": "en",
            "3": "es",
            "4": "fr",
            "5": "hi",
            "6": "it",
            "7": "ja",
            "8": "ko",
            "9": "nl",
            "10": "pl",
            "11": "pt",
            "12": "ru",
            "13": "sv",
            "14": "tr",
            "15": "vi",
            "16": "zh",
        },
        "lang2id": {
            "ar": 0,
            "de": 1,
            "en": 2,
            "es": 3,
            "fr": 4,
            "hi": 5,
            "it": 6,
            "ja": 7,
            "ko": 8,
            "nl": 9,
            "pl": 10,
            "pt": 11,
            "ru": 12,
            "sv": 13,
            "tr": 14,
            "vi": 15,
            "zh": 16,
        },
    },
    "xlm-mlm-100-1280": {
        "do_lowercase_and_remove_accent": False,
        "id2lang": {
            "0": "af",
            "1": "als",
            "2": "am",
            "3": "an",
            "4": "ang",
            "5": "ar",
            "6": "arz",
            "7": "ast",
            "8": "az",
            "9": "bar",
            "10": "be",
            "11": "bg",
            "12": "bn",
            "13": "br",
            "14": "bs",
            "15": "ca",
            "16": "ceb",
            "17": "ckb",
            "18": "cs",
            "19": "cy",
            "20": "da",
            "21": "de",
            "22": "el",
            "23": "en",
            "24": "eo",
            "25": "es",
            "26": "et",
            "27": "eu",
            "28": "fa",
            "29": "fi",
            "30": "fr",
            "31": "fy",
            "32": "ga",
            "33": "gan",
            "34": "gl",
            "35": "gu",
            "36": "he",
            "37": "hi",
            "38": "hr",
            "39": "hu",
            "40": "hy",
            "41": "ia",
            "42": "id",
            "43": "is",
            "44": "it",
            "45": "ja",
            "46": "jv",
            "47": "ka",
            "48": "kk",
            "49": "kn",
            "50": "ko",
            "51": "ku",
            "52": "la",
            "53": "lb",
            "54": "lt",
            "55": "lv",
            "56": "mk",
            "57": "ml",
            "58": "mn",
            "59": "mr",
            "60": "ms",
            "61": "my",
            "62": "nds",
            "63": "ne",
            "64": "nl",
            "65": "nn",
            "66": "no",
            "67": "oc",
            "68": "pl",
            "69": "pt",
            "70": "ro",
            "71": "ru",
            "72": "scn",
            "73": "sco",
            "74": "sh",
            "75": "si",
            "76": "simple",
            "77": "sk",
            "78": "sl",
            "79": "sq",
            "80": "sr",
            "81": "sv",
            "82": "sw",
            "83": "ta",
            "84": "te",
            "85": "th",
            "86": "tl",
            "87": "tr",
            "88": "tt",
            "89": "uk",
            "90": "ur",
            "91": "uz",
            "92": "vi",
            "93": "war",
            "94": "wuu",
            "95": "yi",
            "96": "zh",
            "97": "zh_classical",
            "98": "zh_min_nan",
            "99": "zh_yue",
        },
        "lang2id": {
            "af": 0,
            "als": 1,
            "am": 2,
            "an": 3,
            "ang": 4,
            "ar": 5,
            "arz": 6,
            "ast": 7,
            "az": 8,
            "bar": 9,
            "be": 10,
            "bg": 11,
            "bn": 12,
            "br": 13,
            "bs": 14,
            "ca": 15,
            "ceb": 16,
            "ckb": 17,
            "cs": 18,
            "cy": 19,
            "da": 20,
            "de": 21,
            "el": 22,
            "en": 23,
            "eo": 24,
            "es": 25,
            "et": 26,
            "eu": 27,
            "fa": 28,
            "fi": 29,
            "fr": 30,
            "fy": 31,
            "ga": 32,
            "gan": 33,
            "gl": 34,
            "gu": 35,
            "he": 36,
            "hi": 37,
            "hr": 38,
            "hu": 39,
            "hy": 40,
            "ia": 41,
            "id": 42,
            "is": 43,
            "it": 44,
            "ja": 45,
            "jv": 46,
            "ka": 47,
            "kk": 48,
            "kn": 49,
            "ko": 50,
            "ku": 51,
            "la": 52,
            "lb": 53,
            "lt": 54,
            "lv": 55,
            "mk": 56,
            "ml": 57,
            "mn": 58,
            "mr": 59,
            "ms": 60,
            "my": 61,
            "nds": 62,
            "ne": 63,
            "nl": 64,
            "nn": 65,
            "no": 66,
            "oc": 67,
            "pl": 68,
            "pt": 69,
            "ro": 70,
            "ru": 71,
            "scn": 72,
            "sco": 73,
            "sh": 74,
            "si": 75,
            "simple": 76,
            "sk": 77,
            "sl": 78,
            "sq": 79,
            "sr": 80,
            "sv": 81,
            "sw": 82,
            "ta": 83,
            "te": 84,
            "th": 85,
            "tl": 86,
            "tr": 87,
            "tt": 88,
            "uk": 89,
            "ur": 90,
            "uz": 91,
            "vi": 92,
            "war": 93,
            "wuu": 94,
            "yi": 95,
            "zh": 96,
            "zh_classical": 97,
            "zh_min_nan": 98,
            "zh_yue": 99,
        },
    },
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    text = " ".join(text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output).lower().split(" ")


def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            continue
        output.append(char)
    return "".join(output)


def romanian_preprocessing(text):
    """Sennrich's WMT16 scripts for Romanian preprocessing, used by model `xlm-mlm-enro-1024`"""
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
    text = text.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
    text = text.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/remove-diacritics.py
    text = text.replace("\u0218", "S").replace("\u0219", "s")  # s-comma
    text = text.replace("\u021a", "T").replace("\u021b", "t")  # t-comma
    text = text.replace("\u0102", "A").replace("\u0103", "a")
    text = text.replace("\u00C2", "A").replace("\u00E2", "a")
    text = text.replace("\u00CE", "I").replace("\u00EE", "i")
    return text


class XLMTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer for XLM

        - Moses preprocessing & tokenization for most supported languages

        - Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP)

        - (optionally) lower case & normalize all inputs text

        - argument ``special_tokens`` and function ``set_special_tokens``, can be used to add additional symbols \
        (ex: "__classify__") to a vocabulary

        - `lang2id` attribute maps the languages supported by the model with their ids if provided (automatically set for pretrained vocabularies)

        - `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies)

        - `do_lowercase_and_remove_accent` controle lower casing and accent (automatically set for pretrained vocabularies)
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="</s>",
        mask_token="<special1>",
        additional_special_tokens=[
            "<special0>",
            "<special1>",
            "<special2>",
            "<special3>",
            "<special4>",
            "<special5>",
            "<special6>",
            "<special7>",
            "<special8>",
            "<special9>",
        ],
        lang2id=None,
        id2lang=None,
        do_lowercase_and_remove_accent=True,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        # cache of sm.MosesPunctNormalizer instance
        self.cache_moses_punct_normalizer = dict()
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = dict()
        self.lang_with_custom_tokenizer = set(["zh", "th", "ja"])
        # True for current supported model (v1.2.0), False for XLM-17 & 100
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    def ja_tokenize(self, text):
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea

                self.ja_word_tokenizer = Mykytea.Mykytea(
                    "-model %s/local/share/kytea/model.bin" % os.path.expanduser("~")
                )
            except (AttributeError, ImportError):
                logger.error(
                    "Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper (https://github.com/chezou/Mykytea-python) with the following steps"
                )
                logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
                logger.error("2. autoreconf -i")
                logger.error("3. ./configure --prefix=$HOME/local")
                logger.error("4. make && make install")
                logger.error("5. pip install kytea")
                raise
        return list(self.ja_word_tokenizer.getWS(text))

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _tokenize(self, text, lang="en", bypass_tokenizer=False):
        """
        Tokenize a string given language code. For Chinese, Japanese and Thai, we use a language specific tokenizerself. Otherwise, we use Moses.

        Details of tokenization:
        - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`
        - [pythainlp](https://github.com/PyThaiNLP/pythainlp): Thai tokenizer
            - Install with `pip install pythainlp`
        - [kytea](https://github.com/chezou/Mykytea-python): Japanese tokenizer, wrapper of [KyTea](https://github.com/neubig/kytea)
            - Install with the following steps:
            ```
            git clone git@github.com:neubig/kytea.git && cd kytea
            autoreconf -i
            ./configure --prefix=$HOME/local
            make && make install
            pip install kytea
            ```
        - [jieba](https://github.com/fxsjy/jieba): Chinese tokenizer (*)
            - Install with `pip install jieba`

        (*) The original XLM used [Stanford Segmenter](https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip).
        However, the wrapper (`nltk.tokenize.stanford_segmenter`) is slow due to JVM overhead, and it will be deprecated.
        Jieba is a lot faster and pip-installable. Note there is some mismatch with the Stanford Segmenter. It should be fine
        if you fine-tune the model with Chinese supervisionself. If you want the same exact behaviour, use the original XLM
        [preprocessing script](https://github.com/facebookresearch/XLM/tree/master/tools) to tokenize the sentence externally,
        and set `bypass_tokenizer=True` to bypass the tokenizer.

        Args:
            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)  (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error(
                "Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model."
            )
        if bypass_tokenizer:
            text = text.split()
        elif lang not in self.lang_with_custom_tokenizer:
            text = self.moses_pipeline(text, lang=lang)
            # TODO: make sure we are using `xlm-mlm-enro-1024`, since XLM-100 doesn't have this step
            if lang == "ro":
                text = romanian_preprocessing(text)
            text = self.moses_tokenize(text, lang=lang)
        elif lang == "th":
            text = self.moses_pipeline(text, lang=lang)
            try:
                if "pythainlp" not in sys.modules:
                    from pythainlp.tokenize import word_tokenize as th_word_tokenize
                else:
                    th_word_tokenize = sys.modules["pythainlp"].word_tokenize
            except (AttributeError, ImportError):
                logger.error(
                    "Make sure you install PyThaiNLP (https://github.com/PyThaiNLP/pythainlp) with the following steps"
                )
                logger.error("1. pip install pythainlp")
                raise
            text = th_word_tokenize(text)
        elif lang == "zh":
            try:
                if "jieba" not in sys.modules:
                    import jieba
                else:
                    jieba = sys.modules["jieba"]
            except (AttributeError, ImportError):
                logger.error("Make sure you install Jieba (https://github.com/fxsjy/jieba) with the following steps")
                logger.error("1. pip install jieba")
                raise
            text = " ".join(jieba.cut(text))
            text = self.moses_pipeline(text, lang=lang)
            text = text.split()
        elif lang == "ja":
            text = self.moses_pipeline(text, lang=lang)
            text = self.ja_tokenize(text)
        else:
            raise ValueError("It should not reach here")

        if self.do_lowercase_and_remove_accent and not bypass_tokenizer:
            text = lowercase_and_remove_accent(text)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A XLM sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s> B </s>
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0,))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        An XLM sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file
