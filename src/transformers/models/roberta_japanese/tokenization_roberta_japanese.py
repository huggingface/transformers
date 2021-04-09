# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""

import copy

from ..bert_japanese.tokenization_bert_japanese import MecabTokenizer
from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "cl-tohoku/bert-base-japanese": 512,
    "cl-tohoku/bert-base-japanese-whole-word-masking": 512,
    "cl-tohoku/bert-base-japanese-char": 512,
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "cl-tohoku/bert-base-japanese": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "wordpiece",
    },
    "cl-tohoku/bert-base-japanese-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "wordpiece",
    },
    "cl-tohoku/bert-base-japanese-char": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "character",
    },
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "character",
    },
}


class RobertaJapaneseTokenizer(XLMRobertaTokenizer):
    """RoBERTa tokenizer for Japanese text"""

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            do_word_tokenize=True,
            do_subword_tokenize=True,
            do_zenkaku=False,
            word_tokenizer_type="mecab",
            subword_tokenizer_type="bpe",
            never_split=None,
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
            mecab_kwargs=None,
            **kwargs
    ):
        """
        Constructs a MecabBertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
            **do_word_tokenize**: (`optional`) boolean (default True)
                Whether to do word tokenization.
            **do_subword_tokenize**: (`optional`) boolean (default True)
                Whether to do subword tokenization.
            **word_tokenizer_type**: (`optional`) string (default "basic")
                Type of word tokenizer.
            **subword_tokenizer_type**: (`optional`) string (default "wordpiece")
                Type of subword tokenizer.
            **mecab_kwargs**: (`optional`) dict passed to `MecabTokenizer` constructor (default None)
        """
        super(RobertaJapaneseTokenizer, self).__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            never_split=never_split,
            mecab_kwargs=mecab_kwargs,
            **kwargs,
        )
        # ^^ We call the grandparent's init, not the parent's.

        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.never_split = never_split
        self.mecab_kwargs = copy.deepcopy(mecab_kwargs)

        if do_word_tokenize:
            if word_tokenizer_type == "mecab":
                self.word_tokenizer = MecabTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, do_zenkaku=do_zenkaku, **(mecab_kwargs or {})
                )
            else:
                raise ValueError(f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified.")

        self.do_subword_tokenize = do_subword_tokenize

    @property
    def do_lower_case(self):
        return self.lower_case

    # TODO: (kiyono) No idea how to deal with this
    def __getstate__(self):
        state = dict(self.__dict__)
        if self.word_tokenizer_type == "mecab":
            del state["word_tokenizer"]
        return state

    # TODO: (kiyono) No idea how to deal with this
    def __setstate__(self, state):
        self.__dict__ = state
        if self.word_tokenizer_type == "mecab":
            self.word_tokenizer = MecabTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.mecab_kwargs or {})
            )

    def _tokenize(self, text):

        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            split_tokens = self.sp_model.EncodeAsPieces(' '.join(tokens))
        else:
            split_tokens = tokens

        return split_tokens
