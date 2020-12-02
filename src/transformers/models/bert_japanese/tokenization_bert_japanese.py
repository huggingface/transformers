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


import collections
import copy
import os
import unicodedata
from typing import Optional

from ...utils import logging
from ..bert.tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer, load_vocab


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

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


class BertJapaneseTokenizer(BertTokenizer):
    """BERT tokenizer for Japanese text"""

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
        word_tokenizer_type="basic",
        subword_tokenizer_type="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
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
        super(BertTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
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

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.never_split = never_split
        self.mecab_kwargs = copy.deepcopy(mecab_kwargs)
        if do_word_tokenize:
            if word_tokenizer_type == "basic":
                self.word_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=False
                )
            elif word_tokenizer_type == "mecab":
                self.word_tokenizer = MecabTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, **(mecab_kwargs or {})
                )
            else:
                raise ValueError("Invalid word_tokenizer_type '{}' is specified.".format(word_tokenizer_type))

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            else:
                raise ValueError("Invalid subword_tokenizer_type '{}' is specified.".format(subword_tokenizer_type))

    @property
    def do_lower_case(self):
        return self.lower_case

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.word_tokenizer_type == "mecab":
            del state["word_tokenizer"]
        return state

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
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens


class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""

    def __init__(
        self,
        do_lower_case=False,
        never_split=None,
        normalize_text=True,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
    ):
        """
        Constructs a MecabTokenizer.

        Args:
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of tokens not to split.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **mecab_dic**: (`optional`) string (default "ipadic")
                Name of dictionary to be used for MeCab initialization. If you are using a system-installed dictionary,
                set this option to `None` and modify `mecab_option`.
            **mecab_option**: (`optional`) string
                String passed to MeCab constructor.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

        try:
            import fugashi
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install fugashi to use MecabTokenizer."
                "See https://pypi.org/project/fugashi/ for installation."
            )

        mecab_option = mecab_option or ""

        if mecab_dic is not None:
            if mecab_dic == "ipadic":
                try:
                    import ipadic
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The ipadic dictionary is not installed. "
                        "See https://github.com/polm/ipadic-py for installation."
                    )

                dic_dir = ipadic.DICDIR

            elif mecab_dic == "unidic_lite":
                try:
                    import unidic_lite
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The unidic_lite dictionary is not installed. "
                        "See https://github.com/polm/unidic-lite for installation."
                    )

                dic_dir = unidic_lite.DICDIR

            elif mecab_dic == "unidic":
                try:
                    import unidic
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The unidic dictionary is not installed. "
                        "See https://github.com/polm/unidic-py for installation."
                    )

                dic_dir = unidic.DICDIR
                if not os.path.isdir(dic_dir):
                    raise RuntimeError(
                        "The unidic dictionary itself is not found."
                        "See https://github.com/polm/unidic-py for installation."
                    )

            else:
                raise ValueError("Invalid mecab_dic is specified.")

            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = '-d "{}" -r "{}" '.format(dic_dir, mecabrc) + mecab_option

        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []

        for word in self.mecab(text):
            token = word.surface

            if self.do_lower_case and token not in never_split:
                token = token.lower()

            tokens.append(token)

        return tokens


class CharacterTokenizer:
    """Runs Character tokenziation."""

    def __init__(self, vocab, unk_token, normalize_text=True):
        """
        Constructs a CharacterTokenizer.

        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.normalize_text = normalize_text

    def tokenize(self, text):
        """
        Tokenizes a piece of text into characters.

        For example, :obj:`input = "apple""` wil return as output :obj:`["a", "p", "p", "l", "e"]`.

        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.

        Returns:
            A list of characters.
        """
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        output_tokens = []
        for char in text:
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            output_tokens.append(char)

        return output_tokens
