# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Allegro.pl, Facebook Inc. and the HuggingFace Inc. team.
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
import unicodedata

from ...utils import logging
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/vocab.json"
    },
    "merges_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/merges.txt"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"allegro/herbert-base-cased": 514}
PRETRAINED_INIT_CONFIGURATION = {}

# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class HerbertTokenizer(PreTrainedTokenizer):
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
      punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        tokenizer_file=None,
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sep_token="</s>",
        do_lowercase_and_remove_accent=False,
        **kwargs
    ):

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=None,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            do_lowercase_and_remove_accent=do_lowercase_and_remove_accent,
            **kwargs,
        )
        self.bert_pre_tokenizer = BasicTokenizer(
            do_lower_case=False,
            never_split=self.all_special_tokens,
            tokenize_chinese_chars=False,
            strip_accents=False,
        )

    def _tokenize(self, text):

        pre_tokens = self.bert_pre_tokenizer.tokenize(text)

        split_tokens = []
        for token in pre_tokens:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens
