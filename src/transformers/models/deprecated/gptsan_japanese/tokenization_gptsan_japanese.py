# coding=utf-8
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for GPTSANJapanese."""

import collections
import json
import os
import re
import sys
from typing import List, Optional, Tuple, Union

import numpy as np

from ....tokenization_utils import PreTrainedTokenizer
from ....tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ....utils import PaddingStrategy, logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}


def load_vocab_and_emoji(vocab_file, emoji_file):
    """Loads a vocabulary file and emoji file into a dictionary."""
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())

    vocab = collections.OrderedDict()
    raw_vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()
    token = [[t.rstrip("\n")] if (t == ",\n" or "," not in t) else t.rstrip("\n").split(",") for t in token]
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[",".join(b)] = idx
        for wd in b:
            vocab[wd] = idx

    return vocab, raw_vocab, ids_to_tokens, emoji


class GPTSanJapaneseTokenizer(PreTrainedTokenizer):
    """
    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling
    - Return token_type_ids for Prefix-LM model
    The bagofword token represents a repetition of the previous token and is converted to 3 consecutive tokens when
    decoding In addition, the original Japanese special Sub-Word-Encoding has been released in this repository
    (https://github.com/tanreinama/Japanese-BPEEncoder_V2). The token_type_ids is a mask indicating the prefix input
    position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix_text, or specify a
    sentence of the prefix part and the part after it as a text pair of batch input.

    Example:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17750
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```

    Example for Prefix-LM:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
    [35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ```

    Example for batch encode:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
    [[35993, 35998, 8640, 25948, 35993, 35998, 30647, 35675, 35999, 35999], [35993, 35998, 10382, 9868, 35993, 35998, 30646, 9459, 30646, 35675]]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    >>> # Mask for padding
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|nottoken|>"`):
            The token used for unknown charactor
        pad_token (`str`, *optional*, defaults to `"<|separator|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"<|segmenter|>"`):
            A special token to separate token to prefix part and general input part.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(
        self,
        vocab_file,
        emoji_file,
        unk_token="<|nottoken|>",
        pad_token="<|separator|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        sep_token="<|segmenter|>",
        do_clean_text=False,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find a emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_clean_text = do_clean_text
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            do_clean_text=do_clean_text,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # self.vocab contains support for character fluctuation unique to Japanese, and has a large number of vocab
        return len(self.raw_vocab)

    def get_vocab(self):
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        words = []
        byte_tokens = []
        for word in tokens:
            if word[:6] == "<|byte" and word[-2:] == "|>":
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                    byte_tokens = []
                if word[:7] == "<|emoji" and word[-2:] == "|>":
                    words.append(self.emoji["emoji_inv"][word])
                elif word == "<SP>":
                    words.append(" ")
                elif word == "<BR>":
                    words.append("\n")
                elif word == "<TAB>":
                    words.append("\t")
                elif word == "<BLOCK>":
                    words.append("▀")
                elif word == "<KIGOU>":
                    words.append("ǀ")
                elif word == "<U2000U2BFF>":
                    words.append("‖")
                elif word == "<|bagoftoken|>":
                    if len(words) > 0:
                        words.append(words[-1])
                        words.append(words[-1])
                        words.append(words[-1])
                elif word.startswith("<|") and word.endswith("|>"):
                    words.append("")
                else:
                    words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
            )
        else:
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_index, token in self.ids_to_tokens.items():
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(",".join(token) + "\n")
                index += 1
        with open(emoji_file, "w", encoding="utf-8") as writer:
            json.dump(self.emoji, writer)
        return vocab_file, emoji_file

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # docstyle-ignore
        """
        The tokenizer returns token_type_ids as separators between the Prefix part and the rest.
        token_type_ids is 1 for the Prefix part and 0 for the rest of the token.

        Example:
        ```python
        >>> from transformers import GPTSanJapaneseTokenizer

        >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("ｱｲｳｴ")
        >>> # input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |

        >>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
        >>> # input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
        >>> # token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |

        >>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
        >>> # input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
        ```"""
        prefix_len = 0
        if self.sep_token in self.vocab:
            segid = self.vocab[self.sep_token]
            if segid in token_ids_0:
                prefix_len = token_ids_0.index(segid)
        if token_ids_1 is None:
            total_len = len(token_ids_0)
        else:
            total_len = len(token_ids_0 + token_ids_1)
        return prefix_len * [1] + (total_len - prefix_len) * [0]

    def prepare_for_tokenization(self, text, prefix_text=None, add_sep_token=None, **kwargs):
        # GPTSAN inserts extra SEP tokens in Prefix-LM in addition to SOT for text generation.
        # SOT at the beginning of the text, and SEP at the separator between the Prefix part and the rest.
        if add_sep_token is None:
            add_sep_token = self.sep_token not in text  # If insert un-prefix position explicitly
        prepared = self.bos_token if self.bos_token in self.vocab else ""
        prepared += prefix_text if prefix_text is not None else ""
        if add_sep_token:
            prepared += self.sep_token if self.sep_token in self.vocab else ""
        prepared += text
        return (prepared, kwargs)

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # This tokenizer converts input text pairs into Prefix input and subsequent input
        if isinstance(batch_text_or_text_pairs[0], tuple) or isinstance(tuple(batch_text_or_text_pairs[0]), list):
            # As a single text with an explicit un-prefix position
            batch_prefix_texts = []
            for pref, txt in batch_text_or_text_pairs:
                batch_prefix_texts.append(pref + self.sep_token + txt)
            batch_text_or_text_pairs = batch_prefix_texts

        return super()._batch_encode_plus(
            batch_text_or_text_pairs,
            add_special_tokens,
            padding_strategy,
            truncation_strategy,
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            return_tensors,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose,
            **kwargs,
        )


class SubWordJapaneseTokenizer:
    """
    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling

    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # same as swe
        self.ids_to_tokens = ids_to_tokens  # same as bpe
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        # The original version of this regex displays catastrophic backtracking behaviour. We avoid this using
        # possessive quantifiers in Py >= 3.11. In versions below this, we avoid the vulnerability using a slightly
        # different regex that should generally have the same behaviour in most non-pathological cases.
        if sys.version_info >= (3, 11):
            self.content_repatter6 = re.compile(
                r"(?:\d,\d{3}|[\d億])*+"
                r"(?:\d,\d{3}|[\d万])*+"
                r"(?:\d,\d{3}|[\d千])*+"
                r"(?:千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+"
                r"(?:\(税込\)|\(税抜\)|\+tax)*"
            )
        else:
            self.content_repatter6 = re.compile(
                r"(?:\d,\d{3}|[\d億万千])*"
                r"(?:千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+"
                r"(?:\(税込\)|\(税抜\)|\+tax)*"
            )
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})

    def __len__(self):
        return len(self.ids_to_tokens)

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)
        content = self.content_repatter2.sub("<EMAIL>", content)
        content = self.content_repatter3.sub("<TEL>", content)
        content = self.content_repatter4.sub("<DATE>", content)
        content = self.content_repatter5.sub("<DATE>", content)
        content = self.content_repatter6.sub("<PRICE>", content)
        content = content.translate(self.content_trans1)
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
        return content

    def tokenize(self, text, clean=False):
        text = text.replace(" ", "<SP>")
        text = text.replace("　", "<SP>")
        text = text.replace("\r\n", "<BR>")
        text = text.replace("\n", "<BR>")
        text = text.replace("\r", "<BR>")
        text = text.replace("\t", "<TAB>")
        text = text.replace("—", "ー")
        text = text.replace("−", "ー")
        for k, v in self.emoji["emoji"].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)

        def check_simbol(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                if (
                    (c >= 0xC2A1 and c <= 0xC2BF)
                    or (c >= 0xC780 and c <= 0xC783)
                    or (c >= 0xCAB9 and c <= 0xCBBF)
                    or (c >= 0xCC80 and c <= 0xCDA2)
                ):
                    return True
            return False

        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                if c >= 0xE28080 and c <= 0xE2B07F:
                    return True
            return False

        pos = 0
        result = []
        while pos < len(text):
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
            candidates = []  # (token_id, token, pos)
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.vocab:
                    if wd[0] == "<" and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            if len(candidates) > 0:
                # the smallest token_id is adopted
                _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wd)
                pos = e
            else:
                end = pos + 1
                wd = text[pos:end]
                if check_simbol(wd):
                    result.append("<KIGOU>")
                elif checku2e(wd):
                    result.append("<U2000U2BFF>")
                else:
                    for i in wd.encode("utf-8"):
                        result.append("<|byte%d|>" % i)
                pos = end
        return result

    def convert_id_to_token(self, index):
        return self.ids_to_tokens[index][0]
