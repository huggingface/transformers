# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
 Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
 see tokenization_utils.py
"""
import copy
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tokenizers.pre_tokenizers as pre_tokenizers_fast
from Levenshtein import ratio
from nltk.corpus import words
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

from .convert_slow_tokenizer import convert_slow_tokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from .utils import PaddingStrategy, add_end_docstrings, logging


logger = logging.get_logger(__name__)

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Slow tokenizers have an additional added tokens files
ADDED_TOKENS_FILE = "added_tokens.json"

INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE}


def markdown_compatible(s: str) -> str:
    """
    Make text compatible with Markdown formatting.

    This function makes various text formatting adjustments to make it compatible with Markdown.

    Args:
        s (str): The input text to be made Markdown-compatible.

    Returns:
        str: The Markdown-compatible text.
    """
    # equation tag
    s = re.sub(r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", s, flags=re.M)
    s = re.sub(r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", s, flags=re.M)
    s = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        s,
        flags=re.M,
    )  # multi line
    s = s.replace(r"\. ", ". ")
    # bold formatting
    s = s.replace(r"\bm{", r"\mathbf{").replace(r"{\\bm ", r"\mathbf{")
    # s = s.replace(r"\it{", r"\mathit{").replace(r"{\\it ", r"\mathit{") # not needed
    s = re.sub(r"\\mbox{ ?\\boldmath\$(.*?)\$}", r"\\mathbf{\1}", s)
    # s=re.sub(r'\\begin{table}(.+?)\\end{table}\nTable \d+: (.+?)\n',r'\\begin{table}\1\n\\capation{\2}\n\\end{table}\n',s,flags=re.S)
    # s=re.sub(r'###### Abstract\n(.*?)\n\n',r'\\begin{abstract}\n\1\n\\end{abstract}\n\n',s,flags=re.S)
    # urls
    s = re.sub(
        r"((?:http|ftp|https):\/\/(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))",
        r"[\1](\1)",
        s,
    )
    # algorithms
    s = re.sub(r"```\s*(.+?)\s*```", r"```\n\1\n```", s, flags=re.S)
    # lists

    return s


def find_next_punctuation(s: str, start_inx=0):
    """
    Find the index of the next punctuation mark

    Args:
        s: String to examine
        start_inx: Index where to start
    """

    for i in range(start_inx, len(s)):
        if s[i] in [".", "?", "!", "\n"]:
            return i

    return None


def find_last_punctuation(s: str, start_inx=0):
    """
    Find the index of the last punctuation mark before start_inx

    Args:
        s: String to examine
        start_inx: Index where to look before
    """

    for i in range(start_inx - 1, 0, -1):
        if s[i] in [".", "?", "!", "\n"]:
            return i

    return None


def truncate_repetitions(s: str, min_len=30):
    """
    Attempt to truncate repeating segments in the input string.

    This function looks for the longest repeating substring at the end of the input string and truncates it to appear
    only once. To be considered for removal, repetitions need to be continuous.

    Args:
        s (str): The input raw prediction to be truncated.
        min_len (int): The minimum length of the repeating segment.

    Returns:
        str: The input string with repeated segments truncated.
    """
    s_lower = s.lower()
    s_len = len(s_lower)

    if s_len < 2 * min_len:
        return s

    # try to find a length at which the tail is repeating
    max_rep_len = None
    for rep_len in range(min_len, int(s_len / 2)):
        # check if there is a repetition at the end
        same = True
        for i in range(0, rep_len):
            if s_lower[s_len - rep_len - i - 1] != s_lower[s_len - i - 1]:
                same = False
                break

        if same:
            max_rep_len = rep_len

    if max_rep_len is None:
        return s

    lcs = s_lower[-max_rep_len:]

    # remove all but the last repetition
    st = s
    st_lower = s_lower
    while st_lower.endswith(lcs):
        st = st[:-max_rep_len]
        st_lower = st_lower[:-max_rep_len]

    # this is the tail with the repetitions
    repeating_tail = s_lower[len(st_lower) :]

    # add until next punctuation and make sure last sentence is not repeating
    st_lower_out = st_lower
    while True:
        sentence_end = find_next_punctuation(s_lower, len(st_lower_out))
        sentence_start = find_last_punctuation(s_lower, len(st_lower_out))
        if sentence_end and sentence_start:
            sentence = s_lower[sentence_start:sentence_end]
            st_lower_out = s_lower[: sentence_end + 1]
            if sentence in repeating_tail:
                break
        else:
            break

    s_out = s[: len(st_lower_out)]

    return s_out


def remove_numbers(lines):
    def _clean(s):
        return re.sub(r"(?:[\d_]|\*\*)", "", s).strip()

    if type(lines) is str:
        return _clean(lines)
    out = []
    for l in lines:
        out.append(_clean(l))
    return out


def get_slices(lines, clean_lines):
    """
    Get slices of text based on specific criteria within the lines.

    This function identifies and returns slices of text from the input lines based on certain conditions.

    Args:
        lines (list of str): The list of lines containing the text.
        clean_lines (list of str): A cleaned version of the text (without numbers).

    Returns:
        list of tuple: A list of tuples representing the start and end indices of text slices.
    """
    inds = np.zeros(len(lines))
    for i in range(len(lines) - 1):
        j = i + 1
        while not clean_lines[j] and j < len(lines) - 1:
            j += 1
        if (
            len(clean_lines[i]) < 200
            and len(clean_lines[i]) > 3
            and len(clean_lines[j]) < 200
            and len(clean_lines[j]) > 3
            and not clean_lines[i].startswith("[MISSING_PAGE")
            and (clean_lines[i] == clean_lines[j] or ratio(clean_lines[i], clean_lines[j]) > 0.9)
        ):
            inds[i:j] = 1
    ids = np.where(inds)[0]
    slices = []
    if len(ids) == 0:
        return slices
    j0 = 0
    for j, x in enumerate(np.diff(ids) > 3):
        if x:
            slices.append((ids[j0], ids[j] + 2))
            j0 = j + 1
    slices.append((ids[j0], ids[-1] + 2))
    return [sli for sli in slices if sli[1] - sli[0] > 15]


def remove_slice_from_lines(lines, clean_text, sli) -> str:
    """
    Remove a slice of text from the lines based on specific criteria.

    This function identifies a slice of text within the lines and removes it based on certain conditions.

    Args:
        lines (list of str): The list of lines containing the text.
        clean_text (list of str): A cleaned version of the text (without numbers).
        sli (tuple): A tuple representing the start and end indices of the slice to be removed.

    Returns:
        str: The removed slice of text as a single string.
    """
    base = clean_text[sli[0]]
    section = list(sli)
    check_start_flag = False
    # backwards pass
    for i in range(max(0, sli[0] - 1), max(0, sli[0] - 5), -1):
        if not lines[i]:
            continue
        if lines[i] == "## References":
            section[0] = i
            break
        elif ratio(base, remove_numbers(lines[i])) < 0.9:
            section[0] = i + 1
            potential_ref = remove_numbers(lines[max(0, i - 1)].partition("* [")[-1])
            if len(potential_ref) >= 0.75 * len(base) and ratio(base, potential_ref) < 0.9:
                section[0] = i
            check_start_flag = True
            break
    # forward pass
    for i in range(min(len(lines), sli[1]), min(len(lines), sli[1] + 5)):
        if ratio(base, remove_numbers(lines[i])) < 0.9:
            section[1] = i
            break
    if len(lines) <= section[1]:
        section[1] = len(lines) - 1
    to_delete = "\n".join(lines[section[0] : section[1] + 1])
    # cut off next page content
    itera, iterb = enumerate(lines[section[1] - 1]), enumerate(lines[section[1]])
    while True:
        try:
            (ia, a) = next(itera)
            while a.isnumeric():
                (ia, a) = next(itera)
            (ib, b) = next(iterb)
            while b.isnumeric():
                (ib, b) = next(iterb)
            if a != b:
                break
        except StopIteration:
            break
    if check_start_flag and "* [" in to_delete:
        to_delete = "* [" + to_delete.partition("* [")[-1]
    try:
        delta = len(lines[section[1]]) - ib - 1
        if delta > 0:
            to_delete = to_delete[:-delta]
    except UnboundLocalError:
        pass

    return to_delete.strip()


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class NougatTokenizerFast(PreTrainedTokenizerBase):
    """
    Fast tokenizer for Nougat (wrapping HuggingFace tokenizers library).

    Full copy of [`~tokenization_utils_base.PreTrainedTokenizerFast`] with additional Nougat-specific postprocessing.

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class: PreTrainedTokenizer = None

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.__init__
    def __init__(self, *args, **kwargs):
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)

        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            raise ValueError(
                "Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you "
                "have sentencepiece installed."
            )

        if tokenizer_object is not None:
            fast_tokenizer = copy.deepcopy(tokenizer_object)
        elif fast_tokenizer_file is not None and not from_slow:
            # We have a serialization from tokenizers which let us directly build the backend
            fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
        elif slow_tokenizer is not None:
            # We need to convert a slow tokenizer to build the backend
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        elif self.slow_tokenizer_class is not None:
            # We need to create and convert a slow tokenizer to build the backend
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        else:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece installed to convert a slow tokenizer to a fast one."
            )

        self._tokenizer = fast_tokenizer

        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        self._decode_use_source_tokenizer = False

        _truncation = self._tokenizer.truncation

        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
        else:
            self._tokenizer.no_truncation()

        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])

        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

    @property
    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.is_fast
    def is_fast(self) -> bool:
        return True

    @property
    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.can_save_slow_tokenizer
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return True

    @property
    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.vocab_size
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.get_vocab
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.vocab
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.get_added_vocab
    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        base_vocab = self._tokenizer.get_vocab(with_added_tokens=False)
        full_vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        added_vocab = {tok: index for tok, index in full_vocab.items() if tok not in base_vocab}
        return added_vocab

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.__len__
    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.backend_tokenizer
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        return self._tokenizer

    @property
    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.decoder
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        return self._tokenizer.decoder

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._convert_encoding
    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.convert_tokens_to_ids
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._convert_token_to_id_with_added_voc
    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._convert_id_to_token
    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._add_tokens
    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.num_special_tokens_to_add
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.convert_ids_to_tokens
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.tokenize
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.set_truncation_and_padding
    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
    ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy ([`~utils.PaddingStrategy`]):
                The kind of padding that will be applied to the input
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
                The kind of truncation that will be applied to the input
            max_length (`int`):
                The maximum size of a sequence.
            stride (`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
        """
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # _truncation might contain more keys that the target `transformers`
            # supports. Use only the target keys to trigger `enable_truncation`.
            # This should enable this code to works on various `tokenizers`
            # targets.
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            if current != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._batch_encode_plus
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
    ) -> BatchEncoding:
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})"
            )

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        # Convert encoding to dict
        # `Tokens` has type: Tuple[
        #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
        #                       List[EncodingFast]
        #                    ]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._encode_plus
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self._batch_encode_plus(
            batched_input,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Return tensor is None, then we can remove the leading batch axis
        # Overflowing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.backend_tokenizer.decoder.decode(tokens)

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._decode
    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained
    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens as well as in a unique JSON
        file containing {config + vocab + added-tokens}.
        """
        save_directory = str(save_directory)

        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You"
                " might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        save_slow = (
            (legacy_format is None or legacy_format is True)
            and self.slow_tokenizer_class is not None
            and self.can_save_slow_tokenizer
        )
        save_fast = legacy_format is None or legacy_format is False

        if save_slow:
            added_tokens_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            added_vocab = self.get_added_vocab()
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    f.write(out_str)

            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)

        if save_fast:
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        return file_names

    # Copied from transformers.tokenization_utils_fast.PreTrainedTokenizerFast.train_new_from_iterator
    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
        length=None,
        new_special_tokens=None,
        special_tokens_map=None,
        **kwargs,
    ):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
        as the current one.

        Args:
            text_iterator (generator of `List[str]`):
                The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
                if you have everything in memory.
            vocab_size (`int`):
                The size of the vocabulary you want for your tokenizer.
            length (`int`, *optional*):
                The total number of sequences in the iterator. This is used to provide meaningful progress tracking
            new_special_tokens (list of `str` or `AddedToken`, *optional*):
                A list of new special tokens to add to the tokenizer you are training.
            special_tokens_map (`Dict[str, str]`, *optional*):
                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
                token name to new special token name in this argument.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

        Returns:
            [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on
            `text_iterator`.

        """
        tokenizer_json = json.loads(self._tokenizer.to_str())
        # Remove added tokens for now (uses IDs of tokens)
        added_tokens = tokenizer_json.pop("added_tokens")
        # Remove post processor for now (uses IDs of tokens)
        post_processor = tokenizer_json.pop("post_processor")

        unk_token = None
        # Remove vocab
        if tokenizer_json["model"]["type"] == "BPE":
            tokenizer_json["model"]["vocab"] = {}
            tokenizer_json["model"]["merges"] = []
        elif tokenizer_json["model"]["type"] == "Unigram":
            if tokenizer_json["model"]["unk_id"] is not None:
                unk_id = tokenizer_json["model"]["unk_id"]
                unk_token = tokenizer_json["model"]["vocab"][unk_id][0]
                if special_tokens_map is not None and unk_token in special_tokens_map:
                    unk_token = special_tokens_map[unk_token]
                tokenizer_json["model"]["unk_id"] = 0
                tokenizer_json["model"]["vocab"] = [[unk_token, 0.0]]
        elif tokenizer_json["model"]["type"] in ["WordLevel", "WordPiece"]:
            tokenizer_json["model"]["vocab"] = {}
        else:
            raise ValueError(
                f"This method does not support this type of tokenizer (found {tokenizer_json['model']['type']}) "
                "only BPE, Unigram, WordLevel and WordPiece."
            )

        if (
            special_tokens_map is not None
            and "unk_token" in tokenizer_json["model"]
            and tokenizer_json["model"]["unk_token"] in special_tokens_map
        ):
            tokenizer_json["model"]["unk_token"] = special_tokens_map[tokenizer_json["model"]["unk_token"]]

        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))

        # Get the special tokens from the current tokenizer if none are specified.
        special_tokens = []
        for added_token in added_tokens:
            special = added_token.pop("special", None)
            _ = added_token.pop("id", None)
            if tokenizer_json["model"]["type"] != "Unigram" and not special:
                continue
            if special_tokens_map is not None and added_token["content"] in special_tokens_map:
                added_token["content"] = special_tokens_map[added_token["content"]]
            special_tokens.append(AddedToken(**added_token))

        if new_special_tokens is not None:
            special_tokens.extend(new_special_tokens)

        # Trainer needs to know the end of word / continuing subword thingies in BPE
        if (
            tokenizer_json["model"]["type"] == "BPE"
            and "continuing_subword_prefix" not in kwargs
            and tokenizer_json["model"]["continuing_subword_prefix"] is not None
        ):
            kwargs["continuing_subword_prefix"] = tokenizer_json["model"]["continuing_subword_prefix"]
        if (
            tokenizer_json["model"]["type"] == "BPE"
            and "end_of_word_suffix" not in kwargs
            and tokenizer_json["model"]["end_of_word_suffix"] is not None
        ):
            kwargs["end_of_word_suffix"] = tokenizer_json["model"]["end_of_word_suffix"]
        if tokenizer_json["model"]["type"] == "Unigram" and unk_token is not None:
            kwargs["unk_token"] = unk_token
        if tokenizer_json["pre_tokenizer"] is not None and tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel":
            kwargs["initial_alphabet"] = pre_tokenizers_fast.ByteLevel.alphabet()

        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json["model"]["type"]]
        trainer = trainer_class(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
        tokenizer.train_from_iterator(text_iterator, length=length, trainer=trainer)

        if post_processor is not None:
            trained_tokenizer_json = json.loads(tokenizer.to_str())
            # Almost done, we just have to adjust the token IDs in the post processor
            if "special_tokens" in post_processor:
                for key in post_processor["special_tokens"]:
                    tokens = post_processor["special_tokens"][key]["tokens"]
                    if special_tokens_map is not None:
                        tokens = [special_tokens_map.get(token, token) for token in tokens]
                    post_processor["special_tokens"][key]["tokens"] = tokens
                    post_processor["special_tokens"][key]["ids"] = [tokenizer.token_to_id(token) for token in tokens]

            for special_token in ["cls", "sep"]:
                if special_token in post_processor:
                    token, _ = post_processor[special_token]
                    if special_tokens_map is not None and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    post_processor[special_token] = [token, token_id]

            trained_tokenizer_json["post_processor"] = post_processor
            tokenizer = TokenizerFast.from_str(json.dumps(trained_tokenizer_json))

        kwargs = self.init_kwargs.copy()
        # Map pad/cls/mask token at the Transformers level
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(self, f"_{token}") is not None:
                special_token = getattr(self, token)
                if special_tokens_map is not None and special_token in special_tokens_map:
                    special_token = special_tokens_map[special_token]

                special_token_full = getattr(self, f"_{token}")
                if isinstance(special_token_full, AddedToken):
                    # Create an added token with the same parameters except the content
                    kwargs[token] = AddedToken(
                        special_token,
                        single_word=special_token_full.single_word,
                        lstrip=special_token_full.lstrip,
                        rstrip=special_token_full.rstrip,
                        normalized=special_token_full.normalized,
                    )
                else:
                    kwargs[token] = special_token

        additional_special_tokens = self.additional_special_tokens
        if new_special_tokens is not None:
            additional_special_tokens.extend(new_special_tokens)
        if len(additional_special_tokens) > 0:
            kwargs["additional_special_tokens"] = additional_special_tokens

        return self.__class__(tokenizer_object=tokenizer, **kwargs)

    def remove_hallucinated_references(self, text: str) -> str:
        """
        Remove hallucinated or missing references from the text.

        This function identifies and removes references that are marked as missing or hallucinated from the input text.

        Args:
            text (str): The input text containing references.

        Returns:
            str: The text with hallucinated references removed.
        """
        lines = text.split("\n")
        if len(lines) == 0:
            return ""
        clean_lines = remove_numbers(lines)
        slices = get_slices(lines, clean_lines)
        to_delete = []
        for sli in slices:
            to_delete.append(remove_slice_from_lines(lines, clean_lines, sli))
        for to_delete in reversed(to_delete):
            text = text.replace(to_delete, "\n\n[MISSING_PAGE_POST]\n\n")
        text = re.sub(
            r"## References\n+\[MISSING_PAGE_POST(:\d+)?\]",
            "\n\n[MISSING_PAGE_POST\\1]",
            text,
        )
        return text

    def postprocess_single(self, generation: str, markdown_fix: bool = True) -> str:
        """
        Postprocess a single generated text.

        Args:
            generation (str): The generated text to be postprocessed.
            markdown_fix (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

        Returns:
            str: The postprocessed text.
        """
        generation = re.sub(
            r"(?:\n|^)#+ \d*\W? ?(.{100,})", r"\n\1", generation
        )  # too long section titles probably are none
        generation = generation.strip()
        generation = generation.replace("\n* [leftmargin=*]\n", "\n")
        generation = re.sub(r"^#+ (?:\.?(?:\d|[ixv])+)*\s*(?:$|\n\s*)", "", generation, flags=re.M)
        # most likely hallucinated titles
        lines = generation.split("\n")
        if lines[-1].startswith("#") and lines[-1].lstrip("#").startswith(" ") and len(lines) > 1:
            print("INFO: likely hallucinated title at the end of the page: " + lines[-1])
            generation = "\n".join(lines[:-1])
        # obvious repetition detection
        generation = truncate_repetitions(generation)
        # Reference corrections
        generation = self.remove_hallucinated_references(generation)
        generation = re.sub(r"^\* \[\d+\](\s?[A-W]\.+\s?){10,}.*$", "", generation, flags=re.M)
        generation = re.sub(r"^(\* \[\d+\])\[\](.*)$", r"\1\2", generation, flags=re.M)
        generation = re.sub(r"(^\w\n\n|\n\n\w$)", "", generation)
        # pmc math artifact correction
        generation = re.sub(
            r"([\s.,()])_([a-zA-Z0-9])__([a-zA-Z0-9]){1,3}_([\s.,:()])",
            r"\1\(\2_{\3}\)\4",
            generation,
        )
        generation = re.sub(r"([\s.,\d])_([a-zA-Z0-9])_([\s.,\d;])", r"\1\(\2\)\3", generation)
        # footnote mistakes
        generation = re.sub(
            r"(\nFootnote .*?:) (?:footnotetext|thanks):\W*(.*(?:\n\n|$))",
            r"\1 \2",
            generation,
        )
        # TODO Come up with footnote formatting inside a table
        generation = re.sub(r"\[FOOTNOTE:.+?\](.*?)\[ENDFOOTNOTE\]", "", generation)
        # itemize post processing
        for match in reversed(
            list(
                re.finditer(
                    r"(?:^)(-|\*)?(?!-|\*) ?((?:\d|[ixv])+ )?.+? (-|\*) (((?:\d|[ixv])+)\.(\d|[ixv]) )?.*(?:$)",
                    generation,
                    flags=re.I | re.M,
                )
            )
        ):
            start, stop = match.span()
            delim = match.group(3) + " "
            splits = match.group(0).split(delim)
            replacement = ""
            if match.group(1) is not None:
                splits = splits[1:]
                delim1 = match.group(1) + " "
            else:
                delim1 = ""
                # too many false positives
                continue
            pre, post = generation[:start], generation[stop:]
            for i, item in enumerate(splits):
                level = 0
                potential_numeral, _, rest = item.strip().partition(" ")
                if not rest:
                    continue
                if re.match(r"^[\dixv]+((?:\.[\dixv])?)+$", potential_numeral, flags=re.I | re.M):
                    level = potential_numeral.count(".")

                replacement += (
                    ("\n" if i > 0 else "")
                    + ("\t" * level)
                    + (delim if i > 0 or start == 0 else delim1)
                    + item.strip()
                )
            if post == "":
                post = "\n"
            generation = pre + replacement + post

        if generation.endswith((".", "}")):
            generation += "\n\n"
        if re.match(r"[A-Z0-9,;:]$", generation):
            # add space in case it there is a comma or word ending
            generation += " "
        elif generation.startswith(("#", "**", "\\begin")):
            generation = "\n\n" + generation
        elif generation.split("\n")[-1].startswith(("#", "Figure", "Table")):
            generation = generation + "\n\n"
        else:
            try:
                last_word = generation.split(" ")[-1]
                if last_word in words.words():
                    generation += " "
            except LookupError:
                # add space just in case. Will split words but better than concatenating them
                generation += " "
                # download for the next time
                import nltk

                nltk.download("words")
        # table corrections
        # remove obvious wrong tables
        for l in generation.split("\n"):
            if l.count("\\begin{tabular}") > 15 or l.count("\\multicolumn") > 60 or l.count("&") > 400:
                generation = generation.replace(l, "")
        # whitespace corrections
        generation = generation.replace("\\begin{table} \\begin{tabular}", "\\begin{table}\n\\begin{tabular}")
        generation = generation.replace("\\end{tabular} \\end{table}", "\\end{tabular}\n\\end{table}")
        generation = generation.replace("\\end{table} Tab", "\\end{table}\nTab")
        generation = re.sub(r"(^.+)\\begin{tab", r"\1\n\\begin{tab", generation, flags=re.M)

        generation = generation.replace(r"\begin{tabular}{l l}  & \\ \end{tabular}", "").replace(
            "\\begin{tabular}{}\n\n\\end{tabular}", ""
        )
        generation = generation.replace("\\begin{array}[]{", "\\begin{array}{")
        generation = re.sub(
            r"\\begin{tabular}{([clr ]){2,}}\s*[& ]*\s*(\\\\)? \\end{tabular}",
            "",
            generation,
        )
        generation = re.sub(r"(\*\*S\. A\. B\.\*\*\n+){2,}", "", generation)
        generation = re.sub(r"^#+( [\[\d\w])?$", "", generation, flags=re.M)
        generation = re.sub(r"^\.\s*$", "", generation, flags=re.M)
        generation = re.sub(r"\n{3,}", "\n\n", generation)
        if markdown_fix:
            return markdown_compatible(generation)
        else:
            return generation
