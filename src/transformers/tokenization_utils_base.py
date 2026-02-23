# base
# Copyright 2020 The HuggingFace Inc. team.
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
Base classes common to both the slow and the fast tokenization classes: PreTrainedTokenizerBase (host all the user
fronting encoding methods) Special token mixing (host the special tokens logic) and BatchEncoding (wrap the dictionary
of output with special method for the Fast tokenizers)
"""

from __future__ import annotations

import copy
import json
import os
import re
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Callable, Collection, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Union

import numpy as np
from huggingface_hub import create_repo, is_offline_mode, list_repo_files
from packaging import version

from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
    CHAT_TEMPLATE_DIR,
    CHAT_TEMPLATE_FILE,
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    TensorType,
    add_end_docstrings,
    cached_file,
    copy_func,
    extract_commit_hash,
    is_mlx_available,
    is_numpy_array,
    is_protobuf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_device,
    is_torch_tensor,
    list_repo_templates,
    logging,
    requires_backends,
    to_py_obj,
)
from .utils.chat_parsing_utils import recursive_parse
from .utils.chat_template_utils import render_jinja_template
from .utils.import_utils import PROTOBUF_IMPORT_ERROR


if TYPE_CHECKING:
    if is_torch_available():
        import torch


def import_protobuf_decode_error(error_message=""):
    if is_protobuf_available():
        from google.protobuf.message import DecodeError

        return DecodeError
    else:
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))


def flatten(arr: list):
    res = []
    if len(arr) > 0:
        for sub_arr in arr:
            if isinstance(arr[0], (list, tuple)):
                res.extend(flatten(sub_arr))
            else:
                res.append(sub_arr)
    return res


if is_tokenizers_available() or TYPE_CHECKING:
    from tokenizers import Encoding as EncodingFast

if is_tokenizers_available():
    from tokenizers import AddedToken
else:

    @dataclass(frozen=False, eq=True)
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.

        The `normalized` will default to `not special` if it is not specified, similarly to the definition in
        `tokenizers`.
        """

        def __init__(
            self, content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None
        ):
            self.content = content
            self.single_word = single_word
            self.lstrip = lstrip
            self.rstrip = rstrip
            self.special = special
            self.normalized = normalized if normalized is not None else not special

        def __getstate__(self):
            return self.__dict__

        def __str__(self):
            return self.content


logger = logging.get_logger(__name__)

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = list[str]
EncodedInput = list[int]
TextInputPair = tuple[str, str]
PreTokenizedInputPair = tuple[list[str], list[str]]
EncodedInputPair = tuple[list[int], list[int]]

# Define type aliases for text-related non-text modalities
AudioInput = Union[np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]]

# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (`int`): Index of the first character in the original string.
        end (`int`): Index of the character following the last character in the original string.
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """
    Token span in an encoded string (list of tokens).

    Args:
        start (`int`): Index of the first token in the span.
        end (`int`): Index of the token following the last token in the span.
    """

    start: int
    end: int


class BatchEncoding(UserDict):
    """
    Holds the output of the [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`],
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`] and
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        encoding (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, *optional*):
            If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
            space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this
            information.
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above). Note that this
            parameter has an effect if the parameter `tensor_type` is set, *otherwise has no effect*.
        n_sequences (`Optional[int]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
            initialization.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        encoding: EncodingFast | Sequence[EncodingFast] | None = None,
        tensor_type: None | str | TensorType = None,
        prepend_batch_axis: bool = False,
        n_sequences: int | None = None,
    ):
        super().__init__(data)

        # If encoding is not None, the fast tokenization is used
        if encoding is not None and isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and encoding:
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> int | None:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        return self._n_sequences

    def __getitem__(self, item: int | str) -> Any | EncodingFast:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.

        If the key is a slice, returns the value of the dict associated to `key` ('input_ids', 'attention_mask', etc.)
        with the constraint of slice.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data}
        else:
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def is_fast(self) -> bool:
        """
        TOOD: ita i will rm this `bool`: Whether or not this BatchEncoding was created by a fast tokenizer.
        """
        return self._encodings is not None

    @property
    def encodings(self) -> list[EncodingFast] | None:
        """
        `Optional[list[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> list[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `list[str]`: The list of tokens at that index.
        """
        if not self._encodings:
            raise ValueError(
                "tokens() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index: int = 0) -> list[int | None]:
        """
        Return a list mapping the tokens to the id of their original sentences:

            - `None` for special tokens added around or between sequences,
            - `0` for tokens corresponding to words in the first sequence,
            - `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `list[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens added
            by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding
            sequence.
        """
        if not self._encodings:
            raise ValueError(
                "sequence_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].sequence_ids

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `list[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError(
                "word_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self, batch_or_token_index: int, token_index: int | None = None) -> int:
        """
        Get the index of the sequence represented by the given token. In the general use case, this method returns `0`
        for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

        Can be called as:

        - `self.token_to_sequence(token_index)` if batch size is 1
        - `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_sequence(token_index)

    def token_to_word(self, batch_or_token_index: int, token_index: int | None = None) -> int:
        """
        Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

        Can be called as:

        - `self.token_to_word(token_index)` if batch size is 1
        - `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(
        self, batch_or_word_index: int, word_index: int | None = None, sequence_index: int = 0
    ) -> TokenSpan | None:
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`~tokenization_utils_base.TokenSpan`] with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
        - `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
          1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            ([`~tokenization_utils_base.TokenSpan`], *optional*): Span of tokens in the encoded sequence. Returns
            `None` if no tokens correspond to the word. This can happen especially when the token is a special token
            that has been used to format the tokenization. For example when we add a class token at the very beginning
            of the tokenization.
        """

        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self, batch_or_token_index: int, token_index: int | None = None) -> CharSpan | None:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`~tokenization_utils_base.CharSpan`] with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - `self.token_to_chars(token_index)` if batch size is 1
        - `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in
                the sequence.

        Returns:
            [`~tokenization_utils_base.CharSpan`]: Span of characters in the original string, or None, if the token
            (e.g. <s>, </s>) doesn't correspond to any chars in the origin string.
        """

        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        span_indices = self._encodings[batch_index].token_to_chars(token_index)

        return CharSpan(*span_indices) if span_indices is not None else None

    def char_to_token(self, batch_or_char_index: int, char_index: int | None = None, sequence_index: int = 0) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - `self.char_to_token(char_index)` if batch size is 1
        - `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int`: Index of the token, or None if the char index refers to a whitespace only token and whitespace is
                   trimmed with `trim_offsets=True`.
        """

        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)

    def word_to_chars(
        self, batch_or_word_index: int, word_index: int | None = None, sequence_index: int = 0
    ) -> CharSpan:
        """
        Get the character span in the original string corresponding to given word in a sequence of the batch.

        Character spans are returned as a CharSpan NamedTuple with:

        - start: index of the first character in the original string
        - end: index of the character following the last character in the original string

        Can be called as:

        - `self.word_to_chars(word_index)` if batch size is 1
        - `self.word_to_chars(batch_index, word_index)` if batch size is greater or equal to 1

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            `CharSpan` or `list[CharSpan]`: Span(s) of the associated character or characters in the string. CharSpan
            are NamedTuple with:

                - start: index of the first character associated to the token in the original string
                - end: index of the character following the last character associated to the token in the original
                  string
        """

        if not self._encodings:
            raise ValueError("word_to_chars() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))

    def char_to_word(self, batch_or_char_index: int, char_index: int | None = None, sequence_index: int = 0) -> int:
        """
        Get the word in the original string corresponding to a character in the original string of a sequence of the
        batch.

        Can be called as:

        - `self.char_to_word(char_index)` if batch size is 1
        - `self.char_to_word(batch_index, char_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the character in the original string.
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the character in the
                original string.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int` or `list[int]`: Index or indices of the associated encoded token(s).
        """

        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index, sequence_index)

    def convert_to_tensors(self, tensor_type: str | TensorType | None = None, prepend_batch_axis: bool = False):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        if tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            def as_tensor(value, dtype=None):
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    return torch.from_numpy(np.array(value))
                if len(flatten(value)) == 0 and dtype is None:
                    dtype = torch.int64
                return torch.tensor(value, dtype=dtype)

            is_tensor = torch.is_tensor

        elif tensor_type == TensorType.MLX:
            if not is_mlx_available():
                raise ImportError("Unable to convert output to MLX tensors format, MLX is not installed.")
            import mlx.core as mx

            def as_tensor(value, dtype=None):
                if len(flatten(value)) == 0 and dtype is None:
                    dtype = mx.int32
                return mx.array(value, dtype=dtype)

            def is_tensor(obj):
                return isinstance(obj, mx.array)
        else:

            def as_tensor(value, dtype=None):
                if (
                    isinstance(value, (list, tuple))
                    and len(value) > 0
                    and isinstance(value[0], (list, tuple, np.ndarray))
                ):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                if len(flatten(value)) == 0 and dtype is None:
                    dtype = np.int64
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                    # # at-least2d
                    # if tensor.ndim > 2:
                    #     tensor = tensor.squeeze(0)
                    # elif tensor.ndim < 2:
                    #     tensor = tensor[None, :]

                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> BatchEncoding:
        """
        Send all values to device by calling `v.to(device, non_blocking=non_blocking)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
            non_blocking (`bool`): Whether to perform the copy asynchronously.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            self.data = {
                k: v.to(device=device, non_blocking=non_blocking) if hasattr(v, "to") and callable(v.to) else v
                for k, v in self.data.items()
            }
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self


ENCODE_KWARGS_DOCSTRING = r"""
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
                automatically.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence is provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing tokens returned when
                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
"""

ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (`bool`, *optional*):
                Whether to return token type IDs. If left to the default, will return the token type IDs according to
                the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are token type IDs?](../glossary#token-type-ids)
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
                of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
                of returning overflowing tokens.
            return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
                Whether or not to return special tokens mask information.
            return_offsets_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each token.

                This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
                Python's tokenizer, this method will raise `NotImplementedError`.
            return_length  (`bool`, *optional*, defaults to `False`):
                Whether or not to return the lengths of the encoded inputs.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
              if *"token_type_ids"* is in `self.model_input_names`).

              [What are token type IDs?](../glossary#token-type-ids)

            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
            - **length** -- The length of the inputs (when `return_length=True`)
"""


INIT_TOKENIZER_DOCSTRING = r"""
    Class attributes (overridden by derived classes)

        - **vocab_files_names** (`dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
          vocabulary file required by the model, and as associated values, the filename for saving the associated file
          (string).
        - **pretrained_vocab_files_map** (`dict[str, dict[str, str]]`) -- A dictionary of dictionaries, with the
          high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
          low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
          associated pretrained vocabulary file.
        - **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
          Should be `'right'` or `'left'`.
        - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
          applied. Should be `'right'` or `'left'`.

    Args:
        model_max_length (`int`, *optional*):
            The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
            loaded with [`~tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`], this will be set to the
            value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will
            default to VERY_LARGE_INTEGER (`int(1e30)`).
        padding_side (`str`, *optional*):
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        truncation_side (`str`, *optional*):
            The side on which the model should have truncation applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        chat_template (`str`, *optional*):
            A Jinja template string that will be used to format lists of chat messages. See
            https://huggingface.co/docs/transformers/chat_templating for a full description.
        model_input_names (`list[string]`, *optional*):
            The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
            `"attention_mask"`). Default value is picked from the class attribute of the same name.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the beginning of a sentence.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the end of a sentence.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.
        extra_special_tokens (list of `str` or `tokenizers.AddedToken`, *optional*):
            A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with
            `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end
            of the vocabulary.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. Passing will affect the
            internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
            `<s>` is the `bos_token`, then `tokenizer.tokenize("<s>") = ['<s>`]. Otherwise, if
            `split_special_tokens=True`, then `tokenizer.tokenize("<s>")` will be give `['<','s', '>']`.
"""


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerBase(PushToHubMixin):
    """
    Base class for all tokenizer backends.
    """

    vocab_files_names: dict[str, str] = {}
    pretrained_vocab_files_map: dict[str, dict[str, str]] = {}
    _auto_class: str | None = None

    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"
    slow_tokenizer_class = None

    # Special tokens support (moved from SpecialTokensMixin)
    # V5: Clean separation of named special tokens from extra special tokens
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ]

    def __init__(self, **kwargs):
        self.init_inputs = ()
        for key in kwargs:
            if hasattr(self, key) and callable(getattr(self, key)):
                raise AttributeError(f"{key} conflicts with the method {key} in {self.__class__.__name__}")

        # V5: Convert deprecated additional_special_tokens to extra_special_tokens before storing init_kwargs
        if "additional_special_tokens" in kwargs and "extra_special_tokens" not in kwargs:
            kwargs["extra_special_tokens"] = kwargs.pop("additional_special_tokens")

        self.init_kwargs = copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop("name_or_path", "")
        self._processor_class = kwargs.pop("processor_class", None)

        self._pad_token_type_id = 0
        self.verbose = kwargs.pop("verbose", False)

        # V5: Separate storage for named special tokens and extra special tokens
        self._special_tokens_map = dict.fromkeys(self.SPECIAL_TOKENS_ATTRIBUTES)
        self._extra_special_tokens = []  # List of extra model-specific special tokens

        # V5: track both explicit and auto-detected model-specific tokens
        explicit_model_specific_tokens = kwargs.pop("model_specific_special_tokens", None)
        if explicit_model_specific_tokens is None:
            explicit_model_specific_tokens = {}
        elif not isinstance(explicit_model_specific_tokens, dict):
            raise TypeError("model_specific_special_tokens must be a dictionary of token name to token value")
        auto_model_specific_tokens = {}

        # Directly set hidden values to allow init with tokens not yet in vocab
        for key in list(kwargs.keys()):
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                value = kwargs.pop(key)
                if value is None:
                    continue
                if isinstance(value, (str, AddedToken)):
                    self._special_tokens_map[key] = value
                else:
                    raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")
            elif key == "extra_special_tokens":
                value = kwargs.pop(key)
                if value is None:
                    continue
                if isinstance(value, dict):
                    self._set_model_specific_special_tokens(special_tokens=value)
                elif isinstance(value, (list, tuple)):
                    self._extra_special_tokens = list(value)
                else:
                    raise TypeError("extra_special_tokens must be a list/tuple of tokens or a dict of named tokens")
            elif (
                key.endswith("_token")
                and key not in self.SPECIAL_TOKENS_ATTRIBUTES
                and isinstance(kwargs[key], (str, AddedToken))
            ):
                value = kwargs.pop(key)
                if value is None:
                    continue
                auto_model_specific_tokens[key] = value

        # For backward compatibility we fallback to set model_max_length from max_len if provided
        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER

        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        self.truncation_side = kwargs.pop("truncation_side", self.truncation_side)
        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)

        # By default, clean up tokenization spaces for both fast and slow tokenizers
        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", False)

        # By default, do not split special tokens for both fast and slow tokenizers
        self.split_special_tokens = kwargs.pop("split_special_tokens", False)

        self._in_target_context_manager = False

        self.chat_template = kwargs.pop("chat_template", None)
        if isinstance(self.chat_template, (list, tuple)):
            # Chat templates are stored as lists of dicts with fixed key names,
            # we reconstruct that into a single dict while loading them.
            self.chat_template = {template["name"]: template["template"] for template in self.chat_template}

        model_specific_tokens = {**auto_model_specific_tokens, **explicit_model_specific_tokens}
        if model_specific_tokens:
            self._set_model_specific_special_tokens(special_tokens=model_specific_tokens)

        self.deprecation_warnings = {}

        # Backend information (V5: tracking which backend and files were used)
        self.backend = kwargs.pop("backend", None)
        self.files_loaded = kwargs.pop("files_loaded", [])

    def _set_processor_class(self, processor_class: str):
        """Sets processor class so it can be serialized in `tokenizer_config.json`."""
        self._processor_class = processor_class

    # ---- Special tokens API (moved from SpecialTokensMixin) ----
    def add_special_tokens(
        self,
        special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]],
        replace_extra_special_tokens=True,
    ) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
        model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens can be skipped when decoding using `skip_special_tokens = True`.
        - Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
        - You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        [`BertTokenizer`] `cls_token` is already registered to be `'[CLS]'` and XLM's one is also registered to be
        `'</s>'`).

        Args:
            special_tokens_dict (dictionary *str* to *str*, `tokenizers.AddedToken`, or `Sequence[Union[str, AddedToken]]`):
                Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
                `sep_token`, `pad_token`, `cls_token`, `mask_token`, `extra_special_tokens`].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the `unk_token` to them).
            replace_extra_special_tokens (`bool`, *optional*, defaults to `True`):
                If `True`, the existing list of extra special tokens will be replaced by the list provided in
                `special_tokens_dict`. Otherwise, `extra_special_tokens` will be extended. In the former
                case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged
                as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
                `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
                `extra_special_tokens` are still added tokens, and will not be split by the model.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to add a new classification token to GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        model = GPT2Model.from_pretrained("openai-community/gpt2")

        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"
        ```"""
        if not special_tokens_dict:
            return 0

        # V5: Allowed keys are SPECIAL_TOKENS_ATTRIBUTES + "extra_special_tokens"
        # Backward compatibility: convert "additional_special_tokens" to "extra_special_tokens"
        special_tokens_dict = dict(special_tokens_dict)
        if "additional_special_tokens" in special_tokens_dict:
            special_tokens_dict.setdefault(
                "extra_special_tokens", special_tokens_dict.pop("additional_special_tokens")
            )

        allowed_keys = set(self.SPECIAL_TOKENS_ATTRIBUTES) | {"extra_special_tokens"}
        tokens_to_add = []
        for key, value in special_tokens_dict.items():
            if key not in allowed_keys:
                raise ValueError(f"Key {key} is not a valid special token. Valid keys are: {allowed_keys}")

            if self.verbose:
                logger.info(f"Assigning {value} to the {key} key of the tokenizer")

            if key == "extra_special_tokens":
                if not isinstance(value, (list, tuple)) or not all(isinstance(t, (str, AddedToken)) for t in value):
                    raise ValueError(f"Tokens {value} for key {key} should all be str or AddedToken instances")
                new_tokens = [
                    (
                        AddedToken(t, rstrip=False, lstrip=False, normalized=False, special=True)
                        if isinstance(t, str)
                        else t
                    )
                    for t in value
                    if replace_extra_special_tokens or str(t) not in self.extra_special_tokens
                ]
                if replace_extra_special_tokens and new_tokens:
                    self._extra_special_tokens = list(new_tokens)
                else:
                    self._extra_special_tokens.extend(new_tokens)
                tokens_to_add.extend(new_tokens)
            else:
                if not isinstance(value, (str, AddedToken)):
                    raise ValueError(f"Token {value} for key {key} should be a str or an AddedToken instance")
                if isinstance(value, str):
                    value = AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)
                setattr(self, key, value)
                tokens_to_add.append(value)

        return self.add_tokens(tokens_to_add, special_tokens=True)

    def add_tokens(
        self, new_tokens: str | AddedToken | Sequence[str | AddedToken], special_tokens: bool = False
    ) -> int:
        """
        #TODO remove this from here! PreTrainedTOkeniuzerBase should be agnostic of AddedToken.

        Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and
        tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

        Args:
            new_tokens (`str`, `tokenizers.AddedToken` or a sequence of *str* or `tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
                token to let you personalize its behavior: whether this token should only match against a single word,
                whether this token should strip all potential whitespaces on the left side, whether this token should
                strip all potential whitespaces on the right side, etc.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Specifies if the token is special. This mostly changes the normalization behavior
                See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]
        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: list[str] | list[AddedToken], special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property
    def pad_token_type_id(self) -> int:
        return self._pad_token_type_id

    def __setattr__(self, key, value):
        # Handle _id/_ids suffix (eg. bos_token_id -> bos_token)
        key_without_id = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key

        # Named special tokens (bos_token, eos_token, etc.)
        if key_without_id in self.SPECIAL_TOKENS_ATTRIBUTES:
            if key != key_without_id and value is not None:
                value = self.convert_ids_to_tokens(value)
            if value is not None and not isinstance(value, (str, AddedToken)):
                raise ValueError(f"Cannot set a non-string value as the {key_without_id}")
            self._special_tokens_map[key_without_id] = value
            return

        # Extra special tokens: model-specific special tokens without standard names (eg. <mask_1>)
        if key_without_id == "extra_special_tokens":
            if key != key_without_id and value is not None and isinstance(value, (list, tuple)):
                value = [self.convert_ids_to_tokens(v) for v in value]
            if not isinstance(value, (list, tuple)) and value is not None:
                raise ValueError(f"extra_special_tokens must be a list or tuple, got {type(value)}")
            self._extra_special_tokens = [] if value is None else list(value)
            return

        super().__setattr__(key, value)

    def __getattr__(self, key):
        # Handle _id/_ids suffix (eg. bos_token_id -> bos_token)
        key_without_id = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key

        # Named special tokens (bos_token, eos_token, etc.)
        if key_without_id in self.SPECIAL_TOKENS_ATTRIBUTES:
            token_value = self._special_tokens_map.get(key_without_id)
            if token_value is None:
                if self.verbose:
                    logger.error(f"Using {key}, but it is not set yet.")
                return None
            return self.convert_tokens_to_ids(str(token_value)) if key != key_without_id else str(token_value)

        # Extra special tokens
        if key_without_id == "extra_special_tokens":
            tokens = [str(tok) for tok in self._extra_special_tokens]
            return self.convert_tokens_to_ids(tokens) if key != key_without_id else tokens

        if key not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
        return super().__getattr__(key)

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added.

        For fast tokenizers, data collators call this with `already_has_special_tokens=True` to build a mask over an
        already-formatted sequence. In that case, we compute the mask by checking membership in `all_special_ids`.

        Args:
            token_ids_0: List of IDs for the (possibly already formatted) sequence.
            token_ids_1: Unused when `already_has_special_tokens=True`. Must be None in that case.
            already_has_special_tokens: Whether the sequence is already formatted with special tokens.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of ids is already formatted "
                    "with special tokens for the model."
                )
            special_ids = set(self.all_special_ids)
            return [1 if int(tid) in special_ids else 0 for tid in token_ids_0]

        # Default base implementation for non-formatted sequences is not provided here.
        # Concrete tokenizer classes should override this for their specific formatting rules.
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_special_tokens_mask for non-formatted sequences"
        )

    @property
    def special_tokens_map(self) -> dict[str, str]:
        """
        `dict[str, str]`: A flat dictionary mapping named special token attributes to their string values.

        Only includes the standard named special tokens (bos_token, eos_token, etc.), not extra_special_tokens.
        This provides a clean, flat structure without mixed types.

        Returns:
            A dictionary with keys like 'bos_token', 'eos_token', etc., and string values.

        **V5 Change**: This now returns only named tokens. Use `extra_special_tokens` for the additional tokens.
        """
        return {
            attr: str(self._special_tokens_map[attr])
            for attr in self.SPECIAL_TOKENS_ATTRIBUTES
            if self._special_tokens_map.get(attr) is not None
        }

    # Note: extra_special_tokens and extra_special_tokens_ids are handled by __getattr__ and __setattr__
    # We don't define them as @property to keep the implementation simpler

    @property
    def all_special_tokens(self) -> list[str]:
        """
        `list[str]`: A list of all unique special tokens (named + extra) as strings.

        Includes both named special tokens (bos_token, eos_token, etc.) and extra special tokens.
        Converts tokens of `tokenizers.AddedToken` type to string.
        """
        seen = set()
        all_toks = []

        # Add named special tokens
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            value = self._special_tokens_map.get(attr)
            if value is not None:
                token_str = str(value)
                if token_str not in seen:
                    all_toks.append(token_str)
                    seen.add(token_str)

        # Add extra special tokens
        for token in self._extra_special_tokens:
            token_str = str(token)
            if token_str not in seen:
                all_toks.append(token_str)
                seen.add(token_str)

        return all_toks

    @property
    def all_special_ids(self) -> list[int]:
        """
        `list[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return self.convert_tokens_to_ids(self.all_special_tokens)

    def _set_model_specific_special_tokens(self, special_tokens: dict[str, str | AddedToken]):
        """
        Adds new model-specific special tokens (e.g., for multimodal models).

        These tokens are added to the named special tokens map and will be saved in tokenizer config.
        For example: if the model tokenizer is multimodal, we can support special image or audio tokens.

        Args:
            special_tokens: Dictionary of {token_name: token_value}
        """
        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
        for key, value in special_tokens.items():
            if isinstance(value, (str, AddedToken)):
                self._special_tokens_map[key] = value
            else:
                raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.added_tokens_decoder.items()])
        if added_tokens_decoder_rep:
            added_tokens_decoder_rep = f"\n\t{added_tokens_decoder_rep}\n"
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, model_max_length={self.model_max_length},"
            f" padding_side='{self.padding_side}', truncation_side='{self.truncation_side}',"
            f" special_tokens={self.special_tokens_map},"
            f" added_tokens_decoder={{{added_tokens_decoder_rep}}})"
        )

    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError()

    def get_vocab(self) -> dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `dict[str, int]`: The vocabulary.
        """
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `list[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `list[int]`: The token id or list of token ids.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `list[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `list[str]`: The decoded token(s).
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        trust_remote_code=False,
        **kwargs,
    ):
        r"""
        Instantiate a [`~tokenization_utils_base.PreTrainedTokenizerBase`] (or a derived class) from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  `./my_model_directory/vocab.txt`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
                exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`).
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only rely on local files and not to attempt to download any files.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__` method.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__` method. Can be used to set special tokens like `bos_token`,
                `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `extra_special_tokens`. See parameters in the `__init__` for more details.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        # We can't instantiate directly the base class *PreTrainedTokenizerBase* so let's show our examples on a derived class: BertTokenizer
        # Download vocabulary from huggingface.co and cache.
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

        # Download vocabulary from huggingface.co (user-uploaded) and cache.
        tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

        # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
        tokenizer = BertTokenizer.from_pretrained("./test/saved_model/")

        # If the tokenizer uses a single vocabulary file, you can point directly to this file
        tokenizer = BertTokenizer.from_pretrained("./test/saved_model/my_vocab.txt")

        # You can link tokens to special vocabulary when instantiating
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", unk_token="<unk>")
        # You should be sure '<unk>' is in the vocabulary when doing that.
        # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
        assert tokenizer.unk_token == "<unk>"
        ```"""
        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)
        gguf_file = kwargs.get("gguf_file")

        user_agent = {"file_type": "tokenizer", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        additional_files_names = {}
        init_configuration = {}

        is_local = os.path.isdir(pretrained_model_name_or_path)
        single_file_id = None
        if os.path.isfile(pretrained_model_name_or_path):
            # For legacy support: allow single-file loading if:
            # 1. Only one vocab file is required, OR
            # 2. It's a fast tokenizer with tokenizer_file (which is optional), OR
            # 3. It's a GGUF file
            vocab_files_count = len(cls.vocab_files_names)
            has_optional_tokenizer_file = vocab_files_count > 1 and "tokenizer_file" in cls.vocab_files_names

            if vocab_files_count > 1 and not gguf_file and not has_optional_tokenizer_file:
                raise ValueError(
                    f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is not "
                    "supported for this tokenizer. Use a model identifier or the path to a directory instead."
                )
            file_id = "vocab_file"
            if pretrained_model_name_or_path.endswith("tokenizer.json"):
                file_id = "tokenizer_file"
            vocab_files[file_id] = pretrained_model_name_or_path
            single_file_id = file_id
        else:
            if gguf_file:
                vocab_files["vocab_file"] = gguf_file
            else:
                # At this point pretrained_model_name_or_path is either a directory or a model identifier name
                additional_files_names = {
                    "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
                    "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
                    "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                    # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
                    "tokenizer_file": FULL_TOKENIZER_FILE,
                    "chat_template_file": CHAT_TEMPLATE_FILE,
                }

            vocab_files = {**cls.vocab_files_names, **additional_files_names}

            # Check for versioned tokenizer files
            if "tokenizer_file" in vocab_files:
                fast_tokenizer_file = FULL_TOKENIZER_FILE
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    TOKENIZER_CONFIG_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    _raise_exceptions_for_missing_entries=False,
                    _commit_hash=commit_hash,
                )
                if resolved_config_file is not None:
                    with open(resolved_config_file, encoding="utf-8") as reader:
                        tokenizer_config = json.load(reader)
                        if "fast_tokenizer_files" in tokenizer_config:
                            fast_tokenizer_file = get_fast_tokenizer_file(tokenizer_config["fast_tokenizer_files"])
                    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
                vocab_files["tokenizer_file"] = fast_tokenizer_file

            # This block looks for any extra chat template files
            if is_local:
                template_dir = Path(pretrained_model_name_or_path, CHAT_TEMPLATE_DIR)
                if template_dir.is_dir():
                    for template_file in template_dir.glob("*.jinja"):
                        template_name = template_file.name.removesuffix(".jinja")
                        vocab_files[f"chat_template_{template_name}"] = f"{CHAT_TEMPLATE_DIR}/{template_file.name}"
            else:
                for template in list_repo_templates(
                    pretrained_model_name_or_path,
                    local_files_only=local_files_only,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                ):
                    template = template.removesuffix(".jinja")
                    vocab_files[f"chat_template_{template}"] = f"{CHAT_TEMPLATE_DIR}/{template}.jinja"

        remote_files = []
        if not is_local and not local_files_only:
            try:
                remote_files = list_repo_files(pretrained_model_name_or_path)
            except Exception:
                remote_files = []
        elif pretrained_model_name_or_path and os.path.isdir(pretrained_model_name_or_path):
            remote_files = os.listdir(pretrained_model_name_or_path)

        if "tokenizer_file" in vocab_files and not re.search(vocab_files["tokenizer_file"], "".join(remote_files)):
            # mistral tokenizer names are different, but we can still convert them if
            # mistral common is not there
            other_pattern = r"tekken\.json|tokenizer\.model\.*"
            if match := re.search(other_pattern, "\n".join(remote_files)):
                vocab_files["vocab_file"] = match.group()

        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            elif single_file_id == file_id:
                if os.path.isfile(file_path):
                    resolved_vocab_files[file_id] = file_path
            else:
                try:
                    resolved_vocab_files[file_id] = cached_file(
                        pretrained_model_name_or_path,
                        file_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                        _commit_hash=commit_hash,
                    )
                except OSError:
                    # Re-raise any error raised by cached_file in order to get a helpful error message
                    raise
                except Exception:
                    # For any other exception, we throw a generic error.
                    raise OSError(
                        f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                        "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                        f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                        f"containing all relevant files for a {cls.__name__} tokenizer."
                    )
                commit_hash = extract_commit_hash(resolved_vocab_files[file_id], commit_hash)

        for file_id, file_path in vocab_files.items():
            if file_id not in resolved_vocab_files:
                continue

        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=commit_hash,
            _is_local=is_local,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        _is_local=False,
        trust_remote_code=False,
        **kwargs,
    ):
        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            # used in the past to check if the tokenizer class matches the class in the repo
            init_kwargs.pop("tokenizer_class", None)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration

        if resolved_vocab_files.get("tokenizer_file", None) is not None:
            init_kwargs.pop("add_bos_token", None)
            init_kwargs.pop("add_eos_token", None)

        # If independent chat template file(s) exist, they take priority over template entries in the tokenizer config
        chat_templates = {}
        chat_template_file = resolved_vocab_files.pop("chat_template_file", None)
        extra_chat_templates = [key for key in resolved_vocab_files if key.startswith("chat_template_")]
        if chat_template_file is not None:
            with open(chat_template_file, encoding="utf-8") as chat_template_handle:
                chat_templates["default"] = chat_template_handle.read()
        for extra_chat_template in extra_chat_templates:
            template_file = resolved_vocab_files.pop(extra_chat_template, None)
            if template_file is None:
                continue  # I think this should never happen, but just in case
            template_name = extra_chat_template.removeprefix("chat_template_")
            with open(template_file) as chat_template_handle:
                chat_templates[template_name] = chat_template_handle.read()
        if len(chat_templates) == 1 and "default" in chat_templates:
            init_kwargs["chat_template"] = chat_templates["default"]
        elif chat_templates:
            init_kwargs["chat_template"] = chat_templates

        if not _is_local:
            if "auto_map" in init_kwargs:
                # For backward compatibility with odl format.
                if isinstance(init_kwargs["auto_map"], (tuple, list)):
                    init_kwargs["auto_map"] = {"AutoTokenizer": init_kwargs["auto_map"]}

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # V5: Convert deprecated additional_special_tokens to extra_special_tokens
        if "additional_special_tokens" in init_kwargs:
            init_kwargs.setdefault("extra_special_tokens", init_kwargs.pop("additional_special_tokens"))

        # V5: Collect model-specific tokens (custom *_token keys not in standard attributes)
        default_attrs = set(cls.SPECIAL_TOKENS_ATTRIBUTES)
        model_specific_tokens = {
            key: init_kwargs.pop(key)
            for key in list(init_kwargs.keys())
            if key not in default_attrs and key.endswith("_token") and isinstance(init_kwargs[key], (str, AddedToken))
        }
        # If extra_special_tokens is a dict, merge it into model_specific_tokens
        if isinstance(init_kwargs.get("extra_special_tokens"), dict):
            model_specific_tokens.update(init_kwargs.pop("extra_special_tokens"))
        if model_specific_tokens:
            init_kwargs["model_specific_special_tokens"] = model_specific_tokens

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs or init_kwargs[args_name] is None:
                init_kwargs[args_name] = file_path
        tokenizer_file = resolved_vocab_files.get("tokenizer_file", None)

        init_kwargs["name_or_path"] = pretrained_model_name_or_path
        init_kwargs["is_local"] = _is_local

        #### Handle tokenizer serialization of added and special tokens
        added_tokens_decoder: dict[int, AddedToken] = {}
        added_tokens_map: dict[str, AddedToken] = {}
        # if we have info on the slow added tokens
        if "added_tokens_decoder" in init_kwargs:
            for idx, token in init_kwargs["added_tokens_decoder"].items():
                if isinstance(token, dict):
                    token = AddedToken(**token)
                if isinstance(token, AddedToken):
                    added_tokens_decoder[int(idx)] = token
                    added_tokens_map[str(token)] = token
                else:
                    raise TypeError(
                        f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                    )
        else:
            # Legacy: read special_tokens_map.json and merge into init_kwargs
            if special_tokens_map_file is not None:
                with open(special_tokens_map_file, encoding="utf-8") as f:
                    special_tokens_map = json.load(f)
                for key, value in special_tokens_map.items():
                    if key in kwargs and kwargs[key]:
                        continue  # User-provided kwargs take precedence
                    if isinstance(value, dict) and key != "extra_special_tokens":
                        value = AddedToken(**value, special=True)
                    elif key == "extra_special_tokens" and isinstance(value, list):
                        # Merge list tokens, converting dicts to AddedToken
                        existing = list(init_kwargs.get("extra_special_tokens") or [])
                        for tok in value:
                            tok = AddedToken(**tok, special=True) if isinstance(tok, dict) else tok
                            if tok not in existing:
                                existing.append(tok)
                        value = existing
                    init_kwargs[key] = value
                # Convert dict extra_special_tokens to model_specific_special_tokens
                if isinstance(init_kwargs.get("extra_special_tokens"), dict):
                    init_kwargs.setdefault("model_specific_special_tokens", {}).update(
                        init_kwargs.pop("extra_special_tokens")
                    )

            # slow -> slow|fast, legacy: convert the `"added_tokens.json"` file to `added_tokens_decoder`.
            # this is for legacy purpose. We don't add the tokens after init for efficiency.
            if added_tokens_file is not None:
                # V5: Check both named and extra special tokens
                special_tokens = {str(init_kwargs[k]) for k in cls.SPECIAL_TOKENS_ATTRIBUTES if init_kwargs.get(k)}
                special_tokens.update(str(t) for t in (init_kwargs.get("extra_special_tokens") or []))

                with open(added_tokens_file, encoding="utf-8") as f:
                    added_tok_encoder = json.load(f)
                for str_token, index in added_tok_encoder.items():
                    is_special = str_token in special_tokens
                    added_tokens_decoder[index] = AddedToken(
                        str_token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special
                    )
                    added_tokens_map[str_token] = added_tokens_decoder[index]

            # allows converting a fast -> slow: add the `tokenizer.json`'s `"added_tokens"` to the slow tokenizer
            # if `tokenizer_config.json` is `None`
            if tokenizer_file is not None:
                # This is for slow so can be done before
                with open(tokenizer_file, encoding="utf-8") as tokenizer_file_handle:
                    tokenizer_file_handle = json.load(tokenizer_file_handle)
                    added_tokens = tokenizer_file_handle.pop("added_tokens")
                for serialized_tokens in added_tokens:
                    idx = serialized_tokens.pop("id")
                    added_tokens_decoder[idx] = AddedToken(**serialized_tokens)
                    added_tokens_map[str(added_tokens_decoder[idx])] = added_tokens_decoder[idx]
            # end legacy

        # Passing AddedTokens and not strings to the class to prevent it from casting the string to a different AddedToken
        # convert {'__type': 'AddedToken', 'content': '<ent>', 'lstrip': False, 'normalized': True, ...} to AddedTokens
        init_kwargs["added_tokens_decoder"] = added_tokens_decoder
        init_kwargs = cls.convert_added_tokens(init_kwargs, save=False)
        # V5: Map special tokens from added_tokens_map (named tokens only)
        for key in cls.SPECIAL_TOKENS_ATTRIBUTES:
            if key in init_kwargs and added_tokens_map != {} and init_kwargs[key] is not None:
                init_kwargs[key] = added_tokens_map.get(str(init_kwargs[key]), init_kwargs[key])

        # From pretrained with the legacy fixes
        # for `tokenizers` based tokenizer, we actually want to have vocab and merges pre-extracted from whatever inputs
        # for `none` (PythonBackend) based tokenizer, we also want the vocab file / merge files not extracted.
        # for `sentencepiece` based tokenizer, we pass the sentencepiece model file directly.
        init_kwargs = cls.convert_to_native_format(**init_kwargs)

        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except import_protobuf_decode_error():
            raise RuntimeError(
                "Unable to load tokenizer model from SPM, loading from TikToken will be attempted instead."
                "(Google protobuf error: Tried to load SPM model with non-SPM vocab file).",
            )
        except RuntimeError as e:
            if "sentencepiece_processor.cc" in str(e):
                raise RuntimeError(
                    "Unable to load tokenizer model from SPM, loading from TikToken will be attempted instead."
                    "(SentencePiece RuntimeError: Tried to load SPM model with non-SPM vocab file).",
                ) from e
            else:
                raise e
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )
        return tokenizer

    @classmethod
    def convert_to_native_format(cls, **kwargs):
        return kwargs

    @classmethod
    def convert_added_tokens(cls, obj: AddedToken | Any, save=False, add_type_field=True):
        if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
            obj.pop("__type")
            return AddedToken(**obj)
        if isinstance(obj, AddedToken) and save:
            obj = obj.__getstate__()
            if add_type_field:
                obj["__type"] = "AddedToken"
            else:
                # Don't save "special" for previous tokenizers
                obj.pop("special")
            return obj
        elif isinstance(obj, (list, tuple)):
            return [cls.convert_added_tokens(o, save=save, add_type_field=add_type_field) for o in obj]
        elif isinstance(obj, dict):
            return {k: cls.convert_added_tokens(v, save=save, add_type_field=add_type_field) for k, v in obj.items()}
        return obj

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> tuple[str, ...]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] class method..

        Warning,None This won't save modifications you may have applied to the tokenizer after the instantiation (for
        instance, modifying `tokenizer.do_lower_case` after creation).

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            legacy_format (`bool`, *optional*):
                Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
                format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
                added_tokens files.

                If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with
                "slow" tokenizers (not powered by the *tokenizers* library), so the tokenizer will not be able to be
                loaded in the corresponding "slow" tokenizer.

                If `True`, will save the tokenizer in legacy format. If the "slow" tokenizer doesn't exits, a value
                error is raised.
            filename_prefix (`str`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.

        Returns:
            A tuple of `str`: The files saved.
        """

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)

        tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        tokenizer_config.pop("add_bos_token", None)
        tokenizer_config.pop("add_eos_token", None)

        # Let's save the init kwargs
        target_keys = set(self.init_kwargs.keys())
        target_keys.discard("add_bos_token")
        target_keys.discard("add_eos_token")
        # Let's save the special tokens map (only the strings)
        target_keys.update(["model_max_length"])

        for k in target_keys:
            if hasattr(self, k):
                tokenizer_config[k] = getattr(self, k)

        # Let's make sure we properly save the special tokens
        # V5: Save both named tokens and extra tokens
        tokenizer_config.update(self.special_tokens_map)
        if self._extra_special_tokens:
            tokenizer_config["extra_special_tokens"] = self.extra_special_tokens

        save_jinja_files = kwargs.get("save_jinja_files", True)
        tokenizer_config, saved_raw_chat_template_files = self.save_chat_templates(
            save_directory, tokenizer_config, filename_prefix, save_jinja_files
        )

        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names:
            tokenizer_config.pop(file_id, None)

        # no typefields, this way old fast and slow can load it
        tokenizer_config = self.convert_added_tokens(tokenizer_config, add_type_field=True, save=True)
        # Process added tokens separately: allows previous versions to ignore it!
        added_tokens = {}
        for key, value in self.added_tokens_decoder.items():
            added_tokens[key] = value.__getstate__()
        tokenizer_config["added_tokens_decoder"] = added_tokens

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__

        # tokenizers backend don't need to save added_tokens_decoder and additional_special_tokens
        if any(base.__name__ == "TokenizersBackend" for base in self.__class__.__mro__):
            tokenizer_config.pop("added_tokens_decoder", None)
            tokenizer_config.pop("additional_special_tokens", None)

        # Remove the Fast at the end if we can save the slow tokenizer
        if tokenizer_class.endswith("Fast") and getattr(self, "can_save_slow_tokenizer", False):
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class
        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map
        if getattr(self, "_processor_class", None) is not None:
            tokenizer_config["processor_class"] = self._processor_class
        tokenizer_config.pop("files_loaded", None)
        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=tokenizer_config)

        # remove private information
        if "name_or_path" in tokenizer_config:
            tokenizer_config.pop("name_or_path")
            tokenizer_config.pop("special_tokens_map_file", None)
            tokenizer_config.pop("tokenizer_file", None)
        if "device_map" in tokenizer_config:
            tokenizer_config.pop("device_map")
        if "slow_tokenizer_class" in tokenizer_config:
            tokenizer_config.pop("slow_tokenizer_class")

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        logger.info(f"tokenizer config file saved in {tokenizer_config_file}")

        # Sanitize AddedTokens in special_tokens_map

        file_names = (tokenizer_config_file, *saved_raw_chat_template_files)

        save_files = self._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return save_files

    def _save_pretrained(
        self,
        save_directory: str | os.PathLike,
        file_names: tuple[str, ...],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific [`~tokenization_utils_tokenizers.PreTrainedTokenizerFast._save_pretrained`]
        """
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        save_directory = str(save_directory)

        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        # the new get_added_vocab() also returns special tokens and tokens that have an index < vocab_size
        added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                f.write(out_str)
                logger.info(f"added tokens file saved in {added_tokens_file}")

        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        return file_names + vocab_files + (added_tokens_file,)

    def clean_up_tokenization(self, text: str) -> str:
        """
        Clean up tokenization spaces in a given text.
        This method is mostly for remote code support.

        """

        text = (
            text.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `tuple(str)`: Paths to the files saved.
        """
        raise NotImplementedError

    def tokenize(self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs) -> list[str]:
        """
        Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            pair (`str`, *optional*):
                A second sequence to be encoded with the first.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method. See details in
                [`~PreTrainedTokenizerBase.__call__`]

        Returns:
            `list[str]`: The list of tokens.
        """
        raise NotImplementedError

    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Passed along to the `.tokenize()` method.
        """,
        """
        Returns:
            `list[int]`, `torch.Tensor`, or `np.ndarray`: The tokenized ids of the text.
        """,
    )
    def encode(
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> list[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `list[str]` or `list[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `list[str]` or `list[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        padding_strategy, truncation_strategy, max_length, kwargs_updated = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

        kwargs.update(kwargs_updated)

        encoded_inputs = self._encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        raise NotImplementedError

    @property
    def max_len_single_sentence(self) -> int:
        """
        `int`: The maximum length of a sentence that can be fed to the model.
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> None:
        # For backward compatibility, allow to try to setup 'max_len_single_sentence'.
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
            if not self.deprecation_warnings.get("max_len_single_sentence", False):
                logger.warning(
                    "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
                )
            self.deprecation_warnings["max_len_single_sentence"] = True
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
            )

    @property
    def max_len_sentences_pair(self) -> int:
        """
        `int`: The maximum combined length of a pair of sentences that can be fed to the model.
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> None:
        # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
            if not self.deprecation_warnings.get("max_len_sentences_pair", False):
                logger.warning(
                    "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
                )
            self.deprecation_warnings["max_len_sentences_pair"] = True
        else:
            raise ValueError("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy
        """

        # Backward compatibility for previous behavior:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            truncation = "longest_first"

        # Get padding strategy
        if padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.pad_token is None or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_target (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair_target (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            tokenizer_kwargs (`dict[str, Any]`, *optional*):
                Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
                other kwargs, with explicit parameters taking precedence.
        """
        # To avoid duplicating
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "padding_side": padding_side,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "split_special_tokens": kwargs.pop("split_special_tokens", self.split_special_tokens),
            "verbose": verbose,
        }

        max_target_length = kwargs.pop("max_target_length", None)

        # First merge tokenizer_kwargs, then other kwargs (explicit params take precedence)
        if tokenizer_kwargs is not None:
            all_kwargs.update(tokenizer_kwargs)
        all_kwargs.update(kwargs)
        if text is None and text_target is None:
            raise ValueError("You need to specify either `text` or `text_target`.")

        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=all_kwargs.pop("padding", False),
            truncation=all_kwargs.pop("truncation", None),
            max_length=all_kwargs.pop("max_length", None),
            pad_to_multiple_of=all_kwargs.get("pad_to_multiple_of"),
            verbose=all_kwargs.get("verbose", True),
            **kwargs,
        )

        if text is not None:
            # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
            # input mode in this case.
            if not self._in_target_context_manager and hasattr(self, "_switch_to_input_mode"):
                self._switch_to_input_mode()
            encodings = self._encode_plus(
                text=text,
                text_pair=text_pair,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                **all_kwargs,
            )
        if text_target is not None:
            if hasattr(self, "_switch_to_target_mode"):
                self._switch_to_target_mode()
            target_encodings = self._encode_plus(
                text=text_target,
                text_pair=text_pair_target,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_target_length if max_target_length is not None else max_length,
                **all_kwargs,
            )
            # Leave back tokenizer in input mode
            if hasattr(self, "_switch_to_input_mode"):
                self._switch_to_input_mode()

        if text_target is None:
            return encodings
        elif text is None:
            return target_encodings
        else:
            encodings["labels"] = target_encodings["input_ids"]
            return encodings

    def _encode_plus(
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs: BatchEncoding
        | list[BatchEncoding]
        | dict[str, EncodedInput]
        | dict[str, list[EncodedInput]]
        | list[dict[str, EncodedInput]],
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, or PyTorch tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `dict[str, list[int]]`, `dict[str, list[list[int]]` or `list[dict[str, list[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `dict[str, list[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *dict[str, list[list[int]]]* or *list[dict[str,
                list[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `list[int]` you can have tensors (numpy arrays, or PyTorch tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if (
            isinstance(encoded_inputs, (list, tuple))
            and len(encoded_inputs) > 0
            and isinstance(encoded_inputs[0], Mapping)
        ):
            # Call .keys() explicitly for compatibility with TensorDict and other Mapping subclasses
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has been passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, or pytorch object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(len(v) == batch_size for v in encoded_inputs.values()), (
            "Some items in the output dictionary have a different batch size than others."
        )

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: dict[str, EncodedInput] | BatchEncoding,
        max_length: int | None = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_attention_mask: bool | None = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`list[int]`) or batch of tokenized inputs (`list[list[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in `padding_side` argument:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side

            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError(f"Invalid padding strategy:{padding_side}")

        return encoded_inputs

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`list[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        raise NotImplementedError

    def decode(
        self,
        token_ids: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str | list[str]:
        """
        Converts a sequence of ids into a string, or a list of sequences into a list of strings,
        using the tokenizer and vocabulary with options to remove special tokens and clean up
        tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]`):
                A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the
                `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `Union[str, list[str]]`: The decoded string for a single sequence, or a list of decoded strings for a
            batch of sequences.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        # If we received batched input, decode each sequence
        if isinstance(token_ids, (list, tuple)) and len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
            clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", False)
            return [
                self._decode(
                    token_ids=seq,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    **kwargs,
                )
                for seq in token_ids
            ]

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        This method is provided for backwards compatibility. The `decode` method now handles batched input natively,
        so you can use `decode` directly instead of `batch_decode`.

        Args:
            sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `list[str]`: The list of decoded sentences.
        """
        # Forward to decode() which now handles batched input natively
        result = self.decode(
            token_ids=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        # Ensure we always return a list for backwards compatibility
        if isinstance(result, str):
            return [result]
        return result

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def _eventual_warn_about_too_long_sequence(self, ids: list[int], max_length: int | None, verbose: bool):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`list[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose and self.model_max_length != 0:
            if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors"
                )
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        """
        Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
        library are already mapped with `AutoTokenizer`.

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`):
                The auto class to register this new tokenizer with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = True,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
        determine the format and control tokens to use when converting.

        Args:
            conversation (Union[list[dict[str, str]], list[list[dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (`list[Union[Dict, Callable]]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [tool use guide](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools)
                for more information.
            documents (`list[dict[str, str]]`, *optional*):
                A list of dicts representing documents that will be accessible to the model if it is performing RAG
                (retrieval-augmented generation). If the template does not support RAG, this argument will have no
                effect. We recommend that each document should be a dict containing "title" and "text" keys.
            chat_template (`str`, *optional*):
                A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
                argument, as the model's template will be used by default.
            add_generation_prompt (bool, *optional*):
                If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final
                message in the chat is open-ended, without any EOS tokens. The model will continue this message
                rather than starting a new one. This allows you to "prefill" part of
                the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
            return_dict (`bool`, defaults to `True`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            tokenizer_kwargs (`dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            return_assistant_tokens_mask (`bool`, defaults to `False`):
                Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
                the mask will contain 1. For user and system tokens, the mask will contain 0.
                This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `Union[list[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
            set, will return a dict of tokenizer outputs instead.
        """

        if not tokenize:
            return_dict = False  # dicts are only returned by the tokenizer anyway

        if return_assistant_tokens_mask and not (return_dict and tokenize):
            raise ValueError("`return_assistant_tokens_mask=True` requires `return_dict=True` and `tokenize=True`")

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        chat_template = self.get_chat_template(chat_template, tools)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        template_kwargs = {**self.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present
        rendered_chat, generation_indices = render_jinja_template(
            conversations=conversations,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        if not is_batched:
            rendered_chat = rendered_chat[0]

        if tokenize:
            out = self(
                rendered_chat,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
                **tokenizer_kwargs,
            )
            if return_dict:
                if return_assistant_tokens_mask:
                    assistant_masks = []
                    if is_batched or return_tensors:
                        input_ids = out["input_ids"]
                    else:
                        input_ids = [out["input_ids"]]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break
                            for token_id in range(start_token, end_token + 1 if end_token else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)

                    if not is_batched and not return_tensors:
                        assistant_masks = assistant_masks[0]

                    out["assistant_masks"] = assistant_masks

                    if return_tensors:
                        out.convert_to_tensors(tensor_type=return_tensors)

                return out
            else:
                return out["input_ids"]
        else:
            return rendered_chat

    def encode_message_with_chat_template(
        self,
        message: dict[str, str],
        conversation_history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> list[int]:
        """
        Tokenize a single message. This method is a convenience wrapper around `apply_chat_template` that allows you
        to tokenize messages one by one. This is useful for things like token-by-token streaming.
        This method is not guaranteed to be perfect. For some models, it may be impossible to robustly tokenize
        single messages. For example, if the chat template adds tokens after each message, but also has a prefix that
        is added to the entire chat, it will be impossible to distinguish a chat-start-token from a message-start-token.
        In these cases, this method will do its best to find the correct tokenization, but it may not be perfect.
        **Note:** This method does not support `add_generation_prompt`. If you want to add a generation prompt,
        you should do it separately after tokenizing the conversation.
        Args:
            message (`dict`):
                A dictionary with "role" and "content" keys, representing the message to tokenize.
            conversation_history (`list[dict]`, *optional*):
                A list of dicts with "role" and "content" keys, representing the chat history so far. If you are
                tokenizing messages one by one, you should pass the previous messages in the conversation here.
            **kwargs:
                Additional kwargs to pass to the `apply_chat_template` method.
        Returns:
            `list[int]`: A list of token ids representing the tokenized message.
        """
        if "add_generation_prompt" in kwargs:
            raise ValueError(
                "`encode_message_with_chat_template` does not support `add_generation_prompt`. Please add the generation prompt "
                "separately."
            )

        if conversation_history is None or len(conversation_history) == 0:
            return self.apply_chat_template(
                [message], add_generation_prompt=False, tokenize=True, return_dict=False, **kwargs
            )

        conversation = conversation_history + [message]
        tokens = self.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=True, return_dict=False, **kwargs
        )

        prefix_tokens = self.apply_chat_template(
            conversation_history, add_generation_prompt=False, tokenize=True, return_dict=False, **kwargs
        )
        # It's possible that the prefix tokens are not a prefix of the full list of tokens.
        # For example, if the prefix is `<s>User: Hi` and the full conversation is `<s>User: Hi</s><s>Assistant: Hello`.
        # In this case, we can't simply find the prefix, so we have to do something a bit more subtle.
        # We look for the first place where the tokens differ, and that's our split point.
        # This is not perfect, but it's the best we can do without a token-level API.
        # To make this more robust, we could do a diff and find the longest common subsequence, but this is
        # a good first approximation.
        # This is particularly important for models like Llama3 that have changed their chat template to include
        # EOS tokens after user messages.
        min_len = min(len(prefix_tokens), len(tokens))
        for i in range(min_len):
            if prefix_tokens[i] != tokens[i]:
                return tokens[i:]
        return tokens[min_len:]

    def get_chat_template(self, chat_template: str | None = None, tools: list[dict] | None = None) -> str:
        """
        Retrieve the chat template string used for tokenizing chat messages. This template is used
        internally by the `apply_chat_template` method and can also be used externally to retrieve the model's chat
        template for better generation tracking.

        Args:
            chat_template (`str`, *optional*):
                A Jinja template or the name of a template to use for this conversion.
                It is usually not necessary to pass anything to this argument,
                as the model's template will be used by default.
            tools (`list[Dict]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.

        Returns:
            `str`: The chat template string.
        """
        # First, handle the cases when the model has a dict of multiple templates
        if isinstance(self.chat_template, dict):
            template_dict = self.chat_template
            if chat_template is not None and chat_template in template_dict:
                # The user can pass the name of a template to the chat template argument instead of an entire template
                chat_template = template_dict[chat_template]
            elif chat_template is None:
                if tools is not None and "tool_use" in template_dict:
                    chat_template = template_dict["tool_use"]
                elif "default" in template_dict:
                    chat_template = template_dict["default"]
                else:
                    raise ValueError(
                        "This model has multiple chat templates with no default specified! Please either pass a chat "
                        "template or the name of the template you wish to use to the `chat_template` argument. Available "
                        f"template names are {sorted(template_dict.keys())}."
                    )

        elif chat_template is None:
            # These are the cases when the model has a single template
            # priority: `chat_template` argument > `tokenizer.chat_template`
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use chat template functions because tokenizer.chat_template is not set and no template "
                    "argument was passed! For information about writing templates and setting the "
                    "tokenizer.chat_template attribute, please see the documentation at "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating"
                )

        return chat_template

    def save_chat_templates(
        self,
        save_directory: str | os.PathLike,
        tokenizer_config: dict,
        filename_prefix: str | None,
        save_jinja_files: bool,
    ):
        """
        Writes chat templates out to the save directory if we're using the new format, and removes them from
        the tokenizer config if present. If we're using the legacy format, it doesn't write any files, and instead
        writes the templates to the tokenizer config in the correct format.
        """
        chat_template_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_FILE
        )
        chat_template_dir = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_DIR
        )

        saved_raw_chat_template_files = []
        if save_jinja_files and isinstance(self.chat_template, str):
            # New format for single templates is to save them as chat_template.jinja
            with open(chat_template_file, "w", encoding="utf-8") as f:
                f.write(self.chat_template)
            logger.info(f"chat template saved in {chat_template_file}")
            saved_raw_chat_template_files.append(chat_template_file)
            if "chat_template" in tokenizer_config:
                tokenizer_config.pop("chat_template")  # To ensure it doesn't somehow end up in the config too
        elif save_jinja_files and isinstance(self.chat_template, dict):
            # New format for multiple templates is to save the default as chat_template.jinja
            # and the other templates in the chat_templates/ directory
            for template_name, template in self.chat_template.items():
                if template_name == "default":
                    with open(chat_template_file, "w", encoding="utf-8") as f:
                        f.write(self.chat_template["default"])
                    logger.info(f"chat template saved in {chat_template_file}")
                    saved_raw_chat_template_files.append(chat_template_file)
                else:
                    Path(chat_template_dir).mkdir(exist_ok=True)
                    template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")
                    with open(template_filepath, "w", encoding="utf-8") as f:
                        f.write(template)
                    logger.info(f"chat template saved in {template_filepath}")
                    saved_raw_chat_template_files.append(template_filepath)
            if "chat_template" in tokenizer_config:
                tokenizer_config.pop("chat_template")  # To ensure it doesn't somehow end up in the config too
        elif isinstance(self.chat_template, dict):
            # Legacy format for multiple templates:
            # chat template dicts are saved to the config as lists of dicts with fixed key names.
            tokenizer_config["chat_template"] = [{"name": k, "template": v} for k, v in self.chat_template.items()]
        elif self.chat_template is not None:
            # Legacy format for single templates: Just make them a key in tokenizer_config.json
            tokenizer_config["chat_template"] = self.chat_template
        return tokenizer_config, saved_raw_chat_template_files

    def parse_response(
        self,
        response: str | list[str | int | list[int]] | np.ndarray | torch.Tensor,
        schema: list | dict | None = None,
    ):
        """
        Converts an output string created by generating text from a model into a parsed message dictionary.
        This method is intended for use with chat models, and will read the tokenizer's `response_schema` attribute to
        control parsing, although this can be overridden by passing a `response_schema` argument directly.

        This method is currently **highly experimental** and the schema specification is likely to change in future!
        We recommend not building production code on top of it just yet.

        Args:
            response (`str`):
                The output string generated by the model. This can be either a decoded string or list of strings,
                or token IDs as a list/array.
            schema (`Union[list, dict]`, *optional*):
                A response schema that indicates the expected output format and how parsing should be performed.
                If not provided, the tokenizer's `response_schema` attribute will be used.
        """
        batched = (
            (isinstance(response, list) and not isinstance(response[0], int))
            or getattr(response, "ndim", 0) > 1  # For torch/numpy tensors
        )

        if schema is None:
            if getattr(self, "response_schema", None) is None:
                raise AttributeError("This tokenizer does not have a `response_schema` for parsing chat responses!")
            schema = self.response_schema
        if batched:
            if not (isinstance(response, list) and isinstance(response[0], str)):
                response = self.batch_decode(response)
            return [recursive_parse(single_response, schema) for single_response in response]
        else:
            if not isinstance(response, str):
                response = self.decode(response)
            return recursive_parse(response, schema)


def get_fast_tokenizer_file(tokenization_files: list[str]) -> str:
    """
    Get the tokenization file to use for this version of transformers.

    Args:
        tokenization_files (`list[str]`): The list of available configuration files.

    Returns:
        `str`: The tokenization file to use.
    """
    tokenizer_files_map = {}
    for file_name in tokenization_files:
        search = _re_tokenizer_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            tokenizer_files_map[v] = file_name
    available_versions = sorted(tokenizer_files_map.keys())

    # Defaults to FULL_TOKENIZER_FILE and then try to look at some newer versions.
    tokenizer_file = FULL_TOKENIZER_FILE
    transformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            tokenizer_file = tokenizer_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return tokenizer_file


# Shared helper to locate a SentencePiece model file for a repo/path
def find_sentencepiece_model_file(pretrained_model_name_or_path, **kwargs):
    """
    Find any .model file (SentencePiece model) in the model directory or Hub repo.

    Tries known filenames first ("tokenizer.model", "spm.model"), then scans local dir,
    and as a last resort lists files on the Hub to find any .model.

    Returns the filename (str) relative to the repo root or directory if found, else None.
    """
    from .utils.hub import has_file

    # Try common names first
    for candidate in ("tokenizer.model", "spm.model"):
        try:
            if has_file(
                pretrained_model_name_or_path,
                candidate,
                revision=kwargs.get("revision"),
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
            ):
                return candidate
        except Exception:
            # TODO: tighten to OSError / ProxyError
            continue

    subfolder = kwargs.get("subfolder", "")
    local_files_only = kwargs.get("local_files_only", False)

    # Local directory scan
    if os.path.isdir(pretrained_model_name_or_path):
        dir_path = (
            os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
        )
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".model"):
                    return filename if not subfolder else os.path.join(subfolder, filename)

    # Hub listing if allowed
    if not local_files_only:
        try:
            from huggingface_hub import list_repo_tree

            entries = list_repo_tree(
                repo_id=pretrained_model_name_or_path,
                revision=kwargs.get("revision"),
                path_in_repo=subfolder if subfolder else None,
                recursive=False,
                token=kwargs.get("token"),
            )
            for entry in entries:
                if entry.path.endswith(".model"):
                    return entry.path if not subfolder else entry.path.removeprefix(f"{subfolder}/")
        except Exception as e:
            # TODO: tighten exception class
            logger.debug(f"Could not list Hub repository files: {e}")

    return None


def load_vocab_and_merges(pretrained_model_name_or_path, **kwargs):
    """
    Resolve and load tokenizer vocabulary files from a repo/path.

    Priority order:
    1. Load ``vocab.json`` (WordLevel/WordPiece/BPE fast tokenizers)
    2. Load ``vocab.txt`` when only a WordPiece vocab is available
    3. Optionally load ``merges.txt`` (BPE tokenizers)

    Returns:
        tuple (vocab: dict|None, merges: list[tuple[str,str]]|None, files_loaded: list[str])
    """
    files_loaded = []
    vocab = None
    merges = None
    try:
        resolved_vocab_file = cached_file(
            pretrained_model_name_or_path,
            "vocab.json",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
            subfolder=kwargs.get("subfolder", ""),
        )
    except Exception:
        resolved_vocab_file = None

    if resolved_vocab_file is not None:
        try:
            with open(resolved_vocab_file, "r", encoding="utf-8") as vf:
                vocab = json.load(vf)
            files_loaded.append("vocab.json")
        except Exception:
            vocab = None

    # Fallback to vocab.txt (WordPiece-style vocabularies)
    if vocab is None:
        try:
            resolved_vocab_txt = cached_file(
                pretrained_model_name_or_path,
                "vocab.txt",
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
                subfolder=kwargs.get("subfolder", ""),
            )
        except Exception:
            resolved_vocab_txt = None

        if resolved_vocab_txt is not None:
            try:
                vocab = OrderedDict()
                with open(resolved_vocab_txt, "r", encoding="utf-8") as vf:
                    for index, token in enumerate(vf):
                        token = token.rstrip("\n")
                        vocab[token] = index
                files_loaded.append("vocab.txt")
            except Exception:
                vocab = None

    try:
        resolved_merges_file = cached_file(
            pretrained_model_name_or_path,
            "merges.txt",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
            subfolder=kwargs.get("subfolder", ""),
        )
    except Exception:
        resolved_merges_file = None

    if resolved_merges_file is not None:
        try:
            merges = []
            with open(resolved_merges_file, "r", encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) == 2:
                            merges.append((parts[0], parts[1]))
            files_loaded.append("merges.txt")
        except Exception:
            merges = None

    return vocab, merges, files_loaded


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
PreTrainedTokenizerBase.push_to_hub = copy_func(PreTrainedTokenizerBase.push_to_hub)
if PreTrainedTokenizerBase.push_to_hub.__doc__ is not None:
    PreTrainedTokenizerBase.push_to_hub.__doc__ = PreTrainedTokenizerBase.push_to_hub.__doc__.format(
        object="tokenizer", object_class="AutoTokenizer", object_files="tokenizer files"
    )


def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


def generate_merges(vocab, vocab_scores: dict[str, float] | None = None, skip_tokens: Collection[str] | None = None):
    skip_tokens = set(skip_tokens) if skip_tokens is not None else set()
    reverse = vocab_scores is not None
    vocab_scores = dict(vocab_scores) if reverse else vocab

    merges = []
    for merge, piece_score in vocab_scores.items():
        if merge in skip_tokens:
            continue
        local = []
        for index in range(1, len(merge)):
            piece_l, piece_r = merge[:index], merge[index:]
            if piece_l in skip_tokens or piece_r in skip_tokens:
                continue
            if piece_l in vocab and piece_r in vocab:
                local.append((piece_l, piece_r, piece_score))
        local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)

    merges = sorted(merges, key=lambda val: (val[2], len(val[0]), len(val[1])), reverse=reverse)
    merges = [(val[0], val[1]) for val in merges]
    return merges
