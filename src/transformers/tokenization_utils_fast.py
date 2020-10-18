# coding=utf-8
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
""" Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library).
    For slow (python) tokenizers see tokenization_utils.py
"""

import copy
import json
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast

from .convert_slow_tokenizer import convert_slow_tokenizer
from .file_utils import add_end_docstrings
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from .utils import logging


logger = logging.get_logger(__name__)


# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Slow tokenizers have an additional addedd tokens files
ADDED_TOKENS_FILE = "added_tokens.json"


@add_end_docstrings(
    INIT_TOKENIZER_DOCSTRING,
    """
    .. automethod:: __call__
    """,
)
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase`.

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).
    """

    slow_tokenizer_class: PreTrainedTokenizer = None

    def __init__(self, *args, **kwargs):
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)

        if fast_tokenizer_file is not None:
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
                "Couldn't instantiate the backend tokenizer from one of: "
                "(1) a `tokenizers` library serialization file, "
                "(2) a slow tokenizer instance to convert or "
                "(3) an equivalent slow tokenizer class to instantiate and convert. "
                "You need to have sentencepiece installed to convert a slow tokenizer to a fast one."
            )

        self._tokenizer = fast_tokenizer

        if slow_tokenizer is not None:
            kwargs = copy.deepcopy(slow_tokenizer.init_kwargs)

        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        """
        :obj:`int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        :obj:`tokenizer.get_vocab()[token]` is equivalent to :obj:`tokenizer.convert_tokens_to_ids(token)` when
        :obj:`token` is in the vocab.

        Returns:
            :obj:`Dict[str, int]`: The vocabulary.
        """
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            :obj:`Dict[str, int]`: The added tokens.
        """
        base_vocab = self._tokenizer.get_vocab(with_added_tokens=False)
        full_vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        added_vocab = dict((tok, index) for tok, index in full_vocab.items() if tok not in base_vocab)
        return added_vocab

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        :obj:`tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        :obj:`tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        return self._tokenizer._tokenizer.decoder

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
    ) -> Dict[str, Any]:
        """Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict.

        Overflowing tokens are converted to additional examples (like batches) so the output values of
        the dict are lists (overflows) of lists (tokens).

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

        return encoding_dict

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            token (:obj:`str` or :obj:`List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            :obj:`int` or :obj:`List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        .. note::
            This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not
            put this inside your training loop.

        Args:
            pair (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            :obj:`int`: Number of special tokens added to sequences.
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary
        and added tokens.

        Args:
            ids (:obj:`int` or :obj:`List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            :obj:`str` or :obj:`List[str]`: The decoded token(s).
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

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the backend Rust tokenizer.

        Note that, unlike slow tokenizers (instances of :class:`~transformers.PreTrainedTokenizer`), this method
        will replace the unknown tokens with the :obj:`unk_token`.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            pair (:obj:`str`, `optional`):
                A second sequence to be encoded with the first.
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to add the special tokens associated with the corresponding model.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """
        return self._tokenizer.encode(text, pair, add_special_tokens=add_special_tokens).tokens

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
            padding_strategy (:class:`~transformers.tokenization_utils_base.PaddingStrategy`):
                The kind of padding that will be applied to the input
            truncation_strategy (:class:`~transformers.tokenization_utils_base.TruncationStrategy`):
                The kind of truncation that will be applied to the input
            max_length (:obj:`int`):
                The maximum size of a sequence.
            stride (:obj:`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        """
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            self._tokenizer.enable_truncation(max_length, stride=stride, strategy=truncation_strategy.value)
        else:
            self._tokenizer.no_truncation()

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            self._tokenizer.enable_padding(
                length=max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None,
                direction=self.padding_side,
                pad_id=self.pad_token_id,
                pad_type_id=self.pad_token_type_id,
                pad_token=self.pad_token,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        else:
            self._tokenizer.no_padding()

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
        **kwargs
    ) -> BatchEncoding:

        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(
                "batch_text_or_text_pairs has to be a list (got {})".format(type(batch_text_or_text_pairs))
            )

        if "is_pretokenized" in kwargs:
            warnings.warn(
                "`is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.",
                FutureWarning,
            )
            is_split_into_words = kwargs.pop("is_pretokenized")

        if kwargs:
            raise ValueError(f"Keyword arguments {kwargs} not recognized.")

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        # Avoid thread overhead if only one example.
        if len(batch_text_or_text_pairs) == 1:
            if isinstance(batch_text_or_text_pairs[0], tuple):
                # We got a Tuple with a pair of sequences
                encodings = self._tokenizer.encode(
                    *batch_text_or_text_pairs[0],
                    add_special_tokens=add_special_tokens,
                    is_pretokenized=is_split_into_words,
                )
            else:
                # We got a single sequence
                encodings = self._tokenizer.encode(
                    batch_text_or_text_pairs[0],
                    add_special_tokens=add_special_tokens,
                    is_pretokenized=is_split_into_words,
                )
            encodings = [encodings]
        else:
            encodings = self._tokenizer.encode_batch(
                batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                is_pretokenized=is_split_into_words,
            )

        # Convert encoding to dict
        # `Tokens` has type: List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens = [
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

        # Convert the output to have dict[list] from list[dict]
        sanitized = {}
        for key in tokens[0].keys():
            # To List[List[List[int]]] of shape (batch, overflows, sequence length)
            stack = [e for item in tokens for e in item[key]]
            sanitized[key] = stack

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, enc in enumerate(tokens):
                overflow_to_sample_mapping += [i] * len(enc["input_ids"])
            sanitized["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        return BatchEncoding(sanitized, encodings, tensor_type=return_tensors)

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
        **kwargs
    ) -> BatchEncoding:
        if "is_pretokenized" in kwargs:
            warnings.warn(
                "`is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.",
                FutureWarning,
            )
            is_split_into_words = kwargs.pop("is_pretokenized")

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
        # Overfolwing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.

        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids (:obj:`Union[int, List[int]]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.

        Returns:
            :obj:`str`: The decoded sentence.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def _save_pretrained(
        self,
        save_directory: str,
        file_names: Tuple[str],
        legacy_format: bool = True,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens}
        using the specific :meth:`~transformers.PreTrainedTokenizerFast._save_pretrained`
        """
        if legacy_format:
            added_tokens_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            added_vocab = self.get_added_vocab()
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, ensure_ascii=False)
                    f.write(out_str)

            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)
        else:
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        return file_names
