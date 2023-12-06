# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RWKV5."""

import json
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType, logging, to_py_obj


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "rwkv_vocab_v20230424.txt",
}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "RWKV/rwkv-5-world-169m": "https://huggingface.co/RWKV/rwkv-5-world-169m/blob/main/rwkv_vocab_v20230424.txt",
    },
}


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class RWKVWorldTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, errors="replace", pad_token="0", **kwargs):
        self.add_bos_token = False
        self.encoder = {}
        sorted = []  # must be already sorted
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.encoder[idx] = x

        self.decoder = {}
        for k, v in self.encoder.items():
            self.decoder[v] = int(k)

        self.trie = TRIE()
        for t, i in self.decoder.items():
            _ = self.trie.add(t, val=(t, i))
        self.errors = errors  # how to handle errors in decoding
        self.cache = {}
        self.first_max_length = 0
        super().__init__(
            errors=errors,
            **kwargs,
        )
    
    @property
    def eos_token_id(self) -> Optional[int]:
        return 0
    
    @property
    def eot_token_id(self) -> Optional[int]:
        return 0

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def add_tokens(self, new_tokens, special_tokens: bool = False):
        for token in new_tokens:
            token_id = self.convert_tokens_to_ids(token)
            self.added_tokens_decoder[token_id] = token

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        tokens = []
        for id_ in ids:
            if id_ in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[id_])
            else:
                tokens.append(self._convert_id_to_token(id_))
        return tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.trie.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.encoder[i], tokens)) # noqa

    def _tokenize(self, text, **kwargs):
        """Tokenize a string."""
        return self.encodeBytes(text.encode("utf-8"))

    def _decode_tokens(self, tokens):
        try:
            return self.decodeBytes(tokens).decode("utf-8")
        except Exception:
            return "\ufffd"  # bad utf-8

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        def remove_zeros_from_first_segment(token_ids, first_max_length):
            first_segment = token_ids[:first_max_length]
            first_segment_cleaned = [token for token in first_segment if token != 0]
            return first_segment_cleaned + token_ids[first_max_length:]

        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)
        token_ids = remove_zeros_from_first_segment(token_ids, self.first_max_length)
        if isinstance(token_ids, int):
            if token_ids in self.all_special_ids and skip_special_tokens:
                return ""
            return self.encoder.get(token_ids, self.unk_token)
        elif isinstance(token_ids, list):
            self.first_max_length
            out_str = ""
            out_last = 0
            out_tokens = []
            for i, token in enumerate(token_ids):
                if token == 0:
                    break
                out_tokens += [token]
                tmp = self._decode_tokens(out_tokens[out_last:])
                if "\ufffd" not in tmp:
                    out_str += tmp
                    out_last = i + 1
            return out_str
        else:
            return token_ids

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)

    def prepare_for_tokenization(self, text, **kwargs):
        return (text, kwargs)

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        return PaddingStrategy.LONGEST, TruncationStrategy.DO_NOT_TRUNCATE, -1, kwargs

    def _encode_plus(
        self,
        text: Union[TextInput, EncodedInput],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text, max_length=None, pad_token_id=0):
            def pad_sequence(seq, max_len, pad_tok):
                return [pad_tok] * (max_len - len(seq)) + seq

            if isinstance(text, str):
                tokens = self._tokenize(text)
                if max_length is not None:
                    tokens = pad_sequence(tokens, max_length, pad_token_id)
                return tokens

            elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
                tokenized_texts = [self._tokenize(t) for t in text]
                if max_length is None:
                    max_length = max(len(t) for t in tokenized_texts)
                return [pad_sequence(t, max_length, pad_token_id) for t in tokenized_texts]

            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                if max_length is not None and len(text) < max_length:
                    return pad_sequence(text, max_length, pad_token_id)
                return text

            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids = get_input_ids(text)

        return self.prepare_for_model(
            first_ids,
            pair_ids=None,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[EncodedInput],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text, max_length=None, pad_token_id=0):
            def pad_sequence(seq, max_len, pad_tok):
                return [pad_tok] * (max_len - len(seq)) + seq

            if isinstance(text, str):
                tokens = self._tokenize(text)
                if max_length is not None:
                    tokens = pad_sequence(tokens, max_length, pad_token_id)
                return tokens

            elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
                tokenized_texts = [self._tokenize(t) for t in text]
                if max_length is None:
                    max_length = max(len(t) for t in tokenized_texts)
                return [pad_sequence(t, max_length, pad_token_id) for t in tokenized_texts]

            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                if max_length is not None and len(text) < max_length:
                    return pad_sequence(text, max_length, pad_token_id)
                return text

            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        first_max_length = 0
        second_max_length = 0
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids
            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            first_max_length = max(first_max_length, len(first_ids))
            if second_ids is not None:
                second_max_length = max(second_max_length, len(second_ids))

        self.first_max_length = first_max_length
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids, max_length=first_max_length)
            second_ids = get_input_ids(pair_ids, max_length=second_max_length) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids
