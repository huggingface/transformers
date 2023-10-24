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
"""Tokenization classes for RWKV."""

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
    "vocab_file": "rwkv_vocab_v20230424.json",
}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "RWKV/rwkv-5-world-169m": "https://huggingface.co/RWKV/rwkv-5-world-169m/blob/main/rwkv_vocab_v20230424.json",
    },
}


class DATrie:
    class Node:
        def __init__(self, is_leaf=False, leaf_data=None, tail=""):
            self._is_leaf = is_leaf
            self._leaf_data = leaf_data
            self._tail = tail
            self._next_map = {}

        def is_leaf(self):
            return self._is_leaf

        def set_leaf(self):
            self._is_leaf = True

        def has_next(self, w):
            if w in self._next_map:
                return True
            return False

        def add_node(self, w, node):
            self._next_map[w] = node

        def get_node(self, w):
            if w in self._next_map:
                return self._next_map[w]
            return None

        def get_tail(self):
            return self._tail

        def get_data(self):
            return self._leaf_data

        def set_data(self, data):
            self._leaf_data = data

    def __init__(self, special_ids):
        self.root = self.Node()
        self.data = {}
        self.r_data = {}
        self.special_ids = special_ids

    def insert(self, word, data):
        self.data[word] = data
        self.r_data[data] = word
        idx = 0
        node = self.root
        while idx < len(word):
            w = word[idx]
            is_leaf = idx == (len(word) - 1)
            leaf_data = data if is_leaf else None
            # 不存在则插入
            if not node.has_next(w):
                node.add_node(w, self.Node(is_leaf=is_leaf, leaf_data=leaf_data))
                # last word
            node = node.get_node(w)
            idx += 1
        if not node.is_leaf():
            node.set_leaf()
            node.set_data(data)

    def findStrict(self, word):
        idx = 0
        node = self.root
        while node is not None and idx < len(word):
            w = word[idx]
            if not node.has_next(w):
                return None
                # last word
            node = node.get_node(w)
            idx += 1
        if node.is_leaf():
            return node.get_data()
        return None

    def prefix(self, word):
        idx = 0
        node = self.root
        result = []
        while node is not None and idx < len(word):
            w = word[idx]
            if not node.has_next(w):
                return result
                # last word
            node = node.get_node(w)
            if node.is_leaf():
                result.append([word[: idx + 1], node.get_data()])
            idx += 1
        return result

    def max_prefix(self, content, start_idx):
        idx = start_idx
        node = self.root
        l = len(content)
        result = [
            [
                "",
            ],
        ]
        while node is not None and idx < l:
            w = content[idx]
            if not node.has_next(w):
                return result[-1]
                # last word
            node = node.get_node(w)
            if node.is_leaf():
                result.append([content[start_idx : idx + 1], node.get_data()])
            idx += 1
        return result[-1]

    def max_score(self, content, start_idx):
        idx = start_idx
        node = self.root
        l = len(content)
        result = [
            ["", (3, 0)],
        ]
        while node is not None and idx < l:
            w = content[idx]
            if not node.has_next(w):
                break
                # last word
            node = node.get_node(w)
            if node.is_leaf():
                result.append([content[start_idx : idx + 1], node.get_data()])
            idx += 1
        if len(result) > 1:
            result = sorted(result, key=lambda x: x[1][1])
        return result[-1]

    def match(self, content, add_unk=True, unk_id=-1, **kwargs):
        # length
        l = len(content)
        i = 0
        result_list = []
        while i < l:
            match_word = self.max_prefix(content=content, start_idx=i)
            # print(match_word)
            w = match_word[0]
            if len(w) > 0:
                result_list.append(match_word[1])
                i += len(w)
            else:
                if add_unk:
                    result_list.append(unk_id)
                i += 1
        return result_list

    def id2str(self, ids, escape_special_ids=True, end_ids=[], **kwargs):
        res_str = ""
        for rid in ids:
            if rid in self.r_data:
                if rid in end_ids:
                    break
                if escape_special_ids and rid in self.special_ids:
                    continue
                rstr = self.r_data[rid]
                res_str += rstr
            elif rid == 0:
                break
            else:
                print("ERROR unknown id %d" % rid)
                res_str += "UNK"
        return res_str

    def id2str_v2(self, ids, escape_special_ids=True, end_ids=[], **kwargs):
        res_str = ""
        for rid in ids:
            if rid in self.r_data:
                if rid in end_ids:
                    break
                rstr = self.r_data[rid]
                if escape_special_ids and rid in self.special_ids:
                    continue
                res_str += rstr
            elif rid == 0:
                break
            else:
                print("ERROR unknown id %d" % rid)
                res_str += "UNK"
        return res_str


class RWKVWorldTokenizer(PreTrainedTokenizer):
    """
    Construct a RWKVWorldTokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import RWKVWorldTokenizer

    >>> tokenizer = RWKVWorldTokenizer.from_pretrained("RWKV/rwkv-5-world-169m")
    >>> tokenizer("Hello world")["input_ids"]
    [33155, 40213]

    >>> tokenizer(" Hello world")["input_ids"]
    [36786, 40213]
    ```

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, errors="replace", **kwargs):
        self.add_bos_token = False
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.trie = DATrie(self.all_special_ids)
        for k, v in self.encoder.items():
            self.trie.insert(k, v)
        self.errors = errors  # how to handle errors in decoding
        self.cache = {}

        super().__init__(
            errors=errors,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

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

    def _tokenize(self, text, **kwargs):
        """Tokenize a string."""
        return self.trie.match(text, unk_id=self.unk_token_id, **kwargs)

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)
        if isinstance(token_ids, int):
            if token_ids in self.all_special_ids and skip_special_tokens:
                return ""
            return self.decoder.get(token_ids, self.unk_token)
        elif isinstance(token_ids, list):
            return self.trie.id2str(token_ids, escape_special_ids=skip_special_tokens, **kwargs)
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
        def get_input_ids(text):
            if isinstance(text, str):
                text_id = self.trie.match(text, unk_id=self.unk_token_id)
                return text_id
            elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
                return [self.trie.match(t, unk_id=self.unk_token_id) for t in text]
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
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
        def get_input_ids(text):
            if isinstance(text, str):
                text_id = self.trie.match(text, unk_id=self.unk_token_id)
                return text_id
            elif isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
                return [self.trie.match(t, unk_id=self.unk_token_id) for t in text]
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
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

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
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

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids
