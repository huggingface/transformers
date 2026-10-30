# Copyright 2026 The HuggingFace Inc. team.
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
Tokenization backend for tokenizers>=1.
"""

import os

from tokenizers import Padding, Tokenizer, Truncation

from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from .utils import PaddingStrategy


TOKENIZER_FILE = "tokenizer.json"


class TokenizersV1Backend(PreTrainedTokenizerBase):
    """
    Tokenizer backed by tokenizers>=1.

    It holds the editable `tokenizers.Tokenizer` spec, loaded from a `tokenizer.json`, and the `CompiledTokenizer`
    built from it. Edits, such as adding tokens, go to the spec, which is then compiled again. tokenizers 1.x doesn't
    expose the pipeline components that [`TokenizersBackend`] and the model-specific tokenizer classes are built on, so
    the `tokenizer.json` is used as is.

    Not supported yet: offsets, overflowing tokens and stride, special tokens masks, pretokenized inputs, training.
    """

    vocab_files_names = {"tokenizer_file": TOKENIZER_FILE}

    def __init__(self, tokenizer_file: str, **kwargs):
        self._spec = Tokenizer.from_file(tokenizer_file)
        self._tokenizer = self._spec.compile()

        if (padding := self._spec.padding) is not None:
            kwargs.setdefault("pad_token", padding.pad_token)
            kwargs.setdefault("padding_side", padding.direction)

        super().__init__(**kwargs)

        special_tokens = [*self._special_tokens_map.values(), *self._extra_special_tokens]
        self.add_tokens(
            [
                *kwargs.get("added_tokens_decoder", {}).values(),
                *(AddedToken(t, special=True) if isinstance(t, str) else t for t in special_tokens if t is not None),
            ]
        )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        return self._tokenizer.added_tokens

    def get_added_vocab(self) -> dict[str, int]:
        return {token.content: index for index, token in self.added_tokens_decoder.items()}

    def _add_tokens(self, new_tokens: list[str | AddedToken], special_tokens: bool = False) -> int:
        added_vocab = self.get_added_vocab()
        new_tokens = [token for token in new_tokens if str(token) not in added_vocab]
        if not new_tokens:
            return 0
        if special_tokens:
            added = self._spec.add_special_tokens(new_tokens)
        else:
            added = self._spec.add_tokens(new_tokens)
        self._tokenizer = self._spec.compile()
        return added

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        return self._tokenizer.decode_tokens([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._tokenizer.decode(self.convert_tokens_to_ids(tokens), skip_special_tokens=False)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return len(self._tokenizer.encode(("", "") if pair else "", padding=None, truncation=None))

    def tokenize(self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs) -> list[str]:
        return self._tokenizer.tokenize(
            text if pair is None else (text, pair),
            add_special_tokens=add_special_tokens,
            encode_special_tokens=kwargs.get("split_special_tokens", self.split_special_tokens),
            padding=None,
            truncation=None,
        )

    def _encode_plus(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: bool | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool | None = None,
        **kwargs,
    ) -> BatchEncoding:
        unsupported = {
            "is_split_into_words": is_split_into_words,
            "stride": stride,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
        }
        if requested := [name for name, value in unsupported.items() if value]:
            raise NotImplementedError(f"{', '.join(requested)} is not supported with tokenizers>=1 yet.")

        is_batched = isinstance(text, (list, tuple))
        if is_batched:
            inputs = list(text) if text_pair is None else list(zip(text, text_pair))
        else:
            inputs = [(text, text_pair) if text_pair else text]

        padding = None
        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            padding = Padding(
                direction=padding_side or self.padding_side,
                pad_id=self.pad_token_id,
                pad_type_id=self.pad_token_type_id,
                pad_token=self.pad_token,
                length=max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        truncation = None
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            truncation = Truncation(max_length, strategy=truncation_strategy.value, direction=self.truncation_side)

        encodings = self._tokenizer.encode_batch(
            inputs,
            add_special_tokens=add_special_tokens,
            encode_special_tokens=self.split_special_tokens if split_special_tokens is None else split_special_tokens,
            padding=padding,
            truncation=truncation,
        )

        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        data = {"input_ids": [encoding.ids for encoding in encodings]}
        if return_token_type_ids:
            data["token_type_ids"] = [encoding.type_ids for encoding in encodings]
        if return_attention_mask:
            data["attention_mask"] = [encoding.attention_mask for encoding in encodings]
        if return_length:
            data["length"] = [len(encoding) for encoding in encodings]

        for input_ids in data["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)

        if not is_batched and return_tensors is None:
            data = {key: value[0] if isinstance(value[0], list) else value for key, value in data.items()}
        return BatchEncoding(data, tensor_type=return_tensors)

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces
        # Same rule as `TokenizersBackend`: the cleanup strips legitimate spaces from BPE output
        if clean_up_tokenization_spaces and (
            type(self._spec.model).__name__ != "BPE"
            or self.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output
        ):
            text = self.clean_up_tokenization(text)
        return text

    def _save_pretrained(
        self,
        save_directory: str | os.PathLike,
        file_names: tuple[str, ...],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        tokenizer_file = os.path.join(
            str(save_directory), (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
        )
        self._spec.save(tokenizer_file)
        return file_names + (tokenizer_file,)
