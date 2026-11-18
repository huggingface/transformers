# Copyright 2025 Arc Institute and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenizer for the Evo2 model."""

from __future__ import annotations

from ...tokenization_python import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


def _clamp_token_id(token_id: int) -> int:
    return max(0, min(255, int(token_id)))


class Evo2Tokenizer(PreTrainedTokenizer):
    """
    Construct an Evo 2 character/byte-level tokenizer (UTF-8 byte encoding with a 256-entry vocabulary).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        **kwargs,
    ) -> None:
        self._vocab_size = 256
        self._token_to_id = {chr(i): i for i in range(self._vocab_size)}
        self._id_to_token = {i: chr(i) for i in range(self._vocab_size)}
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        kwargs.setdefault("bos_token", None)
        kwargs.setdefault("eos_token", chr(0))
        kwargs.setdefault("pad_token", chr(1))
        kwargs.setdefault("unk_token", chr(0))
        super().__init__(
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._token_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return [chr(byte) for byte in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str | int) -> int:
        if isinstance(token, int):
            return _clamp_token_id(token)
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        if token in self._token_to_id:
            return self._token_to_id[token]
        if token.lstrip("-").isdigit():
            return _clamp_token_id(int(token))
        return self.unk_token_id if self.unk_token_id is not None else 0

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.added_tokens_decoder:
            return str(self.added_tokens_decoder[index])
        index = _clamp_token_id(index)
        return self._id_to_token[index]

    def convert_tokens_to_string(self, tokens: list[str | int]) -> str:
        bstring = bytearray()
        for token in tokens:
            if isinstance(token, int):
                bstring.append(_clamp_token_id(token))
            elif token in self.added_tokens_encoder:
                bstring.extend(token.encode("utf-8"))
            elif len(token) == 1 and ord(token) < 256:
                bstring.append(ord(token))
            elif token.lstrip("-").isdigit():
                bstring.append(_clamp_token_id(int(token)))
            else:
                bstring.extend(token.encode("utf-8"))
        try:
            return bstring.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(chr(b) for b in bstring)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        length = len(token_ids_0) if token_ids_1 is None else len(token_ids_0) + len(token_ids_1)
        return [0] * length

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        return ()

    def _decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        del clean_up_tokenization_spaces, spaces_between_special_tokens, kwargs
        special_ids = set(self.all_special_ids) if skip_special_tokens else set()
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.added_tokens_decoder:
                tokens.append(str(self.added_tokens_decoder[token_id]))
            else:
                tokens.append(chr(_clamp_token_id(token_id)))
        return self.convert_tokens_to_string(tokens)


__all__ = ["Evo2Tokenizer"]
