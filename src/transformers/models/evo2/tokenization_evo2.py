"""Tokenizer for the Evo2 model."""

from __future__ import annotations

import json
import os
from typing import List, Optional

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

__all__ = ["Evo2Tokenizer"]


def _clamp_token_id(token_id: int) -> int:
    return max(0, min(255, int(token_id)))


class Evo2Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs) -> None:
        self._vocab_size = 256
        self._token_to_id = {chr(i): i for i in range(self._vocab_size)}
        self._id_to_token = {i: chr(i) for i in range(self._vocab_size)}
        self._eos_token_id = 0
        self._pad_token_id = 1
        self._bos_token_id = None
        super().__init__(
            bos_token=None,
            eos_token=chr(0),
            pad_token=chr(1),
            unk_token=None,
            add_bos_token=False,
            add_eos_token=False,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._token_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text: str, **kwargs) -> List[int]:
        del kwargs
        return list(text.encode("utf-8"))

    def _tokenize(self, text: str) -> List[str]:
        return [str(byte) for byte in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str | int) -> int:
        if isinstance(token, int):
            return _clamp_token_id(token)
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        if token in self._token_to_id:
            return self._token_to_id[token]
        return _clamp_token_id(int(token))

    def _convert_id_to_token(self, index: int) -> str:
        index = _clamp_token_id(index)
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index]
        return self._id_to_token[index]

    def convert_tokens_to_string(self, tokens: List[str | int]) -> str:
        byte_values = []
        for token in tokens:
            if isinstance(token, str) and token in self.added_tokens_encoder:
                token = self.added_tokens_encoder[token]
            token_id = _clamp_token_id(int(token))
            byte_values.append(token_id)
        return "".join(chr(byte) for byte in byte_values)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        length = len(token_ids_0) if token_ids_1 is None else len(token_ids_0) + len(token_ids_1)
        return [0] * length

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        file_name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({str(i): i for i in range(self._vocab_size)}, f, ensure_ascii=False, indent=2)
        return (path,)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        del clean_up_tokenization_spaces, spaces_between_special_tokens
        if skip_special_tokens:
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id not in {self.pad_token_id, self.eos_token_id}
            ]
        return "".join(chr(_clamp_token_id(token_id)) for token_id in token_ids)
