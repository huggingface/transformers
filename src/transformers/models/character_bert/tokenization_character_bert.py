# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Tokenization classes for CharacterBERT."""

from __future__ import annotations

from collections.abc import Mapping

from ...tokenization_python import AddedToken, PreTrainedTokenizer
from ...utils import PaddingStrategy, logging
from ..bert.tokenization_bert_legacy import BasicTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {}


class _CharacterMapper:
    beginning_of_sentence_character = 256
    end_of_sentence_character = 257
    beginning_of_word_character = 258
    end_of_word_character = 259
    padding_character = 260
    mask_character = 261

    def __init__(
        self,
        max_word_length: int,
        bos_token: str,
        eos_token: str,
        pad_token: str,
        mask_token: str,
    ):
        self.max_word_length = max_word_length
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.beginning_of_sentence_characters = self._make_bos_eos(self.beginning_of_sentence_character)
        self.end_of_sentence_characters = self._make_bos_eos(self.end_of_sentence_character)
        self.mask_characters = self._make_bos_eos(self.mask_character)
        self.pad_characters = [-1] * self.max_word_length

    def _make_bos_eos(self, character: int) -> list[int]:
        char_ids = [self.padding_character] * self.max_word_length
        char_ids[0] = self.beginning_of_word_character
        char_ids[1] = character
        char_ids[2] = self.end_of_word_character
        return char_ids

    def convert_word_to_char_ids(self, word: str) -> list[int]:
        if word == self.bos_token:
            char_ids = self.beginning_of_sentence_characters
        elif word == self.eos_token:
            char_ids = self.end_of_sentence_characters
        elif word == self.mask_token:
            char_ids = self.mask_characters
        elif word == self.pad_token:
            char_ids = self.pad_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[: self.max_word_length - 2]
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            for index, byte in enumerate(word_encoded, start=1):
                char_ids[index] = byte
            char_ids[len(word_encoded) + 1] = self.end_of_word_character

        # +1 offset to keep 0 as padding value.
        return [char_id + 1 for char_id in char_ids]

    def convert_char_ids_to_word(self, char_ids: list[int]) -> str:
        if len(char_ids) != self.max_word_length:
            raise ValueError(f"Invalid character token length {len(char_ids)}. Expected {self.max_word_length}.")

        if char_ids == self.convert_word_to_char_ids(self.bos_token):
            return self.bos_token
        if char_ids == self.convert_word_to_char_ids(self.eos_token):
            return self.eos_token
        if char_ids == self.convert_word_to_char_ids(self.mask_token):
            return self.mask_token
        if char_ids == self.convert_word_to_char_ids(self.pad_token):
            return self.pad_token

        restored = [char_id - 1 for char_id in char_ids]
        if not restored:
            return ""

        token_bytes = []
        for char_id in restored[1:]:
            if char_id in (self.end_of_word_character, self.padding_character):
                break
            if 0 <= char_id <= 255:
                token_bytes.append(char_id)

        if len(token_bytes) == 0:
            return ""
        return bytes(token_bytes).decode("utf-8", "ignore")


class CharacterBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a CharacterBERT tokenizer.

    This tokenizer performs BERT-style basic tokenization at word level and converts each token into a fixed-length
    sequence of character IDs that CharacterBERT consumes.

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase input text.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
        strip_accents (`bool`, *optional*):
            Whether or not to strip accents.
        max_characters_per_token (`int`, *optional*, defaults to 50):
            Maximum number of characters represented for each token.
        model_max_length (`int`, *optional*, defaults to 512):
            Maximum supported sequence length in tokens.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            Unknown token used when a token cannot be represented.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            Separator token used when building sequence pairs.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            Token used for padding.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            Classifier token added to the start of sequences.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            Token used for masked language modeling.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        do_lower_case: bool = True,
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        max_characters_per_token: int = 50,
        model_max_length: int = 512,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        if max_characters_per_token < 8:
            raise ValueError("`max_characters_per_token` must be at least 8.")

        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.max_characters_per_token = max_characters_per_token

        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )

        self._special_token_ids = {
            str(pad_token): 0,
            str(unk_token): 1,
            str(cls_token): 2,
            str(sep_token): 3,
            str(mask_token): 4,
        }
        self._id_to_special_token = {value: key for key, value in self._special_token_ids.items()}

        self._character_mapper = _CharacterMapper(
            max_word_length=max_characters_per_token,
            bos_token=str(cls_token),
            eos_token=str(sep_token),
            pad_token=str(pad_token),
            mask_token=str(mask_token),
        )

        self._pad_character_ids = self._character_mapper.convert_word_to_char_ids(str(pad_token))
        self._cls_character_ids = self._character_mapper.convert_word_to_char_ids(str(cls_token))
        self._sep_character_ids = self._character_mapper.convert_word_to_char_ids(str(sep_token))

        super().__init__(
            do_lower_case=do_lower_case,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            max_characters_per_token=max_characters_per_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            token_type_ids_pattern="bert_style",
            token_type_ids_include_special_tokens=True,
            special_tokens_pattern="none",
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._special_token_ids)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._special_token_ids)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens)

    def _convert_token_to_id(self, token: str) -> int | list[int]:
        if token in self._special_token_ids:
            return self._special_token_ids[token]
        return self._character_mapper.convert_word_to_char_ids(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_special_token.get(index, str(self.unk_token))

    def _resolve_placeholder_token(self, token_id: int) -> str | None:
        token = self._id_to_special_token.get(token_id)
        if token is not None:
            return token

        added_token = self.added_tokens_decoder.get(token_id)
        if added_token is not None:
            return added_token.content

        return None

    def _normalize_token_id(self, token_id: int | list[int]) -> list[int]:
        if isinstance(token_id, list):
            return token_id

        token = self._resolve_placeholder_token(token_id)
        if token is None:
            raise ValueError(
                "CharacterBERT expects token-level character IDs."
                " Pass raw text or pre-tokenized text instead of pre-indexed integer token IDs."
            )
        return self._character_mapper.convert_word_to_char_ids(token)

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        if isinstance(ids, int):
            token = self._resolve_placeholder_token(ids)
            token = token if token is not None else str(self.unk_token)
            if skip_special_tokens and token in self.all_special_tokens:
                return ""
            return token

        if len(ids) > 0 and isinstance(ids[0], list):
            tokens = []
            for char_token in ids:
                token = self._character_mapper.convert_char_ids_to_word(char_token)
                if skip_special_tokens and token in self.all_special_tokens:
                    continue
                tokens.append(token)
            return tokens

        if len(ids) == self.max_characters_per_token:
            token = self._character_mapper.convert_char_ids_to_word(ids)
            if skip_special_tokens and token in self.all_special_tokens:
                return ""
            return token

        tokens = []
        for token_id in ids:
            if isinstance(token_id, list):
                token = self._character_mapper.convert_char_ids_to_word(token_id)
            else:
                token = self._resolve_placeholder_token(token_id)
                token = token if token is not None else str(self.unk_token)
            if skip_special_tokens and token in self.all_special_tokens:
                continue
            tokens.append(token)
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens).strip()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int | list[int]],
        token_ids_1: list[int | list[int]] | None = None,
    ) -> list[list[int]]:
        first = [self._normalize_token_id(token_id) for token_id in token_ids_0]
        if token_ids_1 is None:
            return [self._cls_character_ids] + first + [self._sep_character_ids]

        second = [self._normalize_token_id(token_id) for token_id in token_ids_1]
        return [self._cls_character_ids] + first + [self._sep_character_ids] + second + [self._sep_character_ids]

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        cls = [0]
        sep = [0]
        if token_ids_1 is None:
            return cls + ([0] * len(token_ids_0)) + sep
        return cls + ([0] * len(token_ids_0)) + sep + ([1] * len(token_ids_1)) + [1]

    def _pad(
        self,
        encoded_inputs: dict[str, list[int]] | Mapping,
        max_length: int | None = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_attention_mask: bool | None = None,
    ) -> dict:
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

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
                encoded_inputs[self.model_input_names[0]] = required_input + [self._pad_character_ids] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self._pad_character_ids] * difference + required_input
            else:
                raise ValueError(f"Invalid padding strategy:{padding_side}")

        return encoded_inputs

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        # CharacterBERT tokenization is algorithmic and does not need an external vocab file.
        return ()


__all__ = ["CharacterBertTokenizer"]
