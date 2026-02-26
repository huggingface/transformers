# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
import os
import re
import shutil
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, overload

import numpy as np
from huggingface_hub import create_repo

from transformers.audio_utils import load_audio_as
from transformers.tokenization_utils_base import (
    VERY_LARGE_INTEGER,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType, add_end_docstrings, logging, to_py_obj
from transformers.utils.import_utils import is_mistral_common_available, is_torch_available, requires


if is_mistral_common_available():
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.validator import ValidationMode
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from mistral_common.tokens.tokenizers.utils import (
        download_tokenizer_from_hf_hub,
        get_one_valid_tokenizer_file,
    )


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


ENCODE_KWARGS_DOCSTRING = r"""
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
                automatically. When Tokenizer is loading with `finetuning` mode it adds both `bos` and `eos`. Else, for "test" mode it only adds `bos`.
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
                  to the maximum acceptable input length for the model if that argument is not provided.
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
"""

ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (`bool`, *optional*):
                Whether to return token type IDs. For `MistralCommonBackend` it returns a list of zeros of the sequence length as only one sequence is supported.

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
            return_length  (`bool`, *optional*, defaults to `False`):
                Whether or not to return the lengths of the encoded inputs.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
            return_offsets_mapping (`Literal[False]`, *optional*): False, kept to match Transformers' signature.
            split_special_tokens (`Literal[False]`, *optional*): False, kept to match Transformers' signature.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

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


class MistralTokenizerType(str, Enum):
    """Enum for the different type of tokenizer."""

    spm = "spm"
    tekken = "tekken"


@overload
def _maybe_remove_lang(text: str, skip_special_tokens: bool) -> str: ...
@overload
def _maybe_remove_lang(text: list[str], skip_special_tokens: bool) -> list[str]: ...
def _maybe_remove_lang(text: str | list[str], skip_special_tokens: bool) -> str | list[str]:
    # in the specific case of Voxtral, the added f"lang:xx" (always a two char language code since it follows ISO 639-1 alpha-2 format)
    # is not considered as a special token by mistral-common and is encoded/ decoded as normal text.
    # Nevertheless we should remove it to ease users life.
    if not skip_special_tokens:
        return text

    if isinstance(text, str):
        return re.sub(r"^lang:[a-z]{2}", "", text)

    return [re.sub(r"^lang:[a-z]{2}", "", string) for string in text]


_MAP_SPECIAL_TOKENS = {
    "bos_token": SpecialTokens.bos.value,
    "eos_token": SpecialTokens.eos.value,
    "pad_token": SpecialTokens.pad.value,
    "unk_token": SpecialTokens.unk.value,
}

_VALID_INIT_KWARGS = {"_from_auto", "backend", "files_loaded"}


@requires(backends=("mistral-common",))
class MistralCommonBackend(PreTrainedTokenizerBase):
    """
    Class to wrap `mistral-common` tokenizers.

    `mistral-common` is the official tokenizer library for Mistral AI models. To use it, you need to install it with:

    ```bash
    pip install transformers[mistral-common]
    ```

    Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.

    For more info on `mistral-common`, see [mistral-common](https://github.com/mistralai/mistral-common).

    This class is a wrapper around a `mistral_common.tokens.tokenizers.mistral.MistralTokenizer`.
    It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer and inherits from the `PreTrainedTokenizerBase` class.

    Here are the key behavior differences with the `PythonBackend` class:

    - Pair of sequences are not supported. The signature has been kept for compatibility but all arguments related to pair of sequences are ignored. The return values for pairs are returned as `None`.
    - The `is_split_into_words` argument is not supported.
    - It is not possible to add new tokens to the tokenizer. Special tokens are handled differently from Transformers. In `mistral-common`, special tokens are never encoded directly. This means that: `tokenizer.encode("<s>")` will not return the ID of the `<s>` token. Instead, it will return a list of IDs corresponding to the tokenization of the string `"<s>"`. For more information, see the [mistral-common documentation](https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens).

    If you have suggestions to improve this class, please open an issue on the [mistral-common GitHub repository](https://github.com/mistralai/mistral-common/issues) if it is related to the tokenizer or on the [Transformers GitHub repository](https://github.com/huggingface/transformers/issues) if it is related to the Hugging Face interface.
    """

    model_input_names: list[str] = ["input_ids", "attention_mask"]
    padding_side: str = "left"
    truncation_side: str = "right"
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
    ]

    def __init__(
        self,
        tokenizer_path: str | os.PathLike | Path,
        mode: ValidationMode = ValidationMode.test,
        model_max_length: int = VERY_LARGE_INTEGER,
        padding_side: str = "left",
        truncation_side: str = "right",
        model_input_names: list[str] | None = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        """
        Constructs a `MistralCommonBackend`.

        - **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
            Should be `'right'` or `'left'`.
        - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
            applied. Should be `'right'` or `'left'`.

        Args:
            tokenizer_path (`str` or `os.PathLike` or `Path`):
                Path to the tokenizer file to load the `MistralTokenizer`.
            mode (`Union[str, ValidationMode]`, *optional*, defaults to `ValidationMode.test`):
                The mode to use for the tokenizer. This will be passed to the `MistralTokenizer` constructor. Possible values are:
                - `"finetuning"` or `ValidationMode.finetuning`: The fine-tuning mode.
                - `"test"` or `ValidationMode.test`: The test mode.
                It changes how the tokenizer validates the input and prepares the request to the model.
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
            model_input_names (`List[str]`, *optional*):
                The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
                `"attention_mask"`). Default value is picked from the class attribute of the same name.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not the model should clean up the spaces that were added when splitting the input text during the
                tokenization process.
        """
        if kwargs and not set(kwargs.keys()).issubset(_VALID_INIT_KWARGS):
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported to init `MistralCommonBackend`.")

        self.init_kwargs = {
            "tokenizer_path": tokenizer_path,
            "mode": mode,
            "model_max_length": model_max_length,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
            "model_input_names": model_input_names,
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
        }
        self._tokenizer_path = Path(tokenizer_path)
        self._mode = self._get_validation_mode(mode)

        self.tokenizer: MistralTokenizer = MistralTokenizer.from_file(str(self._tokenizer_path), mode=self._mode)
        self._tokenizer_type = (
            MistralTokenizerType.tekken
            if isinstance(self.tokenizer.instruct_tokenizer.tokenizer, Tekkenizer)
            else MistralTokenizerType.spm
        )
        self._cache_get_vocab: dict[str, int] | None = None

        self._all_special_ids = self._get_all_special_ids()
        self._all_special_tokens = self.convert_ids_to_tokens(self.all_special_ids)

        super().__init__(
            truncation_side=truncation_side,
            padding_side=padding_side,
            model_max_length=model_max_length,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            extra_special_tokens=None,  # Not used by this backend.
            model_specific_special_tokens=None,  # Not used by this backend.
            model_input_names=model_input_names or self.model_input_names,
            **_MAP_SPECIAL_TOKENS,
            **kwargs,
        )

    @property
    def mode(self) -> ValidationMode:
        """
        `ValidationMode`: The mode used by the tokenizer. Possible values are:
            - `"finetuning"` or `ValidationMode.finetuning`: The finetuning mode.
            - `"test"` or `ValidationMode.test`: The test mode.
            It changes how the tokenizer validates the input and prepares the request to the model.
        """
        return self._mode

    @property
    def all_special_ids(self) -> list[int]:
        """
        `list[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.).
        """
        return sorted(self._all_special_ids)

    @property
    def all_special_tokens(self) -> list[str]:
        """
        `list[str]`: A list of all unique special tokens.
        """
        return self._all_special_tokens

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        `int`: Size of the vocabulary.
        """
        return self.tokenizer.instruct_tokenizer.tokenizer.n_words

    def get_vocab(self) -> dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        This is a lossy conversion. There may be multiple token ids that decode to the same
        string due to partial UTF-8 byte sequences being converted to �.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        if self._cache_get_vocab is None:
            # We reverse the order to make sure that the first token is the one to be returned when there are multiple tokens with the same string representation.
            vocab = self.tokenizer.instruct_tokenizer.tokenizer.vocab()
            self._cache_get_vocab = {token: self._piece_to_id(token, False) for token in vocab}
            # Order the dict.
            self._cache_get_vocab = dict(
                sorted(((k, v) for k, v in self._cache_get_vocab.items()), key=lambda x: x[1])
            )
        return self._cache_get_vocab

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size

    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Not supported by `MistralCommonBackend.encode`.
                Will raise an error if used.
        """,
        """
        Returns:
            `list[int]`, `torch.Tensor`: The tokenized ids of the text.
        """,
    )
    def encode(
        self,
        text: TextInput | EncodedInput,
        text_pair: None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        verbose: bool = True,
        return_offsets_mapping: Literal[False] = False,
        split_special_tokens: Literal[False] = False,
        **kwargs,
    ) -> list[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str` or `list[int]`):
                The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).
            text_pair (`None`, *optional*):
                Not supported by `MistralCommonBackend.encode`. Kept to match `PreTrainedTokenizerBase.encode` signature.
        """
        if return_offsets_mapping or split_special_tokens:
            raise ValueError(
                "`MistralCommonBackend` does not support `return_offsets_mapping` and `split_special_tokens`."
            )

        if truncation in [TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, "only_first", "only_second"]:
            raise ValueError(
                "Truncation strategy `only_first` and `only_second` are not supported by `MistralCommonBackend`."
            )

        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.encode`.")

        if text_pair:
            raise ValueError("`MistralCommonBackend.encode` does not support `text_pair`.")

        return super().encode(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            verbose=verbose,
        )

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.decode`.")

        token_ids = to_py_obj(token_ids)

        if isinstance(token_ids, int):
            token_ids = [token_ids]

        special_token_policy = SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP

        text = self.tokenizer.decode(token_ids, special_token_policy=special_token_policy)

        # Apply tokenizer-specific cleanup if available and requested
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        return _maybe_remove_lang(text=text, skip_special_tokens=skip_special_tokens)

    def decode(
        self,
        token_ids: Union[int, list[int], list[list[int]], np.ndarray, "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str | list[str]:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]`):
                A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the
                `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonBackend.decode`.
                Will raise an error if used.

        Returns:
            `Union[str, list[str]]`: The decoded string for a single sequence, or a list of decoded strings for a
            batch of sequences.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.decode`.")

        return super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def batch_decode(
        self,
        sequences: Union[list[int], list[list[int]], np.ndarray, "torch.Tensor"],
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
                Not supported by `MistralCommonBackend.batch_decode`.
                Will raise an error if used.

        Returns:
            `list[str]`: The list of decoded sentences.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.batch_decode`.")

        return super().batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str: ...
    @overload
    def convert_ids_to_tokens(self, ids: list[int], skip_special_tokens: bool = False) -> list[str]: ...
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

        if isinstance(ids, int):
            return_int = True
            ids = [ids]
        else:
            return_int = False

        tokens: list[str] = []
        for token_id in ids:
            if self.tokenizer.instruct_tokenizer.tokenizer.is_special(token_id) and skip_special_tokens:
                continue
            tokens.append(self.tokenizer.instruct_tokenizer.tokenizer.id_to_piece(token_id))

        if return_int and tokens == []:
            raise ValueError(f"Invalid token id {ids[0]}.")
        elif return_int:
            return tokens[0]

        return tokens

    def _tekken_piece_to_id(self, piece: str, warn: bool) -> int:
        tekken_tokenizer = self.tokenizer.instruct_tokenizer.tokenizer
        assert isinstance(tekken_tokenizer, Tekkenizer), type(tekken_tokenizer)

        piece_bytes = piece.encode("utf-8")
        shift = tekken_tokenizer.num_special_tokens
        try:
            return shift + tekken_tokenizer._tekken_token2id_nospecial[piece_bytes]
        except KeyError:
            piece_str = piece_bytes.decode("utf-8")
            if piece_str in tekken_tokenizer._special_tokens_reverse_vocab:
                return tekken_tokenizer._special_tokens_reverse_vocab[piece_str]
            if warn:
                logger.warning("Failed to convert token %s to id, replacing with <unk>", piece_bytes)
            return tekken_tokenizer.unk_id

    def _piece_to_id(self, piece: str, warn: bool) -> int:
        if self._tokenizer_type == MistralTokenizerType.spm:
            return self.tokenizer.instruct_tokenizer.tokenizer._model.piece_to_id(piece)
        elif self._tokenizer_type == MistralTokenizerType.tekken:
            return self._tekken_piece_to_id(piece, warn)
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

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
            one_token = True
            tokens = [tokens]
        else:
            one_token = False

        ids: list[int] = []
        for token in tokens:
            ids.append(self._piece_to_id(token, True))

        if one_token:
            return ids[0]
        return ids

    def _text_to_ids(self, text: TextInput, add_special_tokens: bool) -> list[int]:
        """
        Converts a string into a sequence of tokens ids, using the tokenizer.
        """
        add_eos = add_special_tokens and self._mode == ValidationMode.finetuning
        tokens_ids = self.tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=add_special_tokens, eos=add_eos)
        return tokens_ids

    def tokenize(
        self,
        text: TextInput,
        return_offsets_mapping: Literal[False] = False,
        split_special_tokens: Literal[False] = False,
        **kwargs,
    ) -> list[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.

        Args:
            text (`str`):
                The sequence to be encoded.
            return_offsets_mapping (`Literal[False]`, *optional*): False, kept to match Transformers' signature.
            split_special_tokens (`Literal[False]`, *optional*): False, kept to match Transformers' signature.
            **kwargs (additional keyword arguments):
                Not supported by `MistralCommonBackend.tokenize`.
                Will raise an error if used.

        Returns:
            `list[str]`: The list of tokens.
        """
        if return_offsets_mapping or split_special_tokens:
            raise ValueError(
                "`MistralCommonBackend` does not support `return_offsets_mapping` and `split_special_tokens`."
            )

        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.tokenize`.")

        return self.convert_ids_to_tokens(self._text_to_ids(text, add_special_tokens=False), skip_special_tokens=False)

    def _get_all_special_ids(self) -> set[int]:
        if self._tokenizer_type == MistralTokenizerType.tekken:
            return self.tokenizer.instruct_tokenizer.tokenizer._special_token_ids
        elif self._tokenizer_type == MistralTokenizerType.spm:
            return {
                token_id
                for token_id in range(self.tokenizer.instruct_tokenizer.tokenizer.n_words)
                if self.tokenizer.instruct_tokenizer.tokenizer.is_special(token_id)
            }
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`list[int]`): List of ids of the sequence.
            token_ids_1 (`None`, *optional*): None, kept to match Transformers' implementation.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if token_ids_1 is not None:
            raise ValueError(
                "`token_ids_1` is not supported by `MistralCommonBackend` and should be `None`, kept for compatibility."
            )

        if already_has_special_tokens:
            return [1 if int(token_id) in self._all_special_ids else 0 for token_id in token_ids_0]

        if self.mode == ValidationMode.test:
            # [BOS] seq0
            return [1] + ([0] * len(token_ids_0))
        else:
            # [BOS] seq0 [EOS]
            return [1] + ([0] * len(token_ids_0)) + [1]

    def _encode_plus(  # type: ignore[override]
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: None = None,
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
        return_length: bool = False,
        verbose: bool = True,
        return_offsets_mapping: Literal[False] = False,
        split_special_tokens: Literal[False] = False,
        **kwargs,
    ) -> BatchEncoding:
        # Detect batched inputs (list of sequences)
        if text_pair is not None:
            raise ValueError("`MistralCommonBackend` does not support `text_pair != None` for `_encode_plus`.")

        if return_offsets_mapping or split_special_tokens:
            raise ValueError(
                "`MistralCommonBackend` does not support `return_offsets_mapping` and `split_special_tokens`."
            )

        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend._encode_plus`.")

        is_batched = isinstance(text, (list, tuple)) and (
            (not text and not is_split_into_words)
            or (text and is_split_into_words and isinstance(text[0], (list, tuple)))
            or (text and not is_split_into_words and isinstance(text[0], (str, list, tuple)))
        )

        if is_batched:
            batch_outputs = {}
            one_overflowed = False
            for current_text in text:
                current_output = self._encode_plus(
                    text=current_text,
                    text_pair=None,
                    add_special_tokens=add_special_tokens,
                    padding_strategy=PaddingStrategy.DO_NOT_PAD,  # we pad in batch afterward
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    stride=stride,
                    is_split_into_words=is_split_into_words,
                    pad_to_multiple_of=None,  # we pad in batch afterward
                    padding_side=None,  # we pad in batch afterward
                    return_tensors=None,  # We convert the whole batch to tensors at the end
                    return_token_type_ids=return_token_type_ids,
                    return_attention_mask=False,  # we pad in batch afterward
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_length=return_length,
                    verbose=verbose,
                )
                for key, value in current_output.items():
                    batch_outputs.setdefault(key, []).append(value)

                # To ensure the list is built for each sample, we need to add this.
                if return_overflowing_tokens and not return_tensors:
                    if "overflowing_tokens" not in current_output:
                        batch_outputs.setdefault("overflowing_tokens", []).append([0])
                        batch_outputs.setdefault("num_truncated_tokens", []).append([0])
                    else:
                        one_overflowed = True

            # Remove overflow-related keys before tensor conversion if return_tensors is set
            # Slow tokenizers don't support returning these as tensors
            if return_overflowing_tokens and (return_tensors or not one_overflowed):
                batch_outputs.pop("overflowing_tokens", None)
                batch_outputs.pop("num_truncated_tokens", None)

            batch_outputs = self.pad(
                batch_outputs,
                padding=padding_strategy.value,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

            return BatchEncoding(batch_outputs, tensor_type=return_tensors)

        def get_input_ids(text):
            if isinstance(text, str):
                return self._text_to_ids(text, False)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(f"Input {text} is not valid. Should be a string, or a list/tuple of integers.")

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
            padding_side=padding_side,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        return_offsets_mapping: Literal[False] = False,
        split_special_tokens: Literal[False] = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            ids (`list[int]`):
                Tokenized input ids of the first sequence.
            pair_ids (`None`, *optional*):
                Not supported by `MistralCommonBackend`. Kept to match the interface of `PreTrainedTokenizerBase`.
        """
        if return_offsets_mapping or split_special_tokens:
            raise ValueError(
                "`MistralCommonBackend` does not support `return_offsets_mapping` and `split_special_tokens`."
            )

        if pair_ids is not None:
            raise ValueError(
                "`pair_ids` is not supported by `MistralCommonBackend` and should be `None`, kept for compatibility."
            )

        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.prepare_for_model`."
            )

        padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # Validation
        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # Truncation
        num_special = self.num_special_tokens_to_add(pair=False) if add_special_tokens else 0
        total_len = len(ids) + len(pair_ids or []) + num_special

        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, _, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=None,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, None)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, None)
        else:
            sequence = ids
            token_type_ids = [0] * len(sequence)

        # Build output
        encoded_inputs = {"input_ids": sequence}
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = (
                self.get_special_tokens_mask(ids, None) if add_special_tokens else [0] * len(sequence)
            )
        if return_overflowing_tokens and not return_tensors and overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length if max_length else 0

        # Check sequence length and warn if needed
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Pad
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        return BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)

    def truncate_sequences(  # type: ignore[override]
        self,
        ids: list[int],
        pair_ids: None = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: str | TruncationStrategy = "longest_first",
        stride: int = 0,
        **kwargs,
    ) -> tuple[list[int], None, list[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`list[int]`):
                Tokenized input ids. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`None`, *optional*):
                Not supported by `MistralCommonBackend`. Kept to match the signature of `PreTrainedTokenizerBase.truncate_sequences`.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `'longest_first'`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[list[int], None, list[int]]`: The truncated `ids` and the list of
            overflowing tokens. `None` is returned to match Transformers signature.
        """

        if pair_ids:
            raise ValueError("`pair_ids` is not supported by `MistralCommonBackend.truncate_sequences`.")

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        if truncation_strategy in [
            TruncationStrategy.ONLY_FIRST,
            TruncationStrategy.ONLY_SECOND,
        ]:
            raise ValueError(f"{truncation_strategy=} is not supported by `MistralCommonBackend`.")

        if num_tokens_to_remove <= 0:
            return ids, None, []

        overflowing_tokens = []

        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            window_len = min(len(ids), stride + num_tokens_to_remove)
            if self.truncation_side == "left":
                overflowing_tokens = ids[:window_len]
                ids = ids[num_tokens_to_remove:]
            else:
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]

        return ids, None, overflowing_tokens

    def apply_chat_template(  # type: ignore[override]
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        tools: list[dict | Callable] | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids.

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (`List[Union[Dict, Callable]]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.
            add_generation_prompt (`bool`, *optional*):
                This argument is a no-op for `MistralCommonBackend`. However, it cannot be used at the same time as `continue_final_message` to keep the API consistent.
                If any conversation ends with an assistant message, it will raise an error. In such cases, use `continue_final_message` instead.
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
            return_dict (`bool`, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
                If at least one conversation contains an image, its pixel values will be returned in the `pixel_values` key.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonBackend.apply_chat_template`.
                Will raise an error if used.

        Returns:
            `Union[str, list[int], list[str], list[list[int]], BatchEncoding]`: The tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.
        """
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.apply_chat_template`."
            )
        if not isinstance(truncation, bool):
            raise TypeError("`truncation` must be a boolean for `apply_chat_template` method.")

        if add_generation_prompt and continue_final_message:
            raise ValueError("Cannot use both `add_generation_prompt` and `continue_final_message`.")

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        if add_generation_prompt:
            for conversation in conversations:
                last_message = conversation[-1]
                if last_message.get("role") == "assistant":
                    raise ValueError(
                        "The last message in the conversation is already an assistant message. Consider using `continue_final_message` instead."
                    )

        def _maybe_adapt_message(message: dict[str, Any]) -> None:
            """Adapt message to `mistral-common` format and leave validation to `mistral-common`."""
            if not isinstance(message, dict):
                return message
            maybe_list_content: str | list[dict[str, str | dict[str, Any]]] | None = message.get("content")
            if not maybe_list_content or isinstance(maybe_list_content, str):
                return message

            normalized_content: list[dict[str, str | dict[str, Any]]] = []
            message = message.copy()
            for content in maybe_list_content:
                content_type = content.get("type", None)
                if not content_type:
                    continue
                elif content_type == "image":
                    maybe_url: str | None = content.get("url")
                    maybe_path: str | None = content.get("path")
                    maybe_base64: str | None = content.get("base64")
                    if maybe_url:
                        image_content = maybe_url
                    elif maybe_path:
                        if not maybe_path.startswith("file://"):
                            maybe_path = Path(maybe_path).resolve().as_uri()
                        image_content = maybe_path
                    elif maybe_base64:
                        if not maybe_base64.startswith("data:image"):
                            maybe_base64 = "data:image/unk;base64," + maybe_base64
                        image_content = maybe_base64
                    else:
                        raise ValueError("Image content must be specified.")
                    normalized_content.append({"type": "image_url", "image_url": {"url": image_content}})
                elif content_type == "audio":
                    maybe_url: str | None = content.get("url")
                    maybe_path: str | None = content.get("path")
                    maybe_base64: str | None = content.get("base64")
                    if maybe_url or maybe_path:
                        audio_data = load_audio_as(maybe_url or maybe_path, return_format="dict", force_mono=True)
                        normalized_content.append({"type": "input_audio", "input_audio": audio_data})
                        continue
                    if not maybe_base64:
                        raise ValueError("Audio content must be specified.")
                    normalized_content.append({"type": "audio_url", "audio_url": {"url": maybe_base64}})
                else:
                    normalized_content.append(content)
            message["content"] = normalized_content
            return message

        outputs = []
        images: list[np.ndarray] = []
        audios: list[np.ndarray] = []

        for conversation in conversations:
            messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]] = []
            for message in conversation:
                message = _maybe_adapt_message(message)
                messages.append(message)

            chat_request = ChatCompletionRequest.from_openai(
                messages=messages,
                tools=tools,
                continue_final_message=continue_final_message,
            )

            tokenized_request = self.tokenizer.encode_chat_completion(chat_request)
            if tokenize:
                outputs.append(tokenized_request.tokens)
            else:
                outputs.append(tokenized_request.text)
            images.extend(tokenized_request.images)
            audios.extend([el.audio_array for el in tokenized_request.audios])

        if not is_batched:
            outputs = outputs[0]

        if tokenize:
            out = self(
                outputs,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
            )
            if return_dict:
                if images:
                    pixel_values: list[np.ndarray] | np.ndarray | torch.Tensor
                    if return_tensors == "pt":
                        if not is_torch_available():
                            raise ImportError(
                                "Unable to convert output to PyTorch tensors format, PyTorch is not installed."
                            )

                        pixel_values = torch.from_numpy(np.stack(images))
                    elif return_tensors == "np":
                        pixel_values = np.array(images)
                    elif return_tensors is None:
                        pixel_values = images
                    else:
                        raise ValueError(f"Unsupported return_tensors type: {return_tensors}")
                    out.data["pixel_values"] = pixel_values
                if audios:
                    if return_tensors is not None:
                        raise NotImplementedError(
                            "When passing audio content in apply_chat_template, `return_tensors` must be None since we cannot batch the audio inputs. The returned audio will be a list of numpy arrays."
                        )
                    # Transformers convention is audio for plural audio (audio does not take a "s")
                    out.data["audio"] = audios
                return out
            else:
                return out["input_ids"]

        else:
            logger.warning(
                "`MistralCommonBackend.apply_chat_template(..., tokenize=False)` is unsafe and may lead to unexpected behavior."
                " Please consider using `tokenize=True` instead and don't encode the output manually."
            )
            return outputs

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: None = None) -> list[int]:
        """
        Build model inputs from a sequence by adding special tokens.

        This method dynamically builds inputs based on the tokenizer's `mode`:
        - `"test"`: seq0 [EOS]
        - `"finetuning"`: [BOS] seq0

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`None`, *optional*): None, kept to match Transformers' signature.

        Returns:
            `list[int]`: List of input IDs with the appropriate special tokens.
        """
        if token_ids_1 is not None:
            raise ValueError(
                "`MistralCommonBackend` does not implement `token_ids_1 != None` for `build_inputs_with_special_tokens`."
            )

        if self.mode == ValidationMode.test:
            # [BOS] seq0
            return [self.bos_token_id] + token_ids_0

        else:
            # [BOS] seq0 [EOS]
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0: list[int], token_ids_1: None = None) -> list[int]:
        """
        Create a mask of zeroes from the token ids with special tokens added.

        Kept to match Transformers' implementation.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`None`, *optional*): None, kept to match Transformers' signature.


        Returns:
            `list[int]`: Token type IDs according to the configured pattern.
        """
        if token_ids_1 is not None:
            raise ValueError(
                "`MistralCommonBackend` does not implement `token_ids_1 != None` for `create_token_type_ids_from_sequences`."
            )

        sequence = self.build_inputs_with_special_tokens(token_ids_0)

        return [0] * len(sequence)

    def num_special_tokens_to_add(self, pair: Literal[False] = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`Literal[False]`, *optional*): False, kept to match Transformer's signature.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        if pair:
            raise ValueError(
                "`MistralCommonBackend` does not implement `pair = True` for `num_special_tokens_to_add`."
            )

        return len(self.build_inputs_with_special_tokens([], None))

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: TextInput | EncodedInput | list[TextInput] | list[EncodedInput] | None = None,
        text_pair: None = None,
        text_target: None = None,
        text_pair_target: None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_offsets_mapping: Literal[False] = False,
        split_special_tokens: Literal[False] = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of int
                (encoded strings).
            text_pair (`None`, *optional*):
                Not supported by `MistralCommonBackend`. Kept to match the signature of `PreTrainedTokenizerBase.__call__`.
            text_target (`None`, *optional*):
                Not supported by `MistralCommonBackend`. Kept to match the signature of `PreTrainedTokenizerBase.__call__`.
            text_pair_target (`None`, *optional*):
                Not supported by `MistralCommonBackend`. Kept to match the signature of `PreTrainedTokenizerBase.__call__`.
        """
        if return_offsets_mapping or split_special_tokens:
            raise ValueError(
                "`MistralCommonBackend` does not support `return_offsets_mapping` and `split_special_tokens`."
            )

        if truncation in [TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, "only_first", "only_second"]:
            raise ValueError(
                "Truncation strategy `only_first` and `only_second` are not supported by `MistralCommonBackend`."
            )

        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.__call__`.")

        if text_pair or text_target or text_pair_target:
            raise ValueError(
                "`text_pair`, `text_target` and `text_pair_target` are not supported by `MistralCommonBackend`."
            )

        return super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        mode: str | ValidationMode = ValidationMode.test,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        model_max_length: int = VERY_LARGE_INTEGER,
        padding_side: str = "left",
        truncation_side: str = "right",
        model_input_names: list[str] | None = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        r"""
        Instantiate a `MistralCommonBackend` from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing the tokenizer config, for instance saved
                  using the [`MistralCommonBackend.tokenization_mistral_common.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
            mode (`Union[str, ValidationMode]`, *optional*, defaults to `ValidationMode.test`):
                Validation mode for the `MistralTokenizer` tokenizer. Possible values are:
                - `"finetuning"` or `ValidationMode.finetuning`: The fine-tuning mode.
                - `"test"` or `ValidationMode.test`: The test mode.
                It changes how the tokenizer validates the input and prepares the request to the model.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
                exist.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`).
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only rely on local files and not to attempt to download any files.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            padding_side (`str`, *optional*, defaults to `"left"`):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            truncation_side (`str`, *optional*, defaults to `"right"`):
                The side on which the model should have truncation applied. Should be selected between ['right', 'left'].
            model_input_names (`List[str]`, *optional*):
                The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
                `"attention_mask"`). Default value is picked from the class attribute of the same name.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not the model should clean up the spaces that were added when splitting the input text during the
                tokenization process.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonBackend.from_pretrained`.
                Will raise an error if used.
        """
        if init_inputs:
            raise ValueError("`init_inputs` are not supported by `MistralCommonBackend.from_pretrained`.")

        # Handle kwargs and AutoTokenizer/AutoProcessor case
        valid_kwargs = _VALID_INIT_KWARGS.union(
            {"trust_remote_code", "_from_pipeline", "_commit_hash", "dtype", "subfolder"}
        )
        if kwargs and not set(kwargs.keys()).issubset(valid_kwargs):
            raise ValueError(
                f"Some kwargs in {list(kwargs.keys())} are not supported by `MistralCommonBackend.from_pretrained`."
            )

        mode = cls._get_validation_mode(mode)

        if not os.path.isdir(pretrained_model_name_or_path):
            tokenizer_path = download_tokenizer_from_hf_hub(
                repo_id=pretrained_model_name_or_path,
                cache_dir=cache_dir,
                token=token,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        else:
            candidate_files = os.listdir(pretrained_model_name_or_path)
            tokenizer_path = os.path.join(pretrained_model_name_or_path, get_one_valid_tokenizer_file(candidate_files))

        return cls(
            tokenizer_path=tokenizer_path,
            mode=mode,
            model_max_length=model_max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_input_names=model_input_names,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def save_pretrained(  # type: ignore[override]
        self,
        save_directory: str | os.PathLike | Path,
        push_to_hub: bool = False,
        token: str | bool | None = None,
        commit_message: str | None = None,
        repo_id: str | None = None,
        private: bool | None = None,
        **kwargs,
    ) -> tuple[str, ...]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        [`~MistralCommonBackend.tokenization_mistral_common.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            token (`str` or *bool*, *optional*, defaults to `None`):
                The token to use to push to the model hub. If `True`, will use the token in the `HF_TOKEN` environment
                variable.
            commit_message (`str`, *optional*): The commit message to use when pushing to the hub.
            repo_id (`str`, *optional*): The name of the repository to which push to the Hub.
            private (`bool`, *optional*): Whether the model repository is private or not.
            kwargs (`Dict[str, Any]`, *optional*):
                Not supported by `MistralCommonBackend.save_pretrained`.
                Will raise an error if used.

        Returns:
            A tuple of `str`: The files saved.
        """
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonBackend.save_pretrained`."
            )

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        shutil.copy(self._tokenizer_path, save_directory)

        if push_to_hub:
            repo_id = repo_id or str(save_directory).split(os.path.sep)[-1]
            repo_id = create_repo(repo_id, token=token, private=private, exist_ok=True).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

        return (str(save_directory / self._tokenizer_path.name),)

    @staticmethod
    def _get_validation_mode(mode: str | ValidationMode) -> ValidationMode:
        """Get the validation mode from a string or a ValidationMode."""
        _invalid_mode_msg = (
            f"Invalid `mistral-common` tokenizer mode: {mode}. Possible values are 'finetuning' or 'test'."
        )
        if isinstance(mode, str):
            try:
                mode = ValidationMode[mode]
            except KeyError:
                raise ValueError(_invalid_mode_msg)
        elif not isinstance(mode, (str, ValidationMode)):
            raise ValueError(_invalid_mode_msg)

        if mode not in [ValidationMode.finetuning, ValidationMode.test]:
            raise ValueError(_invalid_mode_msg)
        return mode

    def __repr__(self) -> str:
        # MistralCommonBackend does not implement added_tokens_decoder, so we need a custom repr
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, model_max_length={self.model_max_length},"
            f" padding_side='{self.padding_side}', truncation_side='{self.truncation_side}',"
            f" special_tokens={self.special_tokens_map})"
        )

    def added_tokens_decoder(self):
        raise NotImplementedError("`MistralCommonBackend` does not implement `added_tokens_decoder`.")

    def add_special_tokens(
        self,
        special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]],
        replace_extra_special_tokens: bool = True,
    ):
        r"""`MistralCommonBackend` does not implement `add_special_tokens` by design.

        If you would like this behaviour to be implemented, please open an issue in the `Transformers` or `mistral-common` repositories to request it.
        """

        raise NotImplementedError("`MistralCommonBackend` does not implement `add_special_tokens`.")

    def add_tokens(  # type: ignore[override]
        self,
        special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]],
        replace_extra_special_tokens: bool = True,
    ):
        """
        `MistralCommonBackend` does not implement `add_special_tokens` by design.

        If you would like this behaviour to be implemented, please open an issue in the `Transformers` or `mistral-common` repositories to request it.
        """

        raise NotImplementedError("`MistralCommonBackend` does not implement `add_tokens`.")

    def convert_added_tokens(cls, obj: AddedToken | Any, save: bool = False, add_type_field: bool = True):  # type: ignore[override]
        """
        `MistralCommonBackend` does not implement `convert_added_tokens` by design.

        If you would like this behaviour to be implemented, please open an issue in the `Transformers` or `mistral-common` repositories to request it.
        """

        raise NotImplementedError("`MistralCommonBackend` does not implement `convert_added_tokens`.")

    def get_chat_template(self, chat_template: str | None = None, tools: list[dict] | None = None) -> str:
        """`MistralCommonBackend` does not implement `get_chat_template` by design as `mistral-common` does not use chat templates."""

        raise NotImplementedError("`MistralCommonBackend` does not implement `get_chat_template`.")

    def save_chat_templates(
        self,
        save_directory: str | os.PathLike,
        tokenizer_config: dict,
        filename_prefix: str | None,
        save_jinja_files: bool,
    ):
        """`MistralCommonBackend` does not implement `save_chat_templates` by design as `mistral-common` does not use chat templates."""

        raise NotImplementedError("`MistralCommonBackend` does not implement `save_chat_templates`.")

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """
        `MistralCommonBackend` does not implement `save_vocabulary` by design.

        This is because `mistral-common` is configured by one tokenizer file. If you'd like to save the vocabulary, please consider using the `save_pretrained` method instead.
        """

        raise NotImplementedError("`MistralCommonBackend` does not implement `save_vocabulary`.")


# Backward compatibility alias for codebases still importing the legacy name.
MistralCommonTokenizer = MistralCommonBackend
