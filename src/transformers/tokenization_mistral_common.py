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
import shutil
import warnings
from collections.abc import Mapping, Sized
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union, overload

import numpy as np

from transformers.audio_utils import load_audio_as
from transformers.tokenization_utils_base import (
    LARGE_INTEGER,
    VERY_LARGE_INTEGER,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType, add_end_docstrings, logging, to_py_obj
from transformers.utils.generic import is_torch_tensor
from transformers.utils.hub import PushToHubMixin
from transformers.utils.import_utils import is_mistral_common_available, is_torch_available, requires


if is_mistral_common_available():
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.validator import ValidationMode
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
    from mistral_common.tokens.tokenizers.image import MultiModalVersion
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


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


@requires(backends=("mistral-common",))
class MistralCommonTokenizer(PushToHubMixin):
    """
    Class to wrap `mistral-common` tokenizers.

    `mistral-common` is the official tokenizer library for Mistral AI models. To use it, you need to install it with:

    ```bash
    pip install transformers[mistral-common]
    ```

    Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.

    For more info on `mistral-common`, see [mistral-common](https://github.com/mistralai/mistral-common).

    This class is a wrapper around a `mistral_common.tokens.tokenizers.mistral.MistralTokenizer`.
    It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer.

    Supports the following methods from the `PreTrainedTokenizerBase` class:

    - [`~MistralCommonTokenizer.get_vocab`]: Returns the vocabulary as a dictionary of token to index.
    - [`~MistralCommonTokenizer.encode`]: Encode a string to a list of integers.
    - [`~MistralCommonTokenizer.decode`]: Decode a list of integers to a string.
    - [`~MistralCommonTokenizer.batch_decode`]: Decode a batch of list of integers to a list of strings.
    - [`~MistralCommonTokenizer.convert_tokens_to_ids`]: Convert a list of tokens to a list of integers.
    - [`~MistralCommonTokenizer.convert_ids_to_tokens`]: Convert a list of integers to a list of tokens.
    - [`~MistralCommonTokenizer.tokenize`]: Tokenize a string.
    - [`~MistralCommonTokenizer.get_special_tokens_mask`]: Get the special tokens mask for a list of tokens.
    - [`~MistralCommonTokenizer.prepare_for_model`]: Prepare a list of inputs for the model.
    - [`~MistralCommonTokenizer.pad`]: Pad a list of inputs to the same length.
    - [`~MistralCommonTokenizer.truncate_sequences`]: Truncate a list of sequences to the same length.
    - [`~MistralCommonTokenizer.apply_chat_template`]: Apply a chat template to a list of messages.
    - [`~MistralCommonTokenizer.__call__`]: Tokenize a string or a list of strings.
    - [`~MistralCommonTokenizer.from_pretrained`]: Download and cache a pretrained tokenizer from the Hugging Face model hub or local directory.
    - [`~MistralCommonTokenizer.save_pretrained`]: Save a tokenizer to a directory, so it can be reloaded using the `from_pretrained` class method.
    - [`~MistralCommonTokenizer.push_to_hub`]: Upload tokenizer to the Hugging Face model hub.

    Here are the key differences with the `PreTrainedTokenizerBase` class:

    - Pair of sequences are not supported. The signature have been kept for compatibility but all arguments related to pair of sequences are ignored. The return values of pairs are returned as `None`.
    - The `is_split_into_words` argument is not supported.
    - The `return_token_type_ids` argument is not supported.
    - It is not possible to add new tokens to the tokenizer. Also the special tokens are handled differently from Transformers. In `mistral-common`, special tokens are never encoded directly. This means that: `tokenizer.encode("<s>")` will not return the ID of the `<s>` token. Instead, it will return a list of IDs corresponding to the tokenization of the string `"<s>"`. For more information, see the [mistral-common documentation](https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens).

    If you have suggestions to improve this class, please open an issue on the [mistral-common GitHub repository](https://github.com/mistralai/mistral-common/issues) if it is related to the tokenizer or on the [Transformers GitHub repository](https://github.com/huggingface/transformers/issues) if it is related to the Hugging Face interface.
    """

    model_input_names: list[str] = ["input_ids", "attention_mask"]
    padding_side: str = "left"
    truncation_side: str = "right"

    def __init__(
        self,
        tokenizer_path: Union[str, os.PathLike, Path],
        mode: ValidationMode = ValidationMode.test,
        model_max_length: int = VERY_LARGE_INTEGER,
        padding_side: str = "left",
        truncation_side: str = "right",
        model_input_names: Optional[list[str]] = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        """
        Constructs a `MistralCommonTokenizer`.

        - **model_input_names** (`List[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
            Should be `'right'` or `'left'`.
        - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
            applied. Should be `'right'` or `'left'`.

        Args:
            tokenizer_path (`str` or `os.PathLike` or `Path`):
                Path to the tokenizer file to load the `MistralTokenizer`.
            mode (`ValidationMode`, *optional*, defaults to `ValidationMode.test`):
                The mode to use for the tokenizer. This will be passed to the `MistralTokenizer` constructor.
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
            model_input_names (`List[string]`, *optional*):
                The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
                `"attention_mask"`). Default value is picked from the class attribute of the same name.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not the model should cleanup the spaces that were added when splitting the input text during the
                tokenization process.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported to init `MistralCommonTokenizer`.")

        self._tokenizer_path = Path(tokenizer_path)
        self.tokenizer: MistralTokenizer = MistralTokenizer.from_file(str(self._tokenizer_path), mode=mode)
        self._tokenizer_type = (
            MistralTokenizerType.tekken
            if isinstance(self.tokenizer.instruct_tokenizer.tokenizer, Tekkenizer)
            else MistralTokenizerType.spm
        )
        self.truncation_side = truncation_side
        self.padding_side = padding_side
        self.model_max_length = model_max_length
        self.cleanup_tokenization_spaces = clean_up_tokenization_spaces
        self.deprecation_warnings = {}  # Use to store when we have already noticed a deprecation warning (avoid overlogging).

        if model_input_names is not None:
            if (
                not isinstance(model_input_names, (list, tuple))
                and len(model_input_names) == 0
                and not all(isinstance(i, str) for i in model_input_names)
            ):
                raise ValueError(
                    "`model_input_names` should be a non-empty list or tuple of str but got an empty value."
                )
            self.model_input_names = model_input_names

        self._cache_get_vocab: Optional[dict[str, int]] = None

    @property
    def bos_token_id(self) -> int:
        """
        Id of the beginning of sentence token in the vocabulary.
        """
        return self.tokenizer.instruct_tokenizer.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        """
        Id of the end of sentence token in the vocabulary.
        """
        return self.tokenizer.instruct_tokenizer.tokenizer.eos_id

    @property
    def unk_token_id(self) -> int:
        """
        Id of the unknown token in the vocabulary.
        """
        return self.tokenizer.instruct_tokenizer.tokenizer.unk_id

    @property
    def pad_token_id(self) -> int:
        """
        Id of the padding token in the vocabulary.
        """
        return self.tokenizer.instruct_tokenizer.tokenizer.pad_id

    @property
    def bos_token(self) -> str:
        """
        String associated to the beginning of sentence token in the vocabulary.
        """
        return self.convert_ids_to_tokens(self.bos_token_id)

    @property
    def eos_token(self) -> str:
        """
        String associated to the end of sentence token in the vocabulary.
        """
        return self.convert_ids_to_tokens(self.eos_token_id)

    @property
    def unk_token(self) -> str:
        """
        String associated to the unknown token in the vocabulary.
        """
        return self.convert_ids_to_tokens(self.unk_token_id)

    @property
    def pad_token(self) -> str:
        """
        String associated to the padding token in the vocabulary.
        """
        return self.convert_ids_to_tokens(self.pad_token_id)

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
            self._cache_get_vocab = {
                token: idx for idx, token in enumerate(self.tokenizer.instruct_tokenizer.tokenizer.vocab())
            }
        return self._cache_get_vocab

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size

    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Not supported by `MistralCommonTokenizer.encode`.
                Will raise an error if used.
        """,
        """
        Returns:
            `List[int]`, `torch.Tensor`: The tokenized ids of the text.
        """,
    )
    def encode(
        self,
        text: Union[TextInput, EncodedInput],
        text_pair: None = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> list[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str` or `List[int]`):
                The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).
            text_pair (`None`, *optional*):
                Not supported by `MistralCommonTokenizer.encode`. Kept to match `PreTrainedTokenizerBase.encode` signature.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.encode`.")
        if text_pair:
            raise ValueError("`MistralCommonTokenizer.encode` does not support `text_pair`.")

        padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
        )

        encoded_inputs = self._encode_plus(
            text,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_attention_mask=False,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_length=False,
            verbose=verbose,
        )

        return encoded_inputs["input_ids"]

    def decode(
        self,
        token_ids: Union[int, list[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer.decode`.
                Will raise an error if used.

        Returns:
            `str`: The decoded sentence.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.decode`.")

        clean_up_tokenization_spaces = clean_up_tokenization_spaces or self.cleanup_tokenization_spaces

        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        special_token_policy = SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP

        decoded_string = self.tokenizer.decode(token_ids, special_token_policy=special_token_policy)
        if clean_up_tokenization_spaces:
            decoded_string = PreTrainedTokenizerBase.clean_up_tokenization(decoded_string)

        return decoded_string

    def batch_decode(
        self,
        sequences: Union[list[int], list[list[int]], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> list[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer.batch_decode`.
                Will raise an error if used.

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

    def _is_control_token(self, token_id: int) -> bool:
        if self._tokenizer_type == MistralTokenizerType.spm:
            return token_id in self.tokenizer.instruct_tokenizer.tokenizer._control_tokens()
        elif self._tokenizer_type == MistralTokenizerType.tekken:
            return token_id < self.tokenizer.instruct_tokenizer.tokenizer.num_special_tokens
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str: ...
    @overload
    def convert_ids_to_tokens(self, ids: list[int], skip_special_tokens: bool = False) -> list[str]: ...
    def convert_ids_to_tokens(
        self, ids: Union[int, list[int]], skip_special_tokens: bool = False
    ) -> Union[str, list[str]]:
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
            one_token = True
            ids = [ids]
        else:
            one_token = False

        tokens: list[str] = []
        for token_id in ids:
            if self._is_control_token(token_id) and skip_special_tokens:
                continue
            tokens.append(self.tokenizer.instruct_tokenizer.tokenizer.id_to_piece(token_id))

        if one_token:
            if tokens == []:
                raise ValueError(f"Invalid token id {ids}.")

            return tokens[0]
        return tokens

    def _piece_to_id(self, piece: str) -> int:
        if self._tokenizer_type == MistralTokenizerType.spm:
            return self.tokenizer.instruct_tokenizer.tokenizer._model.piece_to_id(piece)
        elif self._tokenizer_type == MistralTokenizerType.tekken:
            pieces = self.tokenizer.instruct_tokenizer.tokenizer._model.encode(
                piece, allowed_special="all", disallowed_special=set()
            )
            assert len(pieces) == 1, f"Expected to decode 1 token, got {len(pieces)}"
            return pieces[0]
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """

        if isinstance(tokens, str):
            one_token = True
            tokens = [tokens]
        else:
            one_token = False

        ids: list[int] = []
        for token in tokens:
            ids.append(self._piece_to_id(token))

        if one_token:
            return ids[0]
        return ids

    def _text_to_ids(self, text: TextInput, add_special_tokens: bool) -> list[int]:
        """
        Converts a string into a sequence of tokens ids, using the tokenizer.
        """
        tokens_ids = self.tokenizer.instruct_tokenizer.tokenizer.encode(
            text, bos=add_special_tokens, eos=add_special_tokens
        )
        return tokens_ids

    def tokenize(self, text: TextInput, **kwargs) -> list[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Not supported by `MistralCommonTokenizer.tokenize`.
                Will raise an error if used.

        Returns:
            `List[str]`: The list of tokens.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.tokenize`.")

        return self.convert_ids_to_tokens(self._text_to_ids(text, add_special_tokens=False), skip_special_tokens=False)

    def _encode_plus(
        self,
        text: Union[TextInput, EncodedInput],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer._encode_plus`."
            )

        def get_input_ids(text):
            if isinstance(text, str):
                return self._text_to_ids(text, add_special_tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(f"Input {text} is not valid. Should be a string, or a list/tuple of integers.")

        ids = get_input_ids(text)

        return self.prepare_for_model(
            ids,
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
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text: Union[
            list[TextInput],
            list[EncodedInput],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
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
                return self._text_to_ids(text, add_special_tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError("Input is not valid. Should be a string or a list/tuple of integers.")

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        for ids in batch_text:
            input_ids.append(get_input_ids(ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    def _all_special_ids(self) -> set[int]:
        if self._tokenizer_type == MistralTokenizerType.tekken:
            return {t["rank"] for t in self.tokenizer.instruct_tokenizer.tokenizer._all_special_tokens}
        elif self._tokenizer_type == MistralTokenizerType.spm:
            return self.tokenizer.instruct_tokenizer.tokenizer._control_tokens()
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

    def get_special_tokens_mask(
        self, token_ids_0: list, token_ids_1: None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the sequence.
            token_ids_1 (`List[int]`, *optional*):
                Not supported by `MistralCommonTokenizer`. Kept to match the interface of `PreTrainedTokenizerBase`.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if token_ids_1 is not None:
            raise ValueError(
                "`token_ids_1` is not supported by `MistralCommonTokenizer` and should be `None`, kept for compatibility."
            )
        if already_has_special_tokens:
            raise ValueError(
                "`already_has_special_tokens` is not supported by `MistralCommonTokenizer` and should be `False`."
            )

        all_special_ids = self._all_special_ids()  # cache the ids

        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
        return special_tokens_mask

    def _batch_prepare_for_model(
        self,
        batch_ids: list[Union[PreTokenizedInput, list[int]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            batch_ids: list of tokenized input ids
        """

        batch_outputs = {}
        for ids in batch_ids:
            outputs = self.prepare_for_model(
                ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                padding_side=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: None = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence.
            pair_ids (`None`, *optional*):
                Not supported by `MistralCommonTokenizer`. Kept to match the interface of `PreTrainedTokenizerBase`.
        """
        if pair_ids is not None:
            raise ValueError(
                "`pair_ids` is not supported by `MistralCommonTokenizer` and should be `None`, kept for compatibility."
            )
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.prepare_for_model`."
            )

        padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
        )

        len_ids = len(ids)

        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and len_ids > max_length:
            ids, _, overflowing_tokens = self.truncate_sequences(
                ids,
                num_tokens_to_remove=len_ids - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = len_ids - max_length

        # Build output dictionary
        encoded_inputs[self.model_input_names[0]] = ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, None)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(ids)

        # Padding
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

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _get_padding_truncation_strategies(
        self,
        padding: Union[str, PaddingStrategy, bool] = False,
        truncation: Optional[Union[str, TruncationStrategy, bool]] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Find the correct padding/truncation strategy.
        """

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                        " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                        " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                        " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                        " `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
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
            if truncation in [TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND]:
                raise ValueError(
                    "Truncation strategy `only_first` and `only_second` are not supported by `MistralCommonTokenizer`."
                )
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no"
                                " predefined maximum length. Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has"
                                " no predefined maximum length. Default to no truncation."
                            )
                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
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

    def _pad(
        self,
        encoded_inputs: Union[dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
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
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError(f"Invalid padding strategy:{padding_side}")

        return encoded_inputs

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            list[BatchEncoding],
            dict[str, EncodedInput],
            dict[str, list[EncodedInput]],
            list[dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id`).
        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors), see
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
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
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

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
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
                    "Should be one of a python, numpy, pytorch or tensorflow object."
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

    def truncate_sequences(
        self,
        ids: list[int],
        pair_ids: None = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
        **kwargs,
    ) -> tuple[list[int], None, list[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`None`, *optional*):
                Not supported by `MistralCommonTokenizer`. Kept to match the signature of `PreTrainedTokenizerBase.truncate_sequences`.
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
            `Tuple[List[int], None, List[int]]`: The truncated `ids` and the list of
            overflowing tokens. `None` is returned to match Transformers signature.
        """
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.truncate_sequences`."
            )
        if pair_ids:
            raise ValueError("`pair_ids` is not supported by `MistralCommonTokenizer.truncate_sequences`.")

        if num_tokens_to_remove <= 0:
            return (ids, None, [])

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        if truncation_strategy in [TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND]:
            raise ValueError(
                f"Only {TruncationStrategy.LONGEST_FIRST} and {TruncationStrategy.DO_NOT_TRUNCATE} are supported."
            )

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                else:
                    raise ValueError(f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'.")

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                logger.error(error_msg)

        return (ids, None, overflowing_tokens)

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        tools: Optional[list[Union[dict, Callable]]] = None,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[str, list[int], list[str], list[list[int]], BatchEncoding]:
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
                Not supported by `MistralCommonTokenizer.apply_chat_template`.
                Will raise an error if used.

        Returns:
            `Union[str, List[int], List[str], List[List[int]], BatchEncoding]`: A list of token ids representing the tokenized chat so far, including control
            tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.
        """
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.apply_chat_template`."
            )
        if not isinstance(truncation, bool):
            raise ValueError("`truncation` must be a boolean for `apply_chat_template` method.")

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        def _maybe_adapt_message(message: dict[str, Any]) -> None:
            """Adapt message to `mistral-common` format and leave validation to `mistral-common`."""
            if not isinstance(message, dict):
                return
            maybe_list_content: Optional[Union[str, list[dict[str, Union[str, dict[str, Any]]]]]] = message.get(
                "content", None
            )
            if not maybe_list_content or isinstance(maybe_list_content, str):
                return

            normalized_content: list[dict[str, Union[str, dict[str, Any]]]] = []
            for content in maybe_list_content:
                content_type = content.get("type", None)
                if not content_type:
                    continue
                elif content_type == "image":
                    maybe_url: Optional[str] = content.get("url")
                    maybe_path: Optional[str] = content.get("path")
                    maybe_base64: Optional[str] = content.get("base64")
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
                    maybe_url: Optional[str] = content.get("url")
                    maybe_path: Optional[str] = content.get("path")
                    maybe_base64: Optional[str] = content.get("base64")
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

        outputs = []
        images: list[np.ndarray] = []
        audios: list[np.ndarray] = []

        for conversation in conversations:
            messages: list[dict[str, Union[str, list[dict[str, Union[str, dict[str, Any]]]]]]] = []
            for message in conversation:
                _maybe_adapt_message(message)
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
                    pixel_values: Union[list[np.ndarray], np.ndarray, torch.Tensor]
                    if return_tensors == "pt":
                        if not is_torch_available():
                            raise ImportError(
                                "Unable to convert output to PyTorch tensors format, PyTorch is not installed."
                            )

                        pixel_values = torch.tensor(images)
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
                "`MistralCommonTokenizer.apply_chat_template(..., tokenize=False)` is unsafe and may lead to unexpected behavior."
                " Please consider using `tokenize=True` instead and don't encode the output manually."
            )
            return outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, EncodedInput, list[TextInput], list[EncodedInput], None] = None,
        text_pair: None = None,
        text_target: None = None,
        text_pair_target: None = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of int
                (encoded strings).
            text_pair (`None`, *optional*):
                Not supported by `MistralCommonTokenizer`. Kept to match the signature of `PreTrainedTokenizerBase.__call__`.
            text_target (`None`, *optional*):
                Not supported by `MistralCommonTokenizer`. Kept to match the signature of `PreTrainedTokenizerBase.__call__`.
            text_pair_target (`None`, *optional*):
                Not supported by `MistralCommonTokenizer`. Kept to match the signature of `PreTrainedTokenizerBase.__call__`.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.__call__`.")

        if text_pair or text_target or text_pair_target:
            raise ValueError(
                "`text_pair`, `text_target` and `text_pair_target` are not supported by `MistralCommonTokenizer`."
            )

        if return_tensors in ("tf", "jax"):
            raise ValueError(
                "`MistralCommonTokenizer` does not support `return_tensors='tf'` or `return_tensors='jax'`."
            )

        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], (str, int)):
                    # ... list of strings or int
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings or with a list of ints
                    return len(t[0]) == 0 or isinstance(t[0][0], (str, int))
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must be of type `str` (single example), `List[str]` (batch or single encoded example) "
                "or `List[List[int]]` (batch of encoded examples)."
            )

        is_batched = isinstance(text, (list, tuple)) and isinstance(text[0], (str, list, tuple))

        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        if is_batched:
            return self._batch_encode_plus(
                batch_text=text,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
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
                **kwargs,
            )
        else:
            return self._encode_plus(
                text=text,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
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
                **kwargs,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        mode: ValidationMode = ValidationMode.test,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        model_max_length: int = VERY_LARGE_INTEGER,
        padding_side: str = "left",
        truncation_side: str = "right",
        model_input_names: Optional[list[str]] = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        r"""
        Instantiate a `MistralCommonTokenizer` from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing the tokenizer config, for instance saved
                  using the [`MistralCommonTokenizer.tokenization_mistral_common.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
            mode (`ValidationMode`, *optional*, defaults to `ValidationMode.test`):
                Validation mode for the `MistralTokenizer` tokenizer.
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
            model_input_names (`List[string]`, *optional*):
                The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
                `"attention_mask"`). Default value is picked from the class attribute of the same name.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not the model should cleanup the spaces that were added when splitting the input text during the
                tokenization process.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer.from_pretrained`.
                Will raise an error if used.
        """
        if init_inputs:
            raise ValueError("`init_inputs` are not supported by `MistralCommonTokenizer.from_pretrained`.")

        # Handle kwargs and AutoTokenizer case
        if kwargs and not set(kwargs.keys()).issubset({"_from_auto", "trust_remote_code"}):
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.from_pretrained`."
            )

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
            valid_tokenizer_files = []
            tokenizer_file: str

            instruct_versions = list(TokenizerVersion.__members__)
            mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
            sentencepiece_suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

            for path in os.listdir(pretrained_model_name_or_path):
                pathlib_repo_file = Path(path)
                file_name = pathlib_repo_file.name
                suffix = "".join(pathlib_repo_file.suffixes)
                if file_name == "tekken.json":
                    valid_tokenizer_files.append(file_name)
                elif suffix in sentencepiece_suffixes:
                    valid_tokenizer_files.append(file_name)

            if len(valid_tokenizer_files) == 0:
                raise ValueError(f"No tokenizer file found in directory: {pretrained_model_name_or_path}")
            # If there are multiple tokenizer files, we use tekken.json if it exists, otherwise the versioned one.
            if len(valid_tokenizer_files) > 1:
                if "tekken.json" in valid_tokenizer_files:
                    tokenizer_file = "tekken.json"
                else:
                    tokenizer_file = sorted(valid_tokenizer_files)[-1]
                logger.warning(
                    f"Multiple tokenizer files found in directory: {pretrained_model_name_or_path}. Using {tokenizer_file}."
                )
            else:
                tokenizer_file = valid_tokenizer_files[0]

            tokenizer_path = os.path.join(pretrained_model_name_or_path, tokenizer_file)

        return cls(
            tokenizer_path=tokenizer_path,
            mode=mode,
            model_max_length=model_max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_input_names=model_input_names,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike, Path],
        push_to_hub: bool = False,
        token: Optional[Union[str, bool]] = None,
        commit_message: Optional[str] = None,
        repo_id: Optional[str] = None,
        private: Optional[bool] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ) -> tuple[str]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        [`~MistralCommonTokenizer.tokenization_mistral_common.from_pretrained`] class method.

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
            repo_url (`str`, *optional*): The URL to the Git repository to which push to the Hub.
            organization (`str`, *optional*): The name of the organization in which you would like to push your model.
            kwargs (`Dict[str, Any]`, *optional*):
                Not supported by `MistralCommonTokenizer.save_pretrained`.
                Will raise an error if used.

        Returns:
            A tuple of `str`: The files saved.
        """
        # `save_jinja_files`` must be skipped to be able to save from a processor
        kwargs.pop("save_jinja_files", None)
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.save_pretrained`."
            )

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        shutil.copy(self._tokenizer_path, save_directory)

        if push_to_hub:
            repo_id = repo_id or str(save_directory).split(os.path.sep)[-1]
            repo_id = self._create_repo(
                repo_id, token=token, private=private, repo_url=repo_url, organization=organization
            )
            files_timestamps = self._get_files_timestamps(save_directory)

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

        return (str(save_directory / self._tokenizer_path.name),)
