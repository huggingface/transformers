# Copyright 2022 The Salesforce authors, The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for CodeGen."""

import re
from typing import TYPE_CHECKING, Union

import numpy as np
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import is_torch_available, logging


if TYPE_CHECKING:
    if is_torch_available():
        import torch


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}


class CodeGenTokenizer(TokenizersBackend):
    """
    Construct a CodeGen tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import CodeGenTokenizer

    >>> tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab (`str` or `dict[str, int]`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.
        merges (`str` or `list[str]`, *optional*):
            Custom merges list. If not provided, merges are loaded from `merges_file`.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (CodeGen tokenizer detect beginning of words by the preceding space).
        return_token_type_ids (`bool`, *optional*, defaults to `False`):
            Whether to return token type IDs.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token=None,
        add_prefix_space: bool = False,
        return_token_type_ids: bool = False,
        **kwargs,
    ):
        self.return_token_type_ids = return_token_type_ids
        if self.return_token_type_ids:
            self.model_input_names.append("token_type_ids")

        self.add_prefix_space = add_prefix_space

        self._vocab = vocab if vocab is not None else {}
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.post_processor = processors.ByteLevel(
            add_prefix_space=True, use_regex=True, trim_offsets=False
        )

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            return_token_type_ids=return_token_type_ids,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, list[int], np.ndarray, "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        truncate_before_pattern: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):
                A list of regular expression strings that will be used to truncate the returned string. This can be
                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning
                of a new line). An example pattern could be `["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """

        decoded_text = super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

        prints = list(re.finditer("^print", completion, re.MULTILINE))

        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        defs = list(re.finditer("^def", completion, re.MULTILINE))

        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]

        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion


__all__ = ["CodeGenTokenizer"]
