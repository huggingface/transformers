# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Lumma."""

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

# GPT-4-style split regex used by the Lumma training pipeline (matches the Hub remote tokenizer).
PRETOKENIZE_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*)+|\p{N}| """
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


class LummaTokenizer(TokenizersBackend):
    """
    Construct a Lumma tokenizer. Based on byte-level BPE with a GPT-4-style pretokenizer.

    Special tokens:
    - `<|im_start|>` (BOS, id 1)
    - `<|endoftext|>` (EOS/UNK, id 0)
    - `<|pad|>` (pad, id 3)

    By default, plain text inputs are prefixed with `<|im_start|> ` to match the model's training format.

    This tokenizer inherits from [`TokenizersBackend`]. At load time, `tokenizer.json` from the Hub checkpoint
    is used when available.

    Args:
        vocab (`str` or `dict[str, int]`, *optional*):
            Custom vocabulary dictionary used when building the tokenizer from scratch.
        merges (`str` or `list[str]`, *optional*):
            Custom BPE merges used when building the tokenizer from scratch.
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*, defaults to `"<|im_start|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|pad|>"`):
            The padding token.
        add_prefix_space (`bool`, *optional*):
            Whether to add a prefix space for byte-level pretokenization.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        vocab_file=None,
        merges_file=None,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|im_start|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|pad|>",
        add_prefix_space: bool | None = None,
        **kwargs,
    ):
        self._vocab = vocab if vocab is not None else {"<|endoftext|>": 0}
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.normalizer = normalizers.NFC()
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(PRETOKENIZE_REGEX),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=True,
                    use_regex=False,
                ),
            ]
        )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @staticmethod
    def _prepend_im_start(text):
        if isinstance(text, str):
            return "<|im_start|> " + text
        return text

    def __call__(self, text, *args, **kwargs):
        # Match training: prefix user text with <|im_start|> when special tokens are not added by the caller.
        if not kwargs.get("add_special_tokens", False):
            if isinstance(text, list):
                text = [self._prepend_im_start(t) for t in text]
            else:
                text = self._prepend_im_start(text)
        return super().__call__(text, *args, **kwargs)

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens: bool = True,
        padding=False,
        truncation=None,
        max_length=None,
        stride: int = 0,
        padding_side=None,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(text, str):
            text = "<|im_start|> " + text
        return super().encode(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )


__all__ = ["LummaTokenizer"]
