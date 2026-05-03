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
"""Tokenization classes for the Nandi family."""

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class NandiTokenizer(TokenizersBackend):
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
        self._vocab = (
            vocab
            if vocab is not None
            else {
                "<|endoftext|>": 0,
            }
        )
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
                    use_regex=False
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

    def __call__(self, text, *args, **kwargs):
        add_special_tokens = kwargs.get("add_special_tokens", False)
    
        def add_prefix(t):
            if isinstance(t, str):
                return "<|im_start|> " + t
            return t
    
        # Only inject when special tokens are disabled
        if not add_special_tokens:
            if isinstance(text, list):
                text = [add_prefix(t) for t in text]
            else:
                text = add_prefix(text)
    
        return super().__call__(text, *args, **kwargs)

__all__ = ["NandiTokenizer"]
