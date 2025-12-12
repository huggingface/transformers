# coding=utf-8
# Copyright 2024 Microsoft and The HuggingFace Inc.
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
"""Tokenization classes for VibeVoice."""

from ...utils import logging
from ..qwen2.tokenization_qwen2 import Qwen2Tokenizer


# ---- Optional fast tokenizer import (must be protected for CI safety) ----
try:
    from ..qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
except Exception:
    Qwen2TokenizerFast = None
# -------------------------------------------------------------------------

logger = logging.get_logger(__name__)

__all__ = ["VibeVoiceTokenizer", "VibeVoiceTokenizerFast"]


class VibeVoiceTokenizer(Qwen2Tokenizer):
    """
    Construct a VibeVoice tokenizer. Based on the Qwen2 tokenizer with
    additional special tokens for speech.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        add_special_tokens=True,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        self._add_vibevoice_special_tokens()

    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",
                "<|vision_end|>",
                "<|vision_pad|>",
            ]
        }
        self.add_special_tokens(special_tokens)

        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")
        self._eos_id = self.convert_tokens_to_ids("<|endoftext|>")

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def speech_start_id(self) -> int:
        return self._speech_start_id

    @property
    def speech_end_id(self) -> int:
        return self._speech_end_id

    @property
    def speech_diffusion_id(self) -> int:
        return self._speech_diffusion_id

    @property
    def pad_id(self) -> int:
        return -100


class VibeVoiceTokenizerFast(Qwen2TokenizerFast if Qwen2TokenizerFast is not None else object):
    """
    Construct a fast VibeVoice tokenizer (optional).
    If Qwen2TokenizerFast is unavailable (e.g., in CI environments),
    this class still loads gracefully without breaking imports.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        add_special_tokens=True,
        **kwargs,
    ):
        if Qwen2TokenizerFast is None:
            raise ImportError(
                "Fast tokenizer backend is not available. Install with: pip install transformers[tokenizers]"
            )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        self._add_vibevoice_special_tokens()

    def _add_vibevoice_special_tokens(self):
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",
                "<|vision_end|>",
                "<|vision_pad|>",
            ]
        }
        self.add_special_tokens(special_tokens)

        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")
        self._eos_id = self.convert_tokens_to_ids("<|endoftext|>")

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def speech_start_id(self) -> int:
        return self._speech_start_id

    @property
    def speech_end_id(self) -> int:
        return self._speech_end_id

    @property
    def speech_diffusion_id(self) -> int:
        return self._speech_diffusion_id

    @property
    def pad_id(self) -> int:
        return -100
