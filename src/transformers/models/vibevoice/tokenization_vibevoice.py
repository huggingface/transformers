# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for VibeVoice."""

from ...utils import logging
from ..qwen2.tokenization_qwen2 import Qwen2Tokenizer

logger = logging.get_logger(__name__)


class VibeVoiceTextTokenizer(Qwen2Tokenizer):
    """
    Construct a VibeVoice tokenizer. Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
        add_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether or not to add special tokens when encoding.
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
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens)
                "<|vision_end|>",  # Speech end
                "<|vision_pad|>",  # Speech diffusion pad
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")
        
        self._eos_id = self.convert_tokens_to_ids('<|endoftext|>')

        return num_added
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Id used for padding (returns -100 for loss masking)."""
        return -100


__all__ = ["VibeVoiceTextTokenizer"]