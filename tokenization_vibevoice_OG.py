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


class VibeVoiceTokenizer(Qwen2Tokenizer):
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
        speech_start_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            The token marking the start of speech content (reuses vision token from training).
        speech_end_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            The token marking the end of speech content (reuses vision token from training).
        speech_diffusion_token (`str`, *optional*, defaults to `"<|vision_pad|>"`):
            The token used for speech diffusion/generation placeholders (reuses vision token from training).
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
        speech_start_token="<|vision_start|>",
        speech_end_token="<|vision_end|>",
        speech_diffusion_token="<|vision_pad|>",
        **kwargs,
    ):
        # Define VibeVoice-specific special tokens (using vision tokens as model was trained on them)
        vibevoice_special_tokens = [
            speech_start_token,
            speech_end_token, 
            speech_diffusion_token,
        ]
        
        # Add to additional_special_tokens if provided in kwargs
        additional_special_tokens = kwargs.pop("additional_special_tokens", [])
        additional_special_tokens.extend(vibevoice_special_tokens)
        
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
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        
        # Store token strings for property access
        self._speech_start_token = speech_start_token
        self._speech_end_token = speech_end_token
        self._speech_diffusion_token = speech_diffusion_token

    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self.convert_tokens_to_ids(self._speech_start_token)

    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self.convert_tokens_to_ids(self._speech_end_token)

    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self.convert_tokens_to_ids(self._speech_diffusion_token)


__all__ = ["VibeVoiceTokenizer"]
