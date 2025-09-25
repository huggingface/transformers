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
"""VibeVoice model."""

from ...utils import auto_docstring

from ..qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast



# TODO update rest of generated docstrings
@auto_docstring(
    custom_intro="""
    VibeVoice tokenizer, which is based on the Qwen2's fast tokenizer with additional special tokens for speech.
    """
)
class VibeVoiceTokenizer(Qwen2TokenizerFast):
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        speech_start_token="<|vision_start|>",
        speech_end_token="<|vision_end|>",
        speech_diffusion_token="<|vision_pad|>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens (using vision tokens as model was trained on them)
        vibevoice_special_tokens = [
            speech_start_token,
            speech_end_token, 
            speech_diffusion_token,
        ]
        
        # Add special tokens to the tokenizer
        special_tokens_dict = {"additional_special_tokens": vibevoice_special_tokens}
        self.add_special_tokens(special_tokens_dict)
        
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


__all__ = ["VibeVoiceTokenizerFast"]
