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
"""Tokenizer for PhoneticXeus (IPA CTC tokenizer)."""

from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


class PhoneticXeusTokenizer(Wav2Vec2CTCTokenizer):
    """CTC tokenizer for IPA phone sequences.

    Thin wrapper around [`Wav2Vec2CTCTokenizer`] with defaults matching the PhoneticXeus IPA vocabulary
    (428 tokens). Handles CTC blank collapsing and ID-to-IPA conversion.

    Args:
        vocab_file (`str`):
            Path to `vocab.json` mapping IPA phone strings to integer IDs.
        bos_token (`str`, *optional*, defaults to `"<sos>"`):
            Beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            End of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            Unknown token.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            Padding / CTC blank token.
        word_delimiter_token (`str`, *optional*, defaults to `" "`):
            Token used as word delimiter. Set to `" "` since IPA transcriptions use space between words.
        **kwargs:
            Additional keyword arguments passed to [`Wav2Vec2CTCTokenizer`].
    """

    def __init__(
        self,
        vocab_file,
        bos_token="<sos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<blank>",
        word_delimiter_token=" ",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )


__all__ = ["PhoneticXeusTokenizer"]
