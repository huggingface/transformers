# Copyright 20125 The HuggingFace Inc. team.
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
"""
Utilities for creating fast tokenizers from scratch.
"""

from typing import Optional

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram


def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


class SpmTokenizer:
    """
    Base SentencePiece tokenizer that can be instantiated with model-specific arguments.
    """
    
    def __init__(
        self,
        handle_byte_fallback: bool = True,
        legacy: bool = False,
        add_prefix_space: bool = True,
        special_tokens: Optional[dict] = None,
        vocab: Optional[callable] = None,
        unk_id: Optional[callable] = None,
        normalizer: Optional[callable] = None,
        pre_tokenizer: Optional[callable] = None,
        decoder: Optional[callable] = None,
        post_processor: Optional[callable] = None,
    ):
        self.handle_byte_fallback = handle_byte_fallback
        self.legacy = legacy
        self.add_prefix_space = add_prefix_space
        self.special_tokens = special_tokens or {}
        # Store user-provided callables under private names to avoid clashing with methods
        self._vocab_fn = vocab
        self._unk_id_fn = unk_id
        self._normalizer_fn = normalizer
        self._pre_tokenizer_fn = pre_tokenizer
        self._decoder_fn = decoder
        self._post_processor_fn = post_processor

    def vocab(self):
        if self._vocab_fn is not None:
            return self._vocab_fn()
        # Return empty vocab for training
        return []

    def unk_id(self):
        if self._unk_id_fn is not None:
            return self._unk_id_fn()
        return 0  # Default unk_id

    def tokenizer(self):
        # Always create empty trainable tokenizer
        minimal_vocab = [("<unk>", 0.0)]
        return Tokenizer(Unigram(minimal_vocab, unk_id=self.unk_id(), byte_fallback=self.handle_byte_fallback))

    def normalizer(self):
        if self._normalizer_fn is not None:
            return self._normalizer_fn()
        _normalizers = [
            normalizers.Strip(left=False, right=True),
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        return normalizers.Sequence(_normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        if self._pre_tokenizer_fn is not None:
            return self._pre_tokenizer_fn(replacement, add_prefix_space)
        
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def decoder(self, replacement, add_prefix_space):
        if self._decoder_fn is not None:
            return self._decoder_fn(replacement, add_prefix_space)
        
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self)
        return decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def post_processor(self):
        if self._post_processor_fn is not None:
            return self._post_processor_fn()
        return None

    def create_tokenizer(self) -> Tokenizer:
        """Create and return the configured empty trainable tokenizer."""
        tokenizer = self.tokenizer()

        # Tokenizer assemble
        normalizer = self.normalizer()
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = self.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


__all__ = ["SpmTokenizer", "_get_prepend_scheme"]