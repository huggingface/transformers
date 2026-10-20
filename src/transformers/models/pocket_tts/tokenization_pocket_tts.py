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

import re

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_tokenizers import TokenizersBackend


class PocketTTSTokenizer(TokenizersBackend):
    """
    Construct a Pocket TTS tokenizer (backed by HuggingFace's *tokenizers* library), loaded from the checkpoint's
    `tokenizer.json`, whose normalizer prepares the text the way the model was trained to read it. The one step a
    normalizer cannot do, upper-casing the first character of the prepared text, is done here. Special tokens are left
    as they are, so a prompt rendered by the processor's chat template keeps its placeholders.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """

    def _prepare_text(self, text: str | list | tuple) -> str | list | tuple:
        """Normalizes the text between the special tokens of `text`, and upper-cases its first character."""
        if not isinstance(text, str):
            # a batch, or a pair of texts
            return type(text)(self._prepare_text(item) for item in text)
        special_tokens = sorted(self.added_tokens_encoder, key=len, reverse=True)
        pieces = re.split("(" + "|".join(map(re.escape, special_tokens)) + ")", text) if special_tokens else [text]
        prepared = []
        for piece in pieces:
            if piece not in self.added_tokens_encoder:
                piece = self.backend_tokenizer.normalizer.normalize_str(piece)
                # after the word-start marker the normalizer may prepend
                start = len(piece) - len(piece.lstrip("▁"))
                piece = piece[:start] + piece[start : start + 1].upper() + piece[start + 1 :]
            prepared.append(piece)
        return "".join(prepared)

    def _encode_plus(self, text, text_pair=None, is_split_into_words: bool = False, **kwargs) -> BatchEncoding:
        if is_split_into_words:
            return super()._encode_plus(text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs)
        text = self._prepare_text(text)
        text_pair = self._prepare_text(text_pair) if text_pair is not None else None
        # the text is normalized already: the backend tokenizer encodes it without its normalizer
        normalizer, self.backend_tokenizer.normalizer = self.backend_tokenizer.normalizer, None
        try:
            return super()._encode_plus(text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs)
        finally:
            self.backend_tokenizer.normalizer = normalizer


__all__ = ["PocketTTSTokenizer"]
