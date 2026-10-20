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

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_tokenizers import TokenizersBackend


SENTENCE_END_MARKS = ".!?…"
WEAK_PUNCTUATION = r",;:\-–—"
CLOSING_MARKS = r"\"'”’)\]» "


class PocketTTSTokenizer(TokenizersBackend):
    """
    Construct a Pocket TTS tokenizer (backed by HuggingFace's *tokenizers* library), based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    The text is prepared the way the model was trained to read it (the original's `prepare_text_prompt`), with the
    settings of the checkpoint's language: see [`~PocketTTSTokenizer.normalize_text`]. Every step but the capitalization
    is part of the tokenizer's normalizer. Special tokens are left as they are, so a prompt rendered by the processor's
    chat template keeps its placeholders.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`list[tuple[str, float]]`, *optional*):
            The Unigram vocabulary, as `(piece, score)` pairs. If not provided, a minimal vocabulary is created with the
            special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        normalize (`bool`, *optional*, defaults to `True`):
            Whether to prepare the text before tokenization. Can be overridden per call, e.g. `normalize=False` for the
            transcripts of training samples, which are read as written.
        replace_characters (`dict[str, str]`, *optional*):
            Per-character rewrites (`""` deletes the character), for characters the training text of the language
            never contained.
        remove_semicolons (`bool`, *optional*, defaults to `False`):
            Whether to replace semicolons with commas.
    """

    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: list[tuple[str, float]] | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        normalize: bool = True,
        replace_characters: dict[str, str] | None = None,
        remove_semicolons: bool = False,
        **kwargs,
    ):
        self.normalize = normalize
        self.replace_characters = replace_characters or {}
        self.remove_semicolons = remove_semicolons
        self._tokenizer = self._build_tokenizer(
            vocab or [(str(token), 0.0) for token in (unk_token, bos_token, eos_token, pad_token)]
        )
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            normalize=normalize,
            replace_characters=replace_characters,
            remove_semicolons=remove_semicolons,
            **kwargs,
        )

    def _build_tokenizer(self, vocab: list[tuple[str, float]]) -> Tokenizer:
        tokenizer = Tokenizer(Unigram(vocab, unk_id=0, byte_fallback=True))
        self._text_normalizer = normalizers.Sequence(self._text_normalizers())
        tokenizer.normalizer = self._normalizer(self.normalize)
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        tokenizer.decoder = decoders.Sequence(
            [
                decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )
        return tokenizer

    def _normalizer(self, normalize: bool) -> normalizers.Normalizer:
        """The normalizer of the backend tokenizer: the text preparation, if `normalize`, then the word-start marker."""
        return normalizers.Sequence(
            [self._text_normalizer, normalizers.Prepend("▁")] if normalize else [normalizers.Prepend("▁")]
        )

    def _text_normalizers(self) -> list[normalizers.Normalizer]:
        """The steps of the text preparation a normalizer can do: everything but the capitalization."""
        steps = [normalizers.Strip()]
        if self.replace_characters:
            steps += [normalizers.Replace(source, target) for source, target in self.replace_characters.items()]
            # deleted quotes leave '"Hi?", she said' as 'Hi?, she said': keep the sentence mark only
            steps += [
                normalizers.Replace(Regex(r"\s+"), " "),
                normalizers.Strip(),
                normalizers.Replace(Regex(rf"(?<=[{SENTENCE_END_MARKS}])\s*[,;:]"), ""),
            ]
        steps += [normalizers.Replace("\n", " "), normalizers.Replace("\r", " "), normalizers.Replace("  ", " ")]
        if self.remove_semicolons:
            steps.append(normalizers.Replace(";", ","))
        # The text ends with sentence-final punctuation, followed by its closing marks: a trailing comma, colon or dash
        # becomes a period, any other ending gets one (after up to 8 closing marks).
        last_char = rf"[^{SENTENCE_END_MARKS}{WEAK_PUNCTUATION}{CLOSING_MARKS}]"
        steps += [
            normalizers.Replace(Regex(rf"[{WEAK_PUNCTUATION} ]*[{WEAK_PUNCTUATION}] *(?=[{CLOSING_MARKS}]*$)"), "."),
            normalizers.Replace(
                Regex("(?<=" + "|".join(f"{last_char}[{CLOSING_MARKS}]{{{n}}}" for n in range(9)) + ")$"), "."
            ),
        ]
        return steps

    def _preprocess_text(self, text: str) -> str:
        """The steps of the text preparation the normalizer cannot do: upper-casing the first character it keeps."""
        for index, char in enumerate(text):
            if char.isspace() or self.replace_characters.get(char) == "":
                continue
            if char not in self.replace_characters:
                text = text[:index] + char.upper() + text[index + 1 :]
            break
        return text

    def normalize_text(self, text: str) -> str:
        r"""
        Prepares `text` the way the original's `prepare_text_prompt` does, with this tokenizer's settings: strip, apply
        `replace_characters` (then collapse the whitespace, and drop a comma, semicolon or colon right after a sentence
        mark), turn line breaks into spaces, replace semicolons with commas (`remove_semicolons`), upper-case the first
        character and end the text with sentence-final punctuation.

        Args:
            text (`str`): The text to prepare.

        Returns:
            `str`: The prepared text.
        """
        return self._text_normalizer.normalize_str(self._preprocess_text(text))

    def _preprocess_prompt(self, text: str) -> str:
        """Applies `_preprocess_text` to the text between the special tokens of `text`."""
        special_tokens = sorted(self.added_tokens_encoder, key=len, reverse=True)
        pieces = re.split("(" + "|".join(map(re.escape, special_tokens)) + ")", text) if special_tokens else [text]
        return "".join(
            piece if piece in self.added_tokens_encoder else self._preprocess_text(piece) for piece in pieces
        )

    def _encode_plus(
        self, text, text_pair=None, is_split_into_words: bool = False, normalize: bool | None = None, **kwargs
    ) -> BatchEncoding:
        normalize = self.normalize if normalize is None else normalize
        if normalize and not is_split_into_words:
            text = (
                self._preprocess_prompt(text) if isinstance(text, str) else [self._preprocess_prompt(t) for t in text]
            )
        # the normalizer of this call, the tokenizer's restored after it
        self._tokenizer.normalizer = self._normalizer(normalize)
        try:
            return super()._encode_plus(text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs)
        finally:
            self._tokenizer.normalizer = self._normalizer(self.normalize)


__all__ = ["PocketTTSTokenizer"]
