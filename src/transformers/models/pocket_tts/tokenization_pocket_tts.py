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

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_tokenizers import TokenizersBackend


SENTENCE_END_MARKS = ".!?…"
WEAK_PUNCTUATION = ",;:-–—"
CLOSING_MARKS = "\"'”’)]»"


class PocketTTSTokenizer(TokenizersBackend):
    """
    Construct a Pocket TTS tokenizer (backed by HuggingFace's *tokenizers* library), based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    Before tokenization, the text is prepared the way the model was trained to read it (the original's
    `prepare_text_prompt`), with the settings of the checkpoint's language: see [`~PocketTTSTokenizer.normalize_text`].
    Only the text between special tokens is prepared, so a prompt rendered by the processor's chat template keeps its
    placeholders.

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
            transcripts of training samples, which the original reads as written.
        replace_characters (`dict[str, str]`, *optional*):
            Per-character rewrites (`""` deletes the character), for characters the training text of the language
            never contained.
        remove_semicolons (`bool`, *optional*, defaults to `False`):
            Whether to replace semicolons with commas.
        capitalize_first_letter (`bool`, *optional*, defaults to `True`):
            Whether to upper-case the first character. A model whose text is romanised phonemes must switch it off.
        append_terminal_punctuation (`bool`, *optional*, defaults to `True`):
            Whether to end the text with sentence-final punctuation, as the model was trained on sentences ending with
            one.
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
        capitalize_first_letter: bool = True,
        append_terminal_punctuation: bool = True,
        **kwargs,
    ):
        self.normalize = normalize
        self.replace_characters = replace_characters or {}
        self.remove_semicolons = remove_semicolons
        self.capitalize_first_letter = capitalize_first_letter
        self.append_terminal_punctuation = append_terminal_punctuation

        if vocab is None:
            vocab = [(str(unk_token), 0.0), (str(bos_token), 0.0), (str(eos_token), 0.0), (str(pad_token), 0.0)]
        self._tokenizer = Tokenizer(Unigram(vocab, unk_id=0, byte_fallback=True))
        self._tokenizer.normalizer = normalizers.Prepend("▁")
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        self._tokenizer.decoder = decoders.Sequence(
            [
                decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            normalize=normalize,
            replace_characters=replace_characters,
            remove_semicolons=remove_semicolons,
            capitalize_first_letter=capitalize_first_letter,
            append_terminal_punctuation=append_terminal_punctuation,
            **kwargs,
        )

    def normalize_text(self, text: str) -> str:
        r"""
        Prepares `text` the way the original's `prepare_text_prompt` does, with this tokenizer's settings: strip, apply
        `replace_characters` (then collapse the whitespace, and drop a comma, semicolon or colon right after a sentence
        mark), turn line breaks into spaces, replace semicolons with commas (`remove_semicolons`), upper-case the first
        character (`capitalize_first_letter`) and end the text with sentence-final punctuation
        (`append_terminal_punctuation`).

        Args:
            text (`str`): The text to prepare.

        Returns:
            `str`: The prepared text.
        """
        text = text.strip()
        if self.replace_characters:
            text = " ".join(text.translate(str.maketrans(self.replace_characters)).split())
            # deleted quotes leave '"Hi?", she said' as 'Hi?, she said': keep the sentence mark only
            text = re.sub(rf"([{SENTENCE_END_MARKS}])\s*[,;:]", r"\1", text)
        if not text:
            raise ValueError("Text prompt cannot be empty")
        text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        if self.remove_semicolons:
            text = text.replace(";", ",")
        if self.capitalize_first_letter and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if self.append_terminal_punctuation:
            # a trailing comma, colon or dash becomes a period; any other ending gets one, after the closing marks
            core = text.rstrip(CLOSING_MARKS + " ")
            if core and core[-1] not in SENTENCE_END_MARKS:
                if core[-1] in WEAK_PUNCTUATION:
                    text = core.rstrip(WEAK_PUNCTUATION + " ") + "." + text[len(core) :].strip()
                else:
                    text = text + "."
        return text

    def _normalize_prompt(self, text: str) -> str:
        """Prepares the text between the special tokens of `text`, leaving the special tokens as they are."""
        special_tokens = sorted(self.added_tokens_encoder, key=len, reverse=True)
        pieces = re.split("(" + "|".join(map(re.escape, special_tokens)) + ")", text) if special_tokens else [text]
        return "".join(
            piece if piece in self.added_tokens_encoder or not piece.strip() else self.normalize_text(piece)
            for piece in pieces
        )

    def _encode_plus(
        self, text, text_pair=None, is_split_into_words: bool = False, normalize: bool | None = None, **kwargs
    ) -> BatchEncoding:
        normalize = self.normalize if normalize is None else normalize
        if normalize and not is_split_into_words:
            if isinstance(text, str):
                text = self._normalize_prompt(text)
            elif isinstance(text, (list, tuple)):
                text = [self._normalize_prompt(prompt) for prompt in text]
        return super()._encode_plus(text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs)


__all__ = ["PocketTTSTokenizer"]
