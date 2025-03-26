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

import json
import os
from typing import Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

LANGUAGE_CODES = {
    "en-us": "American English",
    "en-gb": "British English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "pt-br": "Portuguese (Brazil)",
    "ja": "Japanese",
    "zh": "Mandarin Chinese"
}


class StyleTextToSpeech2Tokenizer(PreTrainedTokenizer):
    """
    Construct a StyleTextToSpeech2 tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The begin of sequence token. Note that for FastSpeech2, it is the same as the `eos_token`.
        eos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The end of sequence token. Note that for FastSpeech2, it is the same as the `bos_token`.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        transformer_based_g2p (`bool`, *optional*, defaults to `False`):
            Whether to use a transformer-based G2P model.
        language_code (`str`, *optional*, defaults to `"en-us"`):
            The language code to use for the tokenizer.
        split_text_on (`List[str]`, *optional*, defaults to `['!.?…', ':;', ',—',]`):
            The list of strings to split the text on by order of priority.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="$",
        eos_token="$",
        pad_token="$",
        unk_token="$",
        transformer_based_g2p=False,
        language_code="en-us",
        split_text_on=['!.?…', ':;', ',—',],
        **kwargs,
    ):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        self.language_code = language_code
        self.transformer_based_g2p = transformer_based_g2p
        self._set_g2p(language_code, transformer_based_g2p)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.split_text_on = split_text_on

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )
    
    def _set_g2p(self, language_code, transformer_based_g2p):
        if language_code not in LANGUAGE_CODES:
            raise ValueError(f"Language code {language_code} not supported. Supported languages: {LANGUAGE_CODES.keys()}")

        elif language_code in ["en-us", "en-gb"]:
            is_british = language_code == "en-gb"
            try:
                from misaki import en, espeak
            except ImportError:
                logger.warning(
                    "You need to install misaki to use StyleTextToSpeech2Tokenizer: https://github.com/hexgrad/misaki"
                    "`pip install misaki`"
                )
                raise
            try:
                fallback = espeak.EspeakFallback(british=is_british)
            except Exception as e:
                logger.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logger.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(trf=transformer_based_g2p, british=is_british, fallback=fallback, unk='')

        elif language_code == 'ja':
            try:
                from misaki import ja
                self.g2p = ja.JAG2P()
            except ImportError:
                logger.warning(
                    "You need to install misaki[ja] to use StyleTextToSpeech2Tokenizer with language_code='ja': https://github.com/hexgrad/misaki"
                    "`pip install misaki[ja]`"
                )
                raise

        elif language_code == 'zh':
            try:
                from misaki import zh
                self.g2p = zh.ZHG2P()
            except ImportError:
                logger.warning(
                    "You need to install misaki[zh] to use StyleTextToSpeech2Tokenizer with language_code='zh': https://github.com/hexgrad/misaki"
                    "`pip install misaki[zh]`"
                )
                raise

        else:
            language = LANGUAGE_CODES[language_code]
            logger.warning(
                f"Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'."
            )
            self.g2p = espeak.EspeakG2P(language=language)

    @property
    def vocab_size(self):
        return len(self.decoder)

    def get_vocab(self):
        "Returns vocab as a dict"
        return dict(self.encoder, **self.added_tokens_encoder)

    def prepare_text(self, text):
        """
        Prepare text for tokenization by splitting it into manageable chunks when the tokenized length exceeds the model's maximum length.
        The text is split on the list of strings in `split_text_on` by order of priority. 
        If no split is found, the text is split at the model's maximum length corresponding word.

        Args:
            text (`str` or `List[str]`):
                The text to prepare.
        
        Returns:
            `List[str]`: A list of prepared texts.
        """
        is_batched = isinstance(text, (list, tuple))
        if not is_batched:
            text = [text]

        prepared_texts = []
        for text_item in text:
            _, m_tokens = self.g2p(text_item)

            split_idxs = []
            start_idx, current_idx = 0, 0
            current_n_phonemes = 0

            for current_idx, m_tok in enumerate(m_tokens):
                n_next_phonemes = len(m_tok.phonemes if m_tok.phonemes is not None else '')

                if n_next_phonemes + current_n_phonemes <= self.model_max_length:
                    current_n_phonemes += (n_next_phonemes + 1 if m_tok.whitespace else n_next_phonemes)
                else:
                    for split_text_on_item in self.split_text_on:
                        idx = next((i for i, t in reversed(list(enumerate(m_tokens[start_idx: current_idx]))) if t.phonemes in set(split_text_on_item)), None)
                        if idx is not None:
                            idx += start_idx
                            break

                    if idx is None:
                        end_idx = current_idx
                    else:
                        end_idx = idx + 1
                    split_idxs.append((start_idx, end_idx))

                    start_idx = end_idx
                    current_n_phonemes = sum((len(t.phonemes) + 1) if t.whitespace else len(t.phonemes) for t in m_tokens[end_idx:current_idx + 1])
        
            # include last split
            split_idxs.append((start_idx, len(m_tokens)))

            for start_idx, end_idx in split_idxs:
                prepared_texts.append(
                    ''.join(t.text + (' ' if t.whitespace else '') for t in m_tokens[start_idx:end_idx]).strip()
                )

        return prepared_texts

    def _phonemize(self, text):
        _, m_tokens = self.g2p(text)
        for t in m_tokens:
            t.phonemes = '' if t.phonemes is None else t.phonemes.replace('ɾ', 'T')
        phonemes = ''.join(t.phonemes + (' ' if t.whitespace else '') for t in m_tokens).strip()
        return phonemes
        
    def _tokenize(self, text):
        """Returns a tokenized string."""
        # phonemize
        phonemes = self._phonemize(text)
        tokens = list(phonemes)

        if len(tokens) > self.model_max_length:
            raise ValueError(
                f"Tokenized input text exceeds the model's maximum length of {self.model_max_length} tokens. "
                "Please use the `prepare_text` method of StyleTextToSpeech2Tokenizer to automatically split your text into manageable chunks."
            )
        tokens = [self.bos_token] + tokens + [self.eos_token]
            
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # Override since phonemes cannot be converted back to strings
    def decode(self, token_ids, **kwargs):
        logger.warning(
            "Phonemes cannot be reliably converted to a string due to the one-many mapping, converting to tokens instead."
        )
        return self.convert_ids_to_tokens(token_ids)

    # Override since phonemes cannot be converted back to strings
    def convert_tokens_to_string(self, tokens, **kwargs):
        logger.warning(
            "Phonemes cannot be reliably converted to a string due to the one-many mapping, returning the tokens."
        )
        return tokens

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False))

        return (vocab_file,)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["g2p"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self._set_g2p(self.language_code, self.transformer_based_g2p)


__all__ = ["StyleTextToSpeech2Tokenizer"]
