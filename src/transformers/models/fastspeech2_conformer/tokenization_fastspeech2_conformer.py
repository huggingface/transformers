# coding=utf-8
# Copyright 2023 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for FastSpeech2Conformer."""
import json
import os
from typing import Optional, Tuple

from ...file_utils import requires_backends
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "jaketae/fastspeech2-ljspeech": "https://huggingface.co/jaketae/fastspeech2-ljspeech/resolve/main/vocab.txt",
        "jaketae/fastspeech2-commonvoice": "https://huggingface.co/jaketae/fastspeech2-commonvoice/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "jaketae/fastspeech2-ljspeech": 1024,
    "jaketae/fastspeech2-commonvoice": 1024,
}


class FastSpeech2ConformerTokenizer(PreTrainedTokenizer):
    """
    Construct a FastSpeech2Conformer tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        do_phonemize=True,
        preserve_punctuation=False,
        **kwargs,
    ):
        requires_backends(self, "g2p_en")
        import g2p_en

        # make attribute to avoid creating g2p object every call
        self.g2p = g2p_en.G2p()
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token, **kwargs)
        self.do_phonemize = do_phonemize
        self.preserve_punctuation = preserve_punctuation
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.decoder)

    def get_vocab(self):
        "Returns vocab as a dict"
        return dict(self.encoder, **self.added_tokens_encoder)

    def phonemize(self, text):
        if self.preserve_punctuation:
            return " ".join("|" if p == " " else p for p in self.g2p(text))
        res = [{",": "sp", ";": "sp"}.get(p, p) for p in self.g2p(text)]
        return " ".join(p for p in res if p.isalnum())

    def _tokenize(self, text):
        """Returns a tokenized string."""

        # make sure whitespace is stripped to prevent <unk>
        text = text.strip()

        # phonemize
        if self.do_phonemize:
            text = self.phonemize(text)

        # make sure ' ' is between phonemes
        tokens = text.split(" ")

        tokens = list(filter(lambda p: p.strip() != "", tokens))

        # fastspeech2 needs eos at the end
        tokens.append(self.eos_token)

        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.encoder[self.unk_token])

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # TODO
        raise NotImplementedError

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
