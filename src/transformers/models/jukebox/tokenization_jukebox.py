# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI Jukebox."""


import json
import os
from typing import Any, Dict, Optional, Tuple

import regex as re
from unidecode import unidecode

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "jukebox": "https://huggingface.co/jukebox/resolve/main/vocab.json",
    },
    "vocab_file": {
        "jukebox-1b": "https://huggingface.co/jukebox/resolve/main/vocab-1b.json",
    },
}

PRETRAINED_LYRIC_TOKENS_SIZES = {
    "jukebox": 12,  # corresonds to the dummy-model ?
    "jukebox-1b": 384,
}


class JukeboxTokenizer(PreTrainedTokenizer):
    """
    Constructs a Jukebox tokenizer. Jukebox can be conditioned on 3 different inputs :
        - Artists, unique ids are associated to each artist from the provided dictionary.
        - Genres, unique ids are associated to each genre from the provided dictionary.
        - Lyrics, character based tokenization. Must be initialized with the list of characters that are inside the
        vocabulary.

    This tokenizer is straight forward and does not require trainingg. It should be able to process a different number of inputs:
    as the conditioning of the model can be done on the three different queries. If None is provided, defaults values will be used.:

    Depending on the number of genres on which the model should be conditioned (`n_genres`).
    ```
    >>> from transformers import JukeboxTokenizer
    >>> tokenizer = JukeboxTokenizer.from_pretrained("jukebox")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")['input_ids']
    [6785],[546], [0, ...,   41,       38,       30,
                    77,       46,       41,       49,       40,
                    77,       44,       41,       27,       30] ]
    >>> tokenizer("Alan Jackson", "Country Rock")['input_ids']
    [6785],[546]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    If nothing is provided, the genres and the artist will either be selected randomly or set to None

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to:
    this superclass for more information regarding those methods.

    # TODO: the original paper should support composing from 2 or more artists and genres.
    However the code does not allow that and only supports composing from various genres.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file which should contain a dictionnary where the keys are 'artist', 'genre' and
            'lyrics' and the values are their corresponding vocabulary files.
        max_n_lyric_tokens (`int`, `optional`, defaults to 512):
            Maximum number of lyric tokens to keep.
        n_genres (`int`, `optional`, defaults to 1):
            Maximum number of genres to use for composition.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_lyric_input_size = PRETRAINED_LYRIC_TOKENS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, max_n_lyric_tokens=512, n_genres=1, unk_token="<|endoftext|>", **kwargs):
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        super().__init__(
            unk_token=unk_token,
            **kwargs,
        )
        self.max_n_lyric_tokens = max_n_lyric_tokens
        self.n_genres = n_genres

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            vocabulary = json.load(vocab_handle)
            self.artist_encoder = vocabulary["artist"]
            self.genre_encoder = vocabulary["genre"]
            self.lyrics_encoder = vocabulary["lyrics"]

        self.out_of_vocab = re.compile("[^A-Za-z0-9.,:;!?\-+'\"()\[\] \t\n]+")  # FIXME: should be an argument?

        self.artist_decoder = {v: k for k, v in self.artist_encoder.items()}
        self.genre_decoder = {v: k for k, v in self.genre_encoder.items()}
        self.lyrics_decoder = {v: k for k, v in self.lyrics_encoder.items()}

    @property
    def vocab_size(self):
        return len(self.artist_encoder) + len(self.genre_encoder) + len(self.lyrics_encoder)

    def get_vocab(self):
        return dict(self.artist_encoder, self.genre_encoder, self.lyrics_encoder)

    def get_relevant_lyric_tokens(self, full_tokens, total_length, offset, duration):
        """Extract only the relevant tokens based on the character position. A total of
        `max_n_lyric_tokens` tokens will be returned. If the provided token sequence is smaller, it will be padded,
        othewise, only characters ranging from the midpoint - `max_n_lyric_tokens//2` to the midpoint +
        `max_n_lyric_tokens//2` will be returned. This *focuses* on the most relevant tokens (in time) for the
        sequence.

        Args: # TODO : args to prettify
            full_tokens (`_type_`):
                _description_
            total_length (`_type_`):
                _description_
            offset (`_type_`):
                _description_
            duration (`_type_`):
                _description_
        """
        if len(full_tokens) < self.max_n_lyric_tokens:
            tokens = [0] * (self.max_n_lyric_tokens - len(full_tokens)) + full_tokens
            indices = [-1] * (self.max_n_lyric_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
        else:
            assert 0 <= offset < total_length
            midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
            midpoint = min(
                max(midpoint, self.max_n_lyric_tokens // 2), len(full_tokens) - self.max_n_lyric_tokens // 2
            )
            tokens = full_tokens[midpoint - self.max_n_lyric_tokens // 2 : midpoint + self.max_n_lyric_tokens // 2]
            indices = list(range(midpoint - self.max_n_lyric_tokens // 2, midpoint + self.max_n_lyric_tokens // 2))
        assert len(tokens) == self.max_n_lyric_tokens, f"Expected length {self.max_n_lyric_tokens}, got {len(tokens)}"
        assert (
            len(indices) == self.max_n_lyric_tokens
        ), f"Expected length {self.max_n_lyric_tokens}, got {len(indices)}"
        assert tokens == [full_tokens[index] if index != -1 else 0 for index in indices]
        return tokens, indices

    def _convert_token_to_id(self, artist, genre, lyrics, total_length, offset, duration):
        """Converts the artist, genre and lyrics tokens to their index using the vocabulary.
        The total_length, offset and duration have to be provided in order to select relevant lyrics and add padding to
        the lyrics token sequence.

        Args:
            artist (`_type_`):
                _description_
            genre (`_type_`):
                _description_
            lyrics (`_type_`):
                _description_
            total_length (`_type_`):
                _description_
            offset (`_type_`):
                _description_
            duration (`_type_`):
                _description_
        """
        artist_id = self.artist_encoder.get(artist)
        genre_id = self.genre_encoder.get(genre)
        lyrics
        lyric_ids = [self.genre_encoder.get(character) for character in lyrics]
        lyric_ids = self.get_relevant_lyric_tokens(lyric_ids, total_length, offset, duration)
        return artist_id, genre_id, lyric_ids

    def _tokenize(self, lyrics, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens. Only the lytrics are split into character for the character-based vocabulary.
        """
        # only lyrics are not tokenized, but character based is easily handled
        return [character for character in lyrics]

    def tokenize(self, artist, genre, lyrics, total_length, offset, duration):
        """
        Converts three strings in a 3 sequence of tokens using the tokenizer

        Args:
            artist (`_type_`):
                _description_
            genre (`_type_`):
                _description_
            lyrics (`_type_`):
                _description_
            total_length (`_type_`):
                _description_
            offset (`_type_`):
                _description_
            duration (`_type_`):
                _description_
        """
        artist, genre, lyrics, kwargs = self.prepare_for_tokenization(artist, genre, lyrics, **kwargs)
        # TODO deal with the kwargs here
        lyrics = self._tokenize(lyrics, **kwargs)
        return artist, genre, lyrics

    def _normalize(self, text: str) -> str:
        """Normalizes the input text. This process is for the genres and the artit

        Args:
            text (`str`):
                Artist or Genre string to normalize
        """
        import re

        accepted = frozenset(
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
        )

        rex = re.compile(r"_+")
        text = "".join([c if c in accepted else "_" for c in text.lower()])
        text = rex.sub("_", text).strip("_")
        return text

    def prepare_for_tokenization(
        self, artist: str, genre: str, lyrics: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            artist (`str`):
                The artist name to prepare. This will mostly lower the string
            genre (`str`):
                The gnere name to prepare. This will mostly lower the string.
            lyrics (`str`):
                The lyrics to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs:
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        artist = self._normalize(artist)
        genre = self._normalize(genre)

        lyrics = unidecode(lyrics)
        lyrics = lyrics.replace("\\", "\n")
        lyrics = self.out_of_vocab.sub("", lyrics)
        return (artist, genre, lyrics, kwargs)

    def _convert_id_to_token(self, artist_index, genre_index, lyric_index):
        """Converts an index (integer) in a token (str) using the vocab.
        Args:
            artist_index (`_type_`):
                _description_
            genre_index (`_type_`):
                _description_
            lyric_index (`_type_`):
                _description_
        """
        artist = self.artist_decoder.get(artist_index)
        genre = self.genre_decoder.get(genre_index)
        lyrics = [self.genre_decoder.get(character) for character in lyric_index]
        return artist, genre, lyrics

    # TODO : should add_token be implemeted for artists, genres and lyrics? Should it have
    # a type argument to add an artist token with self.getattr('artist') ?
    # TODO : is a call function required ?

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the tokenizer's vocabulary dictionnary to the provided save_directory.

        Args:
            save_directory (`str`):
                _description_
            filename_prefix (`Optional[str]`, *optional*, defaults to None):
                _description_
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"artist": self.artist_encoder, "genre": self.genre_encoder, "lyrics": self.lyrics_encoder},
                    ensure_ascii=False,
                )
            )

        return vocab_file
