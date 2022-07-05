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
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple

import torch

import regex as re
from tokenizers import normalizers

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "artists_file": "artists.json",
    "lyrics_file": "lyrics.json",
    "genres_file": "genres.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "artists_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/artists.json",
    },
    "genres_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/genres.json",
    },
    "lyrics_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/lyrics.json",
    },
}

PRETRAINED_LYRIC_TOKENS_SIZES = {
    "jukebox": 512,  # corresonds to the dummy-model ?
}


def get_relevant_lyric_tokens(full_tokens, max_n_lyric_tokens, total_length, offset, duration):
    """
    Extract only the relevant tokens based on the character position. A total of `max_n_lyric_tokens` tokens will be
    returned. If the provided token sequence is smaller, it will be padded, othewise, only characters ranging from the
    midpoint - `max_n_lyric_tokens//2` to the midpoint + `max_n_lyric_tokens//2` will be returned. This *focuses* on
    the most relevant tokens (in time) for the sequence.

    Args: # TODO : args to prettify
        full_tokens (`List[int]`):
            List containing the ids of the entire lyrics.
        total_length (`int`):
            Total expected length of the music (not all of it is generated, see duration), in samples.
        offset (`int`):
            Starting sample in the music. If the offset is greater than 0, the lyrics will be shifted take that into
            account
        duration (`int`):
            Expected duration of the generated music, in samples. The duration has to be smaller than the total lenght,
            which represent the overall length of the signal,
    """
    if len(full_tokens) < max_n_lyric_tokens:
        tokens = [0] * (max_n_lyric_tokens - len(full_tokens)) + full_tokens
        indices = [-1] * (max_n_lyric_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        assert 0 <= offset < total_length
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, max_n_lyric_tokens // 2), len(full_tokens) - max_n_lyric_tokens // 2)
        tokens = full_tokens[midpoint - max_n_lyric_tokens // 2 : midpoint + max_n_lyric_tokens // 2]
        indices = list(range(midpoint - max_n_lyric_tokens // 2, midpoint + max_n_lyric_tokens // 2))
    assert len(tokens) == max_n_lyric_tokens, f"Expected length {max_n_lyric_tokens}, got {len(tokens)}"
    assert len(indices) == max_n_lyric_tokens, f"Expected length {max_n_lyric_tokens}, got {len(indices)}"
    assert tokens == [full_tokens[index] if index != -1 else 0 for index in indices]
    return tokens, indices


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
    [[6785],[546], [0, 0, 0, 0, 0, 0, 0,   41,       38,       30,
                    77,       46,       41,       49,       40,
                    77,       44,       41,       27,       30] ]
    >>> tokenizer("Alan Jackson", "Country Rock")['input_ids']
    [6785],[546]]
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
        n_genres (`int`, `optional`, defaults to 1):
            Maximum number of genres to use for composition.
        max_n_lyric_tokens (`int`, `optional`, defaults to 512):
            Maximum number of lyric tokens to keep.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_lyric_input_size = PRETRAINED_LYRIC_TOKENS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        artists_file,
        genres_file,
        lyrics_file,
        version = ["v2","v3","v3"],
        max_n_lyric_tokens=512,
        n_genres=5,
        unk_token="<|endoftext|>",
        **kwargs
    ):
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        super().__init__(
            unk_token=unk_token,
            n_genres=n_genres, 
            max_n_lyric_tokens=max_n_lyric_tokens,
            **kwargs,
        )
        self.version = version
        self.max_n_lyric_tokens = max_n_lyric_tokens
        self.n_genres = n_genres

        with open(artists_file, encoding="utf-8") as vocab_handle:
            vocabulary = json.load(vocab_handle)
            self.artists_encoder = vocabulary["artists"]

        with open(genres_file, encoding="utf-8") as vocab_handle:
            vocabulary = json.load(vocab_handle)
            self.genres_encoder = vocabulary["genres"]

        with open(lyrics_file, encoding="utf-8") as vocab_handle:
            vocabulary = json.load(vocab_handle)
            self.lyrics_encoder = vocabulary["lyrics"]

        oov = '[^A-Za-z0-9.,:;!?\-\'\"()\[\] \t\n]+'
        # In v2, we had a n_vocab=80 and in v3 we missed + and so n_vocab=79 of characters.
        if len(self.lyrics_encoder) == 79:
            oov = oov.replace("\-\'","\-+\'")

        self.out_of_vocab = re.compile(oov) 
        self.artists_decoder = {v: k for k, v in self.artists_encoder.items()}
        self.genres_decoder = {v: k for k, v in self.genres_encoder.items()}
        self.lyrics_decoder = {v: k for k, v in self.lyrics_encoder.items()}

    @property
    def vocab_size(self):
        return len(self.artists_encoder) + len(self.genres_encoder) + len(self.lyrics_encoder)

    def get_vocab(self):
        return dict(self.artists_encoder, self.genres_encoder, self.lyrics_encoder)

    def _convert_token_to_id(self, artist, genres, lyrics, total_length, offset, duration):
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
        artists_id = self.artists_encoder.get(artist)
        genres_ids = [self.genres_encoder.get(genre) for genre in genres]
        genres_ids = genres_ids + [-1] * (self.n_genres - len(genres_ids))
        lyric_ids = [self.lyrics_encoder.get(character) for character in lyrics]
        return artists_id, genres_ids, lyric_ids

    def _tokenize(self, lyrics):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens. Only the lytrics are split into character for the character-based vocabulary.
        """
        # only lyrics are not tokenized, but character based is easily handled
        return [character for character in lyrics]

    def tokenize(self, artist, genre, lyrics, **kwargs):
        """
        Converts three strings in a 3 sequence of tokens using the tokenizer

        Args:
            artist (`_type_`):
                _description_
            genre (`_type_`):
                _description_
            lyrics (`_type_`):
                _description_
        """
        artist, genre, lyrics, kwargs = self.prepare_for_tokenization(artist, genre, lyrics, **kwargs)
        # TODO deal with the kwargs here
        lyrics = self._tokenize(lyrics)
        return artist, genre, lyrics

    def prepare_for_tokenization(
        self, artist: str, genres: str, lyrics: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            artist (`str`):
                The artist name to prepare. This will mostly lower the string
            genres (`str`):
                The gnere name to prepare. This will mostly lower the string.
            lyrics (`str`):
                The lyrics to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs:
                Keyword arguments to use for the tokenization. #TODO v3 could be handled here

        Returns:
            `Tuple[str, str, str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        for idx,version in enumerate(self.version): 
            if version == "v1":
                artist = artist.lower()
                genres = genres.lower()
                lyrics = lyrics.lower()
            else: 
                artist[idx] = self._normalize(artist)
                
                
        artist = self._normalize(artist)
        genres = self._normalize(genres).split("_")  # split is for the full dictionnary with combined genres
        normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
        lyrics = normalizer.normalize_str(lyrics)
        lyrics = lyrics.replace("\\", "\n")
        lyrics = self.out_of_vocab.sub("", lyrics)
        return artist, genres, lyrics, kwargs

    def _normalize(self, text: str) -> str:
        """Normalizes the input text. This process is for the genres and the artit

        Args:
            text (`str`):
                Artist or Genre string to normalize
        """
        import re

        accepted = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
        )

        # In v2, " " is not accepted while it is for v3
        if len(self.lyrics_encoder) == 79:
            accepted += [" "]
        accepted = frozenset(accepted)
        rex = re.compile(r"_+")
        text = "".join([c if c in accepted else "_" for c in text.lower()])
        text = rex.sub("_", text).strip("_")
        return text

    def _convert_id_to_token(self, artists_index, genres_index, lyric_index):
        """Converts an index (integer) in a token (str) using the vocab.
        Args:
            artists_index (`int`):
                Index of the artist in its corresponding dictionnary.
            genres_index (`Union[List[int], int]`):
               Index of the genre in its corresponding dictionnary.
            lyric_index (`List[int]`):
                List of character indices, which each correspond to a character.
        """
        artist = self.artists_decoder.get(artists_index)
        genres = [self.genres_decoder.get(genre) for genre in genres_index]
        lyrics = [self.lyrics_decoder.get(character) for character in lyric_index]
        return artist, genres, lyrics

    def convert_lyric_tokens_to_string(self, lyrics: List[str]) -> str:
        return " ".join(lyrics)

    # TODO : should add_token be implemeted for artists, genres and lyrics? Should it have
    # a type argument to add an artist token with self.getattr('artist') ?

    def __call__(self, artist, genres, lyrics, total_length, offset):
        """Convert the raw string to a list of token ids

        Args:
            artist (`str`):
                _description_
            genre (`str`):
                _description_
            lyrics (`srt`):
                _description_
            total_length (`int`):
                _description_
            offset (`_type_`):
                _description_
        """
        input_ids = [total_length, offset, None]
        artist = artist*len(self.version)
        genres = genres*len(self.version)
        lyrics = lyrics*len(self.version)
        
        artists_tokens, genres_tokens, lyrics_tokens = self.tokenize(artist, genres, lyrics)
        artists_id, genres_ids, lyric_ids = self._convert_token_to_id(
            artists_tokens, genres_tokens, lyrics_tokens, total_length, offset
        )
        input_ids += [artists_id] + genres_ids + relevant_tokens
        attention_masks = [-INFINITY] * (len(full_tokens) - len(relevant_tokens)) + [0] * len(relevant_tokens)
        # TODO properly handle the return pt tensor option
        if return_tensor == "pt":
            return {
                "input_ids": {"y": torch.tensor([input_ids]), "full_tokens": full_tokens},
                "attention_masks": torch.tensor(attention_masks),
            }

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionnary to the provided save_directory.

        Args:
            save_directory (`str`):
                _description_
            filename_prefix (`Optional[str]`, *optional*, defaults to None):
                _description_
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        artists_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["artists_file"]
        )
        with open(artists_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.artists_encoder, ensure_ascii=False))

        genres_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["genres_file"]
        )
        with open(genres_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.genres_encoder, ensure_ascii=False))

        lyrics_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["lyrics_file"]
        )
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.lyrics_encoder, ensure_ascii=False))

        return (artists_file, genres_file, lyrics_file)
