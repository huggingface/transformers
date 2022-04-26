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
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import regex as re
from unidecode import unidecode

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "jukebox": "https://huggingface.co/jukebox/resolve/main/vocab.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "jukebox": 1024,  # (should be the maximum in the pretrained model #TODO: check this)
}


class JukeboxTokenizer(PreTrainedTokenizer):
    """
    Construct a Jukebox tokenizer. Jukebox can be conditioned on 3 different inputs :
        - Artists, unique ids are associated to each artist from the provided dictionary. 
        - Genres, unique ids are associated to each genre from the provided dictionary.
        - Lyrics, character based tokenization. Must be initialized with the list of characters that are inside the vocabulary.  

    This tokenizer is straight forward and does not require trainingg. It should be able to process a different number of inputs
    as the conditioning of the model can be done on the three different queries. If None is provided, defaults values will be used. 

    ```
    >>> from transformers import JukeboxTokenizer
    >>> tokenizer = JukeboxTokenizer.from_pretrained("jukebox")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")['input_ids']
    [15496, 995]
    >>> tokenizer("Alan Jackson", "Country Rock")['input_ids']
    [15496, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    If nothing is provided, the genres and the artist will either be selected randomly or set to None

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        artitst_vocab_file (`str`):
            Path to the vocabulary file which should contain a dictionnary where the keys are 'artist', 'genre' and 'lyrics' 
            and the values are their corresponding vocabulary files. 
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<|endoftext|>",
        **kwargs
    ):
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        super().__init__(
            unk_token=unk_token,
            **kwargs,
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            vocabulary = json.load(vocab_handle)
            self.artist_encoder = vocabulary["artist"]
            self.genre_encoder = vocabulary["genre"]
            self.lyrics_encoder = vocabulary["lyrics"]

        self.out_of_vocab = re.compile('[^A-Za-z0-9.,:;!?\-+\'\"()\[\] \t\n]+')  # FIXME: should be an argument?

        self.artist_decoder = {v: k for k, v in self.artist_encoder.items()}
        self.genre_decoder = {v: k for k, v in self.genre_encoder.items()}
        self.lyrics_decoder = {v: k for k, v in self.lyrics_vocab_file.items()}

    @property
    def vocab_size(self):
        return len(self.artist_encoder) + len(self.genre_encoder) + len(self.lyrics_encoder)

    def get_vocab(self):
        return dict(self.artist_encoder, self.genre_encoder, self.lyrics_encoder , **self.added_tokens_encoder)

    def get_relevant_lyric_tokens(self, full_tokens, n_tokens, total_length, offset, duration):
        if len(full_tokens) < n_tokens:
            tokens = [0] * (n_tokens - len(full_tokens)) + full_tokens
            indices = [-1] * (n_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
        else:
            assert 0 <= offset < total_length
            midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
            midpoint = min(max(midpoint, n_tokens // 2), len(full_tokens) - n_tokens // 2)
            tokens = full_tokens[midpoint - n_tokens // 2:midpoint + n_tokens // 2]
            indices = list(range(midpoint - n_tokens // 2, midpoint + n_tokens // 2))
        assert len(tokens) == n_tokens, f"Expected length {n_tokens}, got {len(tokens)}"
        assert len(indices) == n_tokens, f"Expected length {n_tokens}, got {len(indices)}"
        assert tokens == [full_tokens[index] if index != -1 else 0 for index in indices]
        return tokens, indices

    def _convert_token_to_id(self, artist, genre, lyrics):
        """Converts the artist, genre and lyrics tokens to their index using the vocabulary."""
        artist_id = self.artist_encoder.get(artist)
        genre_id = self.genre_encoder.get(genre)
        lyrics = unidecode(lyrics)
        lyrics = lyrics.replace('\\', '\n')
        lyrics = self.out_of_vocab.sub('', lyrics)  # Remove characters that are outside the vocabulary vocab
        lyric_ids = [self.genre_encoder.get(character) for character in lyrics]
        lyric_ids = self.get_relevant_lyric_tokens(lyric_ids)
        return artist_id, genre_id, lyric_ids

    def _convert_id_to_token(self, artist_index, genre_index, lyric_index):
        """Converts an index (integer) in a token (str) using the vocab."""
        artist = self.artist_decoder.get(artist_index)
        genre = self.genre_decoder.get(genre_index)
        lyrics = [self.genre_decoder.get(character) for character in lyric_index]
        return artist, genre, lyrics

    # TODO : should add_token be implemeted for artists, genres and lyrics? Should it have
    # a type argument to add an artist token with self.getattr('artist')
    # TODO : is a call function required ? 

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({'artist': self.artist_encoder,
                                'genre': self.genre_encoder,
                                'lyrics': self.lyrics_encoder}, ensure_ascii=False))

        return vocab_file
