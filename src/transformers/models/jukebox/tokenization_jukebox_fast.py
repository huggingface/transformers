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
"""Tokenization classes for OpenAI GPT."""


import json
from typing import TYPE_CHECKING, List, Optional, Tuple

from tokenizers import pre_tokenizers

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_jukebox import JukeboxTokenizer


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "jukebox": "https://huggingface.co/jukebox/resolve/main/vocab.json",
    },
    "tokenizer_file": {"jukebox": "https://huggingface.co/jukebox/resolve/main/tokenizer.json"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"jukebox": 1024}


class JukeboxTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Jukebox tokenizer, backed by HuggingFace's tokenizers library. Jukebox can be conditioned on 3
    different inputs :
        - Artists, unique ids are associated to each artist from the provided dictionary.
        - Genres, unique ids are associated to each genre from the provided dictionary.
        - Lyrics, character based tokenization. Must be initialized with the list of characters that are inside the
        vocabulary.

    This tokenizer is straight forward and does not require trainingg. It should be able to process a different number
    of inputs: as the conditioning of the model can be done on the three different queries. If None is provided,
    defaults values will be used.:

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

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
    to: this superclass for more information regarding those methods.

    Args:
        artitst_vocab_file (`str`):
            Path to the vocabulary file which should contain a dictionnary where the keys are 'artist', 'genre' and
            'lyrics' and the values are their corresponding vocabulary files.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = JukeboxTokenizer

    def __init__(
        self, vocab_file=None, tokenizer_file=None, unk_token="<|endoftext|>", add_prefix_space=False, **kwargs
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # TODO: should it be using WordLevel tokenizer ? Don't really know how that works yet
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        """This corresponds to DialoGPT variants of models."""
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])

        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids
