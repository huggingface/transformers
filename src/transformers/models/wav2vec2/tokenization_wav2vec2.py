# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for BlenderbotSmall."""

import json
import os
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}


class Wav2Vec2Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        merges_file (:obj:`str`):
            Path to the merges file.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"__start__"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"__end__"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"__unk__"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"__pad__"`):
            The token used for padding, for example when batching sequences of different lengths.
        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = {
        "vocab_file": {
            "patrickvonplaten/wav2vec2-base-960h": "https://huggingface.co/patrickvonplaten/wav2vec2-base-960h/resolve/main/vocab.json"
        },
        "tokenizer_config_file": {
            "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer.json",
            "patrickvonplaten/wav2vec2-base-960h": "https://huggingface.co/patrickvonplaten/wav2vec2-base-960h/resolve/main/tokenizer_config.json",
        },
    }

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )
        self._word_delimiter_token = word_delimiter_token
        self.do_lower_case = do_lower_case

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def word_delimiter_token(self) -> str:
        """
        :obj:`str`: Padding token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    def __call__(
        self,
        raw_speech: Union[np.array, List[float], List[np.array], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        # make sure input is in list format
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # convert into correct format for padding
        # key has to be called `input_ids`
        encoded_inputs = BatchEncoding({"input_ids": raw_speech})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # hacky way of renaming input_ids to input_values
        padded_inputs["input_values"] = padded_inputs["input_ids"]
        del padded_inputs["input_ids"]

        return padded_inputs

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        grouped_tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, grouped_tokens))

        # replace delimiter token
        string = "".join([" " if token == self.word_delimiter_token else token for token in filtered_tokens]).strip()

        if self.do_lower_case:
            string = string.lower()
        return string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(result)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        return (vocab_file,)
