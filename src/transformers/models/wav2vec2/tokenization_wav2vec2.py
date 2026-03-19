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
"""Tokenization class for Wav2Vec2."""

import json
import os
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Union

import numpy as np

from ...tokenization_python import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
    ModelOutput,
    logging,
    to_py_obj,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    import torch


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}


# Wav2Vec2 has no max input length

WAV2VEC2_KWARGS_DOCSTRING = r"""
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
"""

ListOfDict = list[dict[str, int | str]]


@dataclass
class Wav2Vec2CTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2CTCTokenizer`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `list[dict[str, Union[int, str]]]` or `list[dict[str, Union[int, str]]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each character. Total logit score of the beam associated with
            produced text.
        word_offsets (list of `list[dict[str, Union[int, str]]]` or `list[dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """

    text: list[str] | str
    char_offsets: list[ListOfDict] | ListOfDict = None
    word_offsets: list[ListOfDict] | ListOfDict = None


class Wav2Vec2CTCTokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2CTC tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            The token used for defining the end of a word.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        target_lang (`str`, *optional*):
            A target language the tokenizer should set by default. `target_lang` has to be defined for multi-lingual,
            nested vocabulary such as [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all).

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        target_lang=None,
        **kwargs,
    ):
        self._word_delimiter_token = word_delimiter_token

        self.do_lower_case = do_lower_case
        self.replace_word_delimiter_char = replace_word_delimiter_char
        self.target_lang = target_lang

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)

        # if target lang is defined vocab must be a nested dict
        # with each target lang being one vocabulary
        if target_lang is not None:
            self.encoder = self.vocab[target_lang]
        else:
            self.encoder = self.vocab

        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            replace_word_delimiter_char=replace_word_delimiter_char,
            target_lang=target_lang,
            special_tokens_pattern="none",
            **kwargs,
        )
        # make sure that tokens made of several
        # characters are not split at tokenization
        for token in self.encoder:
            if len(token) > 1:
                self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))

    def set_target_lang(self, target_lang: str):
        """
        Set the target language of a nested multi-lingual dictionary
        """
        if self.vocab == self.encoder:
            raise ValueError(f"{self.vocab} is not a multi-lingual, nested tokenizer. Cannot set target language.")

        if target_lang not in self.vocab:
            raise ValueError(f"{target_lang} does not exist. Choose one of {', '.join(self.vocab.keys())}.")

        self.target_lang = target_lang
        self.init_kwargs["target_lang"] = target_lang
        self.encoder = self.vocab[target_lang]
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Remove conflicting entries from _added_tokens_decoder so vocabulary tokens take precedence
        for token_id in list(self._added_tokens_decoder.keys()):
            if token_id in self.decoder:
                del self._added_tokens_decoder[token_id]

        # make sure that tokens made of several
        # characters are not split at tokenization
        for token in self.encoder:
            if len(token) > 1:
                self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> int | None:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
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

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> dict:
        vocab = dict(self.encoder)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _add_tokens(self, new_tokens: list[str] | list[AddedToken], special_tokens: bool = False) -> int:
        # Overwritten to never strip!
        to_add = []
        for token in new_tokens:
            if isinstance(token, str):
                to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=False))
            else:
                to_add.append(token)

        return super()._add_tokens(to_add, special_tokens)

    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer.
        """
        if self.do_lower_case:
            text = text.upper()

        return list(text.replace(" ", self.word_delimiter_token))

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        """Overridden to prioritize vocabulary tokens over added tokens for nested vocabularies."""
        if isinstance(ids, int):
            if ids in self.decoder:
                return self.decoder[ids]
            return self._added_tokens_decoder[ids].content if ids in self._added_tokens_decoder else self.unk_token

        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.decoder:
                tokens.append(self.decoder[index])
            elif index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                tokens.append(self.unk_token)
        return tokens

    def convert_tokens_to_string(
        self,
        tokens: list[str],
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
    ) -> dict[str, str | float]:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        if len(tokens) == 0:
            return {"text": "", "char_offsets": [], "word_offsets": []}
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
        else:
            chars = tokens
            char_repetitions = len(tokens) * [1]

        # filter self.pad_token which is used as CTC-blank token
        processed_chars = list(filter(lambda char: char != self.pad_token, chars))

        # replace delimiter token
        processed_chars = [
            self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
        ]

        # retrieve offsets
        char_offsets = word_offsets = None
        if output_char_offsets or output_word_offsets:
            char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)

            if len(char_offsets) != len(processed_chars):
                raise ValueError(
                    f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                    " have to be of the same length, but are: "
                    f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                    f" {len(processed_chars)}"
                )

            # set tokens to correct processed token
            for i, char in enumerate(processed_chars):
                char_offsets[i]["char"] = char

            # retrieve word offsets from character offsets
            word_offsets = None
            if output_word_offsets:
                word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)

            # don't output chars if not set to True
            if not output_char_offsets:
                char_offsets = None

        # join to string
        join_char = " " if spaces_between_special_tokens else ""
        string = join_char.join(processed_chars).strip()

        if self.do_lower_case:
            string = string.lower()

        return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}

    @staticmethod
    def _compute_offsets(char_repetitions: list[int], chars: list[str], ctc_token: int) -> list[dict[str, str | int]]:
        end_indices = np.asarray(char_repetitions).cumsum()
        start_indices = np.concatenate(([0], end_indices[:-1]))

        offsets = [
            {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
        ]

        # filter out CTC token
        offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
        return offsets

    @staticmethod
    def _get_word_offsets(offsets: dict[str, str | float], word_delimiter_char: str = " ") -> dict[str, str | float]:
        word_offsets = []

        last_state = "SPACE"
        word = ""
        start_offset = 0
        end_offset = 0
        for i, offset in enumerate(offsets):
            char = offset["char"]
            state = "SPACE" if char == word_delimiter_char else "WORD"

            if state == last_state:
                # If we are in the same state as before, we simply repeat what we've done before
                end_offset = offset["end_offset"]
                word += char
            else:
                # Switching state
                if state == "SPACE":
                    # Finishing a word
                    word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                else:
                    # Starting a new word
                    start_offset = offset["start_offset"]
                    end_offset = offset["end_offset"]
                    word = char

            last_state = state
        if last_state == "WORD":
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        return word_offsets

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if is_split_into_words:
            text = " " + text
        return (text, kwargs)

    def _decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_word_offsets: bool | None = False,
        output_char_offsets: bool | None = False,
    ) -> str:
        """
        special _decode function is needed because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        # Don't skip special tokens in convert_ids_to_tokens so we can handle word_delimiter_token specially
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_tokens and token != self.word_delimiter_token:
                continue
            result.append(token)

        string_output = self.convert_tokens_to_string(
            result,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_word_offsets=output_word_offsets,
            output_char_offsets=output_char_offsets,
        )

        text = string_output["text"]

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        if output_word_offsets or output_char_offsets:
            return Wav2Vec2CTCTokenizerOutput(
                text=text,
                char_offsets=string_output["char_offsets"],
                word_offsets=string_output["word_offsets"],
            )
        else:
            return text

    # overwritten from `tokenization_utils_base.py` because tokenizer can output
    # `ModelOutput` which should not be a list for batched output and
    # because we need docs for `output_char_offsets` here
    def batch_decode(
        self,
        sequences: Union[list[int], list[list[int]], np.ndarray, "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    ) -> list[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the Example of [`~Wav2Vec2CTCTokenizer.decode`] to better understand how to make
                use of `output_char_offsets`. [`~Wav2Vec2CTCTokenizer.batch_decode`] works the same way with batched
                output.

                </Tip>

            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.

                <Tip>

                Please take a look at the Example of [`~Wav2Vec2CTCTokenizer.decode`] to better understand how to make
                use of `output_word_offsets`. [`~Wav2Vec2CTCTokenizer.batch_decode`] works the same way with batched
                output.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `list[str]` or [`~models.wav2vec2.tokenization_wav2vec2.Wav2Vec2CTCTokenizerOutput`]: The list of decoded
            sentences. Will be a [`~models.wav2vec2.tokenization_wav2vec2.Wav2Vec2CTCTokenizerOutput`] when
            `output_char_offsets == True` or `output_word_offsets == True`.
        """
        batch_decoded = [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                output_char_offsets=output_char_offsets,
                output_word_offsets=output_word_offsets,
                **kwargs,
            )
            for seq in sequences
        ]
        if output_char_offsets or output_word_offsets:
            # transform list of dicts to dict of lists
            return Wav2Vec2CTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})

        return batch_decoded

    # overwritten from `tokenization_utils_base.py` because we need docs for `output_char_offsets`
    # and `output_word_offsets` here
    def decode(
        self,
        token_ids: Union[int, list[int], np.ndarray, "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, list[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the example below to better understand how to make use of `output_char_offsets`.

                </Tip>

            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.

                <Tip>

                Please take a look at the example below to better understand how to make use of `output_word_offsets`.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str` or [`~models.wav2vec2.tokenization_wav2vec2.Wav2Vec2CTCTokenizerOutput`]: The list of decoded
            sentences. Will be a [`~models.wav2vec2.tokenization_wav2vec2.Wav2Vec2CTCTokenizerOutput`] when
            `output_char_offsets == True` or `output_word_offsets == True`.

        Example:

        ```python
        >>> # Let's see how to retrieve time steps for a model
        >>> from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
        >>> from datasets import load_dataset
        >>> import datasets
        >>> import torch

        >>> # import model, feature extractor, tokenizer
        >>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        >>> # load first sample of English common_voice
        >>> dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
        >>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        >>> dataset_iter = iter(dataset)
        >>> sample = next(dataset_iter)

        >>> # forward sample through model to get greedily predicted transcription ids
        >>> input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
        >>> logits = model(input_values).logits[0]
        >>> pred_ids = torch.argmax(logits, axis=-1)

        >>> # retrieve word stamps (analogous commands for `output_char_offsets`)
        >>> outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
        >>> # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
        >>> time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

        >>> word_offsets = [
        ...     {
        ...         "word": d["word"],
        ...         "start_time": round(d["start_offset"] * time_offset, 2),
        ...         "end_time": round(d["end_offset"] * time_offset, 2),
        ...     }
        ...     for d in outputs.word_offsets
        ... ]
        >>> # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
        >>> # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en
        >>> word_offsets[:3]
        [{'word': 'THE', 'start_time': 0.7, 'end_time': 0.78}, {'word': 'TRICK', 'start_time': 0.88, 'end_time': 1.08}, {'word': 'APPEARS', 'start_time': 1.2, 'end_time': 1.64}]
        ```"""
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            output_char_offsets=output_char_offsets,
            output_word_offsets=output_word_offsets,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)


__all__ = ["Wav2Vec2CTCTokenizer"]
