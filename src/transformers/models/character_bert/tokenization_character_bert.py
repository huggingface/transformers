# coding=utf-8
# Copyright 2023 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for CharacterBert."""


import collections
import os
import unicodedata
import warnings
from collections.abc import Mapping, Sized
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import LARGE_INTEGER, BatchEncoding, EncodedInput, TruncationStrategy
from ...utils import (
    PaddingStrategy,
    TensorType,
    is_tf_tensor,
    is_torch_tensor,
    logging,
    to_py_obj,
)


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "mlm_vocab_file": "mlm_vocab.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "mlm_vocab_file": {
        "helboukkouri/character-bert-base-uncased": "https://huggingface.co/helboukkouri/character-bert-base-uncased/resolve/main/mlm_vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "helboukkouri/character-bert-base-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "helboukkouri/character-bert-base-uncased": {"max_word_length": 50, "do_lower_case": True},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def special_token_character_ids(index, bow_character_id, eow_character_id, pad_character_id, max_word_length):
    """Generates the character id sequence for a special token with a given index."""
    assert index > 255, "index range 0-255 is reserved for actual characters"
    character_ids = [pad_character_id] * max_word_length
    character_ids[0] = bow_character_id
    character_ids[1] = index
    character_ids[2] = eow_character_id
    # NOTE: we shift everything so that the PAD token
    #       can be assigned the all zeros vector
    character_ids = [(index + 1) for index in character_ids]
    return character_ids


class CharacterBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a CHARACTER_BERT tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        mlm_vocab_file (`str`, *optional*):
            File containing the vocabulary for Masked Language Modeling.
        max_word_length (`int`, *optional*, defaults to 50):
            The maximum token length in characters (or bytes since any non-ascii characters
            are eventually converted into a sequence of UTF-8 bytes).
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        bow_character_id (`int`, *optional*, defaults to 258):
            Id used to delimit the beginning of a token. First element in the character_id vector of the token.
        eow_character_id (`int`, *optional*, defaults to 259):
            Id used to delimit the end of a token. Last element in the character_id vector of the token, before padding.
        pad_character_id (`int`, *optional*, defaults to 260):
            Id used to delimit pad all character_id vectors to the same length, the maximum token length.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original CHARACTER_BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        mlm_vocab_file=None,
        max_word_length=50,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        bow_character_id=258,
        eow_character_id=259,
        pad_character_id=260,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        self.vocab = {}
        if mlm_vocab_file:
            if not os.path.isfile(mlm_vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{mlm_vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = CharacterBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            self.mlm_vocab = load_vocab(mlm_vocab_file)
        else:
            self.mlm_vocab = {}

        self.mlm_ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.mlm_vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        super().__init__(
            mlm_vocab_file=None,
            max_word_length=max_word_length,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            bow_character_id=bow_character_id,
            eow_character_id=eow_character_id,
            pad_character_id=pad_character_id,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # NOTE: this is to force the use of our custom character ids for
        # CLS/SEP/MASK/PAD instead of the default ones
        self._added_tokens_encoder = {}

        # Maximum number of characters (utf-8 bytes) per word
        if max_word_length < 3:
            raise ValueError("maximum word length has to be at least 3")
        self.max_word_length = max_word_length

        # Character delimiting the BEGINNING of a TOKEN
        self.bow_character_id = bow_character_id
        # Character delimiting the END of a TOKEN
        self.eow_character_id = eow_character_id
        # Pads a TOKEN up to the maximum character length
        self.pad_character_id = pad_character_id

        # Token delimiting the BEGINNING of a TEXT
        self.cls_character_ids = special_token_character_ids(
            index=256,
            bow_character_id=self.bow_character_id,
            eow_character_id=self.eow_character_id,
            pad_character_id=self.pad_character_id,
            max_word_length=max_word_length,
        )
        # Token delimiting the END of a TEXT
        self.sep_character_ids = special_token_character_ids(
            index=257,
            bow_character_id=self.bow_character_id,
            eow_character_id=self.eow_character_id,
            pad_character_id=self.pad_character_id,
            max_word_length=max_word_length,
        )
        # Masks a subset of TOKENS for Maked Language Modeling
        self.mask_character_ids = special_token_character_ids(
            index=261,
            bow_character_id=self.bow_character_id,
            eow_character_id=self.eow_character_id,
            pad_character_id=self.pad_character_id,
            max_word_length=max_word_length,
        )

        # Pads a sequence of TOKENS up to a desired length
        # NOTE: since this is zero, actual characters' ids are assigned
        # the range 1-256 instead of 0-255. This means that all character
        # ids are eventually shifted by a +1 so that 0 can be kept for padding.
        self.pad_character_ids = [0] * max_word_length

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def mlm_vocab_size(self):
        return len(self.mlm_vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def get_mlm_vocab(self):
        return self.mlm_vocab

    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                split_tokens.append(token)
        else:
            split_tokens = whitespace_tokenize(text)
        return split_tokens

    def convert_mlm_token_to_id(self, token):
        """Converts a token (str) into an id using the MLM vocab."""
        return self.mlm_vocab.get(token, self.mlm_vocab.get(self.unk_token))

    def convert_mlm_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the MLM vocab."""
        return self.mlm_ids_to_tokens.get(index, self.unk_token)

    # NOTE: the following two methods have misleading names since we are
    # working with character id lists instead of standard token ids.
    # Changing these names breaks a lot of things so we keep them for now.
    def _convert_token_to_id(self, token):
        """Converts a token (str) into a list of character ids (integer)."""
        if token == self.cls_token:
            character_ids = self.cls_character_ids
        elif token == self.sep_token:
            character_ids = self.sep_character_ids
        elif token == self.mask_token:
            character_ids = self.mask_character_ids
        elif token == self.pad_token:
            character_ids = self.pad_character_ids
        else:
            token_bytes = token.encode("utf-8", "ignore")[: (self.max_word_length - 2)]
            character_ids = [self.pad_character_id] * self.max_word_length
            character_ids[0] = self.bow_character_id
            for k, index in enumerate(token_bytes, start=1):
                character_ids[k] = index
            character_ids[len(token_bytes) + 1] = self.eow_character_id
            # NOTE: we shift everything so that the PAD token
            #       can be assigned the all zeros vector
            character_ids = [(index + 1) for index in character_ids]
        return character_ids

    def _convert_id_to_token(self, character_ids):
        """Converts a list of character ids (integer) intp a token (str)."""
        assert len(character_ids) == self.max_word_length, (
            f"Got a character sequence of length {len(character_ids)} while "
            f"`max_word_length={self.max_word_length}`"
        )

        if character_ids == self.cls_character_ids:
            return self.cls_token
        elif character_ids == self.sep_character_ids:
            return self.sep_token
        elif character_ids == self.mask_character_ids:
            return self.mask_token
        elif character_ids == self.pad_character_ids:
            return self.pad_token
        else:
            character_ids_ = [(index - 1) for index in character_ids]
            utf8_codes = list(
                filter(
                    lambda x: (
                        (x != self.pad_character_id) and (x != self.bow_character_id) and (x != self.eow_character_id)
                    ),
                    character_ids_,
                )
            )
            return bytes(utf8_codes).decode("utf-8")

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        # NOTE: the first condition needed to be adapted for character_ids instead
        # of token ids.
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, list) and isinstance(ids[0], int):
            # if ids in self._added_tokens_decoder:
            #    return self._added_tokens_decoder[ids].content
            # else:
            return self._convert_id_to_token(ids)
        tokens = []
        for character_ids in ids:
            tokens.append(self._convert_id_to_token(character_ids))
        return tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # TODO: check the methods below
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[List[int]], token_ids_1: Optional[List[List[int]]] = None
    ) -> List[List[int]]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CHARACTER_BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[List[int]],
        token_ids_1: Optional[List[List[int]]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A CHARACTER_BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        logger.warning("CharacterBERT does not have a token vocabulary. Skipping saving `vocab.txt`.")
        return ()

    def save_mlm_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # NOTE: CharacterBERT has no token vocabulary, this is just to allow
        # saving tokenizer configuration via CharacterBertTokenizer.save_pretrained
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["mlm_vocab_file"],
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.mlm_vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving MLM vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ):
        file_names = super()._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        mlm_vocab_files = self.save_mlm_vocabulary(save_directory, filename_prefix=filename_prefix)
        return file_names + mlm_vocab_files

    # NOTE: this override is a workaround to make the attention mask computation
    # rely on the tokens axis (number of tokens in the sequence) rather than the
    # characters axis (number of characters per word).
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        if self.__class__.__name__.endswith("Fast"):
            if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
                logger.warning_advice(
                    f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                    " using the `__call__` method is faster than using a method to encode the text followed by a call"
                    " to the `pad` method to get a padded encoding."
                )
                self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        # NOTE: this condition is the workaround. Previously:
        # `if required_input and not isinstance(required_input[0], (list, tuple))`
        if required_input and not (
            isinstance(required_input[0], (list, tuple)) and isinstance(required_input[0][0], (list, tuple))
        ):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                        " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                        " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                        " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                        " `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`."
                        )
                    if old_pad_to_max_length is not False:
                        warnings.warn("Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`.")
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is None and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, use"
                    " `truncation=True` to truncate examples to a max length. You can give a specific length with"
                    " `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input"
                    " size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific"
                    " truncation strategy selected among `truncation='only_first'` (will only truncate the first"
                    " sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the"
                    " pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence"
                    " in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no"
                                " predefined maximum length. Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has"
                                " no predefined maximum length. Default to no truncation."
                            )
                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        # NOTE: this is also a workaround because otherwise we check for a negative
        # `self.pad_token_id` while in this case we have a padding character id list
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            (self.pad_token is None)
            or ((not isinstance(self.pad_token_id, (list, tuple))) and (self.pad_token_id < 0))
        ):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original CHARACTER_BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # prevents treating the same character with different unicode codepoints as different characters
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
