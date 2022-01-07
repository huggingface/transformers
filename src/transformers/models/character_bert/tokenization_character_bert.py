# coding=utf-8
# Copyright Hicham EL BOUKKOURI, Olivier FERRET, Thomas LAVERGNE, Hiroshi NOJI,
# Pierre ZWEIGENBAUM, Junichi TSUJII and The HuggingFace Inc. team.
# All rights reserved.
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
"""Tokenization classes for CharacterBERT."""
import json
import os
import unicodedata
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...file_utils import _is_tensorflow, _is_torch, is_tf_available, is_torch_available, to_py_obj
from ...tokenization_utils import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    PreTrainedTokenizer,
    TensorType,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)
from ...tokenization_utils_base import ADDED_TOKENS_FILE
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "mlm_vocab_file": "mlm_vocab.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "mlm_vocab_file": {
        "helboukkouri/character-bert": "https://huggingface.co/helboukkouri/character-bert/resolve/main/mlm_vocab.txt",
        "helboukkouri/character-bert-medical": "https://huggingface.co/helboukkouri/character-bert-medical/resolve/main/mlm_vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "helboukkouri/character-bert": 512,
    "helboukkouri/character-bert-medical": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "helboukkouri/character-bert": {"max_word_length": 50, "do_lower_case": True},
    "helboukkouri/character-bert-medical": {"max_word_length": 50, "do_lower_case": True},
}

PAD_TOKEN_CHAR_ID = 0


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def build_mlm_ids_to_tokens_mapping(mlm_vocab_file):
    """Builds a Masked Language Modeling ids to masked tokens mapping."""
    vocabulary = []
    with open(mlm_vocab_file, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                vocabulary.append(line)
    return OrderedDict(list(enumerate(vocabulary)))


class CharacterBertTokenizer(PreTrainedTokenizer):
    """
    Construct a CharacterBERT tokenizer. Based on characters.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        mlm_vocab_file (`str`, *optional*, defaults to `None`):
            Path to the Masked Language Modeling vocabulary. This is used for converting the output (token ids) of the
            MLM model into tokens.
        max_word_length (`int`, *optional*, defaults to `50`):
            The maximum token length in characters (actually, in bytes as any non-ascii characters will be converted to
            a sequence of utf-8 bytes).
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
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
        strip_accents: (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
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
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            max_word_length=max_word_length,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        # This prevents splitting special tokens during tokenization
        self.unique_no_split_tokens = [self.cls_token, self.mask_token, self.pad_token, self.sep_token, self.unk_token]
        # This is used for converting MLM ids into tokens
        if mlm_vocab_file is None:
            self.ids_to_tokens = None
        else:
            if not os.path.isfile(mlm_vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{mlm_vocab_file}'. "
                    "To load the vocabulary from a pretrained model use "
                    "`tokenizer = CharacterBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            self.ids_to_tokens = build_mlm_ids_to_tokens_mapping(mlm_vocab_file)
        # Tokenization is handled by BasicTokenizer
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # Then, a CharacterMapper is responsible for converting tokens into character ids
        self.max_word_length = max_word_length
        self._mapper = CharacterMapper(max_word_length=max_word_length)

    def __repr__(self) -> str:
        # NOTE: we overwrite this because CharacterBERT does not have self.vocab_size
        return (
            f"CharacterBertTokenizer(name_or_path='{self.name_or_path}', "
            + (f"mlm_vocab_size={self.mlm_vocab_size}, " if self.ids_to_tokens else "")
            + f"model_max_len={self.model_max_length}, is_fast={self.is_fast}, "
            + f"padding_side='{self.padding_side}', special_tokens={self.special_tokens_map_extended})"
        )

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        # return self.vocab_size + len(self.added_tokens_encoder)
        return 0 + len(self.added_tokens_encoder)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        raise NotImplementedError("CharacterBERT does not use a token vocabulary.")

    @property
    def mlm_vocab_size(self):
        if self.ids_to_tokens is None:
            raise ValueError(
                "CharacterBertTokenizer was initialized without a MLM "
                "vocabulary. You can either pass one manually or load a "
                "pre-trained tokenizer using: "
                "`tokenizer = CharacterBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        return len(self.ids_to_tokens)

    def add_special_tokens(self, *args, **kwargs):
        raise NotImplementedError("Adding special tokens is not supported for now.")

    def add_tokens(self, *args, **kwargs):
        # We don't raise an Exception here to allow for ignoring this step.
        # Otherwise, many inherited methods would need to be re-implemented...
        pass

    def get_vocab(self):
        raise NotImplementedError("CharacterBERT does not have a token vocabulary.")

    def get_mlm_vocab(self):
        return {token: i for i, token in self.ids_to_tokens.items()}

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            split_tokens = self.basic_tokenizer.tokenize(text=text, never_split=self.all_special_tokens)
        else:
            split_tokens = whitespace_tokenize(text)  # Default to whitespace tokenization
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).strip()
        return out_string

    def _convert_token_to_id(self, token):
        """Converts a token (str) into a sequence of character ids."""
        return self._mapper.convert_word_to_char_ids(token)

    def _convert_id_to_token(self, index: List[int]):
        # NOTE: keeping the same variable name `ìndex` although this will
        # always be a sequence of indices.
        """Converts an index (actually, a list of indices) in a token (str)."""
        return self._mapper.convert_char_ids_to_word(index)

    def convert_ids_to_tokens(
        self, ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single sequence of character indices or a sequence of character id sequences in a token or a
        sequence of tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, list) and isinstance(ids[0], int):
            if tuple(ids) in self.added_tokens_decoder:
                return self.added_tokens_decoder[tuple(ids)]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for indices in ids:
            indices = list(map(int, indices))
            if skip_special_tokens and tuple(indices) in self.all_special_ids:
                continue
            if tuple(indices) in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[tuple(indices)])
            else:
                tokens.append(self._convert_id_to_token(indices))
        return tokens

    def convert_mlm_id_to_token(self, mlm_id):
        """Converts an index (integer) in a token (str) using the vocab."""
        if self.ids_to_tokens is None:
            raise ValueError(
                "CharacterBertTokenizer was initialized without a MLM "
                "vocabulary. You can either pass one manually or load a "
                "pre-trained tokenizer using: "
                "`tokenizer = CharacterBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        assert (
            mlm_id < self.mlm_vocab_size
        ), "Attempting to convert a MLM id that is greater than the MLM vocabulary size."
        return self.ids_to_tokens[mlm_id]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[List[int]], token_ids_1: Optional[List[List[int]]] = None
    ) -> List[List[int]]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CharacterBERT sequence has the following format:

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
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[List[int]], token_ids_1: Optional[List[List[int]]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A CharacterBERT
        sequence pair mask has the following format:

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
                    `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given
                    sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

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
        `self.pad_token_id` and `self.pad_token_type_id`)

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
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
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
                >= 7.5 (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0][0], (list, tuple)):
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
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
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

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            if isinstance(encoded_inputs["token_type_ids"], list):
                encoded_inputs["attention_mask"] = [1] * len(required_input)
            else:
                encoded_inputs["attention_mask"] = 1

        return encoded_inputs

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        logger.warning("CharacterBERT does not have a token vocabulary. " "Skipping saving `vocab.txt`.")
        return ()

    def save_mlm_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # NOTE: CharacterBERT has no token vocabulary, this is just to allow
        # saving tokenizer configuration via CharacterBertTokenizer.save_pretrained
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + "mlm_vocab.txt"
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as f:
            for _, token in self.ids_to_tokens.items():
                f.write(token + "\n")
        return (vocab_file,)

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific [`~tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained`]
        """
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        save_directory = str(save_directory)

        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        added_vocab = self.get_added_vocab()
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, ensure_ascii=False)
                f.write(out_str)
                logger.info(f"added tokens file saved in {added_tokens_file}")

        vocab_files = self.save_mlm_vocabulary(save_directory, filename_prefix=filename_prefix)

        return file_names + vocab_files + (added_tokens_file,)


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
        strip_accents: (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (*optional*) list of str
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
        orig_tokens = whitespace_tokenize(text)
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
        if never_split is not None and text in never_split:
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


class CharacterMapper:
    """
    NOTE: Adapted from ElmoCharacterMapper:
    https://github.com/allenai/allennlp/blob/main/allennlp/data/token_indexers/elmo_indexer.py Maps individual tokens
    to sequences of character ids, compatible with CharacterBERT.
    """

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding> | short tokens are padded using this + 1
    mask_character = 261  # <mask>

    bos_token = "[CLS]"  # previously: bos_token = "<S>"
    eos_token = "[SEP]"  # previously: eos_token = "</S>"
    pad_token = "[PAD]"
    mask_token = "[MASK]"

    def __init__(
        self,
        max_word_length: int = 50,
    ):
        self.max_word_length = max_word_length
        self.beginning_of_sentence_characters = self._make_char_id_sequence(self.beginning_of_sentence_character)
        self.end_of_sentence_characters = self._make_char_id_sequence(self.end_of_sentence_character)
        self.mask_characters = self._make_char_id_sequence(self.mask_character)
        # This is the character id sequence for the pad token (i.e. [PAD]).
        # We remove 1 because we will add 1 later on and it will be equal to 0.
        self.pad_characters = [PAD_TOKEN_CHAR_ID - 1] * self.max_word_length

    def _make_char_id_sequence(self, character: int):
        char_ids = [self.padding_character] * self.max_word_length
        char_ids[0] = self.beginning_of_word_character
        char_ids[1] = character
        char_ids[2] = self.end_of_word_character
        return char_ids

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word == self.bos_token:
            char_ids = self.beginning_of_sentence_characters
        elif word == self.eos_token:
            char_ids = self.end_of_sentence_characters
        elif word == self.mask_token:
            char_ids = self.mask_characters
        elif word == self.pad_token:
            char_ids = self.pad_characters
        else:
            # Convert characters to indices
            word_encoded = word.encode("utf-8", "ignore")[: (self.max_word_length - 2)]
            # Initialize character_ids with padding
            char_ids = [self.padding_character] * self.max_word_length
            # First character is BeginningOfWord
            char_ids[0] = self.beginning_of_word_character
            # Populate character_ids with computed indices
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            # Last character is EndOfWord
            char_ids[len(word_encoded) + 1] = self.end_of_word_character

        # +1 one for masking so that character padding == 0
        # char_ids domain is therefore: (1, 256) for actual characters
        # and (257-262) for special symbols (BOS/EOS/BOW/EOW/padding/MLM Mask)
        return [c + 1 for c in char_ids]

    def convert_char_ids_to_word(self, char_ids: List[int]) -> str:
        "Converts a sequence of character ids into its corresponding word."

        assert len(char_ids) == self.max_word_length, (
            f"Got character sequence of length {len(char_ids)} while " "`max_word_length={self.max_word_length}`"
        )

        char_ids_ = [(i - 1) for i in char_ids]
        if char_ids_ == self.beginning_of_sentence_characters:
            return self.bos_token
        elif char_ids_ == self.end_of_sentence_characters:
            return self.eos_token
        elif char_ids_ == self.mask_characters:
            return self.mask_token
        elif char_ids_ == self.pad_characters:  # token padding
            return self.pad_token
        else:
            utf8_codes = list(
                filter(
                    lambda x: (x != self.padding_character)
                    and (x != self.beginning_of_word_character)
                    and (x != self.end_of_word_character),
                    char_ids_,
                )
            )
            return bytes(utf8_codes).decode("utf-8")
