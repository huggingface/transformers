# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
""" Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library).
    For slow (python) tokenizers see tokenization_utils.py
"""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import Encoding as EncodingFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.implementations import BaseTokenizer as BaseTokenizerFast

from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)


logger = logging.getLogger(__name__)


class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """ Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherit from PreTrainedTokenizer.

    Handle all the shared methods for tokenization and special tokens as well as methods
    downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

    - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file
      required by the model, and as associated values, the filename for saving the associated file (string).
    - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys
      being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the
      `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the
      associated pretrained vocabulary file.
    - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained
      models, and as associated values, the maximum length of the sequence inputs of this model, or None if the
      model has no maximum input size.
    - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the
      pretrained models, and as associated values, a dictionnary of specific arguments to pass to the
      ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the
      ``from_pretrained()`` method.

    Args:
        - ``tokenizer`` (`BaseTokenizerFast`): A Fast tokenizer from the HuggingFace tokenizer library (in low level Rust language)
        - ``model_max_length``: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer model.
            When the tokenizer is loaded with `from_pretrained`, this will be set to the value stored for the associated
            model in ``max_model_input_sizes`` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).
            no associated max_length can be found in ``max_model_input_sizes``.
        - ``padding_side``: (`Optional`) string: the side on which the model should have padding applied.
            Should be selected between ['right', 'left']
        - ``model_input_names``: (`Optional`) List[string]: the list of the forward pass inputs accepted by the
            model ("token_type_ids", "attention_mask"...).
        - ``bos_token``: (`Optional`) string: a beginning of sentence token.
            Will be associated to ``self.bos_token`` and ``self.bos_token_id``
        - ``eos_token``: (`Optional`) string: an end of sentence token.
            Will be associated to ``self.eos_token`` and ``self.eos_token_id``
        - ``unk_token``: (`Optional`) string: an unknown token.
            Will be associated to ``self.unk_token`` and ``self.unk_token_id``
        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence).
            Will be associated to ``self.sep_token`` and ``self.sep_token_id``
        - ``pad_token``: (`Optional`) string: a padding token.
            Will be associated to ``self.pad_token`` and ``self.pad_token_id``
        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model).
            Will be associated to ``self.cls_token`` and ``self.cls_token_id``
        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language
            modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``
        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens.
            Adding all special tokens here ensure they won't be split by the tokenization process.
            Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``


    .. automethod:: __call__
    """

    def __init__(self, tokenizer: BaseTokenizerFast, **kwargs):
        if not isinstance(tokenizer, BaseTokenizerFast):
            raise ValueError(
                "Tokenizer should be an instance of a Tokenizer " "provided by HuggingFace tokenizers library."
            )
        self._tokenizer: BaseTokenizerFast = tokenizer

        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def get_added_vocab(self) -> Dict[str, int]:
        base_vocab = self._tokenizer.get_vocab(with_added_tokens=False)
        full_vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        added_vocab = dict((tok, index) for tok, index in full_vocab.items() if tok not in base_vocab)
        return added_vocab

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> BaseTokenizerFast:
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        return self._tokenizer._tokenizer.decoder

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """ Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict.

            Overflowing tokens are converted to additional examples (like batches) so the output values of
            the dict are lists (overflows) of lists (tokens).

            Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """ Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False) -> List[str]:
        return self._tokenizer.encode(text, pair, add_special_tokens=add_special_tokens).tokens

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
    ):
        """ Define the truncation and the padding strategies for fast tokenizers
            (provided by HuggingFace tokenizers library) and restore the tokenizer settings afterwards.

            The provided tokenizer has no padding / truncation strategy
            before the managed section. If your tokenizer set a padding / truncation strategy before,
            then it will be reset to no padding/truncation when exiting the managed section.

            Args:
                padding_strategy (:obj:`PaddingStrategy`): The kind of padding that will be applied to the input
                truncation_strategy (:obj:`TruncationStrategy`): The kind of truncation that will be applied to the input
                max_length (:obj:`int`): The maximum size of the sequence
                stride (:obj:`int`): The stride to use when handling overflow
                pad_to_multiple_of (:obj:`int`, `optional`, defaults to `None`)

        """
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            self._tokenizer.enable_truncation(max_length, stride=stride, strategy=truncation_strategy.value)
        else:
            self._tokenizer.no_truncation()

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            self._tokenizer.enable_padding(
                length=max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None,
                direction=self.padding_side,
                pad_id=self.pad_token_id,
                pad_type_id=self.pad_token_type_id,
                pad_token=self.pad_token,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        else:
            self._tokenizer.no_padding()

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_pretokenized: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        if not isinstance(batch_text_or_text_pairs, list):
            raise ValueError(
                "batch_text_or_text_pairs has to be a list (got {})".format(type(batch_text_or_text_pairs))
            )

        if kwargs:
            raise ValueError(f"Keyword arguments {kwargs} not recognized.")

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        # Avoid thread overhead if only one example.
        if len(batch_text_or_text_pairs) == 1:
            if isinstance(batch_text_or_text_pairs[0], tuple):
                # We got a Tuple with a pair of sequences
                encodings = self._tokenizer.encode(
                    *batch_text_or_text_pairs[0],
                    add_special_tokens=add_special_tokens,
                    is_pretokenized=is_pretokenized,
                )
            else:
                # We got a single sequence
                encodings = self._tokenizer.encode(
                    batch_text_or_text_pairs[0],
                    add_special_tokens=add_special_tokens,
                    is_pretokenized=is_pretokenized,
                )
            encodings = [encodings]
        else:
            encodings = self._tokenizer.encode_batch(
                batch_text_or_text_pairs, add_special_tokens=add_special_tokens, is_pretokenized=is_pretokenized
            )

        # Convert encoding to dict
        # `Tokens` has type: List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict]
        sanitized = {}
        for key in tokens[0].keys():
            # To List[List[List[int]]] of shape (batch, overflows, sequence length)
            stack = [e for item in tokens for e in item[key]]
            sanitized[key] = stack

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, enc in enumerate(tokens):
                overflow_to_sample_mapping += [i] * len(enc["input_ids"])
            sanitized["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        return BatchEncoding(sanitized, encodings, tensor_type=return_tensors)

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_pretokenized: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Return tensor is None, then we can remove the leading batch axis
        # Overfolwing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
        if os.path.isdir(save_directory):
            files = self._tokenizer.save_model(save_directory)
        else:
            folder, file = os.path.split(os.path.abspath(save_directory))
            files = self._tokenizer.save_model(folder, name=file)

        return tuple(files)
