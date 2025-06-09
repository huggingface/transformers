# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for Bert2D."""

import math
from typing import List, Optional, Union

# import os # Only needed for __main__
# import shutil # Only needed for __main__
from ...tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TruncationStrategy,
)
from ...utils import logging
from ..bert.tokenization_bert import VOCAB_FILES_NAMES, BertTokenizer


logger = logging.get_logger(__name__)
# Set logger level to DEBUG for more verbose output if needed during development/testing
# logger.setLevel(logging.DEBUG) # Comment out for production


# Helper functions ported from Bert2DTokenizerFast
def is_subword(token: str, subword_prefix="##") -> bool:
    """Returns if a token is a subword"""
    return token.startswith(subword_prefix)


def create_word_ids(
    tokens: List[str], restart_new_sentence=False, seperator_token="[SEP]", padding_token="[PAD]"
) -> List[int]:
    """Creates word ids for given tokens, matching the logic from Bert2DTokenizerFast tests."""
    word_ids: List[int] = []
    current_word_id: int = -1
    sentence_restart_flag = False

    actual_restart_new_sentence = restart_new_sentence and tokens.count(seperator_token) >= 2

    for token in tokens:
        if token == padding_token:  # Pad tokens get word_id 0
            word_ids.append(0)
            # current_word_id = 0 # Resetting current_word_id for padding might be complex if padding is not last
        elif actual_restart_new_sentence and not sentence_restart_flag and token == seperator_token:
            if current_word_id == -1:  # First token is SEP
                current_word_id = 0
                word_ids.append(current_word_id)
            else:  # SEP after some content
                current_word_id += 1
                word_ids.append(current_word_id)

            current_word_id = -1  # Reset for the new sentence (will become 0 at first non-subword)
            sentence_restart_flag = True
        elif not is_subword(token):
            current_word_id += 1
            word_ids.append(current_word_id)
        elif current_word_id == -1:  # First token of a sequence (or after reset SEP) is a subword
            current_word_id = 0
            word_ids.append(current_word_id)
        else:  # Subword of an existing word
            word_ids.append(current_word_id)
    return word_ids


def col_round(x: float) -> int:
    """Colloquial rounding where 0.5 rounds to 1"""
    frac = x - math.floor(x)
    if frac < 0.5:
        return math.floor(x)
    return math.ceil(x)


def get_uniform_id(si: int, max_intermediate_subwords: int, num_intermediate_subwords: int) -> int:
    """Calculates uniform id for the given subword index, si, and max and number of intermediate subwords"""
    if num_intermediate_subwords == 0:
        return 0
    effective_max_pos = max(0, max_intermediate_subwords - 1)
    return col_round(si * effective_max_pos / num_intermediate_subwords)


def get_ids_from_subwords(
    num_subwords_in_current_word: int,
    max_intermediate_subword_positions_per_word: int,
    subword_embedding_order: str,
    intermediate_subword_distribution_strategy: str,
    current_word_starts_with_subword: bool = False,
) -> List[int]:
    """Calculate subword ids for the tokens of a single word."""

    if num_subwords_in_current_word == 0:
        return []

    if current_word_starts_with_subword:
        if num_subwords_in_current_word == 1:
            return [1]
    elif num_subwords_in_current_word == 1:
        return [0]

    if subword_embedding_order == "ending_first":
        subword_ids: List[int] = []

        has_explicit_root = not current_word_starts_with_subword and num_subwords_in_current_word > 0
        has_explicit_last = num_subwords_in_current_word > 1 or (
            current_word_starts_with_subword and num_subwords_in_current_word == 1
        )

        if has_explicit_root:
            subword_ids.append(0)
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word - 1
        else:
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word

        if has_explicit_last:
            num_intermediate_tokens = num_tokens_for_intermediate_and_last - 1
        else:
            num_intermediate_tokens = num_tokens_for_intermediate_and_last

        if num_intermediate_tokens < 0:
            num_intermediate_tokens = 0

        if num_intermediate_tokens > 0:
            if num_intermediate_tokens <= max_intermediate_subword_positions_per_word:
                for si in range(num_intermediate_tokens):
                    subword_ids.append(2 + si)
            else:
                if intermediate_subword_distribution_strategy == "uniform":
                    for si in range(num_intermediate_tokens):
                        subword_ids.append(
                            2
                            + get_uniform_id(si, max_intermediate_subword_positions_per_word, num_intermediate_tokens)
                        )
                elif intermediate_subword_distribution_strategy == "leftover_as_last":
                    for si in range(max_intermediate_subword_positions_per_word):
                        subword_ids.append(2 + si)
                    for _ in range(num_intermediate_tokens - max_intermediate_subword_positions_per_word):
                        subword_ids.append(1)
                else:
                    raise ValueError(
                        f"Unsupported intermediate subword distribution strategy: {intermediate_subword_distribution_strategy}"
                    )

        if has_explicit_last:
            subword_ids.append(1)

        return subword_ids
    else:
        raise ValueError(f"Unsupported subword embedding order: {subword_embedding_order}")


def create_subword_ids(
    tokens: List[str],
    max_intermediate_subword_positions_per_word: int,
    subword_embedding_order: str,
    intermediate_subword_distribution_strategy: str,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token="[PAD]",
) -> List[int]:
    """Creates subword ids for the given tokens and parameters."""

    if not tokens:
        return []

    all_subword_ids: List[int] = []
    current_word_segment_tokens: List[str] = []

    first_content_token_is_subword = False
    if tokens:
        for token_val in tokens:
            if token_val not in [cls_token, sep_token, pad_token]:
                first_content_token_is_subword = is_subword(token_val)
                break

    first_content_word_processed = False

    for token_idx, token in enumerate(tokens):
        if token in [cls_token, sep_token, pad_token]:
            if current_word_segment_tokens:
                is_this_segment_the_very_first_content_word_and_starts_with_subword = (
                    first_content_token_is_subword
                    and not first_content_word_processed
                    and is_subword(current_word_segment_tokens[0])
                )

                generated_ids = get_ids_from_subwords(
                    num_subwords_in_current_word=len(current_word_segment_tokens),
                    max_intermediate_subword_positions_per_word=max_intermediate_subword_positions_per_word,
                    subword_embedding_order=subword_embedding_order,
                    intermediate_subword_distribution_strategy=intermediate_subword_distribution_strategy,
                    current_word_starts_with_subword=is_this_segment_the_very_first_content_word_and_starts_with_subword,
                )
                all_subword_ids.extend(generated_ids)

                if not first_content_word_processed and current_word_segment_tokens:
                    first_content_word_processed = True
                current_word_segment_tokens = []

            all_subword_ids.append(0)
        elif not is_subword(token):
            if current_word_segment_tokens:
                is_this_segment_the_very_first_content_word_and_starts_with_subword = (
                    first_content_token_is_subword
                    and not first_content_word_processed
                    and is_subword(current_word_segment_tokens[0])
                )

                generated_ids = get_ids_from_subwords(
                    num_subwords_in_current_word=len(current_word_segment_tokens),
                    max_intermediate_subword_positions_per_word=max_intermediate_subword_positions_per_word,
                    subword_embedding_order=subword_embedding_order,
                    intermediate_subword_distribution_strategy=intermediate_subword_distribution_strategy,
                    current_word_starts_with_subword=is_this_segment_the_very_first_content_word_and_starts_with_subword,
                )
                all_subword_ids.extend(generated_ids)
                if not first_content_word_processed and current_word_segment_tokens:
                    first_content_word_processed = True
            current_word_segment_tokens = [token]
        else:
            current_word_segment_tokens.append(token)

    if current_word_segment_tokens:
        is_this_segment_the_very_first_content_word_and_starts_with_subword = (
            first_content_token_is_subword
            and not first_content_word_processed
            and is_subword(current_word_segment_tokens[0])
        )

        generated_ids = get_ids_from_subwords(
            num_subwords_in_current_word=len(current_word_segment_tokens),
            max_intermediate_subword_positions_per_word=max_intermediate_subword_positions_per_word,
            subword_embedding_order=subword_embedding_order,
            intermediate_subword_distribution_strategy=intermediate_subword_distribution_strategy,
            current_word_starts_with_subword=is_this_segment_the_very_first_content_word_and_starts_with_subword,
        )
        all_subword_ids.extend(generated_ids)

    return all_subword_ids


class Bert2DTokenizer(BertTokenizer):
    r"""
    Construct a BERT2D tokenizer. Based on WordPiece.

    This tokenizer inherits from [`BertTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods. Bert2DTokenizer adds functionality for generating
    `word_ids` and `subword_ids` which are used for 2D positional embeddings.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`.
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
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        max_intermediate_subword_positions_per_word (`int`, *optional*, defaults to `1`):
            The maximum number of intermediate subword positions per word. This is used to determine how many subword
            positions are allowed for each word in the tokenization process.
        subword_embedding_order (`str`, *optional*, defaults to `"ending_first"`):
            The order in which subword embeddings are processed. Can be `"ending_first"`.
        intermediate_subword_distribution_strategy (`str`, *optional*, defaults to `"uniform"`):
            The strategy for distributing intermediate subword positions. Can be `"uniform"` or `"leftover_as_last"`.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask", "word_ids", "subword_ids"]

    def __init__(
        self,
        vocab_file,
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
        max_intermediate_subword_positions_per_word=1,
        subword_embedding_order="ending_first",
        intermediate_subword_distribution_strategy="uniform",
        **kwargs,
    ):
        # Step 1: Store Bert2D-specific args from init signature into local variables
        local_max_intermediate = max_intermediate_subword_positions_per_word
        local_subword_order = subword_embedding_order
        local_intermediate_strategy = intermediate_subword_distribution_strategy

        # Step 2: Remove Bert2D-specific args from kwargs to prevent passing them to super()
        kwargs.pop("max_intermediate_subword_positions_per_word", None)
        kwargs.pop("subword_embedding_order", None)
        kwargs.pop("intermediate_subword_distribution_strategy", None)

        # Step 3: Call super().__init__(), explicitly passing Bert2DTokenizer.model_input_names
        # This ensures that PreTrainedTokenizerBase uses our desired model_input_names.
        super().__init__(
            vocab_file=vocab_file,
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

        # Step 4: Set Bert2D specific attributes on self using the stored local variables
        self.max_intermediate_subword_positions_per_word = local_max_intermediate
        self.subword_embedding_order = local_subword_order
        self.intermediate_subword_distribution_strategy = local_intermediate_strategy

        if subword_embedding_order != "ending_first":
            logger.warning(
                f"Bert2DTokenizer slow currently only fully supports 'ending_first' for subword_embedding_order. Received: {subword_embedding_order}"
            )

        # Step 5: Update init_kwargs if it exists (for serialization/reconstruction by base classes)
        # This makes sure that when the tokenizer is saved and reloaded, these custom parameters are preserved.
        if hasattr(self, "init_kwargs") and isinstance(self.init_kwargs, dict):
            self.init_kwargs["max_intermediate_subword_positions_per_word"] = (
                self.max_intermediate_subword_positions_per_word
            )
            self.init_kwargs["subword_embedding_order"] = self.subword_embedding_order
            self.init_kwargs["intermediate_subword_distribution_strategy"] = (
                self.intermediate_subword_distribution_strategy
            )
        # else:
        # This case might occur if the superclass doesn't initialize init_kwargs,
        # which would be unusual for PreTrainedTokenizer based classes.
        # logger.warning("self.init_kwargs not found or not a dict during Bert2DTokenizer __init__. Custom params might not be saved.")

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        batch_encoding_super = super().__call__(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # Get lists first to allow modification
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        input_ids_processed = batch_encoding_super["input_ids"]

        is_batched = bool(
            (isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple)))
            or (
                isinstance(text, (list, tuple))
                and text_pair is not None
                and isinstance(text_pair, (list, tuple))
                and text
                and isinstance(text[0], str)
            )
        )
        if not is_batched and isinstance(text, (list, tuple)) and not (text and isinstance(text[0], (list, tuple))):
            if (
                isinstance(input_ids_processed, list)
                and bool(input_ids_processed)
                and isinstance(input_ids_processed[0], list)
            ):
                is_batched = True
        elif not is_batched and text_pair is not None and isinstance(text_pair, (list, tuple)):
            if (
                isinstance(input_ids_processed, list)
                and bool(input_ids_processed)
                and isinstance(input_ids_processed[0], list)
            ):
                is_batched = True

        list_of_input_ids_for_processing: List[List[int]]
        if not is_batched:
            list_of_input_ids_for_processing = [input_ids_processed]
        else:
            list_of_input_ids_for_processing = input_ids_processed

        all_word_ids: List[List[int]] = []
        all_subword_ids: List[List[int]] = []

        for ids_for_one_sequence in list_of_input_ids_for_processing:
            tokens = self.convert_ids_to_tokens(ids_for_one_sequence, skip_special_tokens=False)

            should_restart_word_ids_heuristic = text_pair is not None

            word_ids_for_sequence = create_word_ids(
                tokens,
                restart_new_sentence=should_restart_word_ids_heuristic,
                seperator_token=self.sep_token,
                padding_token=self.pad_token,
            )
            subword_ids_for_sequence = create_subword_ids(
                tokens,
                max_intermediate_subword_positions_per_word=self.max_intermediate_subword_positions_per_word,
                subword_embedding_order=self.subword_embedding_order,
                intermediate_subword_distribution_strategy=self.intermediate_subword_distribution_strategy,
                cls_token=self.cls_token,
                sep_token=self.sep_token,
                pad_token=self.pad_token,
            )
            all_word_ids.append(word_ids_for_sequence)
            all_subword_ids.append(subword_ids_for_sequence)

        if not is_batched:
            batch_encoding_super["word_ids"] = all_word_ids[0]
            batch_encoding_super["subword_ids"] = all_subword_ids[0]
        else:
            batch_encoding_super["word_ids"] = all_word_ids
            batch_encoding_super["subword_ids"] = all_subword_ids

        if return_tensors is not None:
            batch_encoding_super = batch_encoding_super.convert_to_tensors(tensor_type=return_tensors)

        return batch_encoding_super

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.
        This method includes the generation of word_ids and subword_ids specific to Bert2D.
        """
        # Call parent's encode_plus first to get standard tokenization
        result = super().encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # Process as lists first, then convert to tensor if needed
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Check if we have overflow tokens (multiple sequences in result)
        has_overflow = return_overflowing_tokens and "overflowing_tokens" in result

        # Determine if result is batched (could be batched if overflow tokens are present)
        is_batched_output = (
            isinstance(result["input_ids"], list) and result["input_ids"] and isinstance(result["input_ids"][0], list)
        )

        # If we have overflow tokens OR the result is already batched
        if has_overflow or is_batched_output:
            # We'll need to process each sequence separately
            batch_size = len(result["input_ids"]) if is_batched_output else 1
            batch_word_ids = []
            batch_subword_ids = []

            for i in range(batch_size):
                # Get tokens for this sequence
                tokens = self.convert_ids_to_tokens(
                    result["input_ids"][i] if is_batched_output else result["input_ids"], skip_special_tokens=False
                )

                # Determine if this sequence contains multiple sentences by counting SEP tokens
                should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

                word_ids = create_word_ids(
                    tokens,
                    restart_new_sentence=should_restart_word_ids_heuristic,
                    seperator_token=self.sep_token,
                    padding_token=self.pad_token,
                )

                subword_ids = create_subword_ids(
                    tokens,
                    max_intermediate_subword_positions_per_word=self.max_intermediate_subword_positions_per_word,
                    subword_embedding_order=self.subword_embedding_order,
                    intermediate_subword_distribution_strategy=self.intermediate_subword_distribution_strategy,
                    cls_token=self.cls_token,
                    sep_token=self.sep_token,
                    pad_token=self.pad_token,
                )

                batch_word_ids.append(word_ids)
                batch_subword_ids.append(subword_ids)

            # Add to result
            if is_batched_output:
                result["word_ids"] = batch_word_ids
                result["subword_ids"] = batch_subword_ids
            else:
                # If input was single but we have overflow tokens, result should still be a list
                result["word_ids"] = batch_word_ids[0]
                result["subword_ids"] = batch_subword_ids[0]
        else:
            # Standard case - no overflow, single sequence
            tokens = self.convert_ids_to_tokens(result["input_ids"], skip_special_tokens=False)

            # Determine if this sequence contains multiple sentences by counting SEP tokens
            should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

            word_ids = create_word_ids(
                tokens,
                restart_new_sentence=should_restart_word_ids_heuristic,
                seperator_token=self.sep_token,
                padding_token=self.pad_token,
            )

            subword_ids = create_subword_ids(
                tokens,
                max_intermediate_subword_positions_per_word=self.max_intermediate_subword_positions_per_word,
                subword_embedding_order=self.subword_embedding_order,
                intermediate_subword_distribution_strategy=self.intermediate_subword_distribution_strategy,
                cls_token=self.cls_token,
                sep_token=self.sep_token,
                pad_token=self.pad_token,
            )

            # Add custom fields to result
            result["word_ids"] = word_ids
            result["subword_ids"] = subword_ids

        # Convert to tensors if requested
        if return_tensors is not None:
            result = result.convert_to_tensors(tensor_type=return_tensors)

        return result

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[Union[TextInput, PreTokenizedInput]],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare a batch of sequences or a batch of sequence pairs for the model.
        This method includes the generation of word_ids and subword_ids specific to Bert2D.
        """
        # Call the parent's batch_encode_plus first to get standard tokenization
        result = super().batch_encode_plus(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # Process as lists first, then convert to tensor if needed
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Generate word_ids and subword_ids for each item in the batch
        # Use the actual batch size from result["input_ids"], which includes overflow sequences
        batch_size = len(result["input_ids"])
        batch_word_ids = []
        batch_subword_ids = []

        for i in range(batch_size):
            # Get tokens for this batch item
            tokens = self.convert_ids_to_tokens(result["input_ids"][i], skip_special_tokens=False)

            # Determine if this sequence contains multiple sentences by counting SEP tokens
            should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

            word_ids = create_word_ids(
                tokens,
                restart_new_sentence=should_restart_word_ids_heuristic,
                seperator_token=self.sep_token,
                padding_token=self.pad_token,
            )

            subword_ids = create_subword_ids(
                tokens,
                max_intermediate_subword_positions_per_word=self.max_intermediate_subword_positions_per_word,
                subword_embedding_order=self.subword_embedding_order,
                intermediate_subword_distribution_strategy=self.intermediate_subword_distribution_strategy,
                cls_token=self.cls_token,
                sep_token=self.sep_token,
                pad_token=self.pad_token,
            )

            batch_word_ids.append(word_ids)
            batch_subword_ids.append(subword_ids)

        # Add custom fields to result
        result["word_ids"] = batch_word_ids
        result["subword_ids"] = batch_subword_ids

        # Convert to tensors if requested
        if return_tensors is not None:
            result = result.convert_to_tensors(tensor_type=return_tensors)

        return result

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids, for the model.
        This method adds `word_ids` and `subword_ids` specific to Bert2D.
        """
        # Get the standard outputs from the parent class
        prepared_inputs = super().prepare_for_model(
            ids=ids,
            pair_ids=pair_ids,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # Process as lists first
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            prepend_batch_axis=prepend_batch_axis,
            **kwargs,
        )

        # Convert input_ids to tokens to generate word_ids and subword_ids
        tokens = self.convert_ids_to_tokens(prepared_inputs["input_ids"])

        # Heuristic to check if we have a sentence pair
        should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

        # Create and add word_ids
        prepared_inputs["word_ids"] = create_word_ids(
            tokens,
            restart_new_sentence=should_restart_word_ids_heuristic,
            seperator_token=self.sep_token,
            padding_token=self.pad_token,
        )

        # Create and add subword_ids
        prepared_inputs["subword_ids"] = create_subword_ids(
            tokens,
            self.max_intermediate_subword_positions_per_word,
            self.subword_embedding_order,
            self.intermediate_subword_distribution_strategy,
            cls_token=self.cls_token,
            sep_token=self.sep_token,
            pad_token=self.pad_token,
        )

        # Convert to tensors if requested
        if return_tensors is not None:
            prepared_inputs = prepared_inputs.convert_to_tensors(tensor_type=return_tensors)

        return prepared_inputs

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        tools=None,
        documents=None,
        add_generation_prompt=False,
        tokenize=True,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        """
        Override apply_chat_template to fix tensor dimension issues when
        return_tensors="pt" is used with single conversations and return_assistant_tokens_mask=True.
        """
        # Check if we need to apply the fix
        needs_tensor_fix = (
            return_tensors == "pt"
            and return_assistant_tokens_mask
            and return_dict
            and tokenize
            and not isinstance(conversation[0], list)
            if conversation
            else False  # Single conversation, not batched
        )

        if needs_tensor_fix:
            # For single conversations with tensor output, temporarily disable tensor conversion
            # and handle it manually after the call
            result = super().apply_chat_template(
                conversation=conversation,
                chat_template=chat_template,
                tools=tools,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=None,  # Disable tensor conversion temporarily
                return_dict=return_dict,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                tokenizer_kwargs=tokenizer_kwargs,
                **kwargs,
            )

            # Now manually convert to tensors ensuring proper dimensions
            if return_tensors == "pt":
                import torch

                # Convert each field to tensors with proper dimensions
                for key, value in result.items():
                    if isinstance(value, list):
                        # Convert list to tensor and ensure it has a batch dimension
                        tensor = torch.tensor(value)
                        # Ensure we have at least 1D (for sequences) and add batch dimension if needed
                        if tensor.dim() == 0:
                            tensor = tensor.unsqueeze(0)
                        if tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)  # Add batch dimension
                        result[key] = tensor

            return result
        else:
            # For all other cases, use the parent implementation
            return super().apply_chat_template(
                conversation=conversation,
                chat_template=chat_template,
                tools=tools,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_dict=return_dict,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                tokenizer_kwargs=tokenizer_kwargs,
                **kwargs,
            )


__all__ = [
    "Bert2DTokenizer",
    "is_subword",
    "create_word_ids",
    "create_subword_ids",
    "col_round",
    "get_uniform_id",
    "get_ids_from_subwords",
]
