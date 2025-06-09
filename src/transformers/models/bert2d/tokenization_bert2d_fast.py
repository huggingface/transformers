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
"""Fast Tokenization classes for Bert."""

import math
from typing import Dict, List, Optional, Union

from ... import BatchEncoding
from ...tokenization_utils_base import EncodedInput, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import PaddingStrategy, TensorType, logging
from ..bert import BertTokenizerFast


logger = logging.get_logger(__name__)


class Bert2DTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" BERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `\"[UNK]\"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `\"[SEP]\"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `\"[PAD]\"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `\"[CLS]\"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `\"[MASK]\"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `\"##\"`):
            The prefix for subwords.
        max_intermediate_subword_positions_per_word (`int`, *optional*, defaults to `1`):
            The maximum number of intermediate subword positions per word. This is used to determine how many subword
            positions are allowed for each word in the tokenization process.
        subword_embedding_order (`str`, *optional*, defaults to `\"ending_first\"`):
            The order in which subword embeddings are processed. Can be `\"ending_first\"` or `\"starting_first\"`.
        intermediate_subword_distribution_strategy (`str`, *optional*, defaults to `\"uniform\"`):
            The strategy for distributing intermediate subword positions. Can be `\"uniform\"` or `\"random\"`.
            (Note: The original prompt mentioned "uniform" or "random", but the function code provided earlier
            implemented "uniform" or "leftover_as_last". This docstring reflects the prompt's options.)
    """

    model_input_names: List[str] = ["input_ids", "token_type_ids", "word_ids", "subword_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
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
        super().__init__(
            vocab_file=vocab_file,  # Ensure vocab_file is passed correctly
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        self.max_intermediate_subword_positions_per_word = max_intermediate_subword_positions_per_word
        self.subword_embedding_order = subword_embedding_order
        self.intermediate_subword_distribution_strategy = intermediate_subword_distribution_strategy

        # Ensure init_kwargs includes Bert2D specific parameters for correct saving and loading
        self.init_kwargs["max_intermediate_subword_positions_per_word"] = max_intermediate_subword_positions_per_word
        self.init_kwargs["subword_embedding_order"] = subword_embedding_order
        self.init_kwargs["intermediate_subword_distribution_strategy"] = intermediate_subword_distribution_strategy

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
        # Use keyword arguments for the super call (as per previous fix)
        result = super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,  # Pass padding argument to super
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

        # Determine batch characteristics from the processed 'result'
        # Input_ids can be a list (single example) or list of lists (batch)
        is_batched_output = (
            isinstance(result["input_ids"], list) and result["input_ids"] and isinstance(result["input_ids"][0], list)
        )

        actual_batch_size: int
        seq_length: int  # This will be the length after padding/truncation by super().__call__

        if is_batched_output:
            actual_batch_size = len(result["input_ids"])
            seq_length = len(result["input_ids"][0]) if actual_batch_size > 0 else 0
        else:  # Single, non-batched list of ints
            actual_batch_size = 1
            seq_length = len(result["input_ids"])
            # Temporarily wrap single example to use unified loop
            if "input_ids" in result and not isinstance(result["input_ids"][0], list):
                for key in result:
                    if isinstance(result[key], list):  # type: ignore
                        result[key] = [result[key]]  # type: ignore

        # Generate word_ids and subword_ids as lists of lists
        list_of_list_word_ids: List[List[int]] = []
        list_of_list_subword_ids: List[List[int]] = []

        for i in range(actual_batch_size):
            # .tokens() is a method of BatchEncoding from a fast tokenizer
            # It requires an index if the BatchEncoding is from a batch.
            # If BatchEncoding is from a single example, it doesn't take an index.
            # The `result` from super().__call__ is a BatchEncoding.
            # If the original input to __call__ was a batch, result.tokens(i) is correct.
            # If the original input was single, result.tokens() is correct.
            # The BatchEncoding object itself handles this.
            current_tokens = result.tokens(i)  # type: ignore

            # Determine if this sequence contains multiple sentences by checking for SEP tokens
            should_restart_word_ids_heuristic = current_tokens.count(self.sep_token) >= 2

            list_of_list_word_ids.append(
                create_word_ids(
                    current_tokens,
                    restart_new_sentence=should_restart_word_ids_heuristic,
                    seperator_token=self.sep_token,
                    padding_token=self.pad_token,
                )
            )
            list_of_list_subword_ids.append(
                create_subword_ids(
                    current_tokens,
                    self.max_intermediate_subword_positions_per_word,
                    self.subword_embedding_order,
                    self.intermediate_subword_distribution_strategy,
                    cls_token=self.cls_token,
                    sep_token=self.sep_token,
                    pad_token=self.pad_token,
                )
            )

        padding_value_for_ids = 0  # Standard padding for word_ids/subword_ids

        # Pad word_ids and subword_ids to seq_length if padding was enabled
        if padding_strategy_uses_max_length(padding, max_length):
            for i in range(actual_batch_size):
                for id_list in [list_of_list_word_ids[i], list_of_list_subword_ids[i]]:
                    current_len = len(id_list)
                    pad_len = seq_length - current_len
                    if pad_len > 0:
                        if self.padding_side == "right":
                            id_list.extend([padding_value_for_ids] * pad_len)
                        else:  # padding_side == "left"
                            for _ in range(pad_len):
                                id_list.insert(0, padding_value_for_ids)
                    elif pad_len < 0:  # Truncate if longer (should ideally not happen if tokens were truncated)
                        if self.padding_side == "right":  # or truncation_side
                            del id_list[seq_length:]
                        else:
                            del id_list[:-seq_length]

        if not is_batched_output:  # Unwrap back to single list if original was single
            result["word_ids"] = list_of_list_word_ids[0]
            result["subword_ids"] = list_of_list_subword_ids[0]
            # Unwrap other keys if they were wrapped
            for key in list(result.keys()):  # Iterate over a copy of keys
                if isinstance(result[key], list) and len(result[key]) == 1 and key not in ["word_ids", "subword_ids"]:
                    # Check if it was a list of lists that became a list of one list
                    if isinstance(result[key][0], list):
                        result[key] = result[key][0]  # type: ignore
        else:
            result["word_ids"] = list_of_list_word_ids
            result["subword_ids"] = list_of_list_subword_ids

        if return_tensors is not None:
            result = result.convert_to_tensors(tensor_type=return_tensors)  # type: ignore

        return result  # type: ignore

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
            batch_size = len(result["input_ids"]) if is_batched_output else 1 + len(result["overflowing_tokens"])
            batch_word_ids = []
            batch_subword_ids = []

            for i in range(batch_size):
                # Get tokens for this sequence
                tokens = result.tokens(i)

                # Determine if this sequence contains multiple sentences
                should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

                word_ids = create_word_ids(
                    tokens,
                    restart_new_sentence=should_restart_word_ids_heuristic,
                    seperator_token=self.sep_token,
                    padding_token=self.pad_token,
                )

                subword_ids = create_subword_ids(
                    tokens,
                    self.max_intermediate_subword_positions_per_word,
                    self.subword_embedding_order,
                    self.intermediate_subword_distribution_strategy,
                    cls_token=self.cls_token,
                    sep_token=self.sep_token,
                    pad_token=self.pad_token,
                )

                batch_word_ids.append(word_ids)
                batch_subword_ids.append(subword_ids)

            # Add to result
            result["word_ids"] = batch_word_ids
            result["subword_ids"] = batch_subword_ids
        else:
            # Standard case - no overflow, single sequence
            tokens = result.tokens()

            # Determine if this sequence contains multiple sentences
            should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

            word_ids = create_word_ids(
                tokens,
                restart_new_sentence=should_restart_word_ids_heuristic,
                seperator_token=self.sep_token,
                padding_token=self.pad_token,
            )

            subword_ids = create_subword_ids(
                tokens,
                self.max_intermediate_subword_positions_per_word,
                self.subword_embedding_order,
                self.intermediate_subword_distribution_strategy,
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
            tokens = result.tokens(i)

            # Determine if this sequence contains multiple sentences
            should_restart_word_ids_heuristic = tokens.count(self.sep_token) >= 2

            word_ids = create_word_ids(
                tokens,
                restart_new_sentence=should_restart_word_ids_heuristic,
                seperator_token=self.sep_token,
                padding_token=self.pad_token,
            )

            subword_ids = create_subword_ids(
                tokens,
                self.max_intermediate_subword_positions_per_word,
                self.subword_embedding_order,
                self.intermediate_subword_distribution_strategy,
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

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        effective_padding_side = padding_side if padding_side is not None else self.padding_side

        # Separate word_ids and subword_ids if they are lists.
        # Tensors are assumed to be handled by the __call__ method's padding logic
        # or are already in the correct shape if this method is reached.
        word_ids_list = None
        if "word_ids" in encoded_inputs and isinstance(encoded_inputs["word_ids"], list):
            word_ids_list = encoded_inputs.pop("word_ids")

        subword_ids_list = None
        if "subword_ids" in encoded_inputs and isinstance(encoded_inputs["subword_ids"], list):
            subword_ids_list = encoded_inputs.pop("subword_ids")

        # Call the superclass's _pad method to handle standard keys like input_ids, attention_mask, etc.
        # CRITICAL: Pass all relevant arguments, especially `padding_side`.
        padded_standard_inputs = super()._pad(
            encoded_inputs,  # This now only contains standard keys if custom keys were lists and popped
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=effective_padding_side,  # Pass the determined padding_side
            return_attention_mask=return_attention_mask,
        )

        # Now, handle padding for word_ids and subword_ids if they were lists and were popped.
        # This padding should align with how input_ids were padded by the super()._pad call.
        # This logic is primarily for cases where inputs are not yet tensors (e.g. return_tensors=None).
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and max_length is not None:
            main_input_name = self.model_input_names[0]  # usually "input_ids"
            if main_input_name not in padded_standard_inputs:
                # This case should ideally not happen if _pad is called correctly.
                # Fallback to adding custom IDs without padding if main input is missing.
                if word_ids_list is not None:
                    padded_standard_inputs["word_ids"] = word_ids_list
                if subword_ids_list is not None:
                    padded_standard_inputs["subword_ids"] = subword_ids_list
                return padded_standard_inputs

            padded_len = len(padded_standard_inputs[main_input_name])
            padding_val = 0  # Standard padding value for these custom IDs

            if word_ids_list is not None:
                current_len = len(word_ids_list)
                diff = padded_len - current_len
                if diff > 0:
                    if effective_padding_side == "right":
                        padded_standard_inputs["word_ids"] = word_ids_list + [padding_val] * diff
                    else:  # left
                        padded_standard_inputs["word_ids"] = [padding_val] * diff + word_ids_list
                else:  # No padding needed or truncation occurred (list might be longer)
                    padded_standard_inputs["word_ids"] = word_ids_list[:padded_len]  # Ensure it's not longer
            # If word_ids was not a list (e.g., a tensor passed through), and was not popped,
            # it might still be in `encoded_inputs` (the original dict passed to _pad).
            # If so, and it wasn't popped, ensure it's in the output.
            elif "word_ids" in encoded_inputs:
                padded_standard_inputs["word_ids"] = encoded_inputs["word_ids"]

            if subword_ids_list is not None:
                current_len = len(subword_ids_list)
                diff = padded_len - current_len
                if diff > 0:
                    if effective_padding_side == "right":
                        padded_standard_inputs["subword_ids"] = subword_ids_list + [padding_val] * diff
                    else:  # left
                        padded_standard_inputs["subword_ids"] = [padding_val] * diff + subword_ids_list
                else:
                    padded_standard_inputs["subword_ids"] = subword_ids_list[:padded_len]
            elif "subword_ids" in encoded_inputs:
                padded_standard_inputs["subword_ids"] = encoded_inputs["subword_ids"]

        else:  # No padding was applied to standard inputs, or no max_length specified
            if word_ids_list is not None:
                padded_standard_inputs["word_ids"] = word_ids_list
            elif "word_ids" in encoded_inputs:  # Ensure it's carried over if not popped
                padded_standard_inputs["word_ids"] = encoded_inputs["word_ids"]

            if subword_ids_list is not None:
                padded_standard_inputs["subword_ids"] = subword_ids_list
            elif "subword_ids" in encoded_inputs:  # Ensure it's carried over if not popped
                padded_standard_inputs["subword_ids"] = encoded_inputs["subword_ids"]

        return padded_standard_inputs


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
    if num_intermediate_subwords == 0:  # Avoid division by zero if there are no intermediate subwords
        return 0
    # Effective max position is max_intermediate_subwords - 1 because positions are 0-indexed
    # e.g., if max_intermediate_subwords is 1, effective_max_pos is 0.
    # if max_intermediate_subwords is 2, effective_max_pos is 1 (positions 0, 1).
    effective_max_pos = max(0, max_intermediate_subwords - 1)  # Ensure non-negative
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

    # Handle cases where the "word" is just one token
    if current_word_starts_with_subword:  # Word like "##ing"
        if num_subwords_in_current_word == 1:
            return [1]  # Treat as "last" subword if it's the only token and starts with ##
    elif num_subwords_in_current_word == 1:  # Word like "run"
        return [0]  # Treat as "root" subword

    # For multi-token words
    if subword_embedding_order == "ending_first":
        subword_ids: List[int] = []

        has_explicit_root = not current_word_starts_with_subword and num_subwords_in_current_word > 0
        # "Last" subword exists if there's more than one token,
        # OR if it's a single token starting with ## (handled above, but for clarity here)
        has_explicit_last = num_subwords_in_current_word > 1 or (
            current_word_starts_with_subword and num_subwords_in_current_word == 1
        )

        if has_explicit_root:
            subword_ids.append(0)  # Root token (e.g., "run" in "running")
            # Tokens remaining for intermediate and last part
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word - 1
        else:  # Word starts with subword (e.g., "##run" in "##running")
            # All tokens contribute to intermediate and last part (or just last if only one ##token)
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word

        num_intermediate_tokens = 0
        if has_explicit_last:
            # If there's a distinct "last" token, subtract it from the count
            num_intermediate_tokens = num_tokens_for_intermediate_and_last - 1
        else:
            # If no distinct "last" token (e.g. "##word" - only one token, no root),
            # then all (remaining) tokens are considered intermediate.
            # This case should be rare if num_subwords_in_current_word > 1
            num_intermediate_tokens = num_tokens_for_intermediate_and_last

        # Ensure non-negative, can happen if num_subwords_in_current_word is 1 and has_explicit_last is true.
        if num_intermediate_tokens < 0:
            num_intermediate_tokens = 0

        # Assign IDs to intermediate tokens
        if num_intermediate_tokens > 0:
            if num_intermediate_tokens <= max_intermediate_subword_positions_per_word:
                # If fewer or equal intermediate tokens than available slots, assign unique IDs
                for si in range(num_intermediate_tokens):
                    subword_ids.append(2 + si)  # IDs 2, 3, ..., (2+max_intermediate_subword_positions_per_word-1)
            else:  # More intermediate tokens than available slots
                if intermediate_subword_distribution_strategy == "uniform":
                    for si in range(num_intermediate_tokens):
                        subword_ids.append(
                            2
                            + get_uniform_id(si, max_intermediate_subword_positions_per_word, num_intermediate_tokens)
                        )
                elif intermediate_subword_distribution_strategy == "leftover_as_last":
                    # Fill available intermediate slots
                    for si in range(max_intermediate_subword_positions_per_word):
                        subword_ids.append(2 + si)
                    # Assign remaining intermediate tokens as "last" (ID 1)
                    for _ in range(num_intermediate_tokens - max_intermediate_subword_positions_per_word):
                        subword_ids.append(1)
                else:
                    raise ValueError(
                        f"Unsupported intermediate subword distribution strategy: {intermediate_subword_distribution_strategy}"
                    )

        if has_explicit_last:
            subword_ids.append(1)  # Last token (e.g., "##ing" in "running")

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

    # Determine if the very first content token (non-special) is a subword.
    # This helps decide if the first word itself starts with a subword prefix.
    first_content_token_is_subword = False
    if tokens:  # Check if tokens list is not empty
        for token_val in tokens:
            if token_val not in [cls_token, sep_token, pad_token]:
                first_content_token_is_subword = is_subword(token_val)
                break  # Found the first content token

    first_content_word_processed = False  # Flag to track if we've processed the first actual word

    for token_idx, token in enumerate(tokens):
        if token in [cls_token, sep_token, pad_token]:  # Special tokens
            # If there was an ongoing word segment, process it first
            if current_word_segment_tokens:
                # Determine if this segment is the very first content word AND it starts with a subword
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

                if (
                    not first_content_word_processed and current_word_segment_tokens
                ):  # Mark first content word as processed
                    first_content_word_processed = True
                current_word_segment_tokens = []  # Reset for next word

            all_subword_ids.append(0)  # Special tokens get subword_id 0
        elif not is_subword(token):  # Token is a root word (doesn't start with ##)
            # If there was an ongoing word segment (which must have been all subwords), process it
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
            current_word_segment_tokens = [token]  # Start a new word segment with this root token
        else:  # Token is a subword (starts with ##)
            current_word_segment_tokens.append(token)

    # After loop, process any remaining word segment
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


def padding_strategy_uses_max_length(
    padding_strategy: Union[bool, str, PaddingStrategy], max_length: Optional[int]
) -> bool:
    """Helper to determine if padding will occur up to a max_length."""
    if padding_strategy is False or padding_strategy == PaddingStrategy.DO_NOT_PAD:
        return False
    if padding_strategy is True or padding_strategy == PaddingStrategy.LONGEST:
        # Padding to longest in batch still implies a fixed length for that batch
        return True
    if padding_strategy == PaddingStrategy.MAX_LENGTH:
        return max_length is not None
    return False


__all__ = ["Bert2DTokenizerFast"]
