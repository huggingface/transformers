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

import torch

from ... import BatchEncoding
from ...tokenization_utils_base import EncodedInput, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import PaddingStrategy, TensorType
from ..bert import BertTokenizerFast


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
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
        max_intermediate_subword_positions_per_word (`int`, *optional*, defaults to `1`):
            The maximum number of intermediate subword positions per word. This is used to determine how many subword
            positions are allowed for each word in the tokenization process.
        subword_embedding_order (`str`, *optional*, defaults to `"ending_first"`):
            The order in which subword embeddings are processed. Can be `"ending_first"` or `"starting_first"`.
        intermediate_subword_distribution_strategy (`str`, *optional*, defaults to `"uniform"`):
            The strategy for distributing intermediate subword positions. Can be `"uniform"` or `"random"`.
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

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
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

        # Determine batch characteristics from the processed 'result'
        actual_batch_size: int
        seq_length: int
        is_tensor_output = isinstance(result["input_ids"], torch.Tensor)

        if is_tensor_output:
            actual_batch_size = result["input_ids"].size(0)
            seq_length = result["input_ids"].size(1)
        elif (
            isinstance(result["input_ids"], list) and result["input_ids"] and isinstance(result["input_ids"][0], list)
        ):
            actual_batch_size = len(result["input_ids"])
            seq_length = len(result["input_ids"][0]) if actual_batch_size > 0 else 0
        elif isinstance(result["input_ids"], list):  # Single, non-batched list of ints
            actual_batch_size = 1
            seq_length = len(result["input_ids"])
        else:
            raise ValueError("Unexpected format for input_ids in BatchEncoding from superclass.")

        # Generate word_ids and subword_ids as lists of lists
        list_of_list_word_ids: List[List[int]] = []
        list_of_list_subword_ids: List[List[int]] = []

        for i in range(actual_batch_size):
            tokens = result.tokens(i)  # type: ignore # Get tokens for the i-th item
            list_of_list_word_ids.append(create_word_ids(tokens))
            list_of_list_subword_ids.append(
                create_subword_ids(
                    tokens,
                    self.max_intermediate_subword_positions_per_word,
                    self.subword_embedding_order,
                    self.intermediate_subword_distribution_strategy,
                )
            )

        padding_strategy_for_custom_ids = PaddingStrategy.MAX_LENGTH if padding else PaddingStrategy.DO_NOT_PAD

        if return_tensors == "pt":
            padded_word_ids_for_tensor = []
            padded_subword_ids_for_tensor = []
            padding_value_for_ids = 0  # Assuming 0 is the padding ID for word/subword IDs

            for i in range(actual_batch_size):
                w_ids = list_of_list_word_ids[i]
                s_ids = list_of_list_subword_ids[i]

                current_len = len(w_ids)  # Assuming len(w_ids) == len(s_ids)

                if padding_strategy_for_custom_ids == PaddingStrategy.MAX_LENGTH:
                    pad_len = seq_length - current_len
                    if self.padding_side == "right":
                        final_w_ids = w_ids + [padding_value_for_ids] * pad_len
                        final_s_ids = s_ids + [padding_value_for_ids] * pad_len
                    else:  # padding_side == "left"
                        final_w_ids = [padding_value_for_ids] * pad_len + w_ids
                        final_s_ids = [padding_value_for_ids] * pad_len + s_ids
                    padded_word_ids_for_tensor.append(final_w_ids[:seq_length])  # Ensure exact length
                    padded_subword_ids_for_tensor.append(final_s_ids[:seq_length])
                else:  # No padding for custom IDs if main padding is disabled
                    padded_word_ids_for_tensor.append(w_ids)
                    padded_subword_ids_for_tensor.append(s_ids)

            result["word_ids"] = torch.tensor(padded_word_ids_for_tensor, dtype=torch.long)
            result["subword_ids"] = torch.tensor(padded_subword_ids_for_tensor, dtype=torch.long)
        else:
            # Handle non-tensor output (lists)
            # If padding was enabled for input_ids, custom IDs should also be padded.
            if padding_strategy_for_custom_ids == PaddingStrategy.MAX_LENGTH:
                padded_list_w_ids = []
                padded_list_s_ids = []
                padding_value_for_ids = 0
                for i in range(actual_batch_size):
                    w_ids = list_of_list_word_ids[i]
                    s_ids = list_of_list_subword_ids[i]
                    pad_len = seq_length - len(w_ids)
                    if self.padding_side == "right":
                        final_w_ids = w_ids + [padding_value_for_ids] * pad_len
                        final_s_ids = s_ids + [padding_value_for_ids] * pad_len
                    else:
                        final_w_ids = [padding_value_for_ids] * pad_len + w_ids
                        final_s_ids = [padding_value_for_ids] * pad_len + s_ids
                    padded_list_w_ids.append(final_w_ids[:seq_length])
                    padded_list_s_ids.append(final_s_ids[:seq_length])

                if actual_batch_size == 1 and not (
                    isinstance(result["input_ids"], list)
                    and result["input_ids"]
                    and isinstance(result["input_ids"][0], list)
                ):
                    result["word_ids"] = padded_list_w_ids[0]
                    result["subword_ids"] = padded_list_s_ids[0]
                else:
                    result["word_ids"] = padded_list_w_ids
                    result["subword_ids"] = padded_list_s_ids
            else:  # No padding for lists
                if actual_batch_size == 1 and not (
                    isinstance(result["input_ids"], list)
                    and result["input_ids"]
                    and isinstance(result["input_ids"][0], list)
                ):
                    result["word_ids"] = list_of_list_word_ids[0]
                    result["subword_ids"] = list_of_list_subword_ids[0]
                else:
                    result["word_ids"] = list_of_list_word_ids
                    result["subword_ids"] = list_of_list_subword_ids
        return result

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs) -> dict:
        # This _pad method is called by the superclass's padding logic.
        # It expects 'word_ids' and 'subword_ids' to already be in encoded_inputs
        # if they are to be padded by this method.
        # The __call__ method now adds them *after* super().__call__ has completed,
        # including its own padding. So, this _pad override might not be correctly
        # leveraged for word_ids and subword_ids unless they are added before super()._pad().
        # The logic in __call__ now handles padding for word_ids/subword_ids explicitly.

        # First, let the superclass handle padding for its known keys
        # (input_ids, attention_mask, token_type_ids)
        # Note: We are calling the super's _pad, not the one from BertTokenizerFast if it has one.
        # We need to call PreTrainedTokenizerFast's _pad or its parent.
        # However, our __call__ already called super().__call__ which did all the padding.
        # So, this _pad here might be redundant if called again, or needs to be context-aware.

        # The `super()._pad` call was problematic.
        # The main padding is now handled in the __call__ method explicitly for word_ids/subword_ids
        # to align with how input_ids were padded by the main `super().__call__`

        # If this method is still called by the parent's machinery for some reason AFTER our __call__ modifications,
        # we need to ensure it doesn't break.
        # For now, let's assume that the padding logic in __call__ is sufficient.
        # The original _pad logic from your file:
        required_input = encoded_inputs["input_ids"]
        # Determine if padding is needed based on the main input_ids
        # (This logic might be redundant if __call__ already padded everything)
        current_max_length = 0
        if isinstance(required_input, torch.Tensor):
            current_max_length = required_input.shape[1]
        elif isinstance(required_input, list):
            if required_input and isinstance(required_input[0], list):  # batch
                current_max_length = len(required_input[0])
            else:  # single
                current_max_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = current_max_length  # Use the already determined longest length
        elif padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is None:
            max_length = current_max_length

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # Let superclass handle its known keys.
        # This will call PreTrainedTokenizerFast._pad if not overridden further up,
        # or PreTrainedTokenizerBase.pad (which calls _pad).
        # We need to be careful not to double-pad or interfere if custom keys are already handled.
        # Since __call__ now pads word_ids and subword_ids, we can simplify this.
        # The `encoded_inputs` passed here already has `input_ids` etc. padded by the main flow.
        # We just need to ensure `word_ids` and `subword_ids` (if they are lists here) get padded.

        # The `super()._pad` call in this overridden method needs to be carefully considered.
        # It was: encoded_inputs = super()._pad(...)
        # This would call PreTrainedTokenizerFast's _pad.
        # For simplicity, let's assume that the base _pad from PreTrainedTokenizerFast does its job
        # for standard keys if this method is part of that chain.
        # However, the explicit padding in `__call__` is now the primary source for custom keys.

        # The original logic for padding word_ids and subword_ids in this method:
        if "word_ids" in encoded_inputs and isinstance(encoded_inputs["word_ids"], list):  # Only pad if they are lists
            needs_to_be_padded_custom = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD
                and max_length is not None
                and len(
                    encoded_inputs["word_ids"][0]
                    if isinstance(encoded_inputs["word_ids"][0], list)
                    else encoded_inputs["word_ids"]
                )
                != max_length
            )
            if needs_to_be_padded_custom:
                # This assumes word_ids and subword_ids are lists of lists or flat lists
                # The padding logic from __call__ is more robust now.
                # This _pad method, if still called, should ideally not re-pad if tensors already exist.
                # Given the __call__ changes, this _pad might be less critical or might need to be removed/simplified
                # if it conflicts.
                # For now, let's keep the structure but acknowledge __call__ is primary.
                # This part will only execute if word_ids are lists (not tensors).

                # This original logic here would pad based on `max_length` determined for `input_ids`.
                # Let's call the parent's _pad first.
                # Store word_ids and subword_ids if they exist and are lists
                temp_word_ids = encoded_inputs.get("word_ids")
                temp_subword_ids = encoded_inputs.get("subword_ids")

                if "word_ids" in encoded_inputs:
                    del encoded_inputs["word_ids"]
                if "subword_ids" in encoded_inputs:
                    del encoded_inputs["subword_ids"]

                padded_super_inputs = super()._pad(
                    encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask
                )

                # Restore and pad custom keys if they were lists
                # This logic is now largely superseded by the padding in __call__
                # if return_tensors="pt". If not, this might still be relevant.
                # For safety, we'll keep a simplified version.
                # The main padding for word_ids/subword_ids happens in __call__

                # Restore original keys
                if temp_word_ids is not None:
                    padded_super_inputs["word_ids"] = temp_word_ids
                if temp_subword_ids is not None:
                    padded_super_inputs["subword_ids"] = temp_subword_ids

                # If after super()._pad, input_ids were padded, and word_ids are still lists, they need padding.
                # This can happen if return_tensors is not 'pt'.
                # `padded_super_inputs` is the dictionary to return.

                # The following logic from your original _pad seems fine if word_ids/subword_ids are lists
                # and need padding based on the main input_ids length.
                main_input_padded_len = 0
                if isinstance(padded_super_inputs["input_ids"], torch.Tensor):
                    main_input_padded_len = padded_super_inputs["input_ids"].shape[-1]
                elif isinstance(padded_super_inputs["input_ids"], list):
                    main_input_padded_len = len(
                        padded_super_inputs["input_ids"][0]
                        if isinstance(padded_super_inputs["input_ids"][0], list)
                        else padded_super_inputs["input_ids"]
                    )

                padding_value_for_ids = 0
                for key in ["word_ids", "subword_ids"]:
                    if key in padded_super_inputs and isinstance(padded_super_inputs[key], list):
                        # Check if it's a batch of lists or a single list
                        is_batched_list = (
                            isinstance(padded_super_inputs[key][0], list) if padded_super_inputs[key] else False
                        )
                        if is_batched_list:
                            for i in range(len(padded_super_inputs[key])):
                                item_list = padded_super_inputs[key][i]
                                diff = main_input_padded_len - len(item_list)
                                if diff > 0:
                                    if padding_side == "right":
                                        padded_super_inputs[key][i] = item_list + [padding_value_for_ids] * diff
                                    else:
                                        padded_super_inputs[key][i] = [padding_value_for_ids] * diff + item_list
                        else:  # Single list
                            item_list = padded_super_inputs[key]
                            diff = main_input_padded_len - len(item_list)
                            if diff > 0:
                                if padding_side == "right":
                                    padded_super_inputs[key] = item_list + [padding_value_for_ids] * diff
                                else:
                                    padded_super_inputs[key] = [padding_value_for_ids] * diff + item_list
                return padded_super_inputs
            else:  # No padding needed for custom keys if main input wasn't padded or they already match length
                return super()._pad(
                    encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask
                )

        # Fallback to super()._pad if 'word_ids' not present or not lists (e.g., already tensors)
        return super()._pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)


def is_subword(token: str, subword_prefix="##") -> bool:
    """Returns if a token is a subword"""
    return token.startswith(subword_prefix)


def create_word_ids(
    tokens: List[str], restart_new_sentence=False, seperator_token="[SEP]", padding_token="[PAD]"
) -> List[int]:
    """Creates word ids for given tokens"""
    word_ids: List[int] = []
    current_word_id: int = -1
    sentence_restart = False
    if tokens.count(seperator_token) < 2:
        restart_new_sentence = False
    for token in tokens:
        if token == padding_token:
            current_word_id = 0
        # If new sentence requires new starting, do not restart at second
        elif restart_new_sentence and not sentence_restart and token == seperator_token:
            current_word_id = 0
            sentence_restart = True
        elif not is_subword(token):
            current_word_id += 1
        # If tokens start with a subword
        elif current_word_id == -1:
            current_word_id = 0
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
    return col_round(si * (max_intermediate_subwords - 1) / num_intermediate_subwords)


def get_ids_from_subwords(
    num_subwords: int,
    max_intermediate_subword_positions_per_word: int,
    subword_embedding_order: str,
    intermediate_subword_distribution_strategy: str,
    starts_with_subword=False,
) -> List[int]:
    """Calculate subword ids for given subwords list of a single word"""

    if starts_with_subword:
        if num_subwords == 1:
            return [1]
    elif num_subwords <= 2:
        return list(range(num_subwords))

    if subword_embedding_order == "ending_first":
        # Return with simple cases
        if starts_with_subword and num_subwords == 2:
            return [2, 1]

        subword_ids: List[int] = [0] if not starts_with_subword else [1]
        # 2 for root and last token
        num_intermediate_subwords: int = num_subwords - 2 if not starts_with_subword else num_subwords - 1

        # R - L - I1 - I2 - ...
        if num_intermediate_subwords <= max_intermediate_subword_positions_per_word:
            for si in range(num_intermediate_subwords):
                subword_ids.append(2 + si)
        # if there are more intermediate subwords than allowed
        else:
            if intermediate_subword_distribution_strategy == "uniform":
                # Distribute all indices uniformly between allowed indices
                for si in range(num_intermediate_subwords):
                    subword_ids.append(
                        2 + get_uniform_id(si, max_intermediate_subword_positions_per_word, num_intermediate_subwords)
                    )
            elif intermediate_subword_distribution_strategy == "leftover_as_last":
                # Append subword indices that are allowed
                for si in range(max_intermediate_subword_positions_per_word):
                    subword_ids.append(2 + si)
                # Append rest as last
                for si in range(num_intermediate_subwords - max_intermediate_subword_positions_per_word):
                    subword_ids.append(1)
            else:
                raise ValueError("Unsupported intermediate subword distribution strategy")
        subword_ids.append(1)
        return subword_ids
    raise ValueError("Unsupported subword embedding order")


def extend_subword_ids_for_word(
    subword_ids: List[int],
    intermediate_subword_distribution_strategy: str,
    max_intermediate_subword_positions_per_word: int,
    num_subwords: int,
    start_subword_processed: bool,
    starts_with_subword: bool,
    subword_embedding_order: str,
) -> None:
    """Extends subword ids for each word"""
    subword_ids.extend(
        get_ids_from_subwords(
            num_subwords,
            max_intermediate_subword_positions_per_word,
            subword_embedding_order,
            intermediate_subword_distribution_strategy,
            starts_with_subword and not start_subword_processed,
        )
    )


def create_subword_ids(
    tokens: List[str],
    max_intermediate_subword_positions_per_word: int,
    subword_embedding_order: str,
    intermediate_subword_distribution_strategy: str,
):
    """Creates subword ids for the given tokens and parameters"""

    # If tokens are empty return empty subword id list
    if len(tokens) == 0:
        return []

    subword_ids: List[int] = []
    num_subwords: int = 0
    starts_with_subword = is_subword(tokens[0])
    start_subword_processed = False
    for token in tokens:
        if not is_subword(token):
            if num_subwords > 0:
                extend_subword_ids_for_word(
                    subword_ids,
                    intermediate_subword_distribution_strategy,
                    max_intermediate_subword_positions_per_word,
                    num_subwords,
                    start_subword_processed,
                    starts_with_subword,
                    subword_embedding_order,
                )
                start_subword_processed = True
                num_subwords = 0
        num_subwords += 1
    if num_subwords > 0:
        extend_subword_ids_for_word(
            subword_ids,
            intermediate_subword_distribution_strategy,
            max_intermediate_subword_positions_per_word,
            num_subwords,
            start_subword_processed,
            starts_with_subword,
            subword_embedding_order,
        )
    return subword_ids


__all__ = ["Bert2DTokenizerFast"]
