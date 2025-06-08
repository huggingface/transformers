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
from typing import List, Optional, Union, Dict

import torch

from ...tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
    PaddingStrategy,
    TensorType,
)
from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer, VOCAB_FILES_NAMES


logger = logging.get_logger(__name__)

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
    sentence_restart_flag = False # Renamed to avoid conflict with outer scope if any

    # Determine if restart_new_sentence logic should apply
    # It applies if explicitly requested AND there are at least two SEP tokens (implying multiple sentences)
    actual_restart_new_sentence = restart_new_sentence and tokens.count(seperator_token) >= 2

    for token in tokens:
        if token == padding_token:
            # For PAD tokens, word_id is 0, and current_word_id is reset to 0 for subsequent tokens.
            word_ids.append(0)
            current_word_id = 0 
        elif actual_restart_new_sentence and not sentence_restart_flag and token == seperator_token:
            # First SEP token in a multi-sentence pair where restart is enabled.
            # This SEP token itself gets the current_word_id + 1 (or 0 if current is -1).
            # Then, current_word_id resets to 0 for the next sentence.
            if current_word_id == -1 : # Should not happen if CLS is present and handled
                 current_word_id = 0 
                 word_ids.append(current_word_id)
            else:
                 current_word_id +=1 # SEP belongs to the current sentence conceptually
                 word_ids.append(current_word_id)

            current_word_id = 0 # Reset for the new sentence starting after this SEP
            sentence_restart_flag = True
        elif not is_subword(token):
            current_word_id += 1
            word_ids.append(current_word_id)
        elif current_word_id == -1:  # Subword at the beginning (e.g. after [CLS] or if [CLS] is not token 0)
            current_word_id = 0
            word_ids.append(current_word_id)
        else:  # Is a subword, and current_word_id is >= 0
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
    # Ensure max_intermediate_subwords-1 is not negative if max_intermediate_subwords is 0 (though usually >=1)
    effective_max_pos = max(0, max_intermediate_subwords -1)
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
            return [1]  # A single subword token treated as "last"
    elif num_subwords_in_current_word == 1: # Single token, not a subword
        return [0]  # Root

    # For "ending_first" order:
    # ID 0: Root token (first token of a multi-token word, if not starting with subword)
    # ID 1: Last token of a multi-token word
    # ID 2, 3, ...: Intermediate tokens

    if subword_embedding_order == "ending_first":
        subword_ids: List[int] = []
        
        has_explicit_root = not current_word_starts_with_subword and num_subwords_in_current_word > 0
        has_explicit_last = num_subwords_in_current_word > 1 or \
                            (current_word_starts_with_subword and num_subwords_in_current_word == 1)


        if has_explicit_root:
            subword_ids.append(0) # Add root ID
            # Number of tokens remaining after taking out root and potentially last
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word - 1
        else: # Word starts with a subword, or is a single subword
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word
        
        num_intermediate_tokens = 0
        if has_explicit_last:
            num_intermediate_tokens = num_tokens_for_intermediate_and_last - 1
        else: # No explicit last token (e.g. single root token, or single starting subword already handled)
            num_intermediate_tokens = num_tokens_for_intermediate_and_last


        if num_intermediate_tokens < 0 : num_intermediate_tokens = 0


        if num_intermediate_tokens > 0:
            if num_intermediate_tokens <= max_intermediate_subword_positions_per_word:
                for si in range(num_intermediate_tokens):
                    subword_ids.append(2 + si)
            else:  # More intermediate tokens than allowed positions
                if intermediate_subword_distribution_strategy == "uniform":
                    for si in range(num_intermediate_tokens):
                        subword_ids.append(
                            2 + get_uniform_id(si, max_intermediate_subword_positions_per_word, num_intermediate_tokens)
                        )
                elif intermediate_subword_distribution_strategy == "leftover_as_last":
                    for si in range(max_intermediate_subword_positions_per_word):
                        subword_ids.append(2 + si)
                    for _ in range(num_intermediate_tokens - max_intermediate_subword_positions_per_word):
                        subword_ids.append(1)  # Leftovers get ID 1 (last type)
                else:
                    raise ValueError(f"Unsupported intermediate subword distribution strategy: {intermediate_subword_distribution_strategy}")
        
        if has_explicit_last:
             subword_ids.append(1) # Add last ID

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
    
    # This flag is true if the very first content token (non-special) of the *entire input token list* is a subword.
    first_content_token_is_subword = False
    found_first_content_token = False
    for t_idx, token_val in enumerate(tokens):
        if token_val not in [cls_token, sep_token, pad_token]:
            first_content_token_is_subword = is_subword(token_val)
            found_first_content_token = True
            break
    
    # This flag tracks if we have processed the subwords of the first content word.
    first_content_word_processed = False

    for token in tokens:
        if token in [cls_token, sep_token, pad_token]:
            # Process any accumulated word tokens before handling special token
            if current_word_segment_tokens:
                # Determine if this segment is the first *content* word and starts with a subword
                is_this_segment_the_first_content_word_starting_with_subword = \
                    first_content_token_is_subword and \
                    not first_content_word_processed and \
                    is_subword(current_word_segment_tokens[0])

                generated_ids = get_ids_from_subwords(
                    num_subwords_in_current_word=len(current_word_segment_tokens),
                    max_intermediate_subword_positions_per_word=max_intermediate_subword_positions_per_word,
                    subword_embedding_order=subword_embedding_order,
                    intermediate_subword_distribution_strategy=intermediate_subword_distribution_strategy,
                    current_word_starts_with_subword=is_this_segment_the_first_content_word_starting_with_subword,
                )
                all_subword_ids.extend(generated_ids)
                current_word_segment_tokens = []
                if not first_content_word_processed: # Mark first content word as processed
                    first_content_word_processed = True 
            
            all_subword_ids.append(0)  # Special tokens get subword_id 0
        elif not is_subword(token): # Start of a new word (or a single-token word)
            # Process previous word segment (if any)
            if current_word_segment_tokens:
                is_this_segment_the_first_content_word_starting_with_subword = \
                    first_content_token_is_subword and \
                    not first_content_word_processed and \
                    is_subword(current_word_segment_tokens[0])

                generated_ids = get_ids_from_subwords(
                    num_subwords_in_current_word=len(current_word_segment_tokens),
                    max_intermediate_subword_positions_per_word=max_intermediate_subword_positions_per_word,
                    subword_embedding_order=subword_embedding_order,
                    intermediate_subword_distribution_strategy=intermediate_subword_distribution_strategy,
                    current_word_starts_with_subword=is_this_segment_the_first_content_word_starting_with_subword,
                )
                all_subword_ids.extend(generated_ids)
                if not first_content_word_processed:
                    first_content_word_processed = True
            current_word_segment_tokens = [token] # Start new word segment
        else: # Is a subword, continue current word segment
            current_word_segment_tokens.append(token)

    # Process any remaining word tokens at the end
    if current_word_segment_tokens:
        is_this_segment_the_first_content_word_starting_with_subword = \
            first_content_token_is_subword and \
            not first_content_word_processed and \
            is_subword(current_word_segment_tokens[0])
            
        generated_ids = get_ids_from_subwords(
            num_subwords_in_current_word=len(current_word_segment_tokens),
            max_intermediate_subword_positions_per_word=max_intermediate_subword_positions_per_word,
            subword_embedding_order=subword_embedding_order,
            intermediate_subword_distribution_strategy=intermediate_subword_distribution_strategy,
            current_word_starts_with_subword=is_this_segment_the_first_content_word_starting_with_subword,
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
        self.max_intermediate_subword_positions_per_word = max_intermediate_subword_positions_per_word
        if subword_embedding_order != "ending_first":
            logger.warning(f"Bert2DTokenizer slow currently only fully supports 'ending_first' for subword_embedding_order. Received: {subword_embedding_order}")
        self.subword_embedding_order = subword_embedding_order
        self.intermediate_subword_distribution_strategy = intermediate_subword_distribution_strategy

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
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
        
        batch_encoding_super = super().__call__(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding, # Crucial: let superclass handle padding for standard keys
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None, # Get lists first, convert to tensor at the end
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
        
        # Determine if input was batched
        if isinstance(text, str) or (is_split_into_words and isinstance(text, list) and text and isinstance(text[0], str)):
             if text_pair is None:
                 was_batched = False
             else: # text is str, text_pair is str or list
                 if isinstance(text_pair, str) or (is_split_into_words and isinstance(text_pair, list) and text_pair and isinstance(text_pair[0], str)):
                     was_batched = False
                 else: # text is str, text_pair is list of str/list
                     was_batched = True # This case is tricky, superclass might handle it as batch. Assume simple batching.
        elif isinstance(text, list):
            was_batched = True
        else: # Fallback, should be covered by PreTrainedTokenizer logic
            was_batched = isinstance(input_ids_processed, list) and \
                          bool(input_ids_processed) and \
                          isinstance(input_ids_processed[0], list)


        list_of_input_ids_for_processing: List[List[int]]
        if not was_batched:
            list_of_input_ids_for_processing = [input_ids_processed]
        else:
            list_of_input_ids_for_processing = input_ids_processed

        all_word_ids: List[List[int]] = []
        all_subword_ids: List[List[int]] = []

        for ids_for_one_sequence in list_of_input_ids_for_processing:
            tokens = self.convert_ids_to_tokens(ids_for_one_sequence, skip_special_tokens=False)
            
            # restart_new_sentence logic for word_ids
            # BertTokenizer by default creates token_type_ids that distinguish sentences.
            # We can infer sentence pairs if token_type_ids are present and diverse.
            # For simplicity here, we assume if text_pair is provided, restart_new_sentence might be relevant.
            # The fast tokenizer's create_word_ids has a restart_new_sentence param.
            # Let's assume text_pair implies a new sentence for word_id counting if two SEP tokens are present.
            should_restart_word_ids = text_pair is not None

            word_ids_for_sequence = create_word_ids(
                tokens, 
                restart_new_sentence=should_restart_word_ids, # Heuristic
                seperator_token=self.sep_token, 
                padding_token=self.pad_token
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
        
        # `batch_encoding_super` contains already padded/truncated `input_ids`.
        # The `all_word_ids` and `all_subword_ids` are generated based on these.
        # If padding was applied by superclass, their lengths should match `input_ids`.
        # If `padding` was `do_not_pad`, lengths might vary.
        # The explicit padding loop in the previous reasoning might be redundant if tokens
        # from `convert_ids_to_tokens` already reflect the final padded length.
        # Let's verify: `convert_ids_to_tokens` on padded IDs will include PAD tokens.
        # So, `create_word_ids` and `create_subword_ids` will process these PAD tokens.
        # `create_word_ids` assigns 0 to PAD. `create_subword_ids` assigns 0 to PAD.
        # This means padding is implicitly handled by the helper functions if they correctly process PAD_TOKEN.

        final_batch_encoding = BatchEncoding()
        final_batch_encoding.update(batch_encoding_super) # Start with super's results

        if not was_batched:
            final_batch_encoding["word_ids"] = all_word_ids[0]
            final_batch_encoding["subword_ids"] = all_subword_ids[0]
        else:
            final_batch_encoding["word_ids"] = all_word_ids
            final_batch_encoding["subword_ids"] = all_subword_ids

        # Final tensor conversion if requested
        if return_tensors is not None:
            final_batch_encoding = final_batch_encoding.convert_to_tensors(tensor_type=return_tensors)
            
        return final_batch_encoding


__all__ = [
    "Bert2DTokenizer", 
    "is_subword", 
    "create_word_ids", 
    "create_subword_ids", 
    "col_round", 
    "get_uniform_id", 
    "get_ids_from_subwords"
]
