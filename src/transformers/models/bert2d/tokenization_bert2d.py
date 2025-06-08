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
import os # For __main__ test
import shutil # For __main__ test

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
    sentence_restart_flag = False 

    actual_restart_new_sentence = restart_new_sentence and tokens.count(seperator_token) >= 2

    for token in tokens:
        if token == padding_token:
            word_ids.append(0)
            current_word_id = 0 
        elif actual_restart_new_sentence and not sentence_restart_flag and token == seperator_token:
            if current_word_id == -1 :
                 current_word_id = 0 
                 word_ids.append(current_word_id)
            else:
                 current_word_id +=1 
                 word_ids.append(current_word_id)

            current_word_id = 0 
            sentence_restart_flag = True
        elif not is_subword(token):
            current_word_id += 1
            word_ids.append(current_word_id)
        elif current_word_id == -1:  
            current_word_id = 0
            word_ids.append(current_word_id)
        else: 
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
            return [1] 
    elif num_subwords_in_current_word == 1: 
        return [0] 

    if subword_embedding_order == "ending_first":
        subword_ids: List[int] = []
        
        has_explicit_root = not current_word_starts_with_subword and num_subwords_in_current_word > 0
        has_explicit_last = num_subwords_in_current_word > 1 or \
                            (current_word_starts_with_subword and num_subwords_in_current_word == 1)


        if has_explicit_root:
            subword_ids.append(0) 
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word - 1
        else: 
            num_tokens_for_intermediate_and_last = num_subwords_in_current_word
        
        num_intermediate_tokens = 0
        if has_explicit_last: 
            num_intermediate_tokens = num_tokens_for_intermediate_and_last - 1
        else: 
            num_intermediate_tokens = num_tokens_for_intermediate_and_last


        if num_intermediate_tokens < 0 : num_intermediate_tokens = 0


        if num_intermediate_tokens > 0:
            if num_intermediate_tokens <= max_intermediate_subword_positions_per_word:
                for si in range(num_intermediate_tokens):
                    subword_ids.append(2 + si)
            else: 
                if intermediate_subword_distribution_strategy == "uniform":
                    for si in range(num_intermediate_tokens):
                        subword_ids.append(
                            2 + get_uniform_id(si, max_intermediate_subword_positions_per_word, num_intermediate_tokens)
                        )
                elif intermediate_subword_distribution_strategy == "leftover_as_last":
                    for si in range(max_intermediate_subword_positions_per_word):
                        subword_ids.append(2 + si)
                    for _ in range(num_intermediate_tokens - max_intermediate_subword_positions_per_word):
                        subword_ids.append(1) 
                else:
                    raise ValueError(f"Unsupported intermediate subword distribution strategy: {intermediate_subword_distribution_strategy}")
        
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
                is_this_segment_the_very_first_content_word_and_starts_with_subword = \
                    first_content_token_is_subword and \
                    not first_content_word_processed and \
                    is_subword(current_word_segment_tokens[0])

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
                is_this_segment_the_very_first_content_word_and_starts_with_subword = \
                    first_content_token_is_subword and \
                    not first_content_word_processed and \
                    is_subword(current_word_segment_tokens[0])

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
        is_this_segment_the_very_first_content_word_and_starts_with_subword = \
            first_content_token_is_subword and \
            not first_content_word_processed and \
            is_subword(current_word_segment_tokens[0])
            
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
        self.subword_embedding_order = subword_embedding_order
        self.intermediate_subword_distribution_strategy = intermediate_subword_distribution_strategy
        
        if subword_embedding_order != "ending_first":
            logger.warning(f"Bert2DTokenizer slow currently only fully supports 'ending_first' for subword_embedding_order. Received: {subword_embedding_order}")
        

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
            padding=padding, 
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None, 
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
        if not is_batched and isinstance(text, (list, tuple)) and not (text and isinstance(text[0], (list,tuple))):
             if isinstance(input_ids_processed, list) and \
                bool(input_ids_processed) and \
                isinstance(input_ids_processed[0], list):
                 is_batched = True
        elif not is_batched and text_pair is not None and isinstance(text_pair, (list, tuple)):
            if isinstance(input_ids_processed, list) and \
                bool(input_ids_processed) and \
                isinstance(input_ids_processed[0], list):
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
        
        if not is_batched:
            batch_encoding_super["word_ids"] = all_word_ids[0]
            batch_encoding_super["subword_ids"] = all_subword_ids[0]
        else:
            batch_encoding_super["word_ids"] = all_word_ids
            batch_encoding_super["subword_ids"] = all_subword_ids

        if return_tensors is not None:
            batch_encoding_super = batch_encoding_super.convert_to_tensors(tensor_type=return_tensors)
            
        return batch_encoding_super


__all__ = [
    "Bert2DTokenizer", 
    "is_subword", 
    "create_word_ids", 
    "create_subword_ids", 
    "col_round", 
    "get_uniform_id", 
    "get_ids_from_subwords"
]

if __name__ == "__main__":
    print("Running Bert2DTokenizer self-test...")
    
    # Setup a temporary directory and vocab file
    tmpdirname_main = "./tmp_bert2d_main_test"
    os.makedirs(tmpdirname_main, exist_ok=True)
    vocab_tokens_main = [
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", # 0-4
        "un", "##want", "##ed",                     # 5-7
        "runn", "##ing",                            # 8-9
        "hello", "world"                            # 10-11
    ]
    vocab_file_main = os.path.join(tmpdirname_main, VOCAB_FILES_NAMES["vocab_file"])
    with open(vocab_file_main, "w", encoding="utf-8") as vocab_writer:
        vocab_writer.write("\n".join(vocab_tokens_main) + "\n")
    
    print(f"\nCreated dummy vocab file: {vocab_file_main}")

    try:
        print("\nInstantiating Bert2DTokenizer...")
        tokenizer_main = Bert2DTokenizer(vocab_file=vocab_file_main)
        print("Bert2DTokenizer instantiated successfully.")
        print(f"Tokenizer model_input_names: {tokenizer_main.model_input_names}")
        assert "word_ids" in tokenizer_main.model_input_names
        assert "subword_ids" in tokenizer_main.model_input_names
        
        sample_text = "unwanted running"
        print(f"\nTokenizing sample text: '{sample_text}'")
        
        encoded_output = tokenizer_main(sample_text, add_special_tokens=True)
        
        print("\nEncoded output (lists):")
        for key, value in encoded_output.items():
            print(f"  {key}: {value}")
            
        assert "input_ids" in encoded_output, "input_ids missing"
        assert "word_ids" in encoded_output, "word_ids missing from output"
        assert "subword_ids" in encoded_output, "subword_ids missing from output"
        
        # Expected values for "unwanted running" -> [CLS] un ##want ##ed runn ##ing [SEP]
        # input_ids: [1, 5, 6, 7, 8, 9, 2]
        # word_ids:  [0, 1, 1, 1, 2, 2, 3]
        # subword_ids: [0, 0, 2, 1, 0, 1, 0] (default params)

        expected_input_ids = [1, 5, 6, 7, 8, 9, 2]
        expected_word_ids  = [0, 1, 1, 1, 2, 2, 3]
        expected_subword_ids = [0, 0, 2, 1, 0, 1, 0]

        assert encoded_output["input_ids"] == expected_input_ids, f"Mismatch in input_ids. Got {encoded_output['input_ids']}"
        assert encoded_output["word_ids"] == expected_word_ids, f"Mismatch in word_ids. Got {encoded_output['word_ids']}"
        assert encoded_output["subword_ids"] == expected_subword_ids, f"Mismatch in subword_ids. Got {encoded_output['subword_ids']}"
        
        print("\nList output assertions passed.")

        print("\nTesting with return_tensors='pt'...")
        encoded_output_pt = tokenizer_main(sample_text, add_special_tokens=True, return_tensors="pt")
        print("Encoded output (PyTorch Tensors):")
        for key, value in encoded_output_pt.items():
            print(f"  {key}: {value} (shape: {value.shape})")

        assert "word_ids" in encoded_output_pt, "word_ids missing from PT output"
        assert "subword_ids" in encoded_output_pt, "subword_ids missing from PT output"
        assert isinstance(encoded_output_pt["word_ids"], torch.Tensor), "word_ids is not a Tensor"

        # For single, non-batched inputs, BatchEncoding.convert_to_tensors returns 1D tensors
        assert torch.equal(encoded_output_pt["input_ids"], torch.tensor(expected_input_ids)), "PT input_ids mismatch"
        assert torch.equal(encoded_output_pt["word_ids"], torch.tensor(expected_word_ids)), "PT word_ids mismatch"
        assert torch.equal(encoded_output_pt["subword_ids"], torch.tensor(expected_subword_ids)), "PT subword_ids mismatch"
        print("PyTorch tensor output assertions passed.")

        print("\nBert2DTokenizer self-test completed successfully!")

    except Exception as e:
        print(f"\nError during Bert2DTokenizer self-test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(tmpdirname_main):
            shutil.rmtree(tmpdirname_main)
            print(f"\nCleaned up dummy vocab directory: {tmpdirname_main}")

