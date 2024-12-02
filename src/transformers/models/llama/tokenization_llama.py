# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""Tokenization classes for LLaMA."""

import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm
import base64

from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

B_INST, E_INST = "]", "]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""  # fmt: skip


class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
            A simple example:

            - `legacy=True`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
            >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
            [1, 15043, 29871, 1, 869]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
            >>> tokenizer.encode("Hello <s>.")  # 29889 is '.'
            [1, 15043, 29871, 1, 29889]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. Again, this should be set with `from_slow=True` to make sure it's taken into account.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        legacy=None,
        add_prefix_space=True,
        **kwargs,
    ):
        # 1. Set basic attributes first
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.add_prefix_space = add_prefix_space
        self.legacy = legacy if legacy is not None else True

        # 2. Set protected token attributes that sp_model will need
        self._unk_token = unk_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._pad_token = pad_token

        # 3. Initialize sp_model before parent class
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))

        # 4. Now we can safely initialize parent class since sp_model exists for vocab_size
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=sp_model_kwargs or {},
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy if legacy is not None else True,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        try:
            # First try normal loading
            tokenizer.Load(self.vocab_file)
            return tokenizer
        except Exception as e:
            try:
                from sentencepiece import sentencepiece_model_pb2 as model_pb2

                model = model_pb2.ModelProto()
                model.normalizer_spec.add_dummy_prefix = False
                model.normalizer_spec.remove_extra_whitespaces = False

                vocab = []
                special_tokens_dict = {
                    self._unk_token: model_pb2.ModelProto.SentencePiece.Type.UNKNOWN,
                    self._bos_token: model_pb2.ModelProto.SentencePiece.Type.USER_DEFINED,
                    self._eos_token: model_pb2.ModelProto.SentencePiece.Type.USER_DEFINED,
                }

                # Zero-width characters to check for
                zero_width_chars = {
                    '\u200b',  # Zero-width space
                    '\u200c',  # Zero-width non-joiner
                    '\u200d',  # Zero-width joiner
                    '\ufeff',  # Zero-width no-break space
                    '\u2060',  # Word joiner
                }

                unk_token_found = False

                with open(self.vocab_file, 'r', encoding='utf-8') as f:
                    token_set = set()
                    for line_num, line in enumerate(f, 1):
                        try:
                            b64_str, idx = line.strip().split()
                            token_bytes = base64.b64decode(b64_str)
                            token = token_bytes.decode('utf-8', errors='replace')
                            
                            # Strip whitespace as before
                            token = token.strip()

                            # Skip if token is empty after stripping
                            if not token:
                                print(f"Line {line_num}: Empty token after stripping. Skipping.")
                                continue

                            # Check for zero-width characters
                            if any(zwc in token for zwc in zero_width_chars):
                                print(f"Line {line_num}: Token contains zero-width characters. Cleaning.")
                                for zwc in zero_width_chars:
                                    token = token.replace(zwc, '')
                                # Skip if token becomes empty after cleaning
                                if not token:
                                    print(f"Line {line_num}: Token empty after cleaning zero-width chars. Skipping.")
                                    continue

                            # Check for null bytes
                            if '\x00' in token:
                                print(f"Line {line_num}: Token contains null bytes. Skipping.")
                                continue

                            if token in token_set:
                                print(f"Line {line_num}: Duplicate token '{token}' found. Skipping.")
                                continue

                            token_set.add(token)

                            if token == self._unk_token:
                                piece_type = model_pb2.ModelProto.SentencePiece.Type.UNKNOWN
                                unk_token_found = True
                            elif token in special_tokens_dict:
                                piece_type = special_tokens_dict[token]
                            else:
                                piece_type = model_pb2.ModelProto.SentencePiece.Type.NORMAL

                            vocab.append((int(idx), token, piece_type))

                        except Exception as decode_error:
                            print(f"Line {line_num}: Error processing token: {decode_error}")
                            continue

                # Sort vocab by index before adjusting
                vocab.sort(key=lambda x: x[0])
                
                # Debug print to see what's happening with SPIECE_UNDERLINE
                underline_tokens = [(idx, token) for idx, token, _ in vocab if token == '▁']
                print(f"SPIECE_UNDERLINE tokens found: {underline_tokens}")
                
                # Create adjusted vocab preserving specific token indices
                adjusted_vocab = []
                seen_tokens = set()
                next_idx = 0

                # First pass: find the original index of SPIECE_UNDERLINE
                original_underline_idx = None
                for idx, token, _ in vocab:
                    if token == '▁':
                        original_underline_idx = idx
                        break
                
                print(f"Original SPIECE_UNDERLINE index: {original_underline_idx}")

                # Second pass: preserve SPIECE_UNDERLINE index
                for idx, token, piece_type in vocab:
                    if token in seen_tokens:
                        print(f"Skipping duplicate token: {token}")
                        continue
                        
                    if token == '▁':
                        # Keep original index for SPIECE_UNDERLINE
                        new_idx = original_underline_idx
                        print(f"Adding SPIECE_UNDERLINE with index {new_idx}")
                    else:
                        new_idx = next_idx
                        while new_idx == original_underline_idx:
                            next_idx += 1
                            new_idx = next_idx
                        next_idx += 1
                    
                    adjusted_vocab.append((new_idx, token, piece_type))
                    seen_tokens.add(token)
                    
                    if len(adjusted_vocab) < 10:
                        print(f"Added token: '{token}' with index {new_idx}")

                # If unk token not found, add it
                if not unk_token_found:
                    print("Unknown token not found in vocabulary. Adding it.")
                    unk_idx = next_idx
                    while unk_idx == original_underline_idx:
                        next_idx += 1
                        unk_idx = next_idx
                    adjusted_vocab.append((unk_idx, self._unk_token, model_pb2.ModelProto.SentencePiece.Type.UNKNOWN))
                    
                print(f"Number of tokens after adjustment: {len(adjusted_vocab)}")
                
                # Sort by new indices
                adjusted_vocab.sort(key=lambda x: x[0])
                
                # Add tokens to model.pieces
                for idx, token, piece_type in adjusted_vocab:
                    piece = model.pieces.add()
                    piece.piece = token
                    piece.score = 0.0
                    piece.type = piece_type
                    if idx < 10:
                        print(f"Final token added: '{token}' with index {idx}")

                # Set the unknown token ID
                unk_idx = next(i for i, (_, token, _) in enumerate(adjusted_vocab) if token == self._unk_token)
                model.trainer_spec.unk_id = unk_idx
                model.trainer_spec.vocab_size = len(adjusted_vocab)

                model_data = model.SerializeToString()
                tokenizer.LoadFromSerializedProto(model_data)
                return tokenizer

            except Exception as final_e:
                raise ValueError(f"Failed to load tokenizer: {str(final_e)}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens.
        """
        return super().tokenize(text, **kwargs)

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if current_sub_tokens:
                    out_string += self.sp_model.decode(current_sub_tokens)
                out_string += token
                current_sub_tokens = []
                prev_is_special = True
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        if current_sub_tokens:
            out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output