# coding=utf-8
# Copyright 2024 GLM & ZhipuAI team. All rights reserved.
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
"""Tokenization classes for GLM."""

import regex as re
import base64
import os
import tiktoken
from typing import List, Optional, Union, Dict
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding


class GLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
            self,
            vocab_file,
            padding_side="left",
            clean_up_tokenization_spaces=False,
            encode_special_tokens=False,
            **kwargs
    ):
        self.name = "GLMTokenizer"
        self.vocab_file = vocab_file
        pat_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self.pat_str = re.compile(pat_str)
        self.encode_special_tokens = encode_special_tokens

        mergeable_ranks = {}
        with open(vocab_file) as f:
            for line in f:
                token, rank = line.strip().split()
                rank = int(rank)
                token = base64.b64decode(token)
                mergeable_ranks[token] = rank

        self.mergeable_ranks = mergeable_ranks

        self.tokenizer = tiktoken.Encoding(
            name="my_tokenizer",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={}
        )
        self.decoder = {rank: token for token, rank in mergeable_ranks.items()}
        self.n_words = len(self.decoder)

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    @property
    def vocab_size(self):
        return self.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str, int]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, int):
                t = chr(t)
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors="replace")
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type int, bytes or str")
        if temp:
            text += temp.decode("utf-8", errors="replace")
        return text

    def _tokenize(self, text, **kwargs):
        tokens = []
        ids = self.tokenizer.encode(text)
        for t in ids:
            tokens.append(self.decoder[t])
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.mergeable_ranks[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, "")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.convert_tokens_to_ids("[gMASK]"), self.convert_tokens_to_ids("<sop>")]
        return prefix_tokens

    def build_single_message(self, role, metadata, message, tokenize=True):
        assert role in ["system", "user", "assistant", "observation"], role
        if tokenize:
            role_tokens = [self.convert_tokens_to_ids(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n",
                                                                                              disallowed_special=())
            message_tokens = self.tokenizer.encode(message, disallowed_special=())
            tokens = role_tokens + message_tokens
            return tokens
        else:
            return str(f"<|{role}|>{metadata}\n{message}")

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
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
