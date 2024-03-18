# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import os
import unittest

from transformers.models.rwkv5.tokenization_rwkv5 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, require_torch
from transformers.utils import is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


if is_torch_available():
    from transformers import Rwkv5Tokenizer


@require_torch
@require_tokenizers
class RWKV5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Rwkv5Tokenizer
    # TODO we need a tokenizer list to make sure everything is tested
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"add_prefix_space": True}
    test_seq2seq = False

    def setUp(self):
        super().setUp()
        vocab_tokens = [b'\x00', b'\x01', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07', b'\x08', b'\t', b'\n', b'\x0b', b'\x0c', b'\r', b'\x0e', b'\x0f', b'\x10', b'\x11', b'\x12', b'\x13', b'\x14', b'\x15', b'\x16', b'\x17', b'\x18', b'\x19', b'\x1a', b'\x1b', b'\x1c', b'\x1d', b'\x1e', b'\x1f', b' ', b'!', b'"', b'#', b'$', b'%', b'&', b"'", b'(', b')', b'*', b'+', b',', b'-', b'.', b'/', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b':', b';', b'<', b'=', b'>', b'?', b'@', b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z', b'[', b'\\', b']', b'^', b'_', b'`', b'a', b'b', b'c']  # fmt: skip
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w") as vocab_writer:
            for token in vocab_tokens:
                vocab_writer.write(str(token) + "\n")
        self.special_tokens_map = {"unk_token": "<s>"}

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Rwkv5Tokenizer.from_pretrained(self.tmpdirname, **kwargs, trust_remote_code=True)

    def test_pretokenized_inputs(self):
        pass


# TODO add integration tests slow. Maybe also work on fast?