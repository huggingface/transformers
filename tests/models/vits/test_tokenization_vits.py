# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Tests for the VITS tokenizer."""
import json
import os
import tempfile
import unittest

from transformers import VitsMmsTokenizer
from transformers.models.vits.tokenization_vits import VOCAB_FILES_NAMES
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


class VitsTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = VitsMmsTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = "k ' z y u d h e s w – 3 c p - 1 j m i X f l o 0 b r a 4 2 n _ x v t q 5 6 g < > | <pad> <unk>".split(
            " "
        )
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        vocab_tokens[" "] = vocab_tokens["X"]
        del vocab_tokens["X"]

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return VitsMmsTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_add_tokens_tokenizer(self):
        # TODO: fails because we do things differently
        pass

    def test_internal_consistency(self):
        # TODO: we have no decode()!
        pass

    def test_maximum_encoding_length_pair_input(self):
        # TODO: no idea what this is
        pass

    def test_maximum_encoding_length_single_input(self):
        # TODO: no idea what this is
        pass

    def test_pretokenized_inputs(self):
        # TODO: no idea what this is
        pass

    def test_save_and_load_tokenizer(self):
        # TODO: fails on added tokens stuff
        pass

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        # TODO: fails on added tokens stuff
        pass

    @slow
    def test_tokenizer_integration(self):
        # TODO: add this
        pass
