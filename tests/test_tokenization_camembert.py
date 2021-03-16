# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
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

from transformers import CamembertTokenizer, CamembertTokenizerFast
from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")
SAMPLE_BPE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece_bpe.model")

FRAMEWORK = "pt" if is_torch_available() else "tf"


@require_sentencepiece
@require_tokenizers
class CamembertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CamembertTokenizer
    rust_tokenizer_class = CamembertTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = CamembertTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_rust_and_python_bpe_tokenizers(self):
        tokenizer = CamembertTokenizer(SAMPLE_BPE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)
        rust_tokenizer = CamembertTokenizerFast.from_pretrained(self.tmpdirname)

        sequence = "I was born in 92000, and this is falsé."

        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        # <unk> tokens are not the same for `rust` than for `slow`.
        # Because spm gives back raw token instead of `unk` in EncodeAsPieces
        # tokens = tokenizer.tokenize(sequence)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I was born in 92000, and this is falsé."

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)
