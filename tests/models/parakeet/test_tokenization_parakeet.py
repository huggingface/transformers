# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the ParakeetCTC tokenizer."""

import unittest

from transformers.models.parakeet import ParakeetCTCTokenizer

from ...test_tokenization_common import TokenizerTesterMixin


class ParakeetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "bezzam/parakeet-ctc-1.1b-hf"
    tokenizer_class = ParakeetCTCTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False  # Fails due to no pad token

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = ParakeetCTCTokenizer.from_pretrained("bezzam/parakeet-ctc-1.1b-hf")
        tokenizer.save_pretrained(cls.tmpdirname)

    @unittest.skip(reason="Perhaps failing due to CTC-style decoding?")
    def test_pretokenized_inputs(self):
        pass
