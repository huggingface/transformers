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

from transformers.models.parakeet import ParakeetTokenizerFast

from ...test_tokenization_common import TokenizerTesterMixin


class ParakeetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    slow_tokenizer_class = None
    rust_tokenizer_class = ParakeetTokenizerFast
    tokenizer_class = ParakeetTokenizerFast
    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_id = "nvidia/parakeet-ctc-1.1b"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = ParakeetTokenizerFast.from_pretrained("nvidia/parakeet-ctc-1.1b")
        tokenizer.save_pretrained(cls.tmpdirname)

    @unittest.skip(
        reason="This test does not apply to ParakeetTokenizerFast. More details in the test docstring itself."
    )
    def test_added_tokens_do_lower_case(self):
        """
        Precompiled normalization from sentencepiece is `nmt_nfkc_cf` that includes lowercasing. Yet, ParakeetTokenizerFast does not have a do_lower_case attribute.
        This result in the test failing.
        """
        pass

    @unittest.skip(reason="This needs a slow tokenizer. Parakeet does not have one!")
    def test_encode_decode_with_spaces(self):
        return

    @unittest.skip(reason="ParakeetTokenizerFast doesn't have tokenizer_file in its signature.")
    def test_rust_tokenizer_signature(self):
        pass
