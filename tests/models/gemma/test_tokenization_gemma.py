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


import unittest
import tempfile

from transformers import AutoTokenizer, AddedToken, PreTrainedTokenizerFast
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from transformers.create_fast_tokenizer import SentencePieceExtractor
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
)
from tests.test_tokenization_common import TokenizerTesterMixin

input_string = "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61 生活的真谛是 Hi  Hello Hi   <s> Hello<s>how▁▁ and ▁<bos>Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!   "

expected_token_ids = [2, 6750, 1, 235265, 235248, 255969, 235248, 109, 4747, 139, 235335, 139, 216311, 241316, 139,
                      239880,
                      235341, 144, 235269, 235248, 235274, 235284, 235304, 235310, 235248, 235274, 235308, 235248,
                      235308,
                      235269, 235318, 235274, 64001, 235370, 235710, 245467, 235427, 11192, 139, 4521, 11192, 140,
                      235322,
                      235256, 235313, 25957, 235322, 235256, 235313, 1139, 140, 639, 139, 2, 6750, 1, 235265, 235248,
                      255969,
                      235248, 109, 4747, 139, 235335, 139, 216311, 241316, 139, 239880, 235341, 140]
expected_tokens = ['Hey', '<eos>', '.', '▁', '\t\t', '▁', '\n\n', 'you', '▁▁', 'é', '▁▁', '@#', '😈', '▁▁', '🤗', '!',
                   '▁▁▁▁▁▁▁', ',', '▁', '1', '2', '3', '4', '▁', '1', '5', '▁', '5', ',', '6', '1', '▁生活', '的', '真',
                   '谛', '是', '▁Hi', '▁▁', 'Hello', '▁Hi', '▁▁▁', '<', 's', '>', '▁Hello', '<', 's', '>', 'how', '▁▁▁',
                   'and', '▁▁', '<bos>', 'Hey', '<eos>', '.', '▁', '\t\t', '▁', '\n\n', 'you', '▁▁', 'é', '▁▁', '@#',
                   '😈', '▁▁', '🤗', '!', '▁▁▁']


@require_tokenizers
class GemmaTokenizationTest(TokenizerTesterMixin, unittest.TestCase): # TEMP we won't use the mixin in v5
    from_pretrained_id = "google/gemma-7b"
    tokenizer_class = GemmaTokenizerFast
    rust_tokenizer_class = GemmaTokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = False  # we're going to just test the fast one I'll remove this
    space_between_special_tokens = False
    from_pretrained_kwargs = {}
    test_seq2seq = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "google/gemma-7b"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]

    def test_integration_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_integration_expected_token_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)

    def test_save_and_reload(self):
        for tok in self.tokenizers:
            with self.subTest(f"{tok.__class__.__name__}"):
                original_tokens = tok.tokenize(input_string)
                original_ids = tok.encode(input_string)

                # Save tokenizer to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tok.save_pretrained(tmp_dir)

                    # Reload tokenizer from saved directory
                    reloaded_tok = tok.__class__.from_pretrained(tmp_dir)

                    # Test that reloaded tokenizer produces same results
                    reloaded_tokens = reloaded_tok.tokenize(input_string)
                    reloaded_ids = reloaded_tok.encode(input_string)

                    self.assertEqual(original_tokens, reloaded_tokens)
                    self.assertEqual(original_ids, reloaded_ids)
