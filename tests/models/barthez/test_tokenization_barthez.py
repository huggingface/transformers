# Copyright 2019 Hugging Face inc.
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
from transformers import AutoTokenizer
from transformers.tokenization_sentencepiece import SentencePieceExtractor
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers import BarthezTokenizer

from ...test_tokenization_common import TokenizerTesterMixin

# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""

expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁', 'ปี', '▁ir', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']
expected_token_ids = [0, 2078, 75, 10, 1938, 78, 402, 49997, 23, 387, 7648, 4, 124, 663, 75, 41564, 362, 5, 6, 3, 1739, 18324, 1739, 18324, 18324, 0, 901, 0, 1749, 451, 13564, 39363, 3354, 166, 72171, 22, 21077, 64, 12, 18324, 5, 3007, 172, 64, 124, 6, 3, 172, 64, 6, 3, 14833, 2271, 482, 329, 11028, 2]

SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece  
@require_tokenizers
class BarthezTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "moussaKam/mbarthez"
    tokenizer_class = BarthezTokenizer
    rust_tokenizer_class = BarthezTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "moussaKam/mbarthez"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.pad_token = tok_auto.eos_token
        tok_auto.save_pretrained(cls.tmpdirname)

        #Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tok_auto, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab, scores, merges = extractor.extract()
        tok_from_vocab = BarthezTokenizer(vocab=scores)
        tok_from_vocab.pad_token = tok_from_vocab.eos_token

        cls.tokenizers = [tok_auto, tok_from_vocab]

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
