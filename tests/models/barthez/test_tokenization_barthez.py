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

SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece  
@require_tokenizers
class BarthezTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "moussaKam/mbarthez"
    tokenizer_class = BarthezTokenizer
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True


    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁', 'ปี', '▁ir', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']
    integration_expected_token_ids = [0, 2078, 75, 10, 1938, 78, 402, 49997, 23, 387, 7648, 4, 124, 663, 75, 41564, 362, 5, 6, 3, 1739, 18324, 1739, 18324, 18324, 0, 901, 0, 1749, 451, 13564, 39363, 3354, 166, 72171, 22, 21077, 64, 12, 18324, 5, 3007, 172, 64, 124, 6, 3, 172, 64, 6, 3, 14833, 2271, 482, 329, 11028, 2]
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "moussaKam/mbarthez"

        tokenizer = BarthezTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

        #Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tokenizer, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab, scores, merges = extractor.extract()
        tokenizer_from_vocab = BarthezTokenizer(vocab=scores)
        tokenizer_from_vocab.pad_token = tokenizer_from_vocab.eos_token

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]
