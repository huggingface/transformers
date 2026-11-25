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

from transformers import BarthezTokenizer
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
class BarthezTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "moussaKam/mbarthez"
    tokenizer_class = BarthezTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁', 'ปี', '▁ir', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [2078, 75, 10, 1938, 6, 3, 78, 402, 49997, 23, 387, 7648, 4, 124, 663, 75, 41564, 362, 5, 6, 3, 1739, 18324, 1739, 18324, 18324, 0, 901, 0, 1749, 451, 13564, 39363, 3354, 166, 72171, 22, 21077, 64, 12, 18324, 5, 3007, 172, 64, 124, 6, 3, 172, 64, 6, 3, 14833, 2271, 482, 329, 11028]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '<unk>', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁', '<unk>', '▁ir', 'd', '▁', '<unk>', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is falsé. <unk> Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "moussaKam/mbarthez"

        tokenizer = BarthezTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

    def test_bpe_checkpoint_loads_with_bpe_backend(self):
        # Regression test for #48567: `moussaKam/barthez` trains BPE while
        # `moussaKam/mbarthez` trains Unigram, and one class serves both.
        # The BPE vocab used to reach a hardcoded Unigram and raise
        # `TypeError: 'dict' object is not an instance of 'Sequence'`.
        tokenizer = BarthezTokenizer.from_pretrained("moussaKam/barthez")
        self.assertEqual(type(tokenizer._tokenizer.model).__name__, "BPE")
        self.assertEqual(tokenizer.encode("Bonjour le monde"), [0, 17678, 36, 715, 2])

        tokenizer.save_pretrained(self.tmpdirname + "-barthez")
        reloaded = BarthezTokenizer.from_pretrained(self.tmpdirname + "-barthez")
        self.assertEqual(reloaded.encode("Bonjour le monde"), [0, 17678, 36, 715, 2])

    def test_unigram_checkpoint_still_loads_with_unigram_backend(self):
        tokenizer = BarthezTokenizer.from_pretrained("moussaKam/mbarthez")
        self.assertEqual(type(tokenizer._tokenizer.model).__name__, "Unigram")
