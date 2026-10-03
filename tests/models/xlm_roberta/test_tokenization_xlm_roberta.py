# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import AutoTokenizer, XLMRobertaTokenizer
from transformers.testing_utils import require_sentencepiece, require_tokenizers, slow

# import cached_property
from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
class XLMRobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "FacebookAI/xlm-roberta-base"
    tokenizer_class = XLMRobertaTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [3293, 83, 10, 3034, 6, 82803, 87, 509, 103122, 23, 483, 13821, 4, 136, 903, 83, 84047, 446, 5, 6, 62668, 5364, 245875, 354, 2673, 35378, 2673, 35378, 35378, 0, 1274, 0, 2685, 581, 25632, 79315, 5608, 186, 155965, 22, 40899, 71, 12, 35378, 5, 4966, 193, 71, 136, 10249, 193, 71, 48229, 28240, 3642, 621, 398, 20594]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    def test_unigram_dict_vocab_does_not_crash_init(self):
        """Regression test for #47020: dict vocabs from tokenizer.json must not break Unigram __init__."""
        vocab_dict = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "hej": 5}
        tokenizer = XLMRobertaTokenizer(vocab=vocab_dict)
        self.assertIsNotNone(tokenizer._tokenizer)
        self.assertEqual(tokenizer.convert_tokens_to_ids("hej"), 5)

    @slow
    def test_from_pretrained_scandibert_with_dict_vocab_in_tokenizer_json(self):
        """End-to-end load for vesteinn/ScandiBERT, whose tokenizer.json stores Unigram vocab as a dict.

        Network/Hub-dependent, gated by ``RUN_SLOW=1``. Default ``pr-ci/tests_tokenization`` runs
        against an image with ``/mnt/cache`` mounted read-only, which makes any HF Hub metadata write
        raise ``OSError(EROFS)`` even when the snapshot is already cached. The in-memory regression
        coverage above fully exercises the dict-vocab code path that this PR fixes; this slow test is
        kept for maintainers who run with ``RUN_SLOW=1``.
        """
        tokenizer = AutoTokenizer.from_pretrained("vesteinn/ScandiBERT")
        input_ids = tokenizer("Hej", add_special_tokens=False)["input_ids"]
        self.assertGreater(len(input_ids), 0)
