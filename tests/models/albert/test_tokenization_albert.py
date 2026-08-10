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

from transformers import AlbertTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class AlbertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "albert/albert-base-v1"
    tokenizer_class = AlbertTokenizer

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['▁this', '▁is', '▁a', '▁test', '▁', '😊', '▁i', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁false', '.', '▁', '生活的真谛是', '▁hi', '▁hello', '▁hi', '▁hello', '▁hello', '▁', '<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁the', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁hello', '.', '▁but', '▁i', 'rd', '▁and', '▁', 'ป', '▁i', 'rd', '▁', 'ด', '▁hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [48, 25, 21, 1289, 13, 1, 31, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 4997, 9, 13, 1, 4148, 10975, 4148, 10975, 10975, 13, 1, 18, 1, 4148, 1, 18, 1, 1887, 14, 249, 3724, 378, 44, 7428, 13665, 45, 10975, 9, 47, 31, 897, 17, 13, 1, 31, 897, 13, 1, 8409, 184, 50, 42, 845]  # fmt: skip
    integration_expected_decoded_text = "this is a test <unk> i was born in 92000, and this is false. <unk> hi hello hi hello hello <unk>s<unk> hi<unk>s<unk>there the following string should be properly encoded: hello. but ird and <unk> ird <unk> hey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "albert/albert-base-v1"

        tokenizer = AlbertTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

        # Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tokenizer, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab_ids, vocab_scores, merges = extractor.extract()
        tokenizer_from_vocab = AlbertTokenizer(vocab=vocab_scores, merges=merges)
        tokenizer_from_vocab.pad_token = tokenizer_from_vocab.eos_token

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]
