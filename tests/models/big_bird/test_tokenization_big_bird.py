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

from transformers import BigBirdTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SPIECE_UNDERLINE = "▁"

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class BigBirdTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/bigbird-roberta-base"
    tokenizer_class = BigBirdTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊\n', 'I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '\n生活的真谛是\n', 'Hi', '▁Hello', '\n', 'Hi', '▁Hello', '\n\n', '▁', '\n', '▁', '\n', '▁Hello', '\n', '<s>', '\n', 'hi', '<s>', 'there', '\n', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '\n', 'But', '▁', 'ird', '▁and', '▁', 'ปี', '▁', 'ird', '▁', 'ด\n', 'Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [871, 419, 358, 1433, 321, 100, 141, 474, 4743, 388, 961, 11125, 112, 391, 529, 419, 27908, 266, 114, 100, 17351, 18536, 100, 17351, 18536, 100, 321, 100, 321, 100, 18536, 100, 2, 100, 5404, 2, 8218, 100, 565, 1809, 4832, 916, 408, 6206, 30341, 126, 18536, 114, 100, 1638, 321, 1548, 391, 321, 100, 321, 1548, 321, 100, 10915, 804, 490, 446, 1905]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '<unk>', 'I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '<unk>', 'Hi', '▁Hello', '<unk>', 'Hi', '▁Hello', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁Hello', '<unk>', '<s>', '<unk>', 'hi', '<s>', 'there', '<unk>', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '<unk>', 'But', '▁', 'ird', '▁and', '▁', '<unk>', '▁', 'ird', '▁', '<unk>', 'Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk>I was born in 92000, and this is falsé.<unk>Hi Hello<unk>Hi Hello<unk> <unk> <unk> Hello<unk><s><unk>hi<s>there<unk>The following string should be properly encoded: Hello.<unk>But ird and <unk> ird <unk>Hey how are you doing"
