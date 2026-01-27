# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the RemBert tokenizer."""

import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers import RemBertTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers


SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class RemBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/rembert"
    tokenizer_class = RemBertTokenizer
    pre_trained_model_path = "google/rembert"

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活', '的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁', '▁', '▁', '▁', '▁', '▁', '▁Hello', '▁', '<s>', '▁hi', '<s>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'coded', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [1357, 619, 577, 3515, 573, 119091, 623, 820, 18648, 586, 940, 7905, 571, 599, 902, 619, 98696, 780, 572, 573, 6334, 649, 3975, 244511, 1034, 3211, 24624, 3211, 24624, 573, 573, 573, 573, 573, 573, 24624, 573, 3, 1785, 3, 90608, 660, 6802, 15930, 2575, 689, 43272, 592, 185434, 581, 24624, 572, 2878, 1032, 620, 599, 9070, 1032, 620, 60827, 20490, 1865, 781, 734, 9711]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活', '的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁', '▁', '▁', '▁', '▁', '▁', '▁Hello', '▁', '<s>', '▁hi', '<s>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'coded', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello       Hello <s> hi<s>there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"
