# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import Tipsv2Tokenizer
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
class Tipsv2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/tipsv2-b14"
    tokenizer_class = Tipsv2Tokenizer

    # Matches Tipsv2Tokenizer.from_pretrained values
    integration_expected_tokens = ['▁this', '▁is', '▁a', '▁test', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '▁i', '▁was', '▁born', '▁in', '▁92', '000', '<0x2C>', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '▁hi', '▁hello', '▁hi', '▁hello', '▁hello', '<s>', '▁hi', '<s>', '▁there', '▁the', '▁following', '▁string', '▁should', '▁be', '▁proper', 'ly', '▁enc', 'od', 'ed', '<0x3A>', '▁hello', '.', '▁but', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [1267, 512, 262, 3213, 26717, 244, 163, 156, 142, 279, 1300, 10189, 317, 7735, 1043, 48, 346, 1267, 512, 22762, 26773, 26803, 7904, 26859, 26957, 236, 180, 159, 27053, 1697, 5928, 1697, 5928, 5928, 3, 1697, 3, 4858, 345, 10360, 9208, 8037, 343, 4433, 547, 6750, 356, 325, 62, 5928, 26803, 1389, 2988, 26727, 346, 8960, 2988, 26727, 2353, 11856, 1753, 1217, 900, 11099]  # fmt: skip
    expected_tokens_from_ids = ['▁this', '▁is', '▁a', '▁test', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '▁i', '▁was', '▁born', '▁in', '▁92', '000', '<0x2C>', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '▁hi', '▁hello', '▁hi', '▁hello', '▁hello', '<s>', '▁hi', '<s>', '▁there', '▁the', '▁following', '▁string', '▁should', '▁be', '▁proper', 'ly', '▁enc', 'od', 'ed', '<0x3A>', '▁hello', '.', '▁but', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "this is a test 😊 i was born in 92000, and this is falsé. 生活的真谛是 hi hello hi hello hello<s> hi<s> there the following string should be properly encoded: hello. but ird and ปี ird ด hey how are you doing"
