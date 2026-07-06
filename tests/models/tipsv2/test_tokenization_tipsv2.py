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

    # Do not match actual Tipsv2Tokenizer.from_pretrained values because tests hardcode do_lower_case=False
    integration_expected_tokens = ['▁', '<0x54>', 'h', 'is', '▁is', '▁a', '▁test', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '▁', '<0x49>', '▁was', '▁born', '▁in', '▁92', '000', '<0x2C>', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '▁', '<0x48>', 'i', '▁', '<0x48>', 'ello', '▁', '<0x48>', 'i', '▁', '<0x48>', 'ello', '▁', '<0x48>', 'ello', '<s>', '▁hi', '<s>', '▁there', '▁', '<0x54>', 'he', '▁following', '▁string', '▁should', '▁be', '▁proper', 'ly', '▁enc', 'od', 'ed', '<0x3A>', '▁', '<0x48>', 'ello', '.', '▁', '<0x42>', 'ut', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁', '<0x48>', 'ey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [26717, 88, 26732, 309, 512, 262, 3213, 26717, 244, 163, 156, 142, 26717, 77, 1300, 10189, 317, 7735, 1043, 48, 346, 1267, 512, 22762, 26773, 26803, 7904, 26859, 26957, 236, 180, 159, 27053, 26717, 76, 26721, 26717, 76, 3354, 26717, 76, 26721, 26717, 76, 3354, 26717, 76, 3354, 3, 1697, 3, 4858, 26717, 88, 638, 10360, 9208, 8037, 343, 4433, 547, 6750, 356, 325, 62, 26717, 76, 3354, 26803, 26717, 70, 366, 2988, 26727, 346, 8960, 2988, 26727, 2353, 26717, 76, 1943, 1753, 1217, 900, 11099]  # fmt: skip
    expected_tokens_from_ids = ['▁', '<0x54>', 'h', 'is', '▁is', '▁a', '▁test', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '▁', '<0x49>', '▁was', '▁born', '▁in', '▁92', '000', '<0x2C>', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '▁', '<0x48>', 'i', '▁', '<0x48>', 'ello', '▁', '<0x48>', 'i', '▁', '<0x48>', 'ello', '▁', '<0x48>', 'ello', '<s>', '▁hi', '<s>', '▁there', '▁', '<0x54>', 'he', '▁following', '▁string', '▁should', '▁be', '▁proper', 'ly', '▁enc', 'od', 'ed', '<0x3A>', '▁', '<0x48>', 'ello', '.', '▁', '<0x42>', 'ut', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁', '<0x48>', 'ey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    @unittest.skip(reason="Tipsv2 requires pad token to have id 0")
    def test_pad_token_initialization(self):
        pass
