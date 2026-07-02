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
    integration_expected_tokens = ['▁his', '▁is', '▁a', '▁test', '▁', '▁', '▁was', '▁born', '▁in', '▁92', '000', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '是', '▁i', '▁el', 'lo', '▁i', '▁el', 'lo', '▁el', 'lo', '<s>', '▁hi', '<s>', '▁there', '▁he', '▁following', '▁string', '▁should', '▁be', '▁proper', 'ly', '▁enc', 'od', 'ed', '▁el', 'lo', '.', '▁ut', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁ey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [1950, 512, 262, 3213, 26717, 26717, 1300, 10189, 317, 7735, 1043, 346, 1267, 512, 22762, 26773, 26803, 7904, 26859, 26957, 27053, 279, 566, 353, 279, 566, 353, 566, 353, 3, 1697, 3, 4858, 554, 10360, 9208, 8037, 343, 4433, 547, 6750, 356, 325, 566, 353, 26803, 3742, 2988, 26727, 346, 8960, 2988, 26727, 2353, 8264, 1753, 1217, 900, 11099]  # fmt: skip
    expected_tokens_from_ids = ['▁his', '▁is', '▁a', '▁test', '▁', '▁', '▁was', '▁born', '▁in', '▁92', '000', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '是', '▁i', '▁el', 'lo', '▁i', '▁el', 'lo', '▁el', 'lo', '<s>', '▁hi', '<s>', '▁there', '▁he', '▁following', '▁string', '▁should', '▁be', '▁proper', 'ly', '▁enc', 'od', 'ed', '▁el', 'lo', '.', '▁ut', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁ey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "his is a test   was born in 92000 and this is falsé. 生活的真是 i ello i ello ello<s> hi<s> there he following string should be properly encoded ello. ut ird and ปี ird ด ey how are you doing"

    @unittest.skip(reason="Tipsv2 requires pad token to have id 0")
    def test_pad_token_initialization(self):
        pass
