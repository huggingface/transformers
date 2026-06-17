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

    integration_expected_tokens = ['▁', 'T', 'h', 'is', '▁is', '▁a', '▁t', 'est', '▁', '😊', '▁', 'I', '▁w', 'as', '▁b', 'orn', '▁in', '▁9', '20', '00', ',', '▁and', '▁th', 'is', '▁is', '▁f', 'als', 'é', '.', '▁生活', '的', '真', '谛', '是', '▁', 'H', 'i', '▁', 'H', 'el', 'lo', '▁', 'H', 'i', '▁', 'H', 'el', 'lo', '▁', 'H', 'el', 'lo', '<s>', '▁hi', '<s>', '▁the', 're', '▁', 'T', 'he', '▁f', 'ol', 'low', 'ing', '▁st', 'ri', 'ng', '▁s', 'ho', 'uld', '▁be', '▁pr', 'op', 'er', 'ly', '▁enc', 'od', 'ed', ':', '▁', 'H', 'el', 'lo', '.', '▁', 'B', 'ut', '▁i', 'rd', '▁and', '▁ปี', '▁i', 'rd', '▁ด', '▁', 'H', 'ey', '▁how', '▁a', 're', '▁you', '▁do', 'ing']  # fmt: skip
    integration_expected_token_ids = [26717, 2, 26732, 309, 512, 262, 267, 434, 26717, 2, 26717, 2, 287, 300, 270, 1099, 317, 460, 443, 362, 2, 346, 314, 309, 512, 276, 1165, 26773, 26803, 7904, 26859, 26957, 2, 27053, 26717, 2, 26721, 26717, 2, 312, 353, 26717, 2, 26721, 26717, 2, 312, 353, 26717, 2, 312, 353, 3, 1697, 3, 345, 282, 26717, 2, 638, 276, 334, 695, 307, 368, 292, 447, 260, 301, 2769, 343, 497, 411, 263, 547, 6750, 356, 325, 2, 26717, 2, 312, 353, 26803, 26717, 2, 366, 279, 4974, 346, 8960, 279, 4974, 2353, 26717, 2, 1943, 1753, 262, 282, 900, 433, 307]  # fmt: skip
    expected_tokens_from_ids = ['▁', '<unk>', 'h', 'is', '▁is', '▁a', '▁t', 'est', '▁', '<unk>', '▁', '<unk>', '▁w', 'as', '▁b', 'orn', '▁in', '▁9', '20', '00', '<unk>', '▁and', '▁th', 'is', '▁is', '▁f', 'als', 'é', '.', '▁生活', '的', '真', '<unk>', '是', '▁', '<unk>', 'i', '▁', '<unk>', 'el', 'lo', '▁', '<unk>', 'i', '▁', '<unk>', 'el', 'lo', '▁', '<unk>', 'el', 'lo', '<s>', '▁hi', '<s>', '▁the', 're', '▁', '<unk>', 'he', '▁f', 'ol', 'low', 'ing', '▁st', 'ri', 'ng', '▁s', 'ho', 'uld', '▁be', '▁pr', 'op', 'er', 'ly', '▁enc', 'od', 'ed', '<unk>', '▁', '<unk>', 'el', 'lo', '.', '▁', '<unk>', 'ut', '▁i', 'rd', '▁and', '▁ปี', '▁i', 'rd', '▁ด', '▁', '<unk>', 'ey', '▁how', '▁a', 're', '▁you', '▁do', 'ing']  # fmt: skip
    integration_expected_decoded_text = "<unk>his is a test <unk> <unk> was born in 92000<unk> and this is falsé. 生活的真<unk>是 <unk>i <unk>ello <unk>i <unk>ello <unk>ello<s> hi<s> there <unk>he following string should be properly encoded<unk> <unk>ello. <unk>ut ird and ปี ird ด <unk>ey how are you doing"

    @unittest.skip(reason="Tipsv2 requires pad token to have id 0")
    def test_pad_token_initialization(self):
        pass
