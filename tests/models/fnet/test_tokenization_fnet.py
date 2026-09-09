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

from transformers import FNetTokenizer, FNetTokenizerFast
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
@require_tokenizers
class FNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/fnet-base"
    tokenizer_class = FNetTokenizer
    # FNet is encoder-only, and its `model_input_names` deliberately omits `attention_mask`, which the
    # seq2seq batch test asserts on.
    test_seq2seq = False
    # TokenizersExtractor rebuilds the tokenizer from the sentencepiece model alone and does not carry over
    # `do_lower_case=False`, so the extracted tokenizer lowercases where FNet's does not.
    test_tokenizer_from_extractor = False

    # Matches FNetTokenizer.from_pretrained("google/fnet-base"). FNet keeps casing and accents, unlike ALBERT,
    # whose implementation it otherwise reuses.
    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2', '000', ',', '▁and', '▁this', '▁is', '▁f', 'als', 'é', '.', '▁', '生', '活', '的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁enc', 'oded', ':', '▁Hello', '.', '▁But', '▁', 'ird', '▁and', '▁', 'ป', 'ี', '▁', 'ird', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [325, 65, 8, 1123, 16657, 18014, 57, 158, 3446, 38, 917, 16695, 946, 16680, 36, 168, 65, 26, 560, 16747, 16678, 16657, 17093, 17620, 16803, 18107, 31092, 17046, 5364, 9665, 5364, 9665, 9665, 6517, 16664, 16748, 7420, 16762, 16664, 16748, 11448, 97, 1796, 7185, 573, 67, 4622, 1703, 13973, 16717, 9665, 16678, 760, 16657, 1213, 36, 16657, 18004, 17498, 16657, 1213, 16657, 17551, 10239, 409, 108, 60, 1553]  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello Hello <s> hi<s>there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    def test_token_type_ids_are_a_model_input(self):
        # The one behavior FNetTokenizer overrides on top of AlbertTokenizer: FNet's forward takes token_type_ids,
        # so the tokenizer has to emit them.
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.model_input_names, ["input_ids", "token_type_ids"])
        self.assertIn("token_type_ids", tokenizer("A sentence", "And its pair"))

    def test_fast_is_an_alias(self):
        # FNetTokenizer is already backed by `tokenizers`; FNetTokenizerFast is kept only as a public alias.
        self.assertIs(FNetTokenizerFast, FNetTokenizer)
