# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Team. All rights reserved.
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

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.testing_utils import (
    require_tokenizers,
)


@require_tokenizers
class Qwen2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    tokenizer_class = Qwen2Tokenizer

    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.Ċ', 'çĶŁæ´»çļĦ', 'çľŁ', 'è°Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'ĠĠ', 'ĠHello', 'ĊĊ', 'ĠĊĠĠĊ', 'ĠHello', 'Ċ', '<s', '>Ċ', 'hi', '<s', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸Ľ', 'à¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [1986, 374, 264, 1273, 26525, 232, 198, 40, 572, 9223, 304, 220, 24, 17, 15, 15, 15, 11, 323, 419, 374, 31932, 963, 624, 105301, 88051, 116109, 20412, 198, 13048, 220, 21927, 198, 13048, 256, 21927, 271, 48426, 21927, 198, 44047, 397, 6023, 44047, 29, 18532, 198, 785, 2701, 914, 1265, 387, 10277, 20498, 25, 21927, 624, 3983, 220, 2603, 323, 220, 54684, 28319, 256, 220, 2603, 256, 220, 37033, 198, 18665, 1246, 525, 498, 3730]  # fmt: skip
    expected_tokens_from_ids = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.Ċ', 'çĶŁæ´»çļĦ', 'çľŁ', 'è°Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'ĠĠ', 'ĠHello', 'ĊĊ', 'ĠĊĠĠĊ', 'ĠHello', 'Ċ', '<s', '>Ċ', 'hi', '<s', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸Ľ', 'à¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"


    def test_honors_pretokenize_regex_from_kwargs(self):
        """Qwen3.6 / Qwen3.8 ship `pretokenize_regex` with `\\p{M}` in tokenizer_config.json.

        Qwen2Tokenizer used to ignore that key and always apply the module-level
        PRETOKENIZE_REGEX, so AutoTokenizer produced different IDs than the published
        tokenizer.json for scripts that use combining marks (see #49066).
        """
        import json

        from transformers.models.qwen2.tokenization_qwen2 import PRETOKENIZE_REGEX

        refined = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        self.assertNotEqual(refined, PRETOKENIZE_REGEX)
        self.assertIn(r"\p{M}", refined)
        self.assertNotIn(r"\p{M}", PRETOKENIZE_REGEX)

        def split_pattern(tokenizer):
            backend = json.loads(tokenizer.backend_tokenizer.to_str())
            return backend["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]

        with_kwarg = Qwen2Tokenizer(pretokenize_regex=refined)
        self.assertEqual(split_pattern(with_kwarg), refined)

        without_kwarg = Qwen2Tokenizer()
        self.assertEqual(split_pattern(without_kwarg), PRETOKENIZE_REGEX)
