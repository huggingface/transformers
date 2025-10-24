# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import tempfile
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers import AutoTokenizer
from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)


@require_tokenizers
class GemmaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):  # TEMP we won't use the mixin in v5
    from_pretrained_id = "google/gemma-7b"
    tokenizer_class = GemmaTokenizer
    test_slow_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = {}
    test_seq2seq = False

    # Integration test data - Gemma uses a custom input string
    integration_test_input_string = "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61 生活的真谛是 Hi  Hello Hi   <s> Hello<s>how▁▁ and ▁<bos>Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!   "
    integration_expected_tokens = ["Hey", "<eos>", ".", "▁", "\t\t", "▁", "\n\n", "you", "▁▁", "é", "▁▁", "@#", "😈", "▁▁", "🤗", "!", "▁▁▁▁▁▁▁", ",", "▁", "1", "2", "3", "4", "▁", "1", "5", "▁", "5", ",", "6", "1", "▁生活", "的", "真", "谛", "是", "▁Hi", "▁▁", "Hello", "▁Hi", "▁▁▁", "<s>", "▁Hello", "<s>", "how", "▁▁", "▁and", "▁▁", "<bos>", "Hey", "<eos>", ".", "▁", "\t\t", "▁", "\n\n", "you", "▁▁", "é", "▁▁", "@#", "😈", "▁▁", "🤗", "!", "▁▁▁"]
    integration_expected_token_ids = [2, 6750, 1, 235265, 235248, 255969, 235248, 109, 4747, 139, 235335, 139, 216311, 241316, 139, 239880, 235341, 144, 235269, 235248, 235274, 235284, 235304, 235310, 235248, 235274, 235308, 235248, 235308, 235269, 235318, 235274, 64001, 235370, 235710, 245467, 235427, 11192, 139, 4521, 11192, 140, 204, 25957, 204, 1139, 139, 578, 139, 2, 6750, 1, 235265, 235248, 255969, 235248, 109, 4747, 139, 235335, 139, 216311, 241316, 139, 239880, 235341, 140]
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "google/gemma-7b"

        tokenizer = GemmaTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tokenizer]
