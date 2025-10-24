# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoTokenizer, GPTNeoXTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin



@require_tokenizers
class GPTNeoXTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "EleutherAI/gpt-neox-20b"
    tokenizer_class = GPTNeoXTokenizer
    slow_tokenizer_class = None
    test_slow_tokenizer = False
    from_pretrained_kwargs = {}


    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ´»', 'çļĦ', 'çľŁ', 'è°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊĠĊ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸µ', '   ', 'ird', '   ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
    integration_expected_token_ids = [1552, 310, 247, 1071, 187, 42, 369, 5686, 275, 898, 6914, 13, 285, 436, 310, 21649, 860, 15, 187, 20025, 46549, 5225, 48561, 33656, 238, 12105, 187, 12764, 50276, 12092, 187, 12764, 50275, 12092, 46603, 50276, 187, 24387, 187, 29, 84, 31, 187, 5801, 29, 84, 31, 9088, 187, 510, 1563, 2876, 943, 320, 6283, 16202, 27, 24387, 15, 187, 1989, 209, 1817, 285, 209, 2869, 238, 26863, 50275, 1817, 50275, 35071, 187, 8262, 849, 403, 368, 2509]
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        tokenizer = GPTNeoXTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tokenizer]

    def get_tokenizers(self, **kwargs):
        kwargs.setdefault("pad_token", "<|padding|>")
        return super().get_tokenizers(**kwargs)
