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


# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""


expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ´»', 'çļĦ', 'çľŁ', 'è°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊĠĊ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸µ', '   ', 'ird', '   ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
expected_token_ids = [1552, 310, 247, 1071, 187, 42, 369, 5686, 275, 898, 6914, 13, 285, 436, 310, 21649, 860, 15, 187, 20025, 46549, 5225, 48561, 33656, 238, 12105, 187, 12764, 50276, 12092, 187, 12764, 50275, 12092, 46603, 50276, 187, 24387, 187, 29, 84, 31, 187, 5801, 29, 84, 31, 9088, 187, 510, 1563, 2876, 943, 320, 6283, 16202, 27, 24387, 15, 187, 1989, 209, 1817, 285, 209, 2869, 238, 26863, 50275, 1817, 50275, 35071, 187, 8262, 849, 403, 368, 2509]


@require_tokenizers
class GPTNeoXTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "EleutherAI/gpt-neox-20b"
    tokenizer_class = GPTNeoXTokenizer
    rust_tokenizer_class = GPTNeoXTokenizer
    slow_tokenizer_class = None
    test_rust_tokenizer = True
    test_slow_tokenizer = False
    from_pretrained_kwargs = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        tok_auto = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]

    def get_tokenizers(self, **kwargs):
        kwargs.setdefault("pad_token", "<|padding|>")
        return super().get_tokenizers(**kwargs)

    def test_integration_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_integration_expected_token_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)

    def test_save_and_reload(self):
        for tok in self.tokenizers:
            with self.subTest(f"{tok.__class__.__name__}"):
                original_tokens = tok.tokenize(input_string)
                original_ids = tok.encode(input_string)

                # Save tokenizer to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tok.save_pretrained(tmp_dir)

                    # Reload tokenizer from saved directory
                    reloaded_tok = tok.__class__.from_pretrained(tmp_dir)

                    # Test that reloaded tokenizer produces same results
                    reloaded_tokens = reloaded_tok.tokenize(input_string)
                    reloaded_ids = reloaded_tok.encode(input_string)

                    self.assertEqual(original_tokens, reloaded_tokens)
                    self.assertEqual(original_ids, reloaded_ids)
