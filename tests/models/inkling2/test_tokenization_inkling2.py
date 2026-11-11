# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers import Inkling2Tokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class Inkling2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "hf-internal-testing/tiny-inkling2"
    tokenizer_class = Inkling2Tokenizer

    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ġ', 'ðŁĺĬ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '920', '00', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġ', 'fals', 'Ã©', '.', 'Ċ', 'çĶŁæ´»', 'çļĦ', 'çľŁ', 'è°Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'ĠĠ', 'ĠHello', 'ĊĊ', 'ĠĊ', 'ĠĠĊ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸Ľà¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [1072, 149884, 149878, 150170, 220, 113552, 198, 40, 149933, 159937, 149881, 220, 91417, 68018, 11, 149880, 149928, 149884, 220, 48454, 5164, 13, 198, 124056, 25208, 22810, 7139, 15635, 198, 4896, 220, 188766, 198, 4896, 149859, 188766, 68014, 197045, 149865, 188766, 198, 27, 82, 29, 198, 1910, 27, 82, 29, 1179, 198, 371, 150325, 150379, 150201, 149895, 159855, 181352, 25, 188766, 13, 198, 1748, 220, 1949, 149880, 220, 82970, 149859, 220, 1949, 149859, 220, 11669, 198, 34869, 150202, 149914, 149941, 152640]  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"
