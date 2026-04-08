# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers.models.xlnet.tokenization_xlnet import XLNetTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class XLNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "xlnet/xlnet-base-cased"
    tokenizer_class = XLNetTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁false', '.', '▁', '生活的真谛是', '▁Hi', '▁', 'Hello', '▁Hi', '▁', 'Hello', '▁', 'Hello', '<s>', '▁', 'hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁', 'Hello', '.', '▁But', '▁', 'ir', 'd', '▁and', '▁', 'ป', '▁', 'ir', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [122, 27, 24, 934, 17, 0, 35, 30, 1094, 25, 664, 7701, 19, 21, 52, 27, 4417, 9, 17, 0, 4036, 17, 11368, 4036, 17, 11368, 17, 11368, 1, 17, 2582, 1, 105, 32, 405, 4905, 170, 39, 4183, 23147, 60, 17, 11368, 9, 130, 17, 1121, 66, 21, 17, 0, 17, 1121, 66, 17, 0, 14239, 160, 41, 44, 690]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁false', '.', '▁', '<unk>', '▁Hi', '▁', 'Hello', '▁Hi', '▁', 'Hello', '▁', 'Hello', '<s>', '▁', 'hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁', 'Hello', '.', '▁But', '▁', 'ir', 'd', '▁and', '▁', '<unk>', '▁', 'ir', 'd', '▁', '<unk>', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is false. <unk> Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"
