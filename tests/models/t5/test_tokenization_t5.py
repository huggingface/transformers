# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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

from transformers import T5Tokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class T5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google-t5/t5-small"
    tokenizer_class = T5Tokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁', 'a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encode', 'd', ':', '▁Hello', '.', '▁But', '▁', 'i', 'r', 'd', '▁and', '▁', 'ปี', '▁', 'i', 'r', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [100, 19, 3, 9, 794, 3, 2, 27, 47, 2170, 16, 668, 13527, 6, 11, 48, 19, 12553, 7, 154, 5, 3, 2, 2018, 8774, 2018, 8774, 8774, 3, 2, 7, 3155, 7102, 2, 7, 3155, 12137, 37, 826, 6108, 225, 36, 3085, 23734, 26, 10, 8774, 5, 299, 3, 23, 52, 26, 11, 3, 2, 3, 23, 52, 26, 3, 2, 9459, 149, 33, 25, 692]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁', 'a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '<unk>', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<unk>', 's', '>', '▁hi', '<unk>', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encode', 'd', ':', '▁Hello', '.', '▁But', '▁', 'i', 'r', 'd', '▁and', '▁', '<unk>', '▁', 'i', 'r', 'd', '▁', '<unk>', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is falsé. <unk> Hi Hello Hi Hello Hello <unk>s> hi<unk>s>there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"
