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


import json
import os
import tempfile
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)


@require_tokenizers
class GemmaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "hf-internal-testing/dummy-gemma"
    tokenizer_class = GemmaTokenizer

    integration_expected_tokens = ['This', '▁is', '▁a', '▁test', '▁😊', '\n', 'I', '▁was', '▁born', '▁in', '▁', '9', '2', '0', '0', '0', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '\n', '生活的', '真', '谛', '是', '\n', 'Hi', '▁▁', 'Hello', '\n', 'Hi', '▁▁▁', 'Hello', '\n\n', '▁', '\n', '▁▁', '\n', '▁Hello', '\n', '<', 's', '>', '\n', 'hi', '<', 's', '>', 'there', '\n', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '\n', 'But', '▁i', 'rd', '▁and', '▁ปี', '▁▁▁', 'ird', '▁▁▁', 'ด', '\n', 'Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [1596, 603, 476, 2121, 44416, 108, 235285, 729, 7565, 575, 235248, 235315, 235284, 235276, 235276, 235276, 235269, 578, 736, 603, 40751, 235335, 235265, 108, 122182, 235710, 245467, 235427, 108, 2151, 139, 4521, 108, 2151, 140, 4521, 109, 235248, 108, 139, 108, 25957, 108, 235322, 235256, 235313, 108, 544, 235322, 235256, 235313, 11048, 108, 651, 2412, 2067, 1412, 614, 10338, 49748, 235292, 25957, 235265, 108, 1860, 496, 1924, 578, 73208, 140, 5650, 140, 235732, 108, 6750, 1368, 708, 692, 3900]  # fmt: skip
    expected_tokens_from_ids = ['This', '▁is', '▁a', '▁test', '▁😊', '\n', 'I', '▁was', '▁born', '▁in', '▁', '9', '2', '0', '0', '0', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '\n', '生活的', '真', '谛', '是', '\n', 'Hi', '▁▁', 'Hello', '\n', 'Hi', '▁▁▁', 'Hello', '\n\n', '▁', '\n', '▁▁', '\n', '▁Hello', '\n', '<', 's', '>', '\n', 'hi', '<', 's', '>', 'there', '\n', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '\n', 'But', '▁i', 'rd', '▁and', '▁ปี', '▁▁▁', 'ird', '▁▁▁', 'ด', '\n', 'Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    def test_add_bos_token_defaults_to_true(self):
        # A checkpoint may omit `add_bos_token` and ship a post-processor that adds no BOS (e.g. Gemma 4)
        tokenizer = GemmaTokenizer.from_pretrained(self.from_pretrained_id[0])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            config_file = os.path.join(tmp_dir, "tokenizer_config.json")
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
            config.pop("add_bos_token", None)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f)
            tokenizer_file = os.path.join(tmp_dir, "tokenizer.json")
            with open(tokenizer_file, encoding="utf-8") as f:
                backend = json.load(f)
            backend["post_processor"] = {
                "type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {},
            }
            with open(tokenizer_file, "w", encoding="utf-8") as f:
                json.dump(backend, f)
            tokenizer = GemmaTokenizer.from_pretrained(tmp_dir)
        self.assertEqual(tokenizer("Hello")["input_ids"][0], tokenizer.bos_token_id)
