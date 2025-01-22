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
import unittest

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from transformers import AutoProcessor


@require_vision
class LlavaProcessorTest(unittest.TestCase):
    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")
        expected_prompt = "###Human: <image>\nWhat is shown in this image?###Assistant:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)
