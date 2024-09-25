# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from transformers import MllamaProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class MllamaProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "hf-internal-testing/mllama-11b"  # TODO: change
        self.processor = MllamaProcessor.from_pretrained(self.checkpoint)
        self.image1 = Image.new("RGB", (224, 220))
        self.image2 = Image.new("RGB", (512, 128))
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.bos_token = self.processor.bos_token
        self.bos_token_id = self.processor.tokenizer.bos_token_id

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "What do these images show?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The first image shows the statue of Liberty in New York."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And who is that?"},
                ],
            },
        ]

        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        expected_rendered = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "<|image|><|image|>What do these images show?"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "The first image shows the statue of Liberty in New York."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "And who is that?"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        self.assertEqual(rendered, expected_rendered)

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "This is a test sentence."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a response."},
                ],
            },
        ]
        input_ids = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        expected_ids = [
            128000,  # <|begin_of_text|>
            128006,  # <|start_header_id|>
            9125,  # "system"
            128007,  # <|end_of_header|>
            271,  # "\n\n"
            2028,
            374,
            264,
            1296,
            11914,
            13,  # "This is a test sentence."
            128009,  # <|eot_id|>
            128006,  # <|start_header_id|>
            882,  # "user"
            128007,  # <|end_of_header|>
            271,  # "\n\n"
            2028,
            374,
            264,
            2077,
            13,  # "This is a response.",
            128009,  # <|eot_id|>
            128006,  # <|start_header_id|>
            78191,  # "assistant"
            128007,  # <|end_of_header|>
            271,  # "\n\n"
        ]

        self.assertEqual(input_ids, expected_ids)

        # test image in multiple locations
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in two sentences"},
                    {"type": "image"},
                    {"type": "text", "text": " Test sentence   "},
                    {"type": "image"},
                    {"type": "text", "text": "ok\n"},
                ],
            }
        ]

        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        expected_rendered = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Describe this image in two sentences<|image|> Test sentence   <|image|>ok\n<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        self.assertEqual(rendered, expected_rendered)

        input_ids = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        # fmt: off
        expected_ids = [
            128000, 128006, 882, 128007, 271, 75885, 420, 2217, 304, 1403, 23719, 128256,
            3475, 11914, 262, 128256, 564, 198, 128009, 128006, 78191, 128007, 271,
        ]
        # fmt: on
        self.assertEqual(input_ids, expected_ids)

        # text format for content
        messages_list = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in two sentences"},
                ],
            }
        ]
        messages_str = [
            {
                "role": "user",
                "content": "<|image|>Describe this image in two sentences",
            }
        ]

        rendered_list = self.processor.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=False)
        rendered_str = self.processor.apply_chat_template(messages_str, add_generation_prompt=True, tokenize=False)
        self.assertEqual(rendered_list, rendered_str)
