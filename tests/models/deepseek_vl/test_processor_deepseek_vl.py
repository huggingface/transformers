# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import numpy as np

from transformers import AutoProcessor, AutoTokenizer, DeepseekVLProcessor
from transformers.models.deepseek_vl.convert_deepseek_vl_weights_to_hf import CHAT_TEMPLATE
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin

if is_vision_available():
    pass

class DeepseekVLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekVLProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        special_image_tokens = {
            "image_token": "<image_placeholder>"
        }
        processor = self.processor_class.from_pretrained(
            "deepseek-ai/deepseek-vl-7b-chat",
            extra_special_tokens=special_image_tokens,
        )
        processor.save_pretrained(self.tmpdirname)
    
    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)
    
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)
    
    def prepare_processor_dict(self):
        return {
            "chat_template": CHAT_TEMPLATE,
        }
    
    def test_chat_template_single(self):
        """
        Tests that the chat template matches the original implementation when applied to a single message.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        # Single image message
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompt = ["<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:"]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)


        # Single image message with uppercase
        messages = [
            [
                {
                    "role": "USER",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompt = ["<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:"]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)


        # Checking the output dict keys for the prompt
        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Now test the ability to return dict
        messages[0][0]["content"][1].update(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertTrue(self.images_input_name in out_dict)
        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 1)
        self.assertEqual(len(out_dict["attention_mask"]), 1)
        self.assertEqual(len(out_dict[self.images_input_name]), 1)


        # Single prompt with multiple images
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this image"},
                        {"type": "image"},
                        {"type": "text", "text": "with this image"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompt = [
            "<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>:"
        ]
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

        # Multiple turns and multiple images
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this image"},
                        {"type": "image"},
                        {"type": "text", "text": "with this image"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The first image is an equation, the second is a pie chart."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "What about this third image? To which of the previous to is it more similar?",
                        },
                    ],
                },
            ]
        ]

        correct_prompt = [
            "<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>: The first image is an equation, the second is a pie chart.<｜end▁of▁sentence｜><|User|>: <image_placeholder>\nWhat about this third image? To which of the previous to is it more similar?\n\n<|Assistant|>:"
        ]
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

    def test_chat_template_accepts_processing_kwargs(self):
        """Tests that the chat template correctly handles additional processing arguments."""
        # Get processor and skip if it doesn't have a chat template
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        # Create a simple text message for testing
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        ]

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            max_length=256,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 256)

