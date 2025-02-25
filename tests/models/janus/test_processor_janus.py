# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Janus model."""

import tempfile
import unittest

from transformers import JanusProcessor, AutoProcessor, AutoTokenizer
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin
from transformers.models.janus.convert_janus_weights_to_hf import CHAT_TEMPLATE

if is_vision_available():
    from transformers import JanusImageProcessor

# This will be changed to HUB location once the final converted model is uploaded there
TMP_LOCATION = "./hub_files"

class JanusProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = JanusProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        # Similar to Qwen2VLProcessorTest. Tests are done with 1B processor (7B tokenizer is different)
        processor = self.processor_class.from_pretrained(TMP_LOCATION)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def prepare_processor_dict(self):
        # similar to Emu3 and Qwen2VLProcessorTest, but keep the template in the convert script to avoid duplicated code
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

        correct_prompt = ['<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:']

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

        """
        Warning: normally, the other models have a test comparing chat template+tokenization as two separate steps 
        versus as a single step (i.e. processor.apply_chat_template(..., tokenize=True)). However, our processor has
        some extra steps other than simply applying prompt to tokenizer. These include prepending the default system
        prompts and, following the implementation from the Janus codebase, expanding the image token.
        """

        # Checking the output dict keys
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


        # Passing generation prompt explicitly
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ],
                }
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        self.assertEqual(formatted_prompt, correct_prompt)

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

        correct_prompt = ['<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>:']
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
                        {"type": "text",
                         "text": "What about this third image? To which of the previous to is it more similar?"},
                    ],
                },
            ]
        ]

        correct_prompt = ['<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>: The first image is an equation, the second is a pie chart.<｜end▁of▁sentence｜><|User|>: <image_placeholder>\nWhat about this third image? To which of the previous to is it more similar?\n\n<|Assistant|>:']
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

    def test_chat_template_batched(self):
        """
        Tests that the chat template matches the original implementation when applied to a batch of messages.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        # Test 1: Simple single image per message batch
        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
            ],
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

        correct_prompts = ['<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:',
                           '<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:']

        formatted_prompts = processor.apply_chat_template(batched_messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompts, correct_prompts)

        # Similarly to the single case, no test for chat template+tokenization as two separate steps versus as a single step

        # Checking the output dict keys
        out_dict = processor.apply_chat_template(
            batched_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
        )
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Verify image inputs are included in the output dict
        batched_messages[0][0]["content"][1].update(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        batched_messages[1][0]["content"][1].update(
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"}
        )
        out_dict = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True
        )
        self.assertTrue(self.images_input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), 2)  # Batch size for text
        self.assertEqual(len(out_dict["attention_mask"]), 2)  # Batch size for attention mask
        self.assertEqual(len(out_dict[self.images_input_name]), 2)  # Batch size for images

        # Test 2: Two images per message batch with different prompts
        batched_messages = [
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
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe how the previous image compares to the following"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompts = [
            '<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>:',
            '<|User|>: <image_placeholder>\nDescribe how the previous image compares to the following\n<image_placeholder>\n\n<|Assistant|>:']
        formatted_prompts = processor.apply_chat_template(batched_messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompts, correct_prompts)

        # Test 3: Multi-turn conversations with multiple images
        batched_messages = [
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
                        {"type": "text",
                         "text": "What about this third image? To which of the previous to is it more similar?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe how the previous image compares to the following"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The first image is a formula, the second is a plot."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Which of them is closer to the following?"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompts = [
            '<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>: The first image is an equation, the second is a pie chart.<｜end▁of▁sentence｜><|User|>: <image_placeholder>\nWhat about this third image? To which of the previous to is it more similar?\n\n<|Assistant|>:',
            '<|User|>: <image_placeholder>\nDescribe how the previous image compares to the following\n<image_placeholder>\n\n<|Assistant|>: The first image is a formula, the second is a plot.<｜end▁of▁sentence｜><|User|>: Which of them is closer to the following?\n<image_placeholder>\n\n<|Assistant|>:']
        formatted_prompts = processor.apply_chat_template(batched_messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompts, correct_prompts)

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

        # Test 1: Padding to max_length
        # PS: we have to override the parent max_length of 50 to 80 because the output is already 51 tokens
        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            max_length=80,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 80)

        # Test 2: Truncation
        # Verify that the output is truncated to exactly 5 tokens
        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=5,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 5)

        # Test 3: Image processing kwargs
        # Add an image and test image processing parameters
        messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        # Process with image rescaling and verify the pixel values are negative
        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_rescale=True,
            rescale_factor=-1,
            return_tensors="np",
        )
        self.assertLessEqual(out_dict[self.images_input_name][0][0].mean(), 0)

