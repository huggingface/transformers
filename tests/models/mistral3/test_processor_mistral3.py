# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

import requests

from transformers import PixtralProcessor
from transformers.testing_utils import require_read_token, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


@require_vision
@require_read_token
class Mistral3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    """This tests Pixtral processor with the new `spatial_merge_size` argument in Mistral3."""

    processor_class = PixtralProcessor

    @classmethod
    def setUpClass(cls):
        cls.url_0 = "https://www.ilankelman.org/stopsigns/australia.jpg"
        cls.image_0 = Image.open(requests.get(cls.url_0, stream=True).raw)
        cls.url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cls.image_1 = Image.open(requests.get(cls.url_1, stream=True).raw)
        cls.url_2 = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
        cls.image_2 = Image.open(requests.get(cls.url_2, stream=True).raw)

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = PixtralProcessor.from_pretrained(
            "hf-internal-testing/Mistral-Small-3.1-24B-Instruct-2503-only-processor"
        )
        processor.save_pretrained(self.tmpdirname)

    def get_processor(self):
        return self.processor_class.from_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_chat_template_accepts_processing_kwargs(self):
        # override to use slow image processor to return numpy arrays
        processor = self.processor_class.from_pretrained(self.tmpdirname, use_fast=False)
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

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
            truncation=True,
            max_length=50,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 50)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=5,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 5)

        # Now test the ability to return dict
        messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
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

    def test_chat_template(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname, use_fast=False)
        expected_prompt = "<s>[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST][IMG]What is shown in this image?[/INST]"

        messages = [
            {
                "role": "system",
                "content": "",
            },
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

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 198
        image_token_index = 10

        messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    def test_processor_with_single_image(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = "USER: [IMG]\nWhat's the content of the image? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.patch_size = 6

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=self.image_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 30]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [1, 21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=self.url_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 30]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [1, 21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing inputs as a single list
        inputs_image = processor(text=prompt_string, images=[self.image_0], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 30]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [1, 21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test as nested single list
        inputs_image = processor(text=prompt_string, images=[[self.image_0]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 30]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [1, 21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_single_list(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.patch_size = 6

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=[self.image_0, self.image_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 24, 30]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
                    )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=[self.url_0, self.url_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 24, 30]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in as a nested list
        inputs_url = processor(text=prompt_string, images=[[self.image_0, self.image_1]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 24, 30]))

        # fmt: off
        self.assertEqual(
            inputs_url["input_ids"][0].tolist(),
            [1, 21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_multiple_lists(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = [
            "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:",
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ]
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0, self.image_1], [self.image_2]]

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.patch_size = 6

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 2)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 30, 30]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 2)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 30, 30]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing as a single flat list
        inputs_image = processor(
            text=prompt_string, images=[self.image_0, self.image_1, self.image_2], return_tensors="pt", padding=True
        )
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 30, 30]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [1, 21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    def test_processor_returns_full_length_batches(self):
        # to avoid https://github.com/huggingface/transformers/issues/34204
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = [
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ] * 5
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0]] * 5

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.patch_size = 6

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 5)
