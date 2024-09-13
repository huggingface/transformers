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
import requests
import unittest

import torch

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image
    from transformers import AutoProcessor, PixtralProcessor, PixtralImageProcessor, AutoTokenizer


@require_vision
class PixtralProcessorTest(unittest.TestCase):
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
        super().setUp()

        # FIXME - just load the processor directly from the checkpoint
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        image_processor = PixtralImageProcessor()
        self.processor = PixtralProcessor(tokenizer=tokenizer, image_processor=image_processor)

    def test_chat_template(self):
        expected_prompt = "USER: [IMG]\nWhat is shown in this image? ASSISTANT:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    def test_image_token_filling(self):
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 1526
        image_token_index = 32000

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = self.processor(
            text=[self.processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    def test_processor_with_single_image(self):
        prompt_string = "USER: [IMG]\nWhat's the content of the image? ASSISTANT:"

        # Make small for checking image token expansion
        self.processor.image_processor.size = {"longest_edge": 30}
        self.processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = self.processor(text=prompt_string, images=self.image_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["images"], list)
        self.assertTrue(len(inputs_image["images"]) == 1)
        self.assertIsInstance(inputs_image["images"][0], list)
        self.assertTrue(len(inputs_image["images"][0]) == 1)
        self.assertIsInstance(inputs_image["images"][0][0], torch.Tensor)

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            list(input_ids[0]),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [1, 3148, 1001, 29901, 518, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 29962, 13, 5618, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 319, 1799, 9047, 13566, 29901]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = self.processor(text=prompt_string, images=self.url_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_url["images"], list)
        self.assertTrue(len(inputs_url["images"]) == 1)
        self.assertIsInstance(inputs_url["images"][0], list)
        self.assertTrue(len(inputs_url["images"][0]) == 1)
        self.assertIsInstance(inputs_url["images"][0][0], torch.Tensor)

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            list(input_ids[0]),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [1, 3148, 1001, 29901, 518, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 29962, 13, 5618, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 319, 1799, 9047, 13566, 29901]
        )
        # fmt: on

    def test_processor_with_multiple_images_single_list(self):
        prompt_string = "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:"

        # Make small for checking image token expansion
        self.processor.image_processor.size = {"longest_edge": 30}
        self.processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = self.processor(text=prompt_string, images=[self.image_0, self.image_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["images"], list)
        self.assertTrue(len(inputs_image["images"]) == 1)
        self.assertIsInstance(inputs_image["images"][0], list)
        self.assertTrue(len(inputs_image["images"][0]) == 2)
        self.assertIsInstance(inputs_image["images"][0][0], torch.Tensor)

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            list(input_ids[0]),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 3148, 1001, 29901, 518, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 29962, 13, 5618, 29915, 29879, 278, 4328, 1546, 1438, 1023, 4558, 29973, 319, 1799, 9047, 13566, 29901]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = self.processor(text=prompt_string, images=[self.url_0, self.url_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_url["images"], list)
        self.assertTrue(len(inputs_url["images"]) == 1)
        self.assertIsInstance(inputs_url["images"][0], list)
        self.assertTrue(len(inputs_url["images"][0]) == 2)
        self.assertIsInstance(inputs_url["images"][0][0], torch.Tensor)

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            list(input_ids[0]),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 3148, 1001, 29901, 518, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 29962, 13, 5618, 29915, 29879, 278, 4328, 1546, 1438, 1023, 4558, 29973, 319, 1799, 9047, 13566, 29901]
        )
        # fmt: on

    def test_processor_with_multiple_images_multiple_lists(self):
        prompt_string = [
            "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:",
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ]
        image_inputs = [[self.image_0, self.image_1], [self.image_2]]

        # Make small for checking image token expansion
        self.processor.image_processor.size = {"longest_edge": 30}
        self.processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = self.processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 2)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["images"], list)
        self.assertTrue(len(inputs_image["images"]) == 2)
        self.assertIsInstance(inputs_image["images"][0], list)
        self.assertTrue(len(inputs_image["images"][0]) == 2)
        self.assertIsInstance(inputs_image["images"][0][0], torch.Tensor)

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            list(input_ids[0]),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 3148, 1001, 29901, 518, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 29962, 13, 5618, 29915, 29879, 278, 4328, 1546, 1438, 1023, 4558, 29973, 319, 1799, 9047, 13566, 29901]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = self.processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 2)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_url["images"], list)
        self.assertTrue(len(inputs_url["images"]) == 2)
        self.assertIsInstance(inputs_url["images"][0], list)
        self.assertTrue(len(inputs_url["images"][0]) == 2)
        self.assertIsInstance(inputs_url["images"][0][0], torch.Tensor)

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            list(input_ids[0]),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 3148, 1001, 29901, 518, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 29933, 1525, 22311, 3816, 7833, 29954, 3816, 7833, 29954, 3816, 7833, 29954, 29918, 11794, 29962, 13, 5618, 29915, 29879, 278, 4328, 1546, 1438, 1023, 4558, 29973, 319, 1799, 9047, 13566, 29901]
        )
        # fmt: on
