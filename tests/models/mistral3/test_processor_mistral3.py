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

import numpy as np

from transformers import PixtralProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


@require_vision
class Mistral3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    """This tests Pixtral processor with the new `spatial_merge_size` argument in Mistral3."""

    processor_class = PixtralProcessor

    @classmethod
    def setUpClass(cls):
        cls.url_0 = "https://www.ilankelman.org/stopsigns/australia.jpg"
        cls.image_0 = np.random.randint(255, size=(3, 876, 1300), dtype=np.uint8)
        cls.url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cls.image_1 = np.random.randint(255, size=(3, 480, 640), dtype=np.uint8)
        cls.image_2 = np.random.randint(255, size=(3, 1024, 1024), dtype=np.uint8)

        cls.tmpdirname = tempfile.mkdtemp()
        cls.addClassCleanup(lambda tempdir=cls.tmpdirname: shutil.rmtree(tempdir))

        processor_kwargs = cls.prepare_processor_dict()
        processor = PixtralProcessor.from_pretrained(
            "hf-internal-testing/Mistral-Small-3.1-24B-Instruct-2503-only-processor", **processor_kwargs
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

    def get_processor(self):
        return self.processor_class.from_pretrained(self.tmpdirname)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{%- set today = strftime_now(\"%Y-%m-%d\") %}\n{%- set default_system_message = \"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYour knowledge base was last updated on 2023-10-01. The current date is \" + today + \".\\n\\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\\"What are some good restaurants around me?\\\" => \\\"Where are you?\\\" or \\\"When is the next flight to Tokyo\\\" => \\\"Where do you travel from?\\\")\" %}\n\n{{- bos_token }}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- if messages[0] is string %}\n        {%- set system_message = messages[0]['content'] %}\n        {%- set loop_messages = messages[1:] %}\n    {%- else %} \n        {%- set system_message = messages[0]['content'][0]['text'] %}\n        {%- set loop_messages = messages[1:] %}\n    {%- endif %}\n{%- else %}\n    {%- set system_message = default_system_message %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}\n\n{%- for message in loop_messages %}\n    {%- if message['role'] == 'user' %}\n            {%- if message['content'] is string %}\n            {{- '[INST]' + message['content'] + '[/INST]' }}\n            {%- else %}\n                    {{- '[INST]' }}\n                    {%- for block in message['content'] %}\n                            {%- if block['type'] == 'text' %}\n                                    {{- block['text'] }}\n                            {%- elif block['type'] == 'image' or block['type'] == 'image_url' %}\n                                    {{- '[IMG]' }}\n                                {%- else %}\n                                    {{- raise_exception('Only text and image blocks are supported in message content!') }}\n                                {%- endif %}\n                        {%- endfor %}\n                    {{- '[/INST]' }}\n                {%- endif %}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {%- if message['content'] is string %}\n            {{- message['content'] + eos_token }}\n        {%- else %}\n            {{- message['content'][0]['text'] + eos_token }}\n        {%- endif %}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
            "patch_size": 128,
        }  # fmt: skip

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 4
        image_token_index = 10

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
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
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 36]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 4701, 1307, 1278, 3937, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=self.url_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 36]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 4701, 1307, 1278, 3937, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing inputs as a single list
        inputs_image = processor(text=prompt_string, images=[self.image_0], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 36]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 4701, 1307, 1278, 3937, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test as nested single list
        inputs_image = processor(text=prompt_string, images=[[self.image_0]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 24, 36]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 4701, 1307, 1278, 3937, 1063, 1349, 4290, 16002, 41150, 1058]
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
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 24, 36]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
                    )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=[self.url_0, self.url_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 24, 36]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in as a nested list
        inputs_url = processor(text=prompt_string, images=[[self.image_0, self.image_1]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 24, 36]))

        # fmt: off
        self.assertEqual(
            inputs_url["input_ids"][0].tolist(),
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
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
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 36, 36]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 2)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 36, 36]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
             [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing as a single flat list
        inputs_image = processor(
            text=prompt_string, images=[self.image_0, self.image_1, self.image_2], return_tensors="pt", padding=True
        )
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 36, 36]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [1, 21510, 1058, 1032, 10, 10, 10, 12, 10, 10, 10, 13, 10, 10, 10, 12, 10, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
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

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modality="image")
        image_input = self.prepare_image_inputs(batch_size=2)

        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=3,
            )
