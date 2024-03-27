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
from io import BytesIO

import requests

from transformers import Idefics2Processor
from transformers.models.idefics2.processing_idefics2 import build_string_from_input
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    pass

if is_vision_available():
    from PIL import Image


class Idefics2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = Idefics2Processor.from_pretrained("amyeroberts/idefics2")
        self.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        self.image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        self.image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )

    def test_build_string_from_input(self):
        prompt = ["Initial str", self.image1, self.image2, "mid str", self.image3]
        prompt_string = build_string_from_input(
            prompt=prompt, image_seq_len=2, bos_token="<s>", image_token="<im>", fake_image_token="<fake>"
        )
        expected_string = "<s>Initial str<fake><im><im><fake><im><im><fake>mid str<fake><im><im><fake>"
        self.assertEqual(prompt_string, expected_string)

        prompt = [self.image1, self.image3]
        prompt_string = build_string_from_input(
            prompt=prompt, image_seq_len=2, bos_token="<s>", image_token="<im>", fake_image_token="<fake>"
        )
        expected_string = "<s><fake><im><im><fake><im><im><fake>"

        prompt = ["Initial str"]
        prompt_string = build_string_from_input(
            prompt=prompt, image_seq_len=2, bos_token="<s>", image_token="<im>", fake_image_token="<fake>"
        )
        expected_string = "<s>Initial str"

    def test_process_interleaved_images_prompts(self):
        # Test that a single image is processed correctly
        prompt = self.image1
        inputs = self.processor(prompt)

        bos_token = self.processor.tokenizer.bos_token
        image_token = self.processor.image_token
        fake_image_token = self.processor.fake_image_token

        bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(bos_token)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        fake_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(fake_image_token)

        image_seq_len = self.processor.image_seq_len

        # fmt: off
        expected_input_ids = [[bos_token_id] + [fake_image_token_id] + [image_token_id] * image_seq_len + [fake_image_token_id]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 3, 653, 980))
        self.assertEqual(inputs["pixel_attention_mask"].shape, (1, 1, 653, 980))
        # fmt: on

        # Test a single sample with image and text
        prompt = [self.image1, "In this image, we see"]
        inputs = self.processor(prompt)

        # fmt: off
        tokenized_sentence = self.processor.tokenizer("In this image, we see", add_special_tokens=False)
        expected_input_ids = [[bos_token_id] + [fake_image_token_id] + [image_token_id] * image_seq_len + [fake_image_token_id] + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 3, 653, 980))
        self.assertEqual(inputs["pixel_attention_mask"].shape, (1, 1, 653, 980))
        # fmt: on

        # Test that batch is correctly processed
        prompts = [
            [self.image1, "In this image, we see"],
            ["bla, bla", self.image2, self.image3],
        ]
        inputs = self.processor(prompts, padding=True)

        # fmt: off
        tokenized_sentence_1 = self.processor.tokenizer("In this image, we see", add_special_tokens=False)
        tokenized_sentence_2 = self.processor.tokenizer("bla, bla", add_special_tokens=False)
        expected_input_ids_1 = [bos_token_id] + [fake_image_token_id] + [image_token_id] * image_seq_len + [fake_image_token_id] + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = [bos_token_id] + tokenized_sentence_2["input_ids"] + [fake_image_token_id] + [image_token_id] * image_seq_len + [fake_image_token_id] + [image_token_id] * image_seq_len + [fake_image_token_id]
        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)
        padded_expected_input_ids_1 = [0] * pad_len + expected_input_ids_1

        self.assertEqual(
            inputs["input_ids"], [padded_expected_input_ids_1, expected_input_ids_2]
        )
        self.assertEqual(
            inputs["attention_mask"],
            [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * len(expected_input_ids_2)]
        )
        self.assertEqual(inputs['pixel_values'].shape, (2, 2, 3, 767, 980))
        self.assertEqual(inputs['pixel_attention_mask'].shape, (2, 2, 767, 980))
        # fmt: on

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    "What do these images show?",
                    self.image1,
                    "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                ],
            },
            {
                "role": "assistant",
                "content": "The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.",
            },
            {"role": "user", "content": ["And who is that?"]},
        ]

        processor = self.processor
        old_seq_len = processor.image_seq_len
        # Make short sequence length to test that the fake tokens are added correctly
        processor.image_seq_len = 2
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        expected_rendered = (
            "User:What do these images show?<fake_token_around_image><image><image><fake_token_around_image><image><image><fake_token_around_image><end_of_utterance>\n"
            "Assistant:The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>\n"
            "User:And who is that?<end_of_utterance>\n"
            "Assistant:\n"
        )

        self.assertEqual(rendered, expected_rendered)
        # Set back to prevent tests from being stateful
        processor.image_seq_len = old_seq_len
