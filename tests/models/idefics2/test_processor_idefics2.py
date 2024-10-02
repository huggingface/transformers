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
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class Idefics2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b", image_seq_len=2)
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
        self.bos_token = self.processor.tokenizer.bos_token
        self.image_token = self.processor.image_token.content
        self.fake_image_token = self.processor.fake_image_token.content

        self.bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.image_token)
        self.fake_image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.fake_image_token)
        self.image_seq_len = self.processor.image_seq_len

    def test_process_interleaved_images_prompts_no_image_splitting(self):
        old_image_splitting = self.processor.image_processor.do_image_splitting

        self.processor.image_processor.do_image_splitting = False

        # Test that a single image is processed correctly
        inputs = self.processor(images=self.image1)
        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 3, 653, 980))
        self.assertEqual(inputs["pixel_attention_mask"].shape, (1, 1, 653, 980))
        # fmt: on

        # Test a single sample with image and text
        image_str = "<image>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = self.processor(text=text, images=self.image1)

        # fmt: off
        tokenized_sentence = self.processor.tokenizer(text_str, add_special_tokens=False)
        expected_input_ids = [[self.bos_token_id] + [self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len + [self.fake_image_token_id] + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 3, 653, 980))
        self.assertEqual(inputs["pixel_attention_mask"].shape, (1, 1, 653, 980))
        # fmt: on

        # Test that batch is correctly processed
        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "bla, bla"

        text = [
            image_str + text_str_1,
            text_str_2 + image_str + image_str,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = self.processor(text=text, images=images, padding=True)

        # fmt: off
        tokenized_sentence_1 = self.processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = self.processor.tokenizer(text_str_2, add_special_tokens=False)
        expected_input_ids_1 = [self.bos_token_id] + [self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len + [self.fake_image_token_id] + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = [self.bos_token_id] + tokenized_sentence_2["input_ids"] + [self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len + [self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len + [self.fake_image_token_id]
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

        self.processor.image_processor.do_image_splitting = old_image_splitting

    def test_process_interleaved_images_prompts_image_splitting(self):
        old_image_splitting = self.processor.image_processor.do_image_splitting

        self.processor.image_processor.do_image_splitting = True

        # Test that a single image is processed correctly
        inputs = self.processor(images=self.image1)
        self.assertEqual(inputs["pixel_values"].shape, (1, 5, 3, 653, 980))
        self.assertEqual(inputs["pixel_attention_mask"].shape, (1, 5, 653, 980))
        # fmt: on

        # Test a single sample with image and text
        image_str = "<image>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = self.processor(text=text, images=self.image1)

        # fmt: off
        tokenized_sentence = self.processor.tokenizer(text_str, add_special_tokens=False)
        expected_input_ids = [[self.bos_token_id] + ([self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len) * 5 + [self.fake_image_token_id] + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(inputs["pixel_values"].shape, (1, 5, 3, 653, 980))
        self.assertEqual(inputs["pixel_attention_mask"].shape, (1, 5, 653, 980))
        # fmt: on

        # Test that batch is correctly processed
        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "bla, bla"

        text = [
            image_str + text_str_1,
            text_str_2 + image_str + image_str,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = self.processor(text=text, images=images, padding=True)

        # fmt: off
        tokenized_sentence_1 = self.processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = self.processor.tokenizer(text_str_2, add_special_tokens=False)
        expected_input_ids_1 = [self.bos_token_id] + ([self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len) * 5 + [self.fake_image_token_id] + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = [self.bos_token_id] + tokenized_sentence_2["input_ids"] + ([self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len) * 5 + ([self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len) * 5 + [self.fake_image_token_id]
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
        self.assertEqual(inputs['pixel_values'].shape, (2, 10, 3, 767, 980))
        self.assertEqual(inputs['pixel_attention_mask'].shape, (2, 10, 767, 980))
        # fmt: on

        self.processor.image_processor.do_image_splitting = old_image_splitting

    def test_add_special_tokens_processor(self):
        image_str = "<image>"
        text_str = "In this image, we see"
        text = text_str + image_str

        n_image_repeat = 5 if self.processor.image_processor.do_image_splitting else 1

        # fmt: off
        inputs = self.processor(text=text, images=self.image1, add_special_tokens=False)
        tokenized_sentence = self.processor.tokenizer(text_str, add_special_tokens=False)
        expected_input_ids = [tokenized_sentence["input_ids"] + ([self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len) * n_image_repeat + [self.fake_image_token_id]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)

        inputs = self.processor(text=text, images=self.image1)
        expected_input_ids = [[self.bos_token_id] + tokenized_sentence["input_ids"] + ([self.fake_image_token_id] + [self.image_token_id] * self.image_seq_len) * n_image_repeat + [self.fake_image_token_id]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        # fmt: on

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do these images show?"},
                    {"type": "image"},
                    {"type": "image"},
                    "What do these images show?",
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "And who is that?"}]},
        ]

        processor = self.processor
        # Make short sequence length to test that the fake tokens are added correctly
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True)

        expected_rendered = (
            "User: What do these images show?<image><image><end_of_utterance>\n"
            "Assistant: The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>\n"
            "User: And who is that?<end_of_utterance>\n"
            "Assistant:"
        )
        self.assertEqual(rendered, expected_rendered)
