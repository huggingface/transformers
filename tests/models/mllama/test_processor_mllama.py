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

from transformers import AutoImageProcessor, AutoTokenizer, MllamaProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class MllamaProcessorTest(unittest.TestCase):
    def setUp(self):
        # self.processor = MllamaProcessor.from_pretrained("HuggingFaceM4/Mllama-8b")
        tokenizer = AutoTokenizer.from_pretrained("mllama")
        image_processor = AutoImageProcessor.from_pretrained("mllama")
        self.processor = MllamaProcessor(tokenizer=tokenizer, image_processor=image_processor)
        self.image1 = Image.new("RGB", (224, 220))
        self.image2 = Image.new("RGB", (512, 128))
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def test_process_interleaved_images_prompts_image_splitting(self):
        # Test that a single image is processed correctly
        inputs = self.processor(images=self.image2, max_image_tiles=2, size={"width": 224, "height": 224})
        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 2, 3, 224, 224))
        self.assertEqual(inputs["aspect_ratios"].shape, (1, 1, 2))
        self.assertEqual(inputs["aspect_ratios"].squeeze().tolist(), [2, 1])
        self.assertEqual(inputs["num_tiles"], [[2]])

        # Test that text is processed correctly
        text = "<|begin_of_text|>This is a test sentence.<|end_of_text|>"
        inputs = self.processor(text=text)
        expected_ids = [128000, 2028, 374, 264, 1296, 11914, 13, 128001]
        self.assertEqual(inputs["input_ids"][0], expected_ids)
        self.assertEqual(inputs["attention_mask"][0], [1] * len(expected_ids))
        self.assertEqual(inputs["cross_attention_token_mask"], None)

        # Test a single sample with image and text
        image_str = "<|image|>"
        text_str = "This is a test sentence."
        text = image_str + text_str
        inputs = self.processor(
            text=text,
            images=self.image1,
            max_image_tiles=4,
            size={"width": 128, "height": 128},
        )
        expected_ids = [self.image_token_id] + [2028, 374, 264, 1296, 11914, 13]

        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 4, 3, 128, 128))
        self.assertEqual(inputs["aspect_ratios"].shape, (1, 1, 2))
        self.assertEqual(inputs["aspect_ratios"].squeeze().tolist(), [2, 2])
        self.assertEqual(inputs["num_tiles"], [[4]])

        self.assertEqual(inputs["input_ids"][0], expected_ids)
        self.assertEqual(inputs["attention_mask"][0], [1] * len(expected_ids))

        # TODO: len(expected_ids) or -1 ?
        self.assertEqual(inputs["vision_mask"], [[[0, len(expected_ids)]]])

        # Test batch
        text = [
            "<|image|>" + "This is a test sentence.",
            "This is a test sentence.<|image|><|image|>This is a test sentence.",
        ]
        # fmt: off
        expected_ids = [
            [self.image_token_id, 2028, 374, 264, 1296, 11914, 13],
            [2028, 374, 264, 1296, 11914, 13, self.image_token_id, self.image_token_id, 2028, 374, 264, 1296, 11914, 13],
        ]
        # fmt: onn
        images = [[self.image1], [self.image1, self.image2]]
        inputs = self.processor(text=text, images=images, padding=True, max_image_tiles=4, size={"width": 256, "height": 256})

        self.assertEqual(inputs["pixel_values"].shape, (2, 2, 4, 3, 256, 256))
        self.assertEqual(inputs["aspect_ratios"].shape, (2, 2, 2))
        self.assertEqual(inputs["aspect_ratios"].squeeze().tolist(), [[[1, 1], [1, 1]], [[1, 1], [2, 1]]])
        self.assertEqual(inputs["num_tiles"], [[1], [1, 2]])

        for input_ids_i, attention_mask_i, expected_ids_i in zip(inputs["input_ids"], inputs["attention_mask"], expected_ids):
            pad_ids = [id for id, m in zip(input_ids_i, attention_mask_i) if m == 0]
            input_ids = [id for id, m in zip(input_ids_i, attention_mask_i) if m == 1]
            self.assertEqual(input_ids, expected_ids_i)
            self.assertEqual(pad_ids, [self.pad_token_id] * len(pad_ids))

        # TODO: len(expected_ids) or -1 ?
        self.assertEqual(inputs["vision_mask"], [[[0, len(expected_ids[0])]], [[6, len(expected_ids[1])], [7, len(expected_ids[1])]]])

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = messages = [
            {"role": "user", "content": "<|image|><|image|>What do these images show?"},
            {
                "role": "assistant",
                "content": "The first image shows the statue of Liberty in New York.",
            },
            {"role": "user", "content": "And who is that?"},
        ]

        # TODO: make sure processor also work with this chat template, not ony tokenizer
        rendered = self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        expected_rendered = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "<|image|><|image|>What do these images show?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "The first image shows the statue of Liberty in New York.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "And who is that?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        self.assertEqual(rendered, expected_rendered)
