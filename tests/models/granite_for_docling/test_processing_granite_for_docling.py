# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import GraniteForDoclingImageProcessor, GraniteForDoclingProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_of_role|>' + message['role'] + '<|end_of_role|>' }}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for part in message['content'] %}"
    "{% if part['type'] == 'text' %}{{ part['text'] }}{% elif part['type'] == 'image' %}{{ '<image>' }}{% endif %}"
    "{% endfor %}{% endif %}"
    "{{ '<|end_of_text|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_of_role|>assistant<|end_of_role|>' }}{% endif %}"
)


@require_vision
class GraniteForDoclingProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GraniteForDoclingProcessor
    # Every tile expands to `image_seq_len + 2` tokens, so the defaults are too short for the mixin's images
    images_text_kwargs_max_length = 160
    images_text_kwargs_override_max_length = 150
    images_unstructured_max_length = 120

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        special_tokens = ["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>", "<global-img>"]
        special_tokens += [f"<row_{i + 1}_col_{j + 1}>" for i in range(16) for j in range(16)]
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def _setup_image_processor(cls):
        return GraniteForDoclingImageProcessor(size={"height": 16, "width": 16}, max_patches=4)

    @classmethod
    def _setup_test_attributes(cls, processor):
        super()._setup_test_attributes(processor)
        cls.image_token_id = processor.image_token_id
        cls.image_seq_len = processor.image_seq_len

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_len": 2, "chat_template": CHAT_TEMPLATE}

    def prepare_page_image(self):
        # A tall image is tiled on a 1x2 grid (1 column, 2 rows) plus the thumbnail
        return np.random.randint(0, 255, size=(64, 32, 3), dtype=np.uint8)

    def test_fake_image_token_id_after_adding_special_tokens(self):
        processor = self.get_processor()
        self.assertEqual(
            processor.fake_image_token_id, processor.tokenizer.convert_tokens_to_ids(processor.fake_image_token)
        )

    def expected_image_prompt(self, processor, num_rows, num_cols, fine_route=False):
        image_tokens = processor.image_token * (processor.image_seq_len * (4 if fine_route else 1))
        prompt = ""
        for row in range(num_rows):
            for col in range(num_cols):
                prompt += f"{processor.fake_image_token}<row_{row + 1}_col_{col + 1}>{image_tokens}"
            prompt += "\n"
        if num_rows * num_cols > 1:
            prompt += f"\n{processor.fake_image_token}<global-img>{image_tokens}{processor.fake_image_token}"
        return prompt

    @require_torch
    def test_image_token_expansion(self):
        processor = self.get_processor()
        text = f"{processor.image_token}<doclang>"
        inputs = processor(text=text, images=self.prepare_page_image(), return_tensors="pt")

        expected_text = self.expected_image_prompt(processor, num_rows=2, num_cols=1) + "<doclang>"
        expected_ids = processor.tokenizer(expected_text, return_tensors="pt")["input_ids"]
        self.assertEqual(inputs["input_ids"].tolist(), expected_ids.tolist())
        self.assertEqual((inputs["input_ids"] == self.image_token_id).sum().item(), 3 * self.image_seq_len)
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 3, 16, 16))
        self.assertNotIn("tile_fine_mask", inputs)

    @require_torch
    def test_fine_route(self):
        processor = self.get_processor()
        text = f"{processor.image_token}<doclang>"
        inputs = processor(text=text, images=self.prepare_page_image(), return_tensors="pt", fine_route=True)

        expected_text = self.expected_image_prompt(processor, num_rows=2, num_cols=1, fine_route=True) + "<doclang>"
        expected_ids = processor.tokenizer(expected_text, return_tensors="pt")["input_ids"]
        self.assertEqual(inputs["input_ids"].tolist(), expected_ids.tolist())
        self.assertEqual((inputs["input_ids"] == self.image_token_id).sum().item(), 12 * self.image_seq_len)
        self.assertEqual(inputs["tile_fine_mask"].tolist(), [[True, True, True]])

    @require_torch
    def test_chat_template_merges_loose_kwargs_with_processor_kwargs(self):
        # A loose kwarg such as `padding=True` must not discard `processor_kwargs`
        processor = self.get_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.prepare_page_image()},
                    {"type": "text", "text": "<doclang>"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"fine_route": True},
        )
        self.assertEqual(inputs["tile_fine_mask"].tolist(), [[True, True, True]])
        self.assertEqual((inputs["input_ids"] == self.image_token_id).sum().item(), 12 * self.image_seq_len)

    @require_torch
    def test_batched_images_are_padded(self):
        processor = self.get_processor()
        text = [f"{processor.image_token}Page one.", f"{processor.image_token}Page two."]
        images = [self.prepare_page_image(), np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)]
        inputs = processor(text=text, images=images, return_tensors="pt", padding=True, fine_route=True)

        # Three tiles for the tall page, a single tile (no thumbnail) for the square one
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 3, 16, 16))
        self.assertTrue(torch.all(inputs["pixel_values"][1, 1:] == 0))
        self.assertEqual(inputs["tile_fine_mask"].tolist(), [[True, True, True], [True, False, False]])
        image_tokens_per_sample = (inputs["input_ids"] == self.image_token_id).sum(dim=1).tolist()
        self.assertEqual(image_tokens_per_sample, [12 * self.image_seq_len, 4 * self.image_seq_len])

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()
        image_sizes = [(64, 32), (16, 16), (20, 300)]
        images = [np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8) for height, width in image_sizes]
        for fine_route in (False, True):
            num_tokens = processor._get_num_multimodal_tokens(image_sizes=image_sizes, fine_route=fine_route)
            for image, expected_num_tokens, expected_num_patches in zip(
                images, num_tokens["num_image_tokens"], num_tokens["num_image_patches"]
            ):
                inputs = processor(text=processor.image_token, images=image, fine_route=fine_route)
                self.assertEqual(
                    inputs["input_ids"][0].count(self.image_token_id),
                    expected_num_patches * processor.image_seq_len * (4 if fine_route else 1),
                )
                self.assertEqual(len(inputs["input_ids"][0]), expected_num_tokens)
