# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch

from transformers import DeepseekOcr2Processor
from transformers.testing_utils import require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_vision
class DeepseekOcr2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekOcr2Processor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor = image_processor_class()
        image_processor.size = {"height": 64, "width": 64}
        image_processor.tile_size = 512
        return image_processor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained("thisisiron/DeepSeek-OCR-2-hf")
        return tokenizer

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @unittest.skip("DeepseekOcr2Processor pops the image processor output 'num_local_patches'")
    def test_image_processor_defaults(self):
        pass

    def test_image_token_expansion_small_image(self):
        """Small image (< tile_size) should produce no local patches → 257 image tokens."""
        processor = self.get_processor()
        processor.image_processor.size = {"height": 1024, "width": 1024}
        processor.image_processor.tile_size = 768

        # Small image: max(200, 300) < 768 → no local patches
        image = torch.randint(0, 256, (3, 300, 200), dtype=torch.uint8)
        prompt = "<image>\nFree OCR."

        inputs = processor(images=image, text=prompt, return_tensors="pt")

        image_token_id = processor.image_token_id
        num_image_tokens = (inputs["input_ids"] == image_token_id).sum().item()

        # 257 = 256 global + 0 local + 1 separator
        self.assertEqual(num_image_tokens, 257)
        self.assertNotIn("pixel_values_local", inputs)

    def test_image_token_expansion_large_image(self):
        """Large image should produce local patches → more image tokens."""
        processor = self.get_processor()
        processor.image_processor.size = {"height": 1024, "width": 1024}
        processor.image_processor.tile_size = 768

        # Large image: max(2448, 3264) > 768 → local patches
        image = torch.randint(0, 256, (3, 3264, 2448), dtype=torch.uint8)
        prompt = "<image>\nFree OCR."

        inputs = processor(images=image, text=prompt, return_tensors="pt")

        image_token_id = processor.image_token_id
        num_image_tokens = (inputs["input_ids"] == image_token_id).sum().item()
        num_local_patches = inputs["num_local_patches"][0]

        # 3264x2448 image produces 6 local patches (2x3 grid) + 1 global view = 7 total
        # num_image_tokens = 256 global + 144*6 local + 1 separator = 1121
        self.assertEqual(num_local_patches, 6)
        self.assertEqual(num_image_tokens, 1121)
        self.assertIn("pixel_values_local", inputs)
