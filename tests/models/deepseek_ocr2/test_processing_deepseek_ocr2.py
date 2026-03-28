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
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained("thisisiron/DeepSeek-OCR-2-hf")
        return tokenizer

    @unittest.skip("DeepseekOcr2Processor pops the image processor output 'num_local_patches'")
    def test_image_processor_defaults(self):
        pass

    def test_get_num_multimodal_tokens(self):
        """Verify _get_num_multimodal_tokens computes correct token counts.

        Formula: global_tokens + local_tokens * num_crops + 1 (separator)
        - global_tokens = ceil(1024 / 16 / 4)^2 = 256
        - local_tokens  = ceil(768 / 16 / 4)^2  = 144
        """
        processor = self.get_processor()

        # No local patches: 256 + 0 + 1 = 257
        self.assertEqual(processor._get_num_multimodal_tokens(0), 257)

        # 2 crops: 256 + 144*2 + 1 = 545
        self.assertEqual(processor._get_num_multimodal_tokens(2), 545)

        # 6 crops: 256 + 144*6 + 1 = 1121
        self.assertEqual(processor._get_num_multimodal_tokens(6), 1121)

    def test_image_token_expansion_small_image(self):
        """Small image (< tile_size) should produce no local patches → 257 image tokens."""
        processor = self.get_processor()

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

        # Large image: max(2448, 3264) > 768 → local patches
        image = torch.randint(0, 256, (3, 3264, 2448), dtype=torch.uint8)
        prompt = "<image>\nFree OCR."

        inputs = processor(images=image, text=prompt, return_tensors="pt")

        image_token_id = processor.image_token_id
        num_image_tokens = (inputs["input_ids"] == image_token_id).sum().item()
        num_local_patches = inputs["num_local_patches"][0]

        # Token count must match formula
        expected = processor._get_num_multimodal_tokens(num_local_patches)
        self.assertEqual(num_image_tokens, expected)
        self.assertGreater(num_local_patches, 0)
        self.assertIn("pixel_values_local", inputs)
