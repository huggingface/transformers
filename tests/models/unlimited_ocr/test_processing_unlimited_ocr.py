# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import UnlimitedOcrProcessor


@require_vision
class UnlimitedOcrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = UnlimitedOcrProcessor
    # TODO: Change before merge
    model_id = "guarin/Unlimited-OCR"

    # Defaults from mixin are too small as a single image expands to 273 image tokens
    # for this checkpoint (size=1024)
    image_text_kwargs_max_length = 320
    image_text_kwargs_override_max_length = 310
    image_unstructured_max_length = 300

    def get_messages(self, image):
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "document parsing."}],
            }
        ]

    def test_replace_image_tokens(self):
        processor = self.get_processor()

        image = torch.randint(0, 256, (3, 200, 300), dtype=torch.uint8)
        messages = self.get_messages(image)

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        # image resized to 1024, followed by patch size 16 and 4x downsampling = 16 x 16 patches
        # 273 = 16 rows * (16 cols + 1 newline) + 1 view separator
        self.assertEqual(num_image_tokens, 273)
        self.assertNotIn("pixel_values_local", inputs)

    def test_replace_image_tokens_with_local(self):
        processor = self.get_processor()

        image = torch.randint(0, 256, (3, 500, 700), dtype=torch.uint8)
        messages = self.get_messages(image)

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        # global is same as in test above
        # 500 x 700 image is split into 3x4 tiles
        # local tiles are 640, followed by patch size 16 and 4x downsample = 10 x 10 patches
        # 1503 = 273 global + (3 rows * 10) * (4 cols * 10 + 1)) local
        self.assertEqual(num_image_tokens, 1503)
        self.assertIn("pixel_values_local", inputs)

    def test_replace_image_tokens_no_crop(self):
        processor = self.get_processor()

        image = torch.randint(0, 256, (3, 500, 700), dtype=torch.uint8)
        messages = self.get_messages(image)

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"crop_to_patches": False},
        )
        num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        # same as in test_replace_image_tokens
        self.assertEqual(num_image_tokens, 273)
        self.assertNotIn("pixel_values_local", inputs)
