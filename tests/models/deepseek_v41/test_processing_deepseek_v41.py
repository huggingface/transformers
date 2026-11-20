# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoTokenizer, DeepseekV41Processor, is_torch_available
from transformers.testing_utils import require_torch, require_torchvision, require_vision

from ...test_processing_common import ProcessorTesterMixin
from .test_modeling_deepseek_v41 import DeepseekV41ModelTester


if is_torch_available():
    import torch

    from transformers import DeepseekV41Model


@require_torch
@require_vision
@require_torchvision
class DeepseekV41ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekV41Processor

    @classmethod
    def _setup_image_processor(cls):
        # Small limits, so that an image only takes a few positions in the prompt: 2 x 2 patches, merged 3 x 3
        # into a token, and at most 16 positions per image
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(patch_size=2, downsample_ratio=3, max_image_tokens=16, min_pixels=64)

    @classmethod
    def _setup_tokenizer(cls):
        # The tokenizer of the checkpoint, which has the image placeholder and the chat template
        return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")

    @staticmethod
    def _image(height, width):
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    def test_image_placeholder_expansion(self):
        """Each placeholder is repeated once per position of its image: `rows * (columns + 1) + 2`."""
        processor = self.get_processor()
        # 6 x 9 patches make 2 x 3 tokens, and 3 x 5 patches round up to 1 x 2 tokens
        images = [self._image(12, 18), self._image(5, 7)]
        text = f"First {processor.image_token} then {processor.image_token}."
        inputs = processor(text=text, images=images, return_tensors="pt")

        self.assertEqual(inputs["image_grid_thw"].tolist(), [[1, 6, 9], [1, 3, 5]])
        self.assertEqual(inputs["pixel_values"].shape[0], 6 * 9 + 3 * 5)
        input_ids = inputs["input_ids"][0].tolist()
        self.assertEqual(input_ids.count(processor.image_token_id), (2 * (3 + 1) + 2) + (1 * (2 + 1) + 2))

        # The original implementation tokenizes the prompt first and expands the ids afterwards
        expected_ids, num_positions = [], iter([10, 5])
        for token_id in processor.tokenizer(text)["input_ids"]:
            expected_ids += [token_id] * (next(num_positions) if token_id == processor.image_token_id else 1)
        self.assertEqual(input_ids, expected_ids)

    def test_chat_template_with_images(self):
        """The chat template of the checkpoint writes one placeholder per image block."""
        processor = self.get_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._image(12, 18)},
                    {"type": "text", "text": "What is in this picture?"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(prompt.count(processor.image_token), 1)

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        self.assertEqual(inputs["image_grid_thw"].tolist(), [[1, 6, 9]])
        self.assertEqual(int((inputs["input_ids"] == processor.image_token_id).sum()), 10)

    def test_image_positions_match_the_model(self):
        """The model lays each image out on exactly as many positions as the processor made room for."""
        processor = self.get_processor()
        # The tiny vision config has the patch size and the downsample ratio of the image processor above
        model = DeepseekV41Model(DeepseekV41ModelTester(self).get_config()).eval()
        for height, width in [(12, 18), (5, 7), (4, 4), (60, 10), (10, 60), (100, 100)]:
            with self.subTest(height=height, width=width):
                inputs = processor(text=processor.image_token, images=self._image(height, width), return_tensors="pt")
                with torch.no_grad():
                    image_features = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
                num_positions = int((inputs["input_ids"] == processor.image_token_id).sum())
                self.assertEqual(image_features.pooler_output[0].shape[0], num_positions)
