# Copyright 2026 Cohere Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from transformers import (
    AutoTokenizer,
    CohereCompassImageProcessor,
    CohereCompassProcessor,
    CohereCompassVideoProcessor,
)
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_vision
class CohereCompassProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CohereCompassProcessor
    videos_unstructured_max_length = 870
    videos_text_kwargs_max_length = 870
    videos_text_kwargs_override_max_length = 870
    model_id = "CohereLabs/North-Micro-Vision-Instruct"

    @classmethod
    def _setup_image_processor(cls):
        return CohereCompassImageProcessor(
            min_pixels=56 * 56,
            max_pixels=56 * 56,
            patch_size=16,
        )

    @classmethod
    def _setup_video_processor(cls):
        return CohereCompassVideoProcessor(patch_size=16)

    @classmethod
    def _setup_tokenizer(cls):
        # For some reason the tokenizer has saved image processing fields, unset it all!
        tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        return tokenizer

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 640},
            {"num_frames": None, "fps": 2, "expected_dim": 0, "output_length": 640},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 1512},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 1512},
            {"expected_dim": 0, "output_length": 640},
        ]

    def _image(self, height=56, width=56):
        from PIL import Image

        return Image.fromarray(np.full((height, width, 3), 127, dtype=np.uint8))

    def prepare_images_inputs(self, batch_size=None, nested=False):
        if batch_size is None:
            return self._image(64, 64)
        images = [self._image(64, 64) for _ in range(batch_size)]
        return [[image] for image in images] if nested else images

    def prepare_videos_inputs(self, batch_size=None):
        video = np.random.randint(255, size=(8, 3, 64, 64), dtype=np.uint8)
        return video if batch_size is None else [video] * batch_size

    def test_image_placeholder_expansion(self):
        processor = self.get_processor()
        output = processor(
            images=self._image(),
            text="<|VISION_START|><|IMAGE_PAD|><|VISION_END|> describe this image",
            return_tensors="pt",
        )
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2]])
        self.assertEqual((output.input_ids == processor.image_token_id).sum().item(), 1)
        self.assertTrue(output.mm_token_type_ids.equal((output.input_ids == processor.image_token_id).int()))

    def test_multiple_images_preserve_grid_order(self):
        processor = self.get_processor()
        output = processor(
            images=[self._image(56, 56), self._image(56, 112)],
            text=(
                "<|VISION_START|><|IMAGE_PAD|><|VISION_END|> "
                "<|VISION_START|><|IMAGE_PAD|><|VISION_END|> describe this image"
            ),
            return_tensors="pt",
        )
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2], [1, 2, 4]])
        self.assertEqual((output.input_ids == processor.image_token_id).sum().item(), 3)

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(56, 56), (56, 112)])
        self.assertEqual(output["num_image_patches"], [4, 8])
        self.assertEqual(output["num_image_tokens"], [1, 2])
