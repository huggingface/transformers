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

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

    from transformers.models.roma.modeling_roma import RomaKeypointMatchingOutput

if is_vision_available():
    from PIL import Image

    from transformers import RomaImageProcessor


@require_torch
@require_vision
class RomaImageProcessingTest(unittest.TestCase):
    def _make_pair(self):
        image0 = Image.fromarray((np.random.rand(120, 160, 3) * 255).astype("uint8"))
        image1 = Image.fromarray((np.random.rand(90, 100, 3) * 255).astype("uint8"))
        return [[image0, image1]]

    def test_default_properties(self):
        image_processor = RomaImageProcessor()
        self.assertEqual(image_processor.size, {"height": 560, "width": 560})
        self.assertTrue(image_processor.do_resize)
        self.assertTrue(image_processor.do_rescale)
        self.assertTrue(image_processor.do_normalize)
        self.assertFalse(image_processor.do_grayscale)

    def test_from_dict_with_kwargs(self):
        image_processor = RomaImageProcessor.from_dict(
            RomaImageProcessor().to_dict(), size={"height": 70, "width": 70}
        )
        self.assertEqual(image_processor.size, {"height": 70, "width": 70})

    def test_preprocess_pair_shape(self):
        image_processor = RomaImageProcessor()
        encoding = image_processor(self._make_pair(), return_tensors="pt")
        # One pair -> (1, 2, 3, 560, 560).
        self.assertEqual(tuple(encoding["pixel_values"].shape), (1, 2, 3, 560, 560))

    def test_preprocess_is_normalized_rgb(self):
        # RoMa keeps 3 channels (no grayscale) and ImageNet-normalizes, so values are not confined to [0, 1].
        image_processor = RomaImageProcessor()
        encoding = image_processor(self._make_pair(), return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        self.assertEqual(pixel_values.shape[2], 3)
        self.assertTrue((pixel_values < 0).any())

    def test_invalid_number_of_images_raises(self):
        image_processor = RomaImageProcessor()
        # Three loose images cannot be grouped into pairs.
        images = [Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8")) for _ in range(3)]
        with self.assertRaises(ValueError):
            image_processor(images, return_tensors="pt")

    def test_post_process_keypoint_matching(self):
        image_processor = RomaImageProcessor()
        num_samples = 25
        matches = torch.rand(1, num_samples, 4) * 2 - 1  # normalized [-1, 1]
        scores = torch.rand(1, num_samples)
        outputs = RomaKeypointMatchingOutput(matches=matches, matching_scores=scores)

        results = image_processor.post_process_keypoint_matching(
            outputs, target_sizes=[[(120, 160), (90, 100)]], threshold=0.0
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(set(results[0]), {"keypoints0", "keypoints1", "matching_scores"})
        keypoints0 = results[0]["keypoints0"]
        keypoints1 = results[0]["keypoints1"]
        self.assertEqual(keypoints0.shape, (num_samples, 2))
        # Keypoints must lie within the original image bounds (width, height).
        self.assertTrue((keypoints0[:, 0] <= 160).all() and (keypoints0[:, 1] <= 120).all())
        self.assertTrue((keypoints1[:, 0] <= 100).all() and (keypoints1[:, 1] <= 90).all())

    def test_post_process_threshold_filters(self):
        image_processor = RomaImageProcessor()
        matches = torch.rand(1, 30, 4) * 2 - 1
        scores = torch.linspace(0, 1, 30).unsqueeze(0)
        outputs = RomaKeypointMatchingOutput(matches=matches, matching_scores=scores)
        results = image_processor.post_process_keypoint_matching(
            outputs, target_sizes=[[(120, 160), (90, 100)]], threshold=0.5
        )
        self.assertTrue(results[0]["matching_scores"].numel() < 30)
        self.assertTrue((results[0]["matching_scores"] > 0.5).all())
