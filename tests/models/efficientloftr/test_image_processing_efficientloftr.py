# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import time
import unittest

import numpy as np

from tests.models.superglue.test_image_processing_superglue import (
    SuperGlueImageProcessingTest,
    SuperGlueImageProcessingTester,
)
from transformers.testing_utils import (
    require_torch,
    require_vision,
)
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available


if is_torch_available():
    import torch

    from transformers.models.efficientloftr.modeling_efficientloftr import KeypointMatchingOutput

if is_vision_available():
    from transformers import EfficientLoFTRImageProcessor

    if is_torchvision_available():
        from transformers import EfficientLoFTRImageProcessorFast


def random_array(size):
    return np.random.randint(255, size=size)


def random_tensor(size):
    return torch.rand(size)


class EfficientLoFTRImageProcessingTester(SuperGlueImageProcessingTester):
    """Tester for EfficientLoFTRImageProcessor"""

    def __init__(
        self,
        parent,
        batch_size=6,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_grayscale=True,
    ):
        super().__init__(
            parent, batch_size, num_channels, image_size, min_resolution, max_resolution, do_resize, size, do_grayscale
        )

    def prepare_keypoint_matching_output(self, pixel_values):
        """Prepare a fake output for the keypoint matching model with random matches between 50 keypoints per image."""
        max_number_keypoints = 50
        batch_size = len(pixel_values)
        keypoints = torch.zeros((batch_size, 2, max_number_keypoints, 2))
        matches = torch.full((batch_size, 2, max_number_keypoints), -1, dtype=torch.int)
        scores = torch.zeros((batch_size, 2, max_number_keypoints))
        for i in range(batch_size):
            random_number_keypoints0 = np.random.randint(10, max_number_keypoints)
            random_number_keypoints1 = np.random.randint(10, max_number_keypoints)
            random_number_matches = np.random.randint(5, min(random_number_keypoints0, random_number_keypoints1))
            keypoints[i, 0, :random_number_keypoints0] = torch.rand((random_number_keypoints0, 2))
            keypoints[i, 1, :random_number_keypoints1] = torch.rand((random_number_keypoints1, 2))
            random_matches_indices0 = torch.randperm(random_number_keypoints1, dtype=torch.int)[:random_number_matches]
            random_matches_indices1 = torch.randperm(random_number_keypoints0, dtype=torch.int)[:random_number_matches]
            matches[i, 0, random_matches_indices1] = random_matches_indices0
            matches[i, 1, random_matches_indices0] = random_matches_indices1
            scores[i, 0, random_matches_indices1] = torch.rand((random_number_matches,))
            scores[i, 1, random_matches_indices0] = torch.rand((random_number_matches,))
        return KeypointMatchingOutput(keypoints=keypoints, matches=matches, matching_scores=scores)


@require_torch
@require_vision
class EfficientLoFTRImageProcessingTest(SuperGlueImageProcessingTest, unittest.TestCase):
    image_processing_class = EfficientLoFTRImageProcessor if is_vision_available() else None
    fast_image_processing_class = EfficientLoFTRImageProcessorFast if is_torchvision_available() else None

    def setUp(self) -> None:
        super().setUp()
        self.image_processor_tester = EfficientLoFTRImageProcessingTester(self)

    @unittest.skip(reason="Many failing cases. This test needs a more deep investigation.")
    def test_fast_is_faster_than_slow(self):
        """Override the generic test since EfficientLoFTR requires image pairs."""
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast speed test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast speed test as one of the image processors is not defined")

        # Create image pairs for speed test
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=False)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        # Time slow processor
        start_time = time.time()
        for _ in range(10):
            _ = image_processor_slow(dummy_images, return_tensors="pt")
        slow_time = time.time() - start_time

        # Time fast processor
        start_time = time.time()
        for _ in range(10):
            _ = image_processor_fast(dummy_images, return_tensors="pt")
        fast_time = time.time() - start_time

        # Fast should be faster (or at least not significantly slower)
        self.assertLessEqual(
            fast_time, slow_time * 1.2, "Fast processor should not be significantly slower than slow processor"
        )
