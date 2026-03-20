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

import torch

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


class UVDocImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=False,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize

    def prepare_image_processor_dict(self):
        return {
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class UVDocImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.image_processor_tester = UVDocImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip("UVDoc image processors doesn't support 4 channel images")
    def test_call_numpy_4_channels(self):
        pass

    def test_post_process_document_rectification(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)

        batch_size = 2
        height, width = 32, 48
        pred_height, pred_width = 16, 24

        # Create identity grid in normalized coords [-1, 1] for grid_sample
        y_coords = torch.linspace(-1, 1, pred_height)
        x_coords = torch.linspace(-1, 1, pred_width)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        prediction = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Original images as list of tensors (C, H, W) each
        original_images = [torch.rand(3, height, width) for _ in range(batch_size)]

        results = image_processor.post_process_document_rectification(prediction, original_images, scale=255.0)

        self.assertEqual(len(results), batch_size)
        for i, result in enumerate(results):
            self.assertIn("images", result)
            images = result["images"]
            self.assertEqual(images.shape, (height, width, 3))
            self.assertEqual(images.dtype, torch.uint8)
            self.assertTrue(torch.all(images >= 0) and torch.all(images <= 255))

        # Test with custom scale
        results_custom_scale = image_processor.post_process_document_rectification(
            prediction, original_images, scale=1.0
        )
        for result in results_custom_scale:
            self.assertTrue(torch.all(result["images"] <= 1))

    def test_post_process_document_rectification_different_sizes(self):
        """Test post-processing with original images of different sizes (list of tensors)."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)

        # Create predictions for 2 images (model output size is fixed)
        pred_height, pred_width = 16, 24
        y_coords = torch.linspace(-1, 1, pred_height)
        x_coords = torch.linspace(-1, 1, pred_width)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        prediction = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(2, -1, -1, -1)

        # Original images with different sizes: (32, 48) and (64, 96)
        original_images = [
            torch.rand(3, 32, 48),
            torch.rand(3, 64, 96),
        ]

        results = image_processor.post_process_document_rectification(prediction, original_images, scale=255.0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["images"].shape, (32, 48, 3))
        self.assertEqual(results[1]["images"].shape, (64, 96, 3))
        for result in results:
            self.assertEqual(result["images"].dtype, torch.uint8)
            self.assertTrue(torch.all(result["images"] >= 0) and torch.all(result["images"] <= 255))
