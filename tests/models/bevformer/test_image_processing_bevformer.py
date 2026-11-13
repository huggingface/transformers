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
"""Testing suite for the BEVFormer image processor."""

import unittest

import numpy as np
import torch

from transformers import BevformerImageProcessor


class BevformerImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self.image_processor = BevformerImageProcessor()

    def test_image_processor_output_shapes_and_lidar2img(self):
        num_cameras = 6
        height, width = 900, 1600
        rng = np.random.default_rng(42)
        images = (rng.random((num_cameras, height, width, 3)) * 255.0).astype(np.float32)
        lidar2img = (rng.random((num_cameras, 4, 4)) * 10.0).astype(np.float32)

        encoded = self.image_processor(images=images, lidar2img=lidar2img, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        scaled_lidar2img = encoded["lidar2img"]
        image_shapes = encoded["image_shapes"]

        # Expected output shape: (1, 6, 3, 480, 800)
        self.assertEqual(pixel_values.shape, (1, num_cameras, 3, 480, 800))
        self.assertEqual(scaled_lidar2img.shape, (1, num_cameras, 4, 4))
        self.assertEqual(image_shapes.shape, (1, num_cameras, 2))
        self.assertEqual(image_shapes[0, 0, 0].item(), 450)
        self.assertEqual(image_shapes[0, 0, 1].item(), 800)

        # lidar2img scale factor is 0.5 on row 0 and 1
        self.assertTrue(torch.allclose(scaled_lidar2img[0, :, 0, :], 0.5 * torch.from_numpy(lidar2img[:, 0, :])))
        self.assertTrue(torch.allclose(scaled_lidar2img[0, :, 1, :], 0.5 * torch.from_numpy(lidar2img[:, 1, :])))
        self.assertTrue(torch.allclose(scaled_lidar2img[0, :, 2, :], torch.from_numpy(lidar2img[:, 2, :])))
        self.assertTrue(torch.allclose(scaled_lidar2img[0, :, 3, :], torch.from_numpy(lidar2img[:, 3, :])))

    def test_image_processor_without_lidar2img(self):
        num_cameras = 6
        images = (np.random.rand(num_cameras, 900, 1600, 3) * 255.0).astype(np.float32)
        encoded = self.image_processor(images=images, return_tensors="pt")
        self.assertIn("pixel_values", encoded)
        self.assertIn("image_shapes", encoded)
        self.assertNotIn("lidar2img", encoded)
