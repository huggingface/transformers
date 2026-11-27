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

import tempfile
import unittest

import numpy as np
import torch

from transformers import BevformerImageProcessor
from transformers.models.bevformer.modeling_bevformer import BevformerObjectDetectionOutput


class BevformerImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self.image_processor = BevformerImageProcessor()

    def test_image_processor_output_shapes_and_geometry(self):
        num_cameras = 6
        height, width = 900, 1600
        rng = np.random.default_rng(42)
        images = (rng.random((num_cameras, height, width, 3)) * 255.0).astype(np.float32)
        lidar2img = (rng.random((num_cameras, 4, 4)) * 10.0).astype(np.float32)
        can_bus = rng.standard_normal(18).astype(np.float32)

        encoded = self.image_processor(images=images, lidar2img=lidar2img, can_bus=can_bus, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        reference_points_cam = encoded["reference_points_cam"]
        bev_mask = encoded["bev_mask"]
        reference_points_2d = encoded["reference_points_2d"]

        bev_num_queries = self.image_processor.bev_height * self.image_processor.bev_width
        num_pillar_points = self.image_processor.num_points_in_pillar

        # Expected padded image shape: (1, 6, 3, 480, 800)
        self.assertEqual(pixel_values.shape, (1, num_cameras, 3, 480, 800))
        self.assertEqual(reference_points_cam.shape, (1, num_cameras, bev_num_queries, num_pillar_points, 2))
        self.assertEqual(bev_mask.shape, (1, num_cameras, bev_num_queries, num_pillar_points))
        self.assertEqual(bev_mask.dtype, torch.bool)
        self.assertEqual(reference_points_2d.shape, (1, 2, bev_num_queries, 1, 2))
        self.assertEqual(encoded["can_bus"].shape, (1, 18))
        # Non-zero ego motion produces shifted slot 0 != unshifted slot 1
        self.assertFalse(torch.allclose(reference_points_2d[:, 0], reference_points_2d[:, 1]))

    def test_image_processor_without_lidar2img_and_can_bus(self):
        num_cameras = 6
        images = (np.random.rand(num_cameras, 900, 1600, 3) * 255.0).astype(np.float32)
        encoded = self.image_processor(images=images, return_tensors="pt")
        self.assertIn("pixel_values", encoded)
        self.assertIn("reference_points_2d", encoded)
        self.assertNotIn("reference_points_cam", encoded)
        self.assertNotIn("bev_mask", encoded)
        self.assertNotIn("can_bus", encoded)
        # Zero shift when can_bus is omitted
        self.assertTrue(torch.allclose(encoded["reference_points_2d"][:, 0], encoded["reference_points_2d"][:, 1]))

    def test_post_process_denormalizes_point_cloud_range(self):
        # Normalized center at (0.5, 0.5, 0.5) should map to midpoint of [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] = (0, 0, -1)
        logits = torch.full((1, 2, 10), -10.0)
        logits[0, 0, 3] = 10.0  # high confidence class 3
        pred_boxes = torch.zeros((1, 2, 10))
        pred_boxes[0, 0, 0] = 0.5  # cx norm -> 0.0
        pred_boxes[0, 0, 1] = 0.5  # cy norm -> 0.0
        pred_boxes[0, 0, 4] = 0.5  # cz norm -> -1.0
        pred_boxes[0, 0, 5] = 2.0  # h -> z_bottom = -1.0 - 0.5 * 2.0 = -2.0
        pred_boxes[0, 0, 7] = 1.0  # cos_yaw = 1, sin_yaw = 0 -> yaw = 0

        outputs = BevformerObjectDetectionOutput(logits=logits, pred_boxes=pred_boxes)
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["boxes_3d"].shape, (1, 9))
        self.assertEqual(results[0]["labels"].tolist(), [3])
        self.assertTrue(torch.allclose(results[0]["boxes_3d"][0, :3], torch.tensor([0.0, 0.0, -2.0]), atol=1e-5))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.image_processor.save_pretrained(tmpdir)
            reloaded = BevformerImageProcessor.from_pretrained(tmpdir)
        self.assertEqual(reloaded.bev_height, self.image_processor.bev_height)
        self.assertEqual(reloaded.bev_width, self.image_processor.bev_width)
        self.assertEqual(reloaded.point_cloud_range, self.image_processor.point_cloud_range)
        self.assertEqual(reloaded.num_points_in_pillar, self.image_processor.num_points_in_pillar)
        self.assertEqual(reloaded.pad_size_divisor, self.image_processor.pad_size_divisor)
        self.assertFalse(reloaded.do_rescale)
