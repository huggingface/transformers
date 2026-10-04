# Copyright 2026 The HuggingFace Team. All rights reserved.

import unittest

import torch

from transformers import CogVLM2VideoProcessor
from transformers.testing_utils import require_torch, require_torchvision


@require_torch
@require_torchvision
class CogVLM2VideoProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = CogVLM2VideoProcessor()

    def test_video_preprocessing_shape_and_dtype(self):
        video = torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8)
        outputs = self.processor(video, return_tensors="pt")

        self.assertEqual(outputs.pixel_values_videos.shape, (2, 3, 224, 224))
        self.assertEqual(outputs.pixel_values_videos.dtype, torch.float32)
        self.assertEqual(outputs.video_frame_counts.tolist(), [2])
        self.assertTrue(torch.isfinite(outputs.pixel_values_videos).all())

    def test_multiple_videos_preserve_frame_counts(self):
        videos = [
            torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8),
            torch.randint(0, 256, (3, 3, 320, 240), dtype=torch.uint8),
        ]
        outputs = self.processor(videos, return_tensors="pt")

        self.assertEqual(outputs.pixel_values_videos.shape, (5, 3, 224, 224))
        self.assertEqual(outputs.video_frame_counts.tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
