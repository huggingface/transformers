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

import tempfile
import unittest

import numpy as np

from transformers import AutoImageProcessor, LlavaOnevision2ImageProcessor, LlavaOnevision2ImageProcessorPil
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class LlavaOnevision2ImageProcessorTest(unittest.TestCase):
    def test_patch_order_and_backends(self):
        import torch

        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        processors = [
            cls(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})
            for cls in (LlavaOnevision2ImageProcessor, LlavaOnevision2ImageProcessorPil)
        ]
        outputs = [processor(image, return_tensors="pt") for processor in processors]
        self.assertEqual(outputs[0]["pixel_values"].shape, (16, 12))
        self.assertEqual(outputs[0]["image_grid_thw"].tolist(), [[1, 4, 4]])
        torch.testing.assert_close(outputs[0]["pixel_values"], outputs[1]["pixel_values"], atol=1e-6, rtol=1e-6)
        for processor in processors:
            with tempfile.TemporaryDirectory() as directory:
                processor.save_pretrained(directory)
                loaded = AutoImageProcessor.from_pretrained(directory, backend=processor.backend)
                self.assertEqual(loaded.temporal_patch_size, 1)
                self.assertIsInstance(loaded, type(processor))

    def test_dynamic_resolution(self):
        processor = LlavaOnevision2ImageProcessorPil(patch_size=2, size={"shortest_edge": 16, "longest_edge": 64})
        outputs = processor(
            [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 16, 3), dtype=np.uint8)], return_tensors="pt"
        )
        self.assertEqual(outputs["image_grid_thw"].shape, (2, 3))
        self.assertEqual(outputs["image_grid_thw"].prod(-1).sum(), len(outputs["pixel_values"]))
