# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the Cosmos3 Edge image processors."""

import unittest

import numpy as np

from transformers import Cosmos3EdgeImageProcessor, Cosmos3EdgeImageProcessorPil
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class Cosmos3EdgeImageProcessingTest(unittest.TestCase):
    image_processing_classes = (Cosmos3EdgeImageProcessor, Cosmos3EdgeImageProcessorPil)

    def test_per_image_resize_overrides_are_preserved_across_backends(self):
        images = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
        processor_kwargs = {
            "do_rescale": False,
            "do_normalize": False,
            "patch_size": 2,
            "merge_size": 2,
            "min_pixels": 64,
            "max_pixels": 256,
        }

        for image_processing_class in self.image_processing_classes:
            with self.subTest(backend=image_processing_class.__name__):
                output = image_processing_class(**processor_kwargs)(
                    images,
                    per_image_kwargs=[
                        {"min_pixels": 64, "max_pixels": 64},
                        {"min_pixels": 256, "max_pixels": 256},
                    ],
                    return_tensors="pt",
                )

                self.assertEqual(output.image_grid_thw.tolist(), [[1, 4, 4], [1, 8, 8]])
                self.assertEqual(tuple(output.pixel_values.shape), (80, 12))
