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

import importlib.util
import unittest

from PIL import Image

from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor
from transformers.testing_utils import require_torch, require_torchvision


class HunYuanVLImageProcessorTest(unittest.TestCase):
    def test_slow_image_processor_outputs_image_only_inputs(self):
        processor = HunYuanVLImageProcessor(
            min_pixels=32 * 32,
            max_pixels=32 * 32,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )
        image = Image.new("RGB", (32, 32), color="white")

        inputs = processor(images=[image], return_tensors="np")

        self.assertSetEqual(set(inputs.keys()), {"pixel_values", "image_grid_thw"})
        self.assertEqual(inputs["image_grid_thw"].shape, (1, 3))
        self.assertGreater(inputs["pixel_values"].shape[0], 0)

    def test_patch_count_contract_matches_fast_processor(self):
        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("torchvision is required for fast image processor parity checks")

        from transformers.models.hunyuan_vl.image_processing_hunyuan_vl_fast import HunYuanVLImageProcessorFast

        slow_processor = HunYuanVLImageProcessor(
            min_pixels=32 * 32,
            max_pixels=64 * 64,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )
        fast_processor = HunYuanVLImageProcessorFast(
            min_pixels=32 * 32,
            max_pixels=64 * 64,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )

        image_kwargs = {"min_pixels": 32 * 32, "max_pixels": 64 * 64, "patch_size": 16, "merge_size": 1}
        self.assertEqual(
            slow_processor.get_number_of_image_patches(32, 64, image_kwargs),
            fast_processor.get_number_of_image_patches(32, 64, image_kwargs),
        )


@require_torch
@require_torchvision
class HunYuanVLImageProcessorFastTest(unittest.TestCase):
    def test_fast_image_processor_outputs_image_only_inputs(self):
        from transformers.models.hunyuan_vl.image_processing_hunyuan_vl_fast import HunYuanVLImageProcessorFast

        processor = HunYuanVLImageProcessorFast(
            min_pixels=32 * 32,
            max_pixels=32 * 32,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )
        image = Image.new("RGB", (32, 32), color="white")

        inputs = processor(images=[image], return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), {"pixel_values", "image_grid_thw"})
        self.assertEqual(tuple(inputs["image_grid_thw"].shape), (1, 3))
        self.assertGreater(inputs["pixel_values"].shape[0], 0)
