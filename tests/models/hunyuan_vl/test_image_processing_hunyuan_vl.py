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

import json
import unittest

import numpy as np

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers.models.hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil

    if is_torchvision_available():
        from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor


class HunYuanVLImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=32,
        max_resolution=64,
        min_pixels=32 * 32,
        max_pixels=32 * 32,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_resize=True,
        patch_size=16,
        temporal_patch_size=1,
        merge_size=1,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_resize = do_resize
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "do_convert_rgb": self.do_convert_rgb,
        }

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
@require_torchvision
class HunYuanVLImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = HunYuanVLImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def assert_image_processor_output(self, output, batch_size):
        grid_h = grid_w = 2
        patches_per_image = grid_h * grid_w
        expected_output_shape = (
            batch_size * patches_per_image,
            self.image_processor_tester.num_channels
            * self.image_processor_tester.temporal_patch_size
            * self.image_processor_tester.patch_size
            * self.image_processor_tester.patch_size,
        )
        expected_grid_thw = torch.tensor([[1, grid_h, grid_w]] * batch_size)

        self.assertEqual(tuple(output.pixel_values.shape), expected_output_shape)
        self.assertTrue((output.image_grid_thw == expected_grid_thw).all())

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "temporal_patch_size"))
            self.assertTrue(hasattr(image_processing, "merge_size"))

    def test_image_processor_to_json_string(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            obj = json.loads(image_processor.to_json_string())
            for key, value in self.image_processor_dict.items():
                if key not in ["min_pixels", "max_pixels"]:
                    self.assertEqual(obj[key], value)

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            self.assert_image_processor_output(image_processing(image_inputs[0], return_tensors="pt"), batch_size=1)
            self.assert_image_processor_output(
                image_processing(image_inputs, return_tensors="pt"),
                batch_size=self.image_processor_tester.batch_size,
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            self.assert_image_processor_output(image_processing(image_inputs[0], return_tensors="pt"), batch_size=1)
            self.assert_image_processor_output(
                image_processing(image_inputs, return_tensors="pt"),
                batch_size=self.image_processor_tester.batch_size,
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            self.assert_image_processor_output(image_processing(image_inputs[0], return_tensors="pt"), batch_size=1)
            self.assert_image_processor_output(
                image_processing(image_inputs, return_tensors="pt"),
                batch_size=self.image_processor_tester.batch_size,
            )

    @unittest.skip(reason="HunYuanVL image processors are designed for 3-channel RGB images")
    def test_call_numpy_4_channels(self):
        pass

    def test_torchvision_image_processor_outputs_image_only_inputs(self):
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

    def test_patch_count_contract_matches_pil_processor(self):
        torchvision_processor = HunYuanVLImageProcessor(
            min_pixels=32 * 32,
            max_pixels=64 * 64,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )
        pil_processor = HunYuanVLImageProcessorPil(
            min_pixels=32 * 32,
            max_pixels=64 * 64,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )

        image_kwargs = {"min_pixels": 32 * 32, "max_pixels": 64 * 64, "patch_size": 16, "merge_size": 1}
        self.assertEqual(
            torchvision_processor.get_number_of_image_patches(32, 64, image_kwargs),
            pil_processor.get_number_of_image_patches(32, 64, image_kwargs),
        )


@require_vision
class HunYuanVLImageProcessorPilTest(unittest.TestCase):
    def test_pil_image_processor_outputs_image_only_inputs(self):
        processor = HunYuanVLImageProcessorPil(
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
