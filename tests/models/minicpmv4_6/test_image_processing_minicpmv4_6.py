# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


class MiniCPMV4_6ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        min_resolution=64,
        max_resolution=128,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        max_slice_nums=9,
        scale_resolution=448,
        patch_size=14,
        slice_mode=True,
        downsample_mode="16x",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.slice_mode = slice_mode
        self.downsample_mode = downsample_mode

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "max_slice_nums": self.max_slice_nums,
            "scale_resolution": self.scale_resolution,
            "patch_size": self.patch_size,
            "slice_mode": self.slice_mode,
            "downsample_mode": self.downsample_mode,
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
class MiniCPMV4_6ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = MiniCPMV4_6ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "max_slice_nums"))
            self.assertTrue(hasattr(image_processing, "scale_resolution"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "slice_mode"))
            self.assertTrue(hasattr(image_processing, "downsample_mode"))

    def test_call_returns_expected_keys(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            images = self.image_processor_tester.prepare_image_inputs(torchify=True)
            result = image_processor(images, return_tensors="pt")
            self.assertIn("pixel_values", result)
            self.assertIn("tgt_sizes", result)
            self.assertIn("num_patches_per_image", result)
            self.assertIn("grids", result)

    def test_pixel_values_are_tensors(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            images = self.image_processor_tester.prepare_image_inputs(torchify=True)
            result = image_processor(images, return_tensors="pt")
            for pv in result["pixel_values"]:
                self.assertIsInstance(pv, torch.Tensor)

    def test_downsample_modes(self):
        for image_processing_class in self.image_processing_classes.values():
            images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            ip_16x = image_processing_class(**{**self.image_processor_dict, "downsample_mode": "16x"})
            result_16x = ip_16x(images, return_tensors="pt")

            ip_4x = image_processing_class(**{**self.image_processor_dict, "downsample_mode": "4x"})
            result_4x = ip_4x(images, return_tensors="pt")

            for ts_4x, ts_16x in zip(result_4x["tgt_sizes"], result_16x["tgt_sizes"]):
                h4, w4 = (ts_4x[0], ts_4x[1]) if not hasattr(ts_4x[0], "item") else (ts_4x[0].item(), ts_4x[1].item())
                h16, w16 = (
                    (ts_16x[0], ts_16x[1]) if not hasattr(ts_16x[0], "item") else (ts_16x[0].item(), ts_16x[1].item())
                )
                self.assertGreaterEqual(h4 * w4, h16 * w16)
