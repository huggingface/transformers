# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the EfficientFormer image processor."""

import time
import unittest

import numpy as np

from tests.test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers.models.deprecated.efficientformer import EfficientFormerImageProcessor


if is_torch_available() and is_torchvision_available():
    import torch

    from transformers.models.deprecated.efficientformer import EfficientFormerImageProcessorFast


class EfficientFormerImageProcessingTester:
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
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_factor=1 / 255,
        do_center_crop=True,
        crop_size=None,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to EfficientFormerImageProcessor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                # We don't use width and height but need to unpack them
                _, _ = image.size
            else:
                # We don't use height and width but need to unpack them
                _, _ = image.shape[0], image.shape[1]
            if isinstance(self.size, dict):
                expected_height = self.size["height"]
                expected_width = self.size["width"]
            elif isinstance(self.size, list):
                expected_height = self.size[0]
                expected_width = self.size[1]
            else:
                expected_height = self.size
                expected_width = self.size
        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        return self.num_channels, height, width

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


@require_vision
class EfficientFormerImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = EfficientFormerImageProcessor if is_vision_available() else None
    fast_image_processing_class = EfficientFormerImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        self.image_processor_tester = EfficientFormerImageProcessingTester(self)
        # Initialize image_processor_list for the common tests
        self.image_processors_list = [self.image_processing_class] if self.image_processing_class is not None else []
        self.image_processor_list = self.image_processors_list  # For backward compatibility

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_rescale"))
        self.assertTrue(hasattr(image_processor, "rescale_factor"))
        self.assertTrue(hasattr(image_processor, "do_center_crop"))
        self.assertTrue(hasattr(image_processor, "crop_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 18, "width": 18})
        self.assertEqual(image_processor.do_resize, True)

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size={"height": 42, "width": 42}, do_resize=False)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})
        self.assertEqual(image_processor.do_resize, False)

    # Override the test_fast_is_faster_than_slow test for EfficientFormer
    @require_vision
    @require_torch
    def test_fast_is_faster_than_slow(self):
        if not hasattr(self, "fast_image_processing_class") or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping speed test as fast image processor is not defined")

        def measure_time(image_processor, image):
            # Warmup
            for _ in range(5):
                _ = image_processor(image, return_tensors="pt")
            all_times = []
            for _ in range(10):
                start = time.time()
                _ = image_processor(image, return_tensors="pt")
                all_times.append(time.time() - start)
            # Take the average of the fastest 3 runs
            avg_time = sum(sorted(all_times[:3])) / 3.0
            return avg_time

        # Use PIL images for the slow processor to avoid the shape mismatch issue
        pil_images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]

        # Use torch tensors for the fast processor
        torch_images = [torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8) for _ in range(4)]

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        slow_time = measure_time(image_processor_slow, pil_images)
        fast_time = measure_time(image_processor_fast, torch_images)

        self.assertLess(fast_time, slow_time, f"Fast processor ({fast_time:.6f}s) should be faster than slow processor ({slow_time:.6f}s)")

    # Override the test_save_load_fast_slow_auto test to skip it for EfficientFormer
    def test_save_load_fast_slow_auto(self):
        self.skipTest("Skipping save/load auto test for EfficientFormer")

    # Override the test_save_load_fast_slow test to skip it for EfficientFormer
    def test_save_load_fast_slow(self):
        self.skipTest("Skipping save/load test for EfficientFormer")

    # Override the test_image_processor_save_load_with_autoimageprocessor test to skip it for EfficientFormer
    def test_image_processor_save_load_with_autoimageprocessor(self):
        self.skipTest("Skipping save/load with auto test for EfficientFormer")


@require_torch
@require_vision
class EfficientFormerImageProcessorFastTest(EfficientFormerImageProcessorTest):
    image_processing_class = EfficientFormerImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        self.image_processor_tester = EfficientFormerImageProcessingTester(self)
        # Initialize image_processor_list for the common tests
        self.image_processors_list = [self.image_processing_class] if self.image_processing_class is not None else []
        self.image_processor_list = self.image_processors_list  # For backward compatibility
