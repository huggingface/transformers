# Copyright 2023 HuggingFace Inc.
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


import time
import unittest
import warnings

import numpy as np
import pytest
from packaging import version

from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image

    from transformers import VitMatteImageProcessor

    if is_torchvision_available():
        from transformers import VitMatteImageProcessorFast


class VitMatteImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_rescale=True,
        rescale_factor=0.5,
        do_pad=True,
        size_divisibility=10,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.size_divisibility = size_divisibility
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_pad": self.do_pad,
            "size_divisibility": self.size_divisibility,
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
class VitMatteImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = VitMatteImageProcessor if is_vision_available() else None
    fast_image_processing_class = VitMatteImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = VitMatteImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "size_divisibility"))

    def test_call_numpy(self):
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

    def test_call_pytorch(self):
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[1:])
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

        # create batched tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        image_input = torch.stack(image_inputs, dim=0)
        self.assertIsInstance(image_input, torch.Tensor)
        self.assertTrue(image_input.shape[1] == 3)

        trimap_shape = [image_input.shape[0]] + [1] + list(image_input.shape)[2:]
        trimap_input = torch.randint(0, 3, trimap_shape, dtype=torch.uint8)
        self.assertIsInstance(trimap_input, torch.Tensor)
        self.assertTrue(trimap_input.shape[1] == 1)

        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

    def test_call_pil(self):
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.size[::-1])
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

    def test_call_numpy_4_channels(self):
        # Test that can process images which have an arbitrary number of channels

        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processor(
                images=image,
                trimaps=trimap,
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
                return_tensors="pt",
            ).pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)
            self.assertTrue(encoded_images.shape[-3] == 5)

    def test_padding_slow(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image = np.random.randn(3, 249, 491)
        images = image_processing.pad_image(image)
        assert images.shape == (3, 256, 512)

        image = np.random.randn(3, 249, 512)
        images = image_processing.pad_image(image)
        assert images.shape == (3, 256, 512)

    def test_padding_fast(self):
        # extra test because name is different for fast image processor
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        image = torch.rand(3, 249, 491)
        images = image_processing._pad_image(image)
        assert images.shape == (3, 256, 512)

        image = torch.rand(3, 249, 512)
        images = image_processing._pad_image(image)
        assert images.shape == (3, 256, 512)

    def test_image_processor_preprocess_arguments(self):
        # vitmatte require additional trimap input for image_processor
        # that is why we override original common test

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            image = self.image_processor_tester.prepare_image_inputs()[0]
            trimap = np.random.randint(0, 3, size=image.size[::-1])

            with warnings.catch_warnings(record=True) as raised_warnings:
                warnings.simplefilter("always")
                image_processor(image, trimaps=trimap, extra_argument=True)

            messages = " ".join([str(w.message) for w in raised_warnings])
            self.assertGreaterEqual(len(raised_warnings), 1)
            self.assertIn("extra_argument", messages)

    @unittest.skip(reason="TODO: Yoni")
    def test_fast_is_faster_than_slow(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping speed test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping speed test as one of the image processors is not defined")

        def measure_time(image_processor, images, trimaps):
            # Warmup
            for _ in range(5):
                _ = image_processor(images, trimaps=trimaps, return_tensors="pt")
            all_times = []
            for _ in range(10):
                start = time.time()
                _ = image_processor(images, trimaps=trimaps, return_tensors="pt")
                all_times.append(time.time() - start)
            # Take the average of the fastest 3 runs
            avg_time = sum(sorted(all_times[:3])) / 3.0
            return avg_time

        dummy_images = torch.randint(0, 255, (4, 3, 400, 800), dtype=torch.uint8)
        dummy_trimaps = torch.randint(0, 3, (4, 400, 800), dtype=torch.uint8)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        fast_time = measure_time(image_processor_fast, dummy_images, dummy_trimaps)
        slow_time = measure_time(image_processor_slow, dummy_images, dummy_trimaps)

        self.assertLessEqual(fast_time, slow_time)

    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = load_image(url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"))
        dummy_trimap = np.random.randint(0, 3, size=dummy_image.size[::-1])
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, trimaps=dummy_trimap, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, trimaps=dummy_trimap, return_tensors="pt")
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)

    def test_slow_fast_equivalence_batched(self):
        # this only checks on equal resolution, since the slow processor doesn't work otherwise
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        dummy_trimaps = [np.random.randint(0, 3, size=image.shape[1:]) for image in dummy_images]
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, trimaps=dummy_trimaps, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, trimaps=dummy_trimaps, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_fast_image_processor(self):
        # override as trimaps are needed for the image processor
        if self.fast_image_processing_class is None:
            self.skipTest("Skipping compilation test as fast image processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        dummy_trimap = np.random.randint(0, 3, size=input_image.shape[1:])
        image_processor = self.fast_image_processing_class(**self.image_processor_dict)
        output_eager = image_processor(input_image, dummy_trimap, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, dummy_trimap, device=torch_device, return_tensors="pt")

        torch.testing.assert_close(output_eager.pixel_values, output_compiled.pixel_values, rtol=1e-4, atol=1e-4)
