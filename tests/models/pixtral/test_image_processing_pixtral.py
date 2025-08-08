# Copyright 2024 HuggingFace Inc.
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

import numpy as np
import requests
from packaging import version

from transformers.testing_utils import require_torch, require_torch_gpu, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import PixtralImageProcessor

    if is_torchvision_available():
        from transformers import PixtralImageProcessorFast


class PixtralImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        max_num_images_per_sample=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        patch_size=None,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
    ):
        super().__init__()
        size = size if size is not None else {"longest_edge": 24}
        patch_size = patch_size if patch_size is not None else {"height": 8, "width": 8}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.max_num_images_per_sample = max_num_images_per_sample
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "patch_size": self.patch_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_image_shape(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]

        batch_size = len(images)
        return_height, return_width = 0, 0
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            elif isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]

            max_height = max_width = self.size.get("longest_edge")

            ratio = max(height / max_height, width / max_width)
            if ratio > 1:
                height = int(np.floor(height / ratio))
                width = int(np.floor(width / ratio))

            patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]
            num_height_tokens = (height - 1) // patch_height + 1
            num_width_tokens = (width - 1) // patch_width + 1

            return_height = max(num_height_tokens * patch_height, return_height)
            return_width = max(num_width_tokens * patch_width, return_width)

        return batch_size, self.num_channels, return_height, return_width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return images


@require_torch
@require_vision
class PixtralImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = PixtralImageProcessor if is_vision_available() else None
    fast_image_processing_class = PixtralImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = PixtralImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    # The following tests are overridden as PixtralImageProcessor can return images of different sizes
    # and thus doesn't support returning batched tensors

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs_list = self.image_processor_tester.prepare_image_inputs()
            for image in image_inputs_list:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs_list[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs_list[0])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs_list, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs_list)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs_list = self.image_processor_tester.prepare_image_inputs(numpify=True)
            for image in image_inputs_list:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs_list[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs_list[0])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            batch_encoded_images = image_processing(image_inputs_list, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs_list)
            self.assertEqual(tuple(batch_encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs_list = self.image_processor_tester.prepare_image_inputs(torchify=True)
            for image in image_inputs_list:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs_list[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs_list[0])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            batch_encoded_images = image_processing(image_inputs_list, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs_list)
            self.assertEqual(tuple(batch_encoded_images.shape), expected_output_image_shape)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )

        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values[0][0], encoding_fast.pixel_values[0][0])

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        for i in range(len(encoding_slow.pixel_values)):
            self._assert_slow_fast_tensors_equivalence(
                encoding_slow.pixel_values[i][0], encoding_fast.pixel_values[i][0]
            )

    @slow
    @require_torch_gpu
    @require_vision
    def test_can_compile_fast_image_processor(self):
        if self.fast_image_processing_class is None:
            self.skipTest("Skipping compilation test as fast image processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.fast_image_processing_class(**self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(
            output_eager.pixel_values[0][0], output_compiled.pixel_values[0][0], atol=1e-4, rtol=1e-4, mean_atol=1e-5
        )

    @unittest.skip(reason="PixtralImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")  # FIXME Amy
    def test_call_numpy_4_channels(self):
        pass
