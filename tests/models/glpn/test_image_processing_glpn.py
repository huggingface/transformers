# Copyright 2022 HuggingFace Inc.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import GLPNImageProcessor

    if is_torchvision_available():
        from transformers import GLPNImageProcessorFast


class GLPNImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size_divisor=32,
        do_rescale=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size_divisor": self.size_divisor,
            "do_rescale": self.do_rescale,
        }

    def expected_output_image_shape(self, images):
        if isinstance(images[0], Image.Image):
            width, height = images[0].size
        elif isinstance(images[0], np.ndarray):
            height, width = images[0].shape[0], images[0].shape[1]
        else:
            height, width = images[0].shape[1], images[0].shape[2]

        height = height // self.size_divisor * self.size_divisor
        width = width // self.size_divisor * self.size_divisor

        return self.num_channels, height, width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            size_divisor=self.size_divisor,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

    def prepare_depth_outputs(self):
        if not is_torch_available():
            return None
        depth_tensors = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=1,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=True,
            torchify=True,
        )
        depth_tensors = [depth_tensor.squeeze(0) for depth_tensor in depth_tensors]
        stacked_depth_tensors = torch.stack(depth_tensors, dim=0)
        return type("DepthOutput", (), {"predicted_depth": stacked_depth_tensors})


@require_torch
@require_vision
class GLPNImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = GLPNImageProcessor if is_vision_available() else None
    fast_image_processing_class = GLPNImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = GLPNImageProcessingTester(self)
        self.image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size_divisor"))
        self.assertTrue(hasattr(image_processing, "resample"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        # Test not batched input (GLPNImageProcessor doesn't support batching)
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertTrue(tuple(encoded_images.shape) == (1, *expected_output_image_shape))

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input (GLPNImageProcessor doesn't support batching)
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertTrue(tuple(encoded_images.shape) == (1, *expected_output_image_shape))

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input (GLPNImageProcessor doesn't support batching)
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertTrue(tuple(encoded_images.shape) == (1, *expected_output_image_shape))

    def test_call_numpy_4_channels(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        self.image_processing_class.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input (GLPNImageProcessor doesn't support batching)
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertTrue(tuple(encoded_images.shape) == (1, *expected_output_image_shape))
        self.image_processing_class.num_channels = 3

    # override as glpn image processors don't support heterogeneous batching
    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)

    def test_post_process_depth_equivalence(self):
        # Check that both processors produce equivalent post-processed depth maps
        if self.fast_image_processing_class is None:
            self.skipTest("TorchVision not available")

        outputs = self.image_processor_tester.prepare_depth_outputs()
        slow = self.image_processing_class(**self.image_processor_dict)
        fast = self.fast_image_processing_class(**self.image_processor_dict)

        # target_sizes simulate resized inference outputs
        target_sizes = [(240, 320)] * self.image_processor_tester.batch_size
        processed_slow = slow.post_process_depth_estimation(outputs, target_sizes=target_sizes)
        processed_fast = fast.post_process_depth_estimation(outputs, target_sizes=target_sizes)

        # Compare per-sample predicted depth tensors
        for pred_slow, pred_fast in zip(processed_slow, processed_fast):
            depth_slow = pred_slow["predicted_depth"]
            depth_fast = pred_fast["predicted_depth"]
            torch.testing.assert_close(depth_fast, depth_slow, atol=1e-1, rtol=1e-3)
            self.assertLessEqual(torch.mean(torch.abs(depth_fast.float() - depth_slow.float())).item(), 5e-3)
