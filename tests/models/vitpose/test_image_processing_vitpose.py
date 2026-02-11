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
import io
import unittest

import httpx
import numpy as np
import pytest

from transformers.testing_utils import require_torch, require_torch_accelerator, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image

    from transformers import VitPoseImageProcessor


class VitPoseImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_affine_transform=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_affine_transform = do_affine_transform
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "do_affine_transform": self.do_affine_transform,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

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
class VitPoseImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = VitPoseImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = VitPoseImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_affine_transform"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class.from_dict(self.image_processor_dict, backend=backend_name)
            self.assertEqual(image_processor.size, {"height": 20, "width": 20})

            image_processor = self.image_processing_class.from_dict(
                self.image_processor_dict, backend=backend_name, size={"height": 42, "width": 42}
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_call_pil(self):
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]]
            encoded_images = image_processing(image_inputs[0], boxes=boxes, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (2, *expected_output_image_shape))

            # Test batched
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]] * self.image_processor_tester.batch_size
            encoded_images = image_processing(image_inputs, boxes=boxes, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size * 2, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]]
            encoded_images = image_processing(image_inputs[0], boxes=boxes, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (2, *expected_output_image_shape))

            # Test batched
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]] * self.image_processor_tester.batch_size
            encoded_images = image_processing(image_inputs, boxes=boxes, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size * 2, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]]
            encoded_images = image_processing(image_inputs[0], boxes=boxes, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (2, *expected_output_image_shape))

            # Test batched
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]] * self.image_processor_tester.batch_size
            encoded_images = image_processing(image_inputs, boxes=boxes, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size * 2, *expected_output_image_shape)
            )

    def test_call_numpy_4_channels(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            # Test not batched input
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]]
            encoded_images = image_processor(
                image_inputs[0],
                boxes=boxes,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (len(boxes[0]), *expected_output_image_shape))

            # Test batched
            boxes = [[[0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]]] * self.image_processor_tester.batch_size
            encoded_images = image_processor(
                image_inputs,
                boxes=boxes,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size * len(boxes[0]), *expected_output_image_shape),
            )
            self.image_processor_tester.num_channels = 3

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        """VitPose requires boxes parameter for preprocessing."""
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = Image.open(
            io.BytesIO(
                httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg", follow_redirects=True).content
            )
        )
        boxes = [[[0, 0, 1, 1]]]

        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, boxes=boxes, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_encoding = encodings[backend_names[0]].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        """VitPose requires boxes parameter for batched preprocessing."""
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        boxes = [[[0, 0, 1, 1]]] * len(dummy_images)

        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, boxes=boxes, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_encoding = encodings[backend_names[0]].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_torchvision_backend(self):
        """VitPose requires boxes parameter for preprocessing."""
        from packaging import version

        from transformers.testing_utils import torch_device

        if "torchvision" not in self.image_processors_backends_list:
            self.skipTest("Skipping compilation test as torchvision backend is not available")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.image_processing_class(backend="torchvision", **self.image_processor_dict)
        boxes = [[[0, 0, 1, 1]]]
        output_eager = image_processor(input_image, boxes=boxes, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, boxes=boxes, device=torch_device, return_tensors="pt")
        self._assert_tensors_equivalence(
            output_eager.pixel_values, output_compiled.pixel_values, atol=1e-4, rtol=1e-4, mean_atol=1e-5
        )
