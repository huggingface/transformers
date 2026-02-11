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
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import GLPNImageProcessor


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

    def setUp(self):
        super().setUp()
        self.image_processor_tester = GLPNImageProcessingTester(self)
        self.image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size_divisor"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))

    def test_call_pil(self):
        # Initialize image_processing
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
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
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
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
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input (GLPNImageProcessor doesn't support batching)
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertTrue(tuple(encoded_images.shape) == (1, *expected_output_image_shape))

    def test_call_numpy_4_channels(self):
        for backend_name in self.image_processors_backends_list:
            # Initialize image_processing
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
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

    # Override as GLPN image processors don't support heterogeneous batching (use equal_resolution=True)
    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

        # Create processors for each backend
        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @require_vision
    @require_torch
    def test_backends_equivalence_post_process_depth(self):
        """Check that all backends produce equivalent post-processed depth maps."""
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        outputs = self.image_processor_tester.prepare_depth_outputs()
        target_sizes = [(240, 320)] * self.image_processor_tester.batch_size

        # Create processors and run post-processing for each backend
        processed = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            processed[backend_name] = image_processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)

        # Compare all backends to the first one (reference backend)
        backend_names = list(processed.keys())
        reference_backend = backend_names[0]
        for backend_name in backend_names[1:]:
            for pred_ref, pred_other in zip(processed[reference_backend], processed[backend_name]):
                depth_ref = pred_ref["predicted_depth"].float()
                depth_other = pred_other["predicted_depth"].float()
                self._assert_tensors_equivalence(depth_ref, depth_other)
