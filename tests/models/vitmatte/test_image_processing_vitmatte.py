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


import inspect
import unittest
import warnings

import numpy as np
import pytest

from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


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
        size_divisor=10,
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
        self.size_divisor = size_divisor
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
            "size_divisor": self.size_divisor,
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
    def setUp(self):
        super().setUp()
        self.image_processor_tester = VitMatteImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "size_divisor"))

    def test_call_numpy(self):
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

    def test_call_pytorch(self):
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[1:])
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisor == 0)
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

        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

    def test_call_pil(self):
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.size[::-1])
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(images=image, trimaps=trimap, return_tensors="pt").pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-3] == 4)

    def test_call_numpy_4_channels(self):
        # Test that can process images which have an arbitrary number of channels

        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        # Test not batched input (image processor does not support batched inputs)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processor(
                images=image,
                trimaps=trimap,
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
                return_tensors="pt",
            ).pixel_values

            # Verify that width and height can be divided by size_divisibility and that correct dimensions got merged
            self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisor == 0)
            self.assertTrue(encoded_images.shape[-3] == 5)

    def test_padding(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processing = image_processing_class(**self.image_processor_dict)
            if backend_name == "pil":
                image = np.random.randn(3, 249, 491)
                images = image_processing.pad_image(image)
                assert images.shape == (3, 256, 512)

                image = np.random.randn(3, 249, 512)
                images = image_processing.pad_image(image)
                assert images.shape == (3, 256, 512)
            else:  # torchvision
                image = torch.rand(3, 249, 491)
                images = image_processing._pad_image(image)
                assert images.shape == (3, 256, 512)

                image = torch.rand(3, 249, 512)
                images = image_processing._pad_image(image)
                assert images.shape == (3, 256, 512)

    def test_image_processor_preprocess_arguments(self):
        is_tested = False

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)

            # validation done by _valid_processor_keys attribute
            if hasattr(image_processor, "_valid_processor_keys") and hasattr(image_processor, "preprocess"):
                preprocess_parameter_names = inspect.getfullargspec(image_processor.preprocess).args
                preprocess_parameter_names.remove("self")
                preprocess_parameter_names.sort()
                valid_processor_keys = image_processor._valid_processor_keys
                valid_processor_keys.sort()
                self.assertEqual(preprocess_parameter_names, valid_processor_keys)
                is_tested = True

            # validation done by @filter_out_non_signature_kwargs decorator
            if hasattr(image_processor.preprocess, "_filter_out_non_signature_kwargs"):
                inputs = self.image_processor_tester.prepare_image_inputs()
                image = inputs[0]
                trimap = np.random.randint(0, 3, size=image.size[::-1])

                with warnings.catch_warnings(record=True) as raised_warnings:
                    warnings.simplefilter("always")
                    image_processor(image, trimaps=trimap, extra_argument=True)

                messages = " ".join([str(w.message) for w in raised_warnings])
                self.assertGreaterEqual(len(raised_warnings), 1)
                self.assertIn("extra_argument", messages)
                is_tested = True

            # ViTMatte-specific: validation for processors requiring trimaps (no _filter_out_non_signature_kwargs)
            if "trimaps" in inspect.signature(image_processor.preprocess).parameters:
                inputs = self.image_processor_tester.prepare_image_inputs()
                image = inputs[0]
                trimap = np.random.randint(0, 3, size=image.size[::-1])

                # Extra kwargs are rejected (TypeError for strict validation, or warning)
                with self.assertRaises(TypeError):
                    image_processor(image, trimaps=trimap, extra_argument=True)
                is_tested = True

        if not is_tested:
            self.skipTest(reason="No validation found for `preprocess` method")

    def test_backends_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_image(url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"))
        dummy_trimap = np.random.randint(0, 3, size=dummy_image.size[::-1])

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, trimaps=dummy_trimap, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    def test_backends_equivalence_batched(self):
        # this only checks on equal resolution, since the slow processor doesn't work otherwise
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        dummy_trimaps = [np.random.randint(0, 3, size=image.shape[1:]) for image in dummy_images]

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, trimaps=dummy_trimaps, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_torchvision_backend(self):
        # override as trimaps are needed for the image processor
        if "torchvision" not in self.image_processing_classes:
            self.skipTest("Skipping compilation test as torchvision image processor is not defined")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        dummy_trimap = np.random.randint(0, 3, size=input_image.shape[1:])
        image_processor = self.image_processing_classes["torchvision"](**self.image_processor_dict)
        output_eager = image_processor(input_image, dummy_trimap, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, dummy_trimap, device=torch_device, return_tensors="pt")

        torch.testing.assert_close(output_eager.pixel_values, output_compiled.pixel_values, rtol=1e-4, atol=1e-4)
