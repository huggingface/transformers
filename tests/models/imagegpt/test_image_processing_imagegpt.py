# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
import os
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import ImageGPTImageProcessor


class ImageGPTImageProcessingTester(unittest.TestCase):
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
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize

    def prepare_image_processor_dict(self):
        return {
            # here we create 2 clusters for the sake of simplicity
            "clusters": np.asarray(
                [
                    [0.8866443634033203, 0.6618829369544983, 0.3891746401786804],
                    [-0.6042559146881104, -0.02295008860528469, 0.5423797369003296],
                ]
            ),
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
        }

    def expected_output_image_shape(self, images):
        return (self.size["height"] * self.size["width"],)

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
class ImageGPTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = ImageGPTImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = ImageGPTImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "clusters"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_image_processor_to_json_string(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for key, value in self.image_processor_dict.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, obj[key]))
            else:
                self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_processor.json")
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processing_class.from_json_file(json_file_path).to_dict()

        image_processor_first = image_processor_first.to_dict()
        for key, value in image_processor_first.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, image_processor_second[key]))
            else:
                self.assertEqual(image_processor_first[key], value)

    def test_image_processor_from_and_save_pretrained(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor_first.save_pretrained(tmpdirname)
            image_processor_second = self.image_processing_class.from_pretrained(tmpdirname).to_dict()

        image_processor_first = image_processor_first.to_dict()
        for key, value in image_processor_first.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, image_processor_second[key]))
            else:
                self.assertEqual(image_processor_first[key], value)

    @unittest.skip("ImageGPT requires clusters at initialization")
    def test_init_without_params(self):
        pass

    # Override the test from ImageProcessingTestMixin as ImageGPT model takes input_ids as input
    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").input_ids
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(encoded_images)
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").input_ids
        self.assertEqual(
            tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
        )

    # Override the test from ImageProcessingTestMixin as ImageGPT model takes input_ids as input
    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").input_ids
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(encoded_images)
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").input_ids
        self.assertEqual(
            tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
        )

    @unittest.skip("ImageGPT assumes clusters for 3 channels")
    def test_call_numpy_4_channels(self):
        pass

    # Override the test from ImageProcessingTestMixin as ImageGPT model takes input_ids as input
    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)

        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").input_ids
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").input_ids
        self.assertEqual(
            tuple(encoded_images.shape),
            (self.image_processor_tester.batch_size, *expected_output_image_shape),
        )


def prepare_images():
    # we use revision="refs/pr/1" until the PR is merged
    # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1
    dataset = load_dataset("hf-internal-testing/fixtures_image_utils", split="test", revision="refs/pr/1")

    image1 = dataset[4]["image"]
    image2 = dataset[5]["image"]

    images = [image1, image2]

    return images


@require_vision
@require_torch
class ImageGPTImageProcessorIntegrationTest(unittest.TestCase):
    @slow
    def test_image(self):
        image_processing = ImageGPTImageProcessor.from_pretrained("openai/imagegpt-small")

        images = prepare_images()

        # test non-batched
        encoding = image_processing(images[0], return_tensors="pt")

        self.assertIsInstance(encoding.input_ids, torch.LongTensor)
        self.assertEqual(encoding.input_ids.shape, (1, 1024))

        expected_slice = [306, 191, 191]
        self.assertEqual(encoding.input_ids[0, :3].tolist(), expected_slice)

        # test batched
        encoding = image_processing(images, return_tensors="pt")

        self.assertIsInstance(encoding.input_ids, torch.LongTensor)
        self.assertEqual(encoding.input_ids.shape, (2, 1024))

        expected_slice = [303, 13, 13]
        self.assertEqual(encoding.input_ids[1, -3:].tolist(), expected_slice)
