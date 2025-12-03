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


import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image

    from transformers import Glm4vImageProcessor
    from transformers.models.glm4v.image_processing_glm4v import smart_resize

    if is_torchvision_available():
        from transformers import Glm4vImageProcessorFast


class Glm4vImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        temporal_patch_size=2,
        patch_size=14,
        merge_size=2,
    ):
        size = size if size is not None else {"longest_edge": 20, "shortest_edge": 10}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
        }

    def expected_output_image_shape(self, images):
        grid_t = 1
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        seq_len = 0
        for image in images:
            if isinstance(image, list) and isinstance(image[0], Image.Image):
                image = np.stack([np.array(frame) for frame in image])
            elif hasattr(image, "shape"):
                pass
            else:
                image = np.array(image)
            if hasattr(image, "shape") and len(image.shape) >= 3:
                if isinstance(image, np.ndarray):
                    if len(image.shape) == 4:
                        height, width = image.shape[1:3]
                    elif len(image.shape) == 3:
                        height, width = image.shape[:2]
                    else:
                        height, width = self.min_resolution, self.min_resolution
                else:
                    height, width = image.shape[-2:]
            else:
                height, width = self.min_resolution, self.min_resolution

            resized_height, resized_width = smart_resize(
                self.temporal_patch_size,
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.size["shortest_edge"],
                max_pixels=self.size["longest_edge"],
            )
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            seq_len += grid_t * grid_h * grid_w
        return (seq_len, hidden_dim)

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
class ViTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Glm4vImageProcessor if is_vision_available() else None
    fast_image_processing_class = Glm4vImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Glm4vImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 10, "longest_edge": 20})

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size={"shortest_edge": 42, "longest_edge": 42}
            )
            self.assertEqual(image_processor.size, {"shortest_edge": 42, "longest_edge": 42})

    # batch size is flattened
    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processor_list:
            # Test that can process images which have an arbitrary number of channels
            # Initialize image_processing
            image_processor = image_processing_class(**self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            # Test not batched input
            encoded_images = image_processor(
                image_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processor(
                image_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
