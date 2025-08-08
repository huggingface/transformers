# coding=utf-8
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

from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image

    from transformers import Idefics3ImageProcessor

    if is_torchvision_available():
        from transformers import Idefics3ImageProcessorFast


if is_torch_available():
    import torch


class Idefics3ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_images=1,
        image_size=18,
        min_resolution=30,
        max_resolution=40,
        do_resize=True,
        size=None,
        max_image_size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
        do_pad=True,
        do_image_splitting=True,
        resample=PILImageResampling.LANCZOS,
    ):
        self.size = size if size is not None else {"longest_edge": max_resolution}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_images = num_images
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.resample = resample
        self.do_image_splitting = do_image_splitting
        self.max_image_size = max_image_size if max_image_size is not None else {"longest_edge": 20}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.do_pad = do_pad

    def prepare_image_processor_dict(self):
        return {
            "do_convert_rgb": self.do_convert_rgb,
            "do_resize": self.do_resize,
            "size": self.size,
            "max_image_size": self.max_image_size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
            "do_image_splitting": self.do_image_splitting,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to Idefics3ImageProcessor,
        assuming do_resize is set to True. The expected size in that case the max image size.
        """
        return self.max_image_size["longest_edge"], self.max_image_size["longest_edge"]

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        effective_nb_images = (
            self.num_images * 5 if self.do_image_splitting else 1
        )  # 5 is a squared image divided into 4 + global image resized
        return effective_nb_images, self.num_channels, height, width

    def prepare_image_inputs(
        self,
        batch_size=None,
        min_resolution=None,
        max_resolution=None,
        num_channels=None,
        num_images=None,
        size_divisor=None,
        equal_resolution=False,
        numpify=False,
        torchify=False,
    ):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.

        One can specify whether the images are of the same resolution or not.
        """
        assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

        batch_size = batch_size if batch_size is not None else self.batch_size
        min_resolution = min_resolution if min_resolution is not None else self.min_resolution
        max_resolution = max_resolution if max_resolution is not None else self.max_resolution
        num_channels = num_channels if num_channels is not None else self.num_channels
        num_images = num_images if num_images is not None else self.num_images

        images_list = []
        for i in range(batch_size):
            images = []
            for j in range(num_images):
                if equal_resolution:
                    width = height = max_resolution
                else:
                    # To avoid getting image width/height 0
                    if size_divisor is not None:
                        # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                        min_resolution = max(size_divisor, min_resolution)
                    width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
                images.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))
            images_list.append(images)

        if not numpify and not torchify:
            # PIL expects the channel dimension as last dimension
            images_list = [[Image.fromarray(np.moveaxis(image, 0, -1)) for image in images] for images in images_list]

        if torchify:
            images_list = [[torch.from_numpy(image) for image in images] for images in images_list]

        if numpify:
            # Numpy images are typically in channels last format
            images_list = [[image.transpose(1, 2, 0) for image in images] for images in images_list]

        return images_list


@require_torch
@require_vision
class Idefics3ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Idefics3ImageProcessor if is_vision_available() else None
    fast_image_processing_class = Idefics3ImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Idefics3ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_image_splitting"))
            self.assertTrue(hasattr(image_processing, "max_image_size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "do_image_splitting"))

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for sample_images in image_inputs:
                for image in sample_images:
                    self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy_4_channels(self):
        # Idefics3 always processes images as RGB, so it always returns images with 3 channels
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processor_dict = self.image_processor_dict
            image_processing = image_processing_class(**image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            for sample_images in image_inputs:
                for image in sample_images:
                    self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for images in image_inputs:
                for image in images:
                    self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for images in image_inputs:
                for image in images:
                    self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size, *expected_output_image_shape),
            )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )
        dummy_image = dummy_image.resize((100, 150))
        image_processor_slow = self.image_processing_class(
            **self.image_processor_dict, resample=PILImageResampling.BICUBIC
        )
        image_processor_fast = self.fast_image_processing_class(
            **self.image_processor_dict, resample=PILImageResampling.BICUBIC
        )

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt", return_row_col_info=True)
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt", return_row_col_info=True)

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.pixel_attention_mask.float(), encoding_fast.pixel_attention_mask.float()
        )
        self.assertEqual(encoding_slow.rows, encoding_fast.rows)
        self.assertEqual(encoding_slow.cols, encoding_fast.cols)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(
            equal_resolution=False, num_images=5, torchify=True
        )
        # pop some images to have non homogenous batches:
        indices_to_pop = [i if np.random.random() < 0.5 else None for i in range(len(dummy_images))]
        for i in indices_to_pop:
            if i is not None:
                dummy_images[i].pop()

        image_processor_slow = self.image_processing_class(
            **self.image_processor_dict, resample=PILImageResampling.BICUBIC
        )
        image_processor_fast = self.fast_image_processing_class(
            **self.image_processor_dict, resample=PILImageResampling.BICUBIC
        )

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt", return_row_col_info=True)
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt", return_row_col_info=True)

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values, atol=3e-1)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.pixel_attention_mask.float(), encoding_fast.pixel_attention_mask.float()
        )
        self.assertEqual(encoding_slow.rows, encoding_fast.rows)
        self.assertEqual(encoding_slow.cols, encoding_fast.cols)

    def test_get_num_patches_without_images(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            num_patches_and_row_cols = image_processing.get_number_of_image_patches(
                height=100, width=100, images_kwargs={}
            )
            self.assertEqual(num_patches_and_row_cols, (5, 2, 2))

            num_patches_and_row_cols = image_processing.get_number_of_image_patches(
                height=300, width=500, images_kwargs={"do_image_splitting": False}
            )
            self.assertEqual(num_patches_and_row_cols, (1, 1, 1))

            num_patches_and_row_cols = image_processing.get_number_of_image_patches(
                height=300, width=500, images_kwargs={"do_image_splitting": True}
            )
            self.assertEqual(num_patches_and_row_cols, (5, 2, 2))

            num_patches_and_row_cols = image_processing.get_number_of_image_patches(
                height=300,
                width=600,
                images_kwargs={"do_image_splitting": True, "max_image_size": {"longest_edge": 30}},
            )
            self.assertEqual(num_patches_and_row_cols, (3, 1, 2))
