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

import unittest

import numpy as np

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision


if is_torch_available():
    import torch

if is_vision_available():
    import PIL.Image

    from transformers import ImageFeatureExtractionMixin


def get_random_image(height, width):
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return PIL.Image.fromarray(random_array)


@require_vision
class ImageFeatureExtractionTester(unittest.TestCase):
    def test_conversion_image_to_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)

        # Conversion with defaults (rescale + channel first)
        array1 = feature_extractor.to_numpy_array(image)
        self.assertTrue(array1.dtype, np.float32)
        self.assertEqual(array1.shape, (3, 16, 32))

        # Conversion with rescale and not channel first
        array2 = feature_extractor.to_numpy_array(image, channel_first=False)
        self.assertTrue(array2.dtype, np.float32)
        self.assertEqual(array2.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array1, array2.transpose(2, 0, 1)))

        # Conversion with no rescale and channel first
        array3 = feature_extractor.to_numpy_array(image, rescale=False)
        self.assertTrue(array3.dtype, np.uint8)
        self.assertEqual(array3.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array1, array3.astype(np.float32) / 255.0))

        # Conversion with no rescale and not channel first
        array4 = feature_extractor.to_numpy_array(image, rescale=False, channel_first=False)
        self.assertTrue(array4.dtype, np.uint8)
        self.assertEqual(array4.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array2, array4.astype(np.float32) / 255.0))

    def test_conversion_array_to_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        array = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)

        # By default, rescale (for an array of ints) and channel permute
        array1 = feature_extractor.to_numpy_array(array)
        self.assertTrue(array1.dtype, np.float32)
        self.assertEqual(array1.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array1, array.transpose(2, 0, 1).astype(np.float32) / 255.0))

        # Same with no permute
        array2 = feature_extractor.to_numpy_array(array, channel_first=False)
        self.assertTrue(array2.dtype, np.float32)
        self.assertEqual(array2.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array2, array.astype(np.float32) / 255.0))

        # Force rescale to False
        array3 = feature_extractor.to_numpy_array(array, rescale=False)
        self.assertTrue(array3.dtype, np.uint8)
        self.assertEqual(array3.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array3, array.transpose(2, 0, 1)))

        # Force rescale to False and no channel permute
        array4 = feature_extractor.to_numpy_array(array, rescale=False, channel_first=False)
        self.assertTrue(array4.dtype, np.uint8)
        self.assertEqual(array4.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array4, array))

        # Now test the default rescale for a float array (defaults to False)
        array5 = feature_extractor.to_numpy_array(array2)
        self.assertTrue(array5.dtype, np.float32)
        self.assertEqual(array5.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array5, array1))

    @require_torch
    def test_conversion_torch_to_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.randint(0, 256, (16, 32, 3))
        array = tensor.numpy()

        # By default, rescale (for a tensor of ints) and channel permute
        array1 = feature_extractor.to_numpy_array(array)
        self.assertTrue(array1.dtype, np.float32)
        self.assertEqual(array1.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array1, array.transpose(2, 0, 1).astype(np.float32) / 255.0))

        # Same with no permute
        array2 = feature_extractor.to_numpy_array(array, channel_first=False)
        self.assertTrue(array2.dtype, np.float32)
        self.assertEqual(array2.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array2, array.astype(np.float32) / 255.0))

        # Force rescale to False
        array3 = feature_extractor.to_numpy_array(array, rescale=False)
        self.assertTrue(array3.dtype, np.uint8)
        self.assertEqual(array3.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array3, array.transpose(2, 0, 1)))

        # Force rescale to False and no channel permute
        array4 = feature_extractor.to_numpy_array(array, rescale=False, channel_first=False)
        self.assertTrue(array4.dtype, np.uint8)
        self.assertEqual(array4.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array4, array))

        # Now test the default rescale for a float tensor (defaults to False)
        array5 = feature_extractor.to_numpy_array(array2)
        self.assertTrue(array5.dtype, np.float32)
        self.assertEqual(array5.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array5, array1))

    def test_conversion_image_to_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)

        # On an image, `to_pil_image1` is a noop.
        image1 = feature_extractor.to_pil_image(image)
        self.assertTrue(isinstance(image, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image), np.array(image1)))

    def test_conversion_array_to_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        array = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)

        # By default, no rescale (for an array of ints)
        image1 = feature_extractor.to_pil_image(array)
        self.assertTrue(isinstance(image1, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image1), array))

        # If the array is channel-first, proper reordering of the channels is done.
        image2 = feature_extractor.to_pil_image(array.transpose(2, 0, 1))
        self.assertTrue(isinstance(image2, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image2), array))

        # If the array has floating type, it's rescaled by default.
        image3 = feature_extractor.to_pil_image(array.astype(np.float32) / 255.0)
        self.assertTrue(isinstance(image3, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image3), array))

        # You can override the default to rescale.
        image4 = feature_extractor.to_pil_image(array.astype(np.float32), rescale=False)
        self.assertTrue(isinstance(image4, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image4), array))

        # And with floats + channel first.
        image5 = feature_extractor.to_pil_image(array.transpose(2, 0, 1).astype(np.float32) / 255.0)
        self.assertTrue(isinstance(image5, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image5), array))

    @require_torch
    def test_conversion_tensor_to_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.randint(0, 256, (16, 32, 3))
        array = tensor.numpy()

        # By default, no rescale (for a tensor of ints)
        image1 = feature_extractor.to_pil_image(tensor)
        self.assertTrue(isinstance(image1, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image1), array))

        # If the tensor is channel-first, proper reordering of the channels is done.
        image2 = feature_extractor.to_pil_image(tensor.permute(2, 0, 1))
        self.assertTrue(isinstance(image2, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image2), array))

        # If the tensor has floating type, it's rescaled by default.
        image3 = feature_extractor.to_pil_image(tensor.float() / 255.0)
        self.assertTrue(isinstance(image3, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image3), array))

        # You can override the default to rescale.
        image4 = feature_extractor.to_pil_image(tensor.float(), rescale=False)
        self.assertTrue(isinstance(image4, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image4), array))

        # And with floats + channel first.
        image5 = feature_extractor.to_pil_image(tensor.permute(2, 0, 1).float() / 255.0)
        self.assertTrue(isinstance(image5, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image5), array))

    def test_resize_image_and_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = np.array(image)

        # Size can be an int or a tuple of ints.
        resized_image = feature_extractor.resize(image, 8)
        self.assertTrue(isinstance(resized_image, PIL.Image.Image))
        self.assertEqual(resized_image.size, (8, 8))

        resized_image1 = feature_extractor.resize(image, (8, 16))
        self.assertTrue(isinstance(resized_image1, PIL.Image.Image))
        self.assertEqual(resized_image1.size, (8, 16))

        # Passing and array converts it to a PIL Image.
        resized_image2 = feature_extractor.resize(array, 8)
        self.assertTrue(isinstance(resized_image2, PIL.Image.Image))
        self.assertEqual(resized_image2.size, (8, 8))
        self.assertTrue(np.array_equal(np.array(resized_image), np.array(resized_image2)))

        resized_image3 = feature_extractor.resize(image, (8, 16))
        self.assertTrue(isinstance(resized_image3, PIL.Image.Image))
        self.assertEqual(resized_image3.size, (8, 16))
        self.assertTrue(np.array_equal(np.array(resized_image1), np.array(resized_image3)))

    @require_torch
    def test_resize_tensor(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.randint(0, 256, (16, 32, 3))
        array = tensor.numpy()

        # Size can be an int or a tuple of ints.
        resized_image = feature_extractor.resize(tensor, 8)
        self.assertTrue(isinstance(resized_image, PIL.Image.Image))
        self.assertEqual(resized_image.size, (8, 8))

        resized_image1 = feature_extractor.resize(tensor, (8, 16))
        self.assertTrue(isinstance(resized_image1, PIL.Image.Image))
        self.assertEqual(resized_image1.size, (8, 16))

        # Check we get the same results as with NumPy arrays.
        resized_image2 = feature_extractor.resize(array, 8)
        self.assertTrue(np.array_equal(np.array(resized_image), np.array(resized_image2)))

        resized_image3 = feature_extractor.resize(array, (8, 16))
        self.assertTrue(np.array_equal(np.array(resized_image1), np.array(resized_image3)))

    def test_normalize_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = np.array(image)
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # PIL Image are converted to NumPy arrays for the normalization
        normalized_image = feature_extractor.normalize(image, mean, std)
        self.assertTrue(isinstance(normalized_image, np.ndarray))
        self.assertEqual(normalized_image.shape, (3, 16, 32))

        # During the conversion rescale and channel first will be applied.
        expected = array.transpose(2, 0, 1).astype(np.float32) / 255.0
        np_mean = np.array(mean).astype(np.float32)[:, None, None]
        np_std = np.array(std).astype(np.float32)[:, None, None]
        expected = (expected - np_mean) / np_std
        self.assertTrue(np.array_equal(normalized_image, expected))

    def test_normalize_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        array = np.random.random((16, 32, 3))
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # mean and std can be passed as lists or NumPy arrays.
        expected = (array - np.array(mean)) / np.array(std)
        normalized_array = feature_extractor.normalize(array, mean, std)
        self.assertTrue(np.array_equal(normalized_array, expected))

        normalized_array = feature_extractor.normalize(array, np.array(mean), np.array(std))
        self.assertTrue(np.array_equal(normalized_array, expected))

        # Normalize will detect automatically if channel first or channel last is used.
        array = np.random.random((3, 16, 32))
        expected = (array - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
        normalized_array = feature_extractor.normalize(array, mean, std)
        self.assertTrue(np.array_equal(normalized_array, expected))

        normalized_array = feature_extractor.normalize(array, np.array(mean), np.array(std))
        self.assertTrue(np.array_equal(normalized_array, expected))

    @require_torch
    def test_normalize_tensor(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.rand(16, 32, 3)
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # mean and std can be passed as lists or tensors.
        expected = (tensor - torch.tensor(mean)) / torch.tensor(std)
        normalized_tensor = feature_extractor.normalize(tensor, mean, std)
        self.assertTrue(torch.equal(normalized_tensor, expected))

        normalized_tensor = feature_extractor.normalize(tensor, torch.tensor(mean), torch.tensor(std))
        self.assertTrue(torch.equal(normalized_tensor, expected))

        # Normalize will detect automatically if channel first or channel last is used.
        tensor = torch.rand(3, 16, 32)
        expected = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        normalized_tensor = feature_extractor.normalize(tensor, mean, std)
        self.assertTrue(torch.equal(normalized_tensor, expected))

        normalized_tensor = feature_extractor.normalize(tensor, torch.tensor(mean), torch.tensor(std))
        self.assertTrue(torch.equal(normalized_tensor, expected))

    def test_center_crop_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_image = feature_extractor.center_crop(image, size)
            self.assertTrue(isinstance(cropped_image, PIL.Image.Image))

            # PIL Image.size is transposed compared to NumPy or PyTorch (width first instead of height first).
            expected_size = (size, size) if isinstance(size, int) else (size[1], size[0])
            self.assertEqual(cropped_image.size, expected_size)

    def test_center_crop_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = feature_extractor.to_numpy_array(image)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_array = feature_extractor.center_crop(array, size)
            self.assertTrue(isinstance(cropped_array, np.ndarray))

            expected_size = (size, size) if isinstance(size, int) else size
            self.assertEqual(cropped_array.shape[-2:], expected_size)

            # Check result is consistent with PIL.Image.crop
            cropped_image = feature_extractor.center_crop(image, size)
            self.assertTrue(np.array_equal(cropped_array, feature_extractor.to_numpy_array(cropped_image)))

    @require_torch
    def test_center_crop_tensor(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = feature_extractor.to_numpy_array(image)
        tensor = torch.tensor(array)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_tensor = feature_extractor.center_crop(tensor, size)
            self.assertTrue(isinstance(cropped_tensor, torch.Tensor))

            expected_size = (size, size) if isinstance(size, int) else size
            self.assertEqual(cropped_tensor.shape[-2:], expected_size)

            # Check result is consistent with PIL.Image.crop
            cropped_image = feature_extractor.center_crop(image, size)
            self.assertTrue(torch.equal(cropped_tensor, torch.tensor(feature_extractor.to_numpy_array(cropped_image))))
