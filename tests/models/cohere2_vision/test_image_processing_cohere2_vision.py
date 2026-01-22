# Copyright 2025 HuggingFace Inc.
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

    if is_torchvision_available():
        from transformers import Cohere2VisionImageProcessorFast


class Cohere2VisionImageProcessingTester(unittest.TestCase):
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
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
    ):
        super().__init__()
        size = size if size is not None else {"height": 30, "width": 30}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
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
class Cohere2VisionProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    fast_image_processing_class = Cohere2VisionImageProcessorFast if is_torchvision_available() else None
    test_slow_image_processor = False

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Cohere2VisionImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (10, 3, 30, 30))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (70, 3, 30, 30))

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (10, 3, 30, 30))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (70, 3, 30, 30))

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (10, 3, 30, 30))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(tuple(encoded_images.shape), (70, 3, 30, 30))

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processor_list:
            # Test that can process images which have an arbitrary number of channels
            # Initialize image_processing
            image_processor = image_processing_class(**self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)

            # Test not batched input
            encoded_images = image_processor(
                image_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            self.assertEqual(tuple(encoded_images.shape), (10, 4, 30, 30))

            # Test batched
            encoded_images = image_processor(
                image_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 1.0, 1.0, 1.0),
            ).pixel_values
            self.assertEqual(tuple(encoded_images.shape), (70, 4, 30, 30))

    def test_crop_to_patches_aspect_ratio(self):
        """Test that row/column ordering is correct when cropping non-square images to patches.

        This test verifies that patches can be stitched back to reconstruct the original image,
        which validates that the row/column ordering in get_optimal_tiled_canvas is correct.
        If row/column are swapped, the image would be resized to wrong dimensions and patches
        would not match the original content.
        """
        for image_processing_class in self.image_processor_list:
            patch_size = 64
            image_processor = image_processing_class(
                do_resize=True,
                size={"height": patch_size, "width": patch_size},
                do_normalize=False,  # Disable normalization to preserve pixel values
                do_rescale=False,  # Disable rescaling to preserve pixel values
                crop_to_patches=True,
                min_patches=1,
                max_patches=6,  # Allow up to 6 patches to test asymmetric grids like 2x3
            )

            # Create a 2:3 aspect ratio image (2 rows x 3 columns of patches)
            # This asymmetric grid will fail if rows/columns are swapped
            num_rows, num_cols = 2, 3
            image_height = patch_size * num_rows  # 128
            image_width = patch_size * num_cols  # 192

            # Create image with unique color for each patch position
            test_image = Image.new("RGB", (image_width, image_height))
            for row in range(num_rows):
                for col in range(num_cols):
                    patch_idx = row * num_cols + col  # 0-5
                    color = (patch_idx * 40 + 20, 0, 0)  # Unique red values: 20, 60, 100, 140, 180, 220
                    for y in range(patch_size):
                        for x in range(patch_size):
                            test_image.putpixel(
                                (col * patch_size + x, row * patch_size + y),
                                color,
                            )

            # Process image
            result = image_processor(test_image, return_tensors="pt")
            patches = result.pixel_values
            num_patches_result = result.num_patches

            # Should produce 7 patches (6 grid patches + 1 thumbnail)
            self.assertEqual(num_patches_result.tolist(), [7])
            self.assertEqual(tuple(patches.shape), (7, 3, patch_size, patch_size))

            # Verify each patch has the correct color (excluding thumbnail which is last)
            # Patches should be ordered row by row: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
            for patch_idx in range(6):
                expected_red = patch_idx * 40 + 20
                actual_red = patches[patch_idx, 0, 0, 0].item()  # Red channel, top-left pixel
                self.assertEqual(
                    actual_red,
                    expected_red,
                    f"Patch {patch_idx} has wrong color. Expected red={expected_red}, got {actual_red}. "
                    f"This indicates row/column ordering is incorrect.",
                )

            # Stitch patches back and verify against original
            stitched = torch.zeros(3, image_height, image_width)
            for patch_idx in range(6):
                row = patch_idx // num_cols
                col = patch_idx % num_cols
                stitched[
                    :,
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ] = patches[patch_idx]

            original_tensor = torch.tensor(np.array(test_image)).permute(2, 0, 1).float()
            self.assertTrue(
                torch.allclose(stitched, original_tensor),
                "Patches do not stitch back to original image - row/column ordering may be wrong",
            )

    def test_get_number_of_image_patches_aspect_ratio(self):
        """Test that get_number_of_image_patches returns correct count for non-square images.

        This directly tests the row/column unpacking fix by verifying patch counts match
        the expected grid layout. If rows/columns are swapped, the wrong grid would be
        chosen for asymmetric images.
        """
        for image_processing_class in self.image_processor_list:
            patch_size = 64
            image_processor = image_processing_class(
                size={"height": patch_size, "width": patch_size},
                crop_to_patches=True,
                min_patches=1,
                max_patches=12,
            )

            # Test 1: Tall image (4 rows x 1 column) should give 5 patches (4 + thumbnail)
            tall_patches = image_processor.get_number_of_image_patches(
                height=patch_size * 4,  # 256
                width=patch_size,  # 64
                images_kwargs={},
            )
            self.assertEqual(tall_patches, 5, "Tall image (4:1) should produce 5 patches")

            # Test 2: Wide image (1 row x 4 columns) should give 5 patches (4 + thumbnail)
            wide_patches = image_processor.get_number_of_image_patches(
                height=patch_size,  # 64
                width=patch_size * 4,  # 256
                images_kwargs={},
            )
            self.assertEqual(wide_patches, 5, "Wide image (1:4) should produce 5 patches")

            # Test 3: Asymmetric image (2 rows x 3 columns) should give 7 patches
            asym_patches = image_processor.get_number_of_image_patches(
                height=patch_size * 2,  # 128
                width=patch_size * 3,  # 192
                images_kwargs={"max_patches": 6},
            )
            self.assertEqual(asym_patches, 7, "Asymmetric image (2:3) should produce 7 patches")

            # Test 4: Opposite asymmetric (3 rows x 2 columns) should also give 7 patches
            asym_patches2 = image_processor.get_number_of_image_patches(
                height=patch_size * 3,  # 192
                width=patch_size * 2,  # 128
                images_kwargs={"max_patches": 6},
            )
            self.assertEqual(asym_patches2, 7, "Asymmetric image (3:2) should produce 7 patches")
