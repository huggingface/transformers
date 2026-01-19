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


if is_vision_available():
    from PIL import Image


if is_torch_available():
    import torch

    if is_torchvision_available():
        from transformers import Lfm2VlImageProcessorFast
        from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import (
            find_closest_aspect_ratio,
            round_by_factor,
        )


class Lfm2VlImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_images=1,
        min_resolution=256,
        max_resolution=1024,
        downsample_factor=2,
        do_image_splitting=False,
        min_tiles=2,
        max_tiles=10,
        use_thumbnail=True,
        min_image_tokens=64,
        max_image_tokens=256,
        encoder_patch_size=16,
        tile_size=512,
        max_pixels_tolerance=2.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_images = num_images
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

        self.downsample_factor = downsample_factor
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance

    def prepare_image_processor_dict(self):
        return {
            "downsample_factor": self.downsample_factor,
            "do_image_splitting": self.do_image_splitting,
            "min_tiles": self.min_tiles,
            "max_tiles": self.max_tiles,
            "use_thumbnail": self.use_thumbnail,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "encoder_patch_size": self.encoder_patch_size,
            "tile_size": self.tile_size,
            "max_pixels_tolerance": self.max_pixels_tolerance,
        }

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
        return [[image] for image in images]


@require_torch
@require_vision
class Lfm2VlImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    test_slow_image_processor = False
    fast_image_processing_class = Lfm2VlImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Lfm2VlImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "downsample_factor"))
            self.assertTrue(hasattr(image_processing, "min_tiles"))
            self.assertTrue(hasattr(image_processing, "max_tiles"))
            self.assertTrue(hasattr(image_processing, "use_thumbnail"))
            self.assertTrue(hasattr(image_processing, "min_image_tokens"))
            self.assertTrue(hasattr(image_processing, "max_image_tokens"))
            self.assertTrue(hasattr(image_processing, "encoder_patch_size"))
            self.assertTrue(hasattr(image_processing, "tile_size"))
            self.assertTrue(hasattr(image_processing, "max_pixels_tolerance"))

    @require_vision
    def test_smart_resize(self):
        # verify that smart resize output dims are divisible by encoder_patch_size * downsample_factor
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        width, height = image_processing.smart_resize(
            height=500,
            width=300,
            downsample_factor=image_processing.downsample_factor,
            min_image_tokens=image_processing.min_image_tokens,
            max_image_tokens=image_processing.max_image_tokens,
            encoder_patch_size=image_processing.encoder_patch_size,
        )
        mod = image_processing.encoder_patch_size * image_processing.downsample_factor
        self.assertEqual(width % mod, 0)
        self.assertEqual(height % mod, 0)

    @require_vision
    def test_get_grid_layout(self):
        # splitting a 512×512 image into tiles of size processor.image_processor.tile_size
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        rows, cols, _, _, num_patches = image_processing._get_grid_layout(
            height=1024,
            width=1024,
            min_tiles=image_processing.min_tiles,
            max_tiles=image_processing.max_tiles,
            tile_size=image_processing.tile_size,
        )
        self.assertEqual(num_patches, 4)
        self.assertEqual(num_patches, rows * cols)

        rows, cols, _, _, num_patches = image_processing._get_grid_layout(
            height=1024,
            width=1024,
            min_tiles=8,
            max_tiles=8,
            tile_size=image_processing.tile_size,
        )
        self.assertEqual(num_patches, 8)
        self.assertEqual(num_patches, rows * cols)

    def test_find_closest_aspect_ratio(self):
        # should pick (1,1) over (2,1) for a square image
        result = find_closest_aspect_ratio(1.0, [(1, 1), (2, 1)], width=100, height=100, image_size=100)
        self.assertEqual(result, (1, 1))

        result = find_closest_aspect_ratio(0.5, [(1, 1), (1, 2)], width=100, height=200, image_size=200)
        self.assertEqual(result, (1, 2))

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for sample_images in image_inputs:
            for image in sample_images:
                self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_call_numpy_4_channels(self):
        # Lfm2Vl always processes images as RGB, so it always returns images with 3 channels
        # Initialize image_processing
        image_processor_dict = self.image_processor_dict
        image_processing = self.fast_image_processing_class(**image_processor_dict)
        # create random numpy tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        for sample_images in image_inputs:
            for image in sample_images:
                self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for images in image_inputs:
            for image in images:
                self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.fast_image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        for images in image_inputs:
            for image in images:
                self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (1, image_processing.max_num_patches, 3 * image_processing.encoder_patch_size**2),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            tuple(encoded_images.shape),
            (
                self.image_processor_tester.batch_size,
                image_processing.max_num_patches,
                3 * image_processing.encoder_patch_size**2,
            ),
        )

    def test_small_image_no_tiling_no_thumbnail(self):
        """Small image with tiling disabled should use smart resize, no thumbnail."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=False,
            use_thumbnail=True,  # even if enabled, should not be used for small/non-tiled images
        )
        # Create a small image (256x256)
        small_image = Image.new("RGB", (256, 256), color="red")
        result = image_processing([[small_image]], return_tensors="pt", return_row_col_info=True)

        # With tiling disabled, should be 1 tile (no thumbnail)
        self.assertEqual(result.image_rows[0].item(), 1)
        self.assertEqual(result.image_cols[0].item(), 1)
        # Should have exactly 1 image in batch (no thumbnail)
        self.assertEqual(result.pixel_values.shape[0], 1)

    def test_small_image_tiling_enabled_no_thumbnail(self):
        """Small image with tiling enabled should not be tiled (too small), no thumbnail."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=True,
            min_tiles=2,
            max_tiles=10,
        )
        # Create a small image that won't exceed the max_image_tokens threshold
        small_image = Image.new("RGB", (256, 256), color="blue")
        result = image_processing([[small_image]], return_tensors="pt", return_row_col_info=True)

        # Small image should not be tiled (1x1 grid), no thumbnail added
        self.assertEqual(result.image_rows[0].item(), 1)
        self.assertEqual(result.image_cols[0].item(), 1)
        # Should have exactly 1 image in batch (no thumbnail)
        self.assertEqual(result.pixel_values.shape[0], 1)

    def test_large_image_no_tiling_smart_resize(self):
        """Large image with tiling disabled should use smart resize, no thumbnail."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=False,
            use_thumbnail=True,  # even if enabled, should not be used
        )
        # Create a large image (2048x2048)
        large_image = Image.new("RGB", (2048, 2048), color="green")
        result = image_processing([[large_image]], return_tensors="pt", return_row_col_info=True)

        # With tiling disabled, should be 1 tile even for large images
        self.assertEqual(result.image_rows[0].item(), 1)
        self.assertEqual(result.image_cols[0].item(), 1)
        # Should have exactly 1 image in batch (no thumbnail, smart resize only)
        self.assertEqual(result.pixel_values.shape[0], 1)

    def test_large_image_tiling_enabled_thumbnail_disabled(self):
        """Large image with tiling enabled but thumbnail disabled should tile without thumbnail."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=False,
            min_tiles=2,
            max_tiles=10,
            tile_size=512,
        )
        # Create a large image that will require tiling
        large_image = Image.new("RGB", (2048, 2048), color="yellow")
        result = image_processing([[large_image]], return_tensors="pt", return_row_col_info=True)

        # Large image should be tiled into multiple tiles
        num_rows = result.image_rows[0].item()
        num_cols = result.image_cols[0].item()
        num_tiles = num_rows * num_cols
        self.assertGreater(num_tiles, 1, "Large image should be tiled into multiple tiles")

        # Count actual patches - with thumbnail disabled, should equal number of tiles
        num_images_in_batch = result.pixel_values.shape[0]
        self.assertEqual(
            num_images_in_batch, num_tiles, "Number of patches should equal number of tiles (no thumbnail)"
        )

    def test_large_image_tiling_enabled_thumbnail_enabled(self):
        """Large image with tiling and thumbnail enabled should tile AND add thumbnail."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=True,
            min_tiles=2,
            max_tiles=10,
            tile_size=512,
        )
        # Create a large image that will require tiling
        large_image = Image.new("RGB", (2048, 2048), color="purple")
        result = image_processing([[large_image]], return_tensors="pt", return_row_col_info=True)

        # Large image should be tiled into multiple tiles
        num_rows = result.image_rows[0].item()
        num_cols = result.image_cols[0].item()
        num_tiles = num_rows * num_cols
        self.assertGreater(num_tiles, 1, "Large image should be tiled into multiple tiles")

        # With thumbnail enabled, we should have tiles + 1 thumbnail
        num_images_in_batch = result.pixel_values.shape[0]
        self.assertEqual(num_images_in_batch, num_tiles + 1, "Number of patches should equal tiles + 1 (thumbnail)")

    # ==================== Non-Square Aspect Ratio Tests ====================

    def test_landscape_image_aspect_ratio(self):
        """Test that landscape images (wider than tall) are processed correctly."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=True,
            min_tiles=2,
            max_tiles=10,
            tile_size=512,
        )
        # Create a landscape image (1920x1080, ~16:9 aspect ratio)
        landscape_image = Image.new("RGB", (1920, 1080), color="blue")
        result = image_processing([[landscape_image]], return_tensors="pt", return_row_col_info=True)

        num_rows = result.image_rows[0].item()
        num_cols = result.image_cols[0].item()

        # Landscape image should have more columns than rows
        self.assertGreaterEqual(num_cols, num_rows, "Landscape image should have cols >= rows")

    def test_extreme_aspect_ratio_wide(self):
        """Test extremely wide image (panorama-like)."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=True,
            min_tiles=2,
            max_tiles=10,
            tile_size=512,
        )
        # Create an extremely wide image (3000x500)
        wide_image = Image.new("RGB", (3000, 500), color="red")
        result = image_processing([[wide_image]], return_tensors="pt", return_row_col_info=True)

        num_rows = result.image_rows[0].item()
        num_cols = result.image_cols[0].item()

        # Very wide image should have significantly more cols than rows
        self.assertGreater(num_cols, num_rows, "Very wide image should have cols > rows")

    def test_extreme_aspect_ratio_tall(self):
        """Test extremely tall image."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=True,
            min_tiles=2,
            max_tiles=10,
            tile_size=512,
        )
        # Create an extremely tall image (500x3000)
        tall_image = Image.new("RGB", (500, 3000), color="yellow")
        result = image_processing([[tall_image]], return_tensors="pt", return_row_col_info=True)

        num_rows = result.image_rows[0].item()
        num_cols = result.image_cols[0].item()

        # Very tall image should have significantly more rows than cols
        self.assertGreater(num_rows, num_cols, "Very tall image should have rows > cols")

    # ==================== Output Validation Tests ====================

    def test_image_sizes_returned_with_row_col_info(self):
        """Test that image_sizes is returned when return_row_col_info=True."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)
        image = Image.new("RGB", (512, 256), color="green")
        result = image_processing([[image]], return_tensors="pt", return_row_col_info=True)

        # Check all row/col info is returned
        self.assertIn("image_rows", result)
        self.assertIn("image_cols", result)
        self.assertIn("image_sizes", result)

        # image_sizes should contain [height, width] for the resized image
        image_sizes = result.image_sizes
        self.assertIsInstance(image_sizes, torch.Tensor)
        self.assertEqual(image_sizes.shape[0], 1)  # one sample
        self.assertEqual(image_sizes.shape[1], 2)  # [height, width]

    def test_output_consistency_across_formats(self):
        """Test that outputs are consistent regardless of input format (PIL, numpy, torch)."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        # Create same image in different formats
        pil_image = Image.new("RGB", (256, 256), color="white")
        np_image = np.array(pil_image)
        torch_image = torch.from_numpy(np_image).permute(2, 0, 1)

        result_pil = image_processing([[pil_image]], return_tensors="pt")
        result_np = image_processing([[np_image]], return_tensors="pt")
        result_torch = image_processing([[torch_image]], return_tensors="pt")

        # All should produce same shapes
        self.assertEqual(result_pil.pixel_values.shape, result_np.pixel_values.shape)
        self.assertEqual(result_pil.pixel_values.shape, result_torch.pixel_values.shape)
        self.assertEqual(result_pil.spatial_shapes.tolist(), result_np.spatial_shapes.tolist())
        self.assertEqual(result_pil.spatial_shapes.tolist(), result_torch.spatial_shapes.tolist())

    # ==================== Multiple Images Per Sample Tests ====================

    def test_multiple_images_per_sample(self):
        """Test processing multiple images in a single sample: [[img1, img2, img3]]."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        img1 = Image.new("RGB", (256, 256), color="red")
        img2 = Image.new("RGB", (256, 256), color="green")
        img3 = Image.new("RGB", (256, 256), color="blue")

        result = image_processing([[img1, img2, img3]], return_tensors="pt")

        # Should have 3 images processed
        self.assertEqual(result.pixel_values.shape[0], 3)
        self.assertEqual(result.spatial_shapes.shape[0], 3)
        self.assertEqual(result.pixel_attention_mask.shape[0], 3)

    def test_mixed_image_counts_across_batch(self):
        """Test batch with different number of images per sample: [[img1], [img2, img3]]."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        img1 = Image.new("RGB", (256, 256), color="red")
        img2 = Image.new("RGB", (256, 256), color="green")
        img3 = Image.new("RGB", (256, 256), color="blue")

        # First sample has 1 image, second sample has 2 images
        result = image_processing([[img1], [img2, img3]], return_tensors="pt")

        # Total should be 3 images (1 + 2)
        self.assertEqual(result.pixel_values.shape[0], 3)
        self.assertEqual(result.spatial_shapes.shape[0], 3)

    def test_multiple_images_different_sizes(self):
        """Test multiple images per sample with different sizes."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        img_small = Image.new("RGB", (256, 256), color="red")
        img_medium = Image.new("RGB", (512, 512), color="green")
        img_large = Image.new("RGB", (768, 768), color="blue")

        result = image_processing([[img_small, img_medium, img_large]], return_tensors="pt")

        # Should have 3 images processed
        self.assertEqual(result.pixel_values.shape[0], 3)
        # All should have same max_num_patches due to padding
        self.assertEqual(result.pixel_values.shape[1], image_processing.max_num_patches)

    # ==================== Parameter Variations Tests ====================

    def test_forced_grid_config_min_equals_max(self):
        """Test forcing a specific grid configuration with min_tiles == max_tiles."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            min_tiles=4,
            max_tiles=4,  # Force exactly 4 tiles
            tile_size=512,
            use_thumbnail=False,
        )
        # Large image that would normally get more tiles
        wide_image = Image.new("RGB", (3000, 500), color="red")
        result = image_processing([[wide_image]], return_tensors="pt", return_row_col_info=True)

        num_rows = result.image_rows[0].item()
        num_cols = result.image_cols[0].item()
        num_tiles = num_rows * num_cols

        # Should be exactly 4 tiles
        self.assertEqual(num_tiles, 4, "Should have exactly 4 tiles when min_tiles == max_tiles == 4")

    # ==================== Input Validation Tests ====================

    def test_min_tiles_greater_than_max_tiles_raises_error(self):
        """Test that min_tiles > max_tiles raises ValueError."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            min_tiles=10,
            max_tiles=2,  # Invalid: min > max
        )
        image = Image.new("RGB", (1024, 1024), color="red")

        with self.assertRaises(ValueError) as context:
            image_processing([[image]], return_tensors="pt")

        self.assertIn("min_tiles", str(context.exception).lower())

    # ==================== Edge Case Images Tests ====================

    def test_very_small_image(self):
        """Test image smaller than encoder_patch_size."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=False,
            encoder_patch_size=16,
        )
        # Image smaller than patch size
        tiny_image = Image.new("RGB", (8, 8), color="red")
        result = image_processing([[tiny_image]], return_tensors="pt")

        # Should still process without error
        self.assertIn("pixel_values", result)
        self.assertEqual(result.pixel_values.dim(), 3)

    def test_grayscale_image(self):
        """Test that grayscale (1-channel) images are converted to RGB."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        # Create grayscale image
        grayscale_image = Image.new("L", (256, 256), color=128)
        result = image_processing([[grayscale_image]], return_tensors="pt")

        # Should process and output 3 channels (converted to RGB)
        self.assertIn("pixel_values", result)
        # pixel_values shape is (batch, num_patches, patch_size^2 * 3)
        expected_patch_dim = 3 * image_processing.encoder_patch_size**2
        self.assertEqual(result.pixel_values.shape[2], expected_patch_dim)

    def test_rgba_4_channel_image(self):
        """Test that RGBA (4-channel) images are converted to RGB."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        # Create RGBA image with alpha channel
        rgba_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        result = image_processing([[rgba_image]], return_tensors="pt", do_convert_rgb=True)

        # Should process and output 3 channels (alpha dropped)
        self.assertIn("pixel_values", result)
        expected_patch_dim = 3 * image_processing.encoder_patch_size**2
        self.assertEqual(result.pixel_values.shape[2], expected_patch_dim)

    def test_numpy_4_channel_rgba(self):
        """Test actual 4-channel numpy array input - convert to PIL for RGB conversion."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        # Create 4-channel numpy array (RGBA) and convert to PIL Image for RGB conversion
        rgba_np = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        rgba_pil = Image.fromarray(rgba_np, mode="RGBA")
        result = image_processing([[rgba_pil]], return_tensors="pt", do_convert_rgb=True)

        # Should convert to 3 channels
        self.assertIn("pixel_values", result)
        expected_patch_dim = 3 * image_processing.encoder_patch_size**2
        self.assertEqual(result.pixel_values.shape[2], expected_patch_dim)

    def test_single_pixel_image(self):
        """Test 1x1 pixel image (extreme edge case)."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        single_pixel = Image.new("RGB", (1, 1), color="blue")
        result = image_processing([[single_pixel]], return_tensors="pt")

        # Should process without error
        self.assertIn("pixel_values", result)

    # ==================== Helper Function Unit Tests ====================

    def test_round_by_factor(self):
        """Test round_by_factor function."""
        # Exact multiples should return themselves
        self.assertEqual(round_by_factor(32, 16), 32)
        self.assertEqual(round_by_factor(64, 16), 64)

        # Values should round to nearest multiple
        self.assertEqual(round_by_factor(30, 16), 32)  # 30 -> 32 (closer to 32 than 16)
        self.assertEqual(round_by_factor(20, 16), 16)  # 20 -> 16 (closer to 16 than 32)
        self.assertEqual(round_by_factor(24, 16), 32)  # 24 -> 32 (equidistant, rounds up)

        # Test with different factors
        self.assertEqual(round_by_factor(100, 32), 96)  # 100 -> 96
        self.assertEqual(round_by_factor(50, 32), 64)  # 50 -> 64

        # Test with factor of 1
        self.assertEqual(round_by_factor(17, 1), 17)

    def test_is_image_too_large_small_image(self):
        """Test _is_image_too_large with small image."""
        image_processing = self.fast_image_processing_class(
            max_image_tokens=256,
            encoder_patch_size=16,
            downsample_factor=2,
            max_pixels_tolerance=2.0,
        )

        is_large = image_processing._is_image_too_large(
            height=512,
            width=512,
            max_image_tokens=256,
            encoder_patch_size=16,
            downsample_factor=2,
            max_pixels_tolerance=2.0,
        )
        self.assertFalse(is_large)

    def test_is_image_too_large_large_image(self):
        """Test _is_image_too_large with large image."""
        image_processing = self.fast_image_processing_class(
            max_image_tokens=256,
            encoder_patch_size=16,
            downsample_factor=2,
            max_pixels_tolerance=1.0,
        )

        is_large = image_processing._is_image_too_large(
            height=565,
            width=565,
            max_image_tokens=256,
            encoder_patch_size=16,
            downsample_factor=2,
            max_pixels_tolerance=1.0,
        )
        self.assertTrue(is_large)

    # ==================== Batch Processing Tests ====================

    def test_batch_mixed_image_sizes(self):
        """Test batch processing with different image sizes requiring different processing paths."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        # Create images with significantly different sizes
        small_image = Image.new("RGB", (256, 256), color="red")
        medium_image = Image.new("RGB", (512, 512), color="green")
        large_image = Image.new("RGB", (1024, 1024), color="blue")

        # Process as batch
        result = image_processing([[small_image], [medium_image], [large_image]], return_tensors="pt")

        # All should be processed and padded to same size
        self.assertEqual(result.pixel_values.shape[0], 3)
        # All should have same max_num_patches
        self.assertEqual(result.pixel_values.shape[1], image_processing.max_num_patches)
        # Patch dimension should be patch_size^2 * 3 channels
        expected_patch_dim = 3 * image_processing.encoder_patch_size**2
        self.assertEqual(result.pixel_values.shape[2], expected_patch_dim)

        # Spatial shapes should all be square (equal height and width)
        shapes = result.spatial_shapes.tolist()
        for shape in shapes:
            self.assertEqual(shape[0], shape[1], "Square images should have equal height and width")

        # pixel_attention_mask should have correct shape
        self.assertEqual(result.pixel_attention_mask.shape[0], 3)
        self.assertEqual(result.pixel_attention_mask.shape[1], image_processing.max_num_patches)

    def test_batch_mixed_aspect_ratios(self):
        """Test batch with mixed aspect ratios."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        square = Image.new("RGB", (512, 512), color="red")
        landscape = Image.new("RGB", (1024, 512), color="green")
        portrait = Image.new("RGB", (512, 1024), color="blue")

        result = image_processing([[square], [landscape], [portrait]], return_tensors="pt")

        # All should be processed
        self.assertEqual(result.pixel_values.shape[0], 3)
        self.assertEqual(result.spatial_shapes.shape[0], 3)

        # Spatial shapes should reflect aspect ratios: [height, width]
        shapes = result.spatial_shapes.tolist()
        square_shape, landscape_shape, portrait_shape = shapes

        # Square: height == width
        self.assertEqual(square_shape[0], square_shape[1], "Square image should have equal spatial dimensions")

        # Landscape: width > height
        self.assertGreater(landscape_shape[1], landscape_shape[0], "Landscape image should have width > height")

        # Portrait: height > width
        self.assertGreater(portrait_shape[0], portrait_shape[1], "Portrait image should have height > width")

        # pixel_attention_mask should match batch size and max_num_patches
        self.assertEqual(result.pixel_attention_mask.shape[0], 3)
        self.assertEqual(result.pixel_attention_mask.shape[1], image_processing.max_num_patches)

    def test_disable_grouping_single_image(self):
        """Test disable_grouping parameter with single image."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        image = Image.new("RGB", (512, 512), color="purple")

        # Process with and without disable_grouping
        result_grouped = image_processing([[image]], return_tensors="pt", disable_grouping=False)
        result_ungrouped = image_processing([[image]], return_tensors="pt", disable_grouping=True)

        # Both should produce all expected output keys
        for result in [result_grouped, result_ungrouped]:
            self.assertIn("pixel_values", result)
            self.assertIn("spatial_shapes", result)
            self.assertIn("pixel_attention_mask", result)

        # Both should have same output shapes for single image
        self.assertEqual(result_grouped.pixel_values.shape, result_ungrouped.pixel_values.shape)
        self.assertEqual(result_grouped.spatial_shapes.shape, result_ungrouped.spatial_shapes.shape)
        self.assertEqual(result_grouped.pixel_attention_mask.shape, result_ungrouped.pixel_attention_mask.shape)

        # Verify specific shapes
        self.assertEqual(result_ungrouped.pixel_values.shape[0], 1)
        self.assertEqual(result_ungrouped.pixel_values.shape[1], image_processing.max_num_patches)
        expected_patch_dim = 3 * image_processing.encoder_patch_size**2
        self.assertEqual(result_ungrouped.pixel_values.shape[2], expected_patch_dim)

    def test_disable_grouping_batch(self):
        """Test disable_grouping parameter with batch of images."""
        image_processing = self.fast_image_processing_class(do_image_splitting=False)

        # Images of same size - normally would be grouped
        img1 = Image.new("RGB", (256, 256), color="red")
        img2 = Image.new("RGB", (256, 256), color="green")
        img3 = Image.new("RGB", (256, 256), color="blue")

        # Process with disable_grouping=True
        result = image_processing([[img1], [img2], [img3]], return_tensors="pt", disable_grouping=True)

        # Should produce valid output for all images
        self.assertEqual(result.pixel_values.shape[0], 3)

    def test_batch_with_tiling(self):
        """Test batch processing when some images need tiling."""
        image_processing = self.fast_image_processing_class(
            do_image_splitting=True,
            use_thumbnail=True,
            min_tiles=2,
            max_tiles=4,
            tile_size=512,
        )

        # Small image (no tiling needed) and large image (will be tiled)
        small = Image.new("RGB", (256, 256), color="red")
        large = Image.new("RGB", (1024, 1024), color="blue")  # 2x2 tiles at 512

        result = image_processing([[small], [large]], return_tensors="pt", return_row_col_info=True)

        # Calculate tiles for each image
        small_tiles = result.image_rows[0].item() * result.image_cols[0].item()
        large_tiles = result.image_rows[1].item() * result.image_cols[1].item()

        # Small image: single tile (no splitting needed)
        self.assertEqual(small_tiles, 1, "Small 256x256 image should have 1 tile (no splitting)")

        # Large image: 2x2 = 4 tiles for 1024x1024 with tile_size=512
        self.assertEqual(large_tiles, 4, "Large 1536x1536 image should have 4 tiles (2x2)")

        # Total images: small (1) + large tiles (4) + thumbnail for large (1) = 6
        # Thumbnail is only added when there's more than 1 tile
        expected_total = 1 + 4 + 1  # small + large_tiles + large_thumbnail
        self.assertEqual(result.pixel_values.shape[0], expected_total)
        self.assertEqual(result.spatial_shapes.shape[0], expected_total)
        self.assertEqual(result.pixel_attention_mask.shape[0], expected_total)
