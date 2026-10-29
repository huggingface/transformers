# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

import math
import unittest
from unittest.mock import patch

import numpy as np

from transformers.image_utils import get_image_size
from transformers.testing_utils import require_cv2, require_torch, require_vision
from transformers.utils import import_utils, is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image, ImageDraw


class MiniCPMOcrImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        # Random test inputs kwargs
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("min_resolution", 64)
        kwargs.setdefault("max_resolution", 128)

        # Image processor init kwargs
        kwargs.setdefault("scale_resolution", 448)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("downsample_mode", "4x")

        super().__init__(**kwargs)

    def expected_output_image_shape(self, image_inputs):
        """Return the expected NaViT-packed shape [C, P, total_L] for pixel_values[0]."""
        total_L = 0
        for image in image_inputs:
            if isinstance(image, Image.Image):
                height, width = image.height, image.width
            else:
                height, width = get_image_size(image)

            aspect_ratio = width / height
            new_height = int(self.scale_resolution / math.sqrt(aspect_ratio))
            new_width = int(new_height * aspect_ratio)

            # MiniCPM-OCR aligns the resize grid to `patch_size * 2` in `"4x"` mode and
            # `patch_size * 4` in `"16x"` mode, matching the number of 2x2 spatial merges.
            divisor = self.patch_size * (2 if self.downsample_mode == "4x" else 4)
            best_height = max(round(new_height / divisor) * divisor, divisor)
            best_width = max(round(new_width / divisor) * divisor, divisor)

            total_L += best_height * best_width // self.patch_size

        return [self.num_channels, self.patch_size, total_L]


@require_torch
@require_vision
class MiniCPMOcrImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = MiniCPMOcrImageProcessingTester

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_shape))

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_shape))

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_shape))

    @unittest.skip("NaViT expected_output_image_shape cannot infer channel dim for 4-channel images")
    def test_call_numpy_4_channels(self):
        pass

    def test_downsample_modes(self):
        for image_processing_class in self.image_processing_classes.values():
            images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            processor_16x = image_processing_class(**{**self.image_processor_dict, "downsample_mode": "16x"})
            result_16x = processor_16x(images, return_tensors="pt")

            processor_4x = image_processing_class(**{**self.image_processor_dict, "downsample_mode": "4x"})
            result_4x = processor_4x(images, return_tensors="pt")

            for target_sizes_4x, target_sizes_16x in zip(result_4x["target_sizes"], result_16x["target_sizes"]):
                height_4x, width_4x = target_sizes_4x
                height_16x, width_16x = target_sizes_16x
                self.assertGreaterEqual(height_4x * width_4x, height_16x * width_16x)

    def test_resize_alignment_follows_downsample_mode(self):
        """`"4x"` aligns the resize grid to `patch_size * 2`, `"16x"` to `patch_size * 4`."""
        for image_processing_class in self.image_processing_classes.values():
            patch_size = 14
            # 2 * patch_size = 28 is a valid "4x" grid but not a valid "16x" one (needs 56).
            image = Image.new("RGB", (28 * 3, 28 * 3))

            processor_4x = image_processing_class(
                slice_mode=False, scale_resolution=84, patch_size=patch_size, downsample_mode="4x"
            )
            target_sizes_4x = processor_4x(image, return_tensors="pt")["target_sizes"]
            for height, width in target_sizes_4x.tolist():
                self.assertEqual(height % 2, 0)
                self.assertEqual(width % 2, 0)

            processor_16x = image_processing_class(
                slice_mode=False, scale_resolution=84, patch_size=patch_size, downsample_mode="16x"
            )
            target_sizes_16x = processor_16x(image, return_tensors="pt")["target_sizes"]
            for height, width in target_sizes_16x.tolist():
                self.assertEqual(height % 4, 0)
                self.assertEqual(width % 4, 0)

    def test_crop_to_patches_aspect_ratio(self):
        """Test that row/column ordering is correct when cropping non-square images to patches"""
        for image_processing_class in self.image_processing_classes.values():
            patch_size = 14
            image_processor = image_processing_class(slice_mode=True, max_slice_nums=7, scale_resolution=10)

            for num_rows, num_cols in [(2, 3), (3, 2), (1, 6), (6, 1)]:
                image_height = patch_size * num_rows
                image_width = patch_size * num_cols
                test_image = Image.new("RGB", (image_width, image_height))
                result = image_processor(test_image, return_tensors="pt")

                # Should produce 7 patches (6 grid patches + 1 thumbnail)
                self.assertEqual(result.grids, [[num_rows, num_cols]])
                self.assertEqual(tuple(result.pixel_values.shape), (1, 3, patch_size, 392))

    @staticmethod
    def _dense_text_image(width=200, height=80, num_glyphs=40, glyph_height=8):
        """Small, dense, text-like glyphs: the `"448"` tier."""
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        rng = np.random.default_rng(0)
        for _ in range(num_glyphs):
            left = int(rng.integers(0, max(1, width - 6)))
            top = int(rng.integers(0, max(1, height - glyph_height)))
            draw.rectangle([left, top, left + 3, top + glyph_height], fill=(0, 0, 0))
        return image

    @staticmethod
    def _tiny_image(width=80, height=60):
        """Few large blobs below `TINY_PIXELS`: the `"84"` tier.

        Noise would not do: hundreds of 1-2px components read as small dense text.
        """
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle([4, 6, 32, 45], fill=(0, 0, 0))
        draw.rectangle([44, 9, 73, 48], fill=(0, 0, 0))
        return image

    def test_small_image_upsize_never_keeps_native_resolution(self):
        """Small images stay at their native size, aligned to the patch grid only."""
        for image_processing_class in self.image_processing_classes.values():
            image = self._tiny_image()
            upsizing = image_processing_class(slice_mode=False, downsample_mode="4x")
            native = image_processing_class(slice_mode=False, downsample_mode="4x", small_image_upsize="never")
            upsized_area = np.prod(upsizing(image, return_tensors="pt")["target_sizes"].numpy(), axis=1).sum()
            native_area = np.prod(native(image, return_tensors="pt")["target_sizes"].numpy(), axis=1).sum()
            self.assertGreater(upsized_area, native_area)

    @require_cv2
    def test_adaptive_upsize_upscales_small_dense_text(self):
        """Small dense text is upscaled even though `"never"` keeps small images native."""
        for image_processing_class in self.image_processing_classes.values():
            image = self._dense_text_image()
            native = image_processing_class(slice_mode=False, downsample_mode="4x", small_image_upsize="never")
            adaptive = image_processing_class(slice_mode=False, downsample_mode="4x", small_image_upsize="adaptive")
            native_area = np.prod(native(image, return_tensors="pt")["target_sizes"].numpy(), axis=1).sum()
            adaptive_area = np.prod(adaptive(image, return_tensors="pt")["target_sizes"].numpy(), axis=1).sum()
            self.assertGreater(adaptive_area, native_area)

    @require_cv2
    def test_adaptive_upsize_resolution_tiers(self):
        """The classifier maps dense text to `scale_resolution`, tiny images to `TINY_RESOLUTION`, the rest to
        native resolution."""
        from transformers.models.minicpm_ocr.image_processing_minicpm_ocr import (
            TINY_RESOLUTION,
            adaptive_upsize_resolution,
        )

        scale_resolution = self.image_processor_tester.scale_resolution
        blank = Image.new("RGB", (200, 120), (255, 255, 255))
        for image, expected in [
            (self._dense_text_image(), scale_resolution),
            (self._tiny_image(), TINY_RESOLUTION),
            (blank, None),
        ]:
            resolution = adaptive_upsize_resolution(
                np.array(image).transpose(2, 0, 1).astype(np.uint8), scale_resolution
            )
            self.assertEqual(resolution, expected)

    @require_cv2
    def test_adaptive_tile_target_sizes_match_pixel_values(self):
        """Per-tile adaptive resolution must keep `target_sizes` in sync with `pixel_values`."""
        for image_processing_class in self.image_processing_classes.values():
            # 800x800 with a dense-text top half and a blank bottom half: with the default
            # `scale_resolution=448` this slices into a 2x2 grid whose cells classify as
            # `"448"` (dense text -> baseline tile) and `"native"` (blank -> smaller tile).
            image = Image.new("RGB", (800, 800), (255, 255, 255))
            image.paste(self._dense_text_image(width=800, height=400, num_glyphs=900), (0, 0))
            patch_size = self.image_processor_tester.patch_size

            adaptive = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "slice_mode": True,
                    "small_image_upsize": "adaptive",
                }
            )
            result = adaptive(image, return_tensors="pt")
            rows = result["target_sizes"].tolist()

            self.assertEqual(result["grids"], [[2, 2]])
            self.assertEqual(len(rows), 5)  # one source row + one row per tile
            # Dense-text tiles keep the baseline grid while blank tiles keep their native
            # (smaller) grid, so the rows are not uniform.
            self.assertEqual(len({tuple(row) for row in rows}), 2)
            tile_areas = [height * width for height, width in rows[1:]]
            self.assertLess(min(tile_areas), max(tile_areas))
            # The claimed patch count matches what `pixel_values` actually carries.
            self.assertEqual(
                sum(height * width for height, width in rows),
                result["pixel_values"].shape[-1] // patch_size,
            )

            # Without adaptive tiling every row is the baseline grid and the invariant
            # holds the same way.
            plain = image_processing_class(**{**self.image_processor_dict, "slice_mode": True})
            result = plain(image, return_tensors="pt")
            rows = result["target_sizes"].tolist()
            self.assertEqual(len({tuple(row) for row in rows}), 1)
            self.assertEqual(
                sum(height * width for height, width in rows),
                result["pixel_values"].shape[-1] // patch_size,
            )

    def test_adaptive_upsize_requires_cv2(self):
        """Without OpenCV the feature raises instead of silently using a different rule."""
        for name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(
                slice_mode=False,
                downsample_mode="4x",
                small_image_upsize="adaptive",
            )

            missing_cv2 = {"cv2": (lambda: False, import_utils.BACKENDS_MAPPING["cv2"][1])}
            with patch.dict(import_utils.BACKENDS_MAPPING, missing_cv2):
                with self.assertRaises(ImportError, msg=f"{name} should require OpenCV"):
                    image_processor(self._dense_text_image(), return_tensors="pt")
