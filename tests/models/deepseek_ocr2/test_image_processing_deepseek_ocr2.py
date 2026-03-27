# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import DeepseekOcr2ImageProcessor
    from transformers.image_utils import PILImageResampling

    if is_torchvision_available():
        from transformers import DeepseekOcr2ImageProcessorFast


class DeepseekOcr2ImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=500,
        max_resolution=800,
        do_resize=True,
        size=None,
        tile_size=384,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
    ):
        super().__init__()
        size = size if size is not None else {"height": 512, "width": 512}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.tile_size = tile_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "tile_size": self.tile_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
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
class DeepseekOcr2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = DeepseekOcr2ImageProcessor if is_vision_available() else None
    fast_image_processing_class = DeepseekOcr2ImageProcessorFast if is_vision_available() and is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = DeepseekOcr2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "tile_size"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass

    def test_crop_to_patches(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)[0]
        processed_images, (num_cols, num_rows) = image_processor.crop_image_to_patches(
            image,
            min_patches=1,
            max_patches=6,
            tile_size={"height": self.image_processor_tester.tile_size, "width": self.image_processor_tester.tile_size},
        )
        self.assertGreater(len(processed_images), 0)
        # Patches are returned in channels-last format (H, W, C) for numpy input
        self.assertEqual(processed_images[0].shape[0], self.image_processor_tester.tile_size)
        self.assertEqual(processed_images[0].shape[1], self.image_processor_tester.tile_size)

    def test_preprocess_global_only(self):
        """Test preprocessing without crop_to_patches (global view only)."""
        image_processor = self.image_processing_class(**self.image_processor_dict, crop_to_patches=False)
        images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=False)
        result = image_processor(images, return_tensors="pt")
        self.assertIn("pixel_values", result)
        self.assertEqual(len(result["num_local_patches"]), len(images))
        # Without crop_to_patches, all num_local_patches should be 0
        for n in result["num_local_patches"]:
            self.assertEqual(n, 0)

    def test_preprocess_with_crop_to_patches(self):
        """Test preprocessing with crop_to_patches enabled."""
        image_processor = self.image_processing_class(**self.image_processor_dict, crop_to_patches=True)
        # Use larger images to trigger local patch extraction (must be > tile_size)
        images = prepare_image_inputs(
            batch_size=2,
            num_channels=3,
            min_resolution=500,
            max_resolution=700,
            equal_resolution=True,
        )
        result = image_processor(images, return_tensors="pt")
        self.assertIn("pixel_values", result)
        # With large images and crop_to_patches, should have local patches
        has_local = any(n > 0 for n in result["num_local_patches"])
        self.assertTrue(has_local)
        if has_local:
            self.assertIn("pixel_values_local", result)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        # Use BICUBIC for slow to match fast (torchvision doesn't support LANCZOS for tensors)
        slow_dict = {**self.image_processor_dict, "resample": PILImageResampling.BICUBIC}
        image_processor_slow = self.image_processing_class(**slow_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        # Single large image (has local patches, > tile_size)
        dummy_images = prepare_image_inputs(
            batch_size=1, num_channels=3, min_resolution=500, max_resolution=700, equal_resolution=True
        )
        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.pixel_values_local, encoding_fast.pixel_values_local
        )
        self.assertTrue(
            torch.equal(encoding_slow.num_local_patches, encoding_fast.num_local_patches),
            "num_local_patches mismatch",
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        slow_dict = {**self.image_processor_dict, "resample": PILImageResampling.BICUBIC}
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**slow_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors=None)
        encoding_fast = image_processor_fast(dummy_images, return_tensors=None)

        # Global views: compare per-image (sizes may vary)
        for i in range(len(encoding_slow.pixel_values)):
            self._assert_slow_fast_tensors_equivalence(
                torch.from_numpy(encoding_slow.pixel_values[i]), encoding_fast.pixel_values[i]
            )

        # num_local_patches
        s_nlp = encoding_slow["num_local_patches"]
        f_nlp = encoding_fast["num_local_patches"]
        self.assertEqual(list(s_nlp), list(f_nlp), "num_local_patches mismatch")

        # Local patches (flat list)
        s_local = encoding_slow.get("pixel_values_local")
        f_local = encoding_fast.get("pixel_values_local")
        if s_local is not None and f_local is not None:
            self.assertEqual(len(s_local), len(f_local), "local patch count mismatch")
            for i in range(len(s_local)):
                self._assert_slow_fast_tensors_equivalence(
                    torch.from_numpy(s_local[i]), f_local[i]
                )
