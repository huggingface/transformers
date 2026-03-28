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
from transformers.utils import is_torch_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch


class DeepseekOcr2ImageProcessingTester:
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
    def setUp(self):
        super().setUp()
        self.image_processor_tester = DeepseekOcr2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
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
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            tile_size = self.image_processor_tester.tile_size
            if backend_name == "pil":
                image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)[0]
                processed_images = image_processor.crop_image_to_patches(
                    image, min_patches=1, max_patches=6, tile_size=tile_size
                )
                self.assertGreater(len(processed_images), 0)
                self.assertEqual(processed_images[0].shape[:2], (tile_size, tile_size))
            else:
                image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)[0]
                stacked_patches, n_patches = image_processor.crop_image_to_patches(
                    image.unsqueeze(0).float(), min_patches=1, max_patches=6, tile_size=tile_size
                )
                self.assertGreater(n_patches, 0)
                self.assertEqual(stacked_patches.shape[-2:], (tile_size, tile_size))

    def test_preprocess_global_only(self):
        """Test preprocessing without crop_to_patches (global view only)."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict, crop_to_patches=False)
            images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=False)
            result = image_processor(images, return_tensors="pt")
            self.assertIn("pixel_values", result)
            self.assertEqual(len(result["num_local_patches"]), len(images))
            for n in result["num_local_patches"]:
                self.assertEqual(n, 0)

    def test_preprocess_with_crop_to_patches(self):
        """Test preprocessing with crop_to_patches enabled."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict, crop_to_patches=True)
            images = prepare_image_inputs(
                batch_size=2, num_channels=3, min_resolution=500, max_resolution=700, equal_resolution=True
            )
            result = image_processor(images, return_tensors="pt")
            self.assertIn("pixel_values", result)
            has_local = any(n > 0 for n in result["num_local_patches"])
            self.assertTrue(has_local)
            if has_local:
                self.assertIn("pixel_values_local", result)

    def test_backends_equivalence(self):
        """Override to also compare pixel_values_local and num_local_patches."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)[0]

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(
                encodings[reference_backend].pixel_values, encodings[backend_name].pixel_values
            )
            torch.testing.assert_close(
                encodings[reference_backend].num_local_patches, encodings[backend_name].num_local_patches
            )
            if encodings[reference_backend].get("pixel_values_local") is not None:
                self._assert_tensors_equivalence(
                    encodings[reference_backend].pixel_values_local,
                    encodings[backend_name].pixel_values_local,
                )

    def test_backends_equivalence_batched(self):
        """Override to also compare pixel_values_local and num_local_patches (variable shape)."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors=None)

        backend_names = list(encodings.keys())
        reference_backend = "pil"
        ref_encoding = encodings[reference_backend]

        for backend_name in [b for b in backend_names if b != reference_backend]:
            other_encoding = encodings[backend_name]
            # Global views
            for i in range(len(ref_encoding.pixel_values)):
                self._assert_tensors_equivalence(
                    torch.from_numpy(ref_encoding.pixel_values[i]), other_encoding.pixel_values[i]
                )
            # num_local_patches
            self.assertEqual(
                list(ref_encoding["num_local_patches"]),
                list(other_encoding["num_local_patches"]),
            )
            # Local patches
            ref_local = ref_encoding.get("pixel_values_local")
            other_local = other_encoding.get("pixel_values_local")
            if ref_local is not None and other_local is not None:
                self.assertEqual(len(ref_local), len(other_local))
                for i in range(len(ref_local)):
                    self._assert_tensors_equivalence(torch.from_numpy(ref_local[i]), other_local[i])
