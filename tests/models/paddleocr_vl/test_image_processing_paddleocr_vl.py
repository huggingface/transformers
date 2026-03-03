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

import itertools
import json
import tempfile
import unittest

import numpy as np

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.models.paddleocr_vl.image_processing_paddleocr_vl import smart_resize
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class PaddleOCRVLImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=56,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        temporal_patch_size=1,
        patch_size=14,
        merge_size=2,
        do_convert_rgb=True,
    ):
        # Use small pixel bounds so tests run quickly with small images
        size = size if size is not None else {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
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
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "min_pixels": self.size["shortest_edge"],
            "max_pixels": self.size["longest_edge"],
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_image_shape(self, images):
        """
        Returns the expected pixel_values shape for a batch of images.
        PaddleOCRVL outputs patches of shape (N_patches_total, C, patch_size, patch_size).
        """
        seq_len = 0
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] <= 4:
                    # channels-last: (H, W, C)
                    height, width = image.shape[:2]
                else:
                    # channels-first: (C, H, W)
                    height, width = image.shape[-2:]
            elif is_torch_available() and isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]
            else:
                height, width = self.min_resolution, self.min_resolution

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.size["shortest_edge"],
                max_pixels=self.size["longest_edge"],
            )
            grid_h = resized_height // self.patch_size
            grid_w = resized_width // self.patch_size
            seq_len += grid_h * grid_w  # temporal_patch_size=1, so grid_t=1

        return (seq_len, self.num_channels, self.patch_size, self.patch_size)

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
class PaddleOCRVLImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = PaddleOCRVLImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "temporal_patch_size"))
            self.assertTrue(hasattr(image_processing, "merge_size"))

    def test_image_processor_to_json_string(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            obj = json.loads(image_processor.to_json_string())
            for key, value in self.image_processor_dict.items():
                # min_pixels/max_pixels are stored as size in the config
                if key not in ["min_pixels", "max_pixels"]:
                    self.assertEqual(obj[key], value)

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(
                image_processor.size,
                {
                    "shortest_edge": self.image_processor_dict["min_pixels"],
                    "longest_edge": self.image_processor_dict["max_pixels"],
                },
            )

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, min_pixels=28 * 28, max_pixels=56 * 56
            )
            self.assertEqual(image_processor.size, {"shortest_edge": 28 * 28, "longest_edge": 56 * 56})

    def test_select_best_resolution(self):
        best_resolution = smart_resize(561, 278, factor=28)
        self.assertEqual(best_resolution, (560, 280))

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Single image
            encoded = image_processing(image_inputs[0], return_tensors="pt")
            expected_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded.pixel_values.shape), expected_shape)
            self.assertEqual(encoded.image_grid_thw.shape, (1, 3))

            # Batched
            encoded = image_processing(image_inputs, return_tensors="pt")
            expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded.pixel_values.shape), expected_shape)
            self.assertEqual(encoded.image_grid_thw.shape, (self.image_processor_tester.batch_size, 3))

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Single image
            encoded = image_processing(image_inputs[0], return_tensors="pt")
            expected_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded.pixel_values.shape), expected_shape)
            self.assertEqual(encoded.image_grid_thw.shape, (1, 3))

            # Batched
            encoded = image_processing(image_inputs, return_tensors="pt")
            expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded.pixel_values.shape), expected_shape)
            self.assertEqual(encoded.image_grid_thw.shape, (self.image_processor_tester.batch_size, 3))

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Single image
            encoded = image_processing(image_inputs[0], return_tensors="pt")
            expected_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded.pixel_values.shape), expected_shape)
            self.assertEqual(encoded.image_grid_thw.shape, (1, 3))

            # Batched
            encoded = image_processing(image_inputs, return_tensors="pt")
            expected_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded.pixel_values.shape), expected_shape)
            self.assertEqual(encoded.image_grid_thw.shape, (self.image_processor_tester.batch_size, 3))

    def test_call_equal_resolution(self):
        """With equal-resolution images, the batched output shapes are fully deterministic."""
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            # equal_resolution=True → all images are max_resolution × max_resolution = 80×80
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # smart_resize(80, 80, factor=28, min_pixels=56*56, max_pixels=28*28*1280) → (84, 84)
            # grid_h = grid_w = 84 / 14 = 6, N_per_image = 36
            expected_n_patches_per_image = 6 * 6
            batch_size = self.image_processor_tester.batch_size

            process_out = image_processing(image_inputs, return_tensors="pt")
            self.assertEqual(
                tuple(process_out.pixel_values.shape),
                (batch_size * expected_n_patches_per_image, 3, 14, 14),
            )
            expected_grid = torch.tensor([[1, 6, 6]] * batch_size)
            self.assertTrue((process_out.image_grid_thw == expected_grid).all())

    @unittest.skip(reason="PaddleOCRVLImageProcessor converts to RGB, 4-channel images not consistently supported")
    def test_call_numpy_4_channels(self):
        pass

    def test_custom_image_size(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                image_processing.save_pretrained(tmpdirname)
                image_processor_loaded = image_processing_class.from_pretrained(
                    tmpdirname, max_pixels=56 * 56, min_pixels=28 * 28
                )

            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            # equal_resolution=True → all images are max_resolution × max_resolution = 80×80
            # smart_resize(80, 80, factor=28, min_pixels=28*28=784, max_pixels=56*56=3136):
            #   h_bar=84, 84*84=7056 > 3136, so reduce:
            #   beta = sqrt(80*80/3136) = sqrt(2.041) = 1.429
            #   h_bar = floor(80/1.429/28)*28 = floor(2.0)*28 = 2*28 = 56
            #   grid_h=4, grid_w=4 → N=16 per image
            process_out = image_processor_loaded(image_inputs, return_tensors="pt")
            expected_n = 16  # 4*4 grid (56/14 = 4)
            self.assertEqual(process_out.pixel_values.shape[0], self.image_processor_tester.batch_size * expected_n)
            self.assertEqual(process_out.pixel_values.shape[1:], (3, 14, 14))

    def test_custom_pixels(self):
        # Use pixel values >= 784 (28*28) to avoid smart_resize producing 0-size outputs
        # for images in the 56x80 px range used in the tester (factor=28 requires output >= 28px)
        pixel_choices = frozenset(itertools.product((1000, 5000, 50000), (1000, 5000, 50000)))
        for image_processing_class in self.image_processing_classes.values():
            image_processor_dict = self.image_processor_dict.copy()
            for a_pixels, b_pixels in pixel_choices:
                image_processor_dict["min_pixels"] = min(a_pixels, b_pixels)
                image_processor_dict["max_pixels"] = max(a_pixels, b_pixels)
                image_processor = image_processing_class(**image_processor_dict)
                image_inputs = self.image_processor_tester.prepare_image_inputs()
                # Just verify no error is raised
                image_processor(image_inputs, return_tensors="pt")

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(image_inputs, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
            self.assertEqual(reference_encoding.image_grid_thw.dtype, encodings[backend_name].image_grid_thw.dtype)
            self._assert_tensors_equivalence(
                reference_encoding.image_grid_thw.float(), encodings[backend_name].image_grid_thw.float()
            )

    @require_vision
    @require_torch
    def test_backends_equivalence_batched(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
            self._assert_tensors_equivalence(
                reference_encoding.image_grid_thw.float(), encodings[backend_name].image_grid_thw.float()
            )

    def test_get_num_patches_without_images(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            # 100×100 → smart_resize(100, 100, factor=28, min_pixels=56*56, max_pixels=28*28*1280)
            # h_bar=112, w_bar=112, grid_h=8, grid_w=8 → 64 patches
            num_patches = image_processing.get_number_of_image_patches(height=100, width=100, images_kwargs={})
            self.assertEqual(num_patches, 64)

            # 200×50 → h_bar=196, w_bar=56, grid_h=14, grid_w=4 → 56 patches
            num_patches = image_processing.get_number_of_image_patches(height=200, width=50, images_kwargs={})
            self.assertEqual(num_patches, 56)

            # With custom patch_size=28 → factor=28*2=56
            # 100×100 → smart_resize(100, 100, factor=56, min_pixels=56*56, max_pixels=28*28*1280)
            # h_bar=round(100/56)*56=2*56=112, grid_h=112/28=4, grid_w=4 → 16 patches
            num_patches = image_processing.get_number_of_image_patches(
                height=100, width=100, images_kwargs={"patch_size": 28}
            )
            self.assertEqual(num_patches, 16)
