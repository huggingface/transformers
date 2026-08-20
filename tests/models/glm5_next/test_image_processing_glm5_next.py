# Copyright 2026 the HuggingFace Team. All rights reserved.
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

if is_torchvision_available():
    from transformers import Glm5NextImageProcessor
    from transformers.models.glm5_next.image_processing_glm5_next import smart_resize

if is_vision_available():
    from PIL import Image


class Glm5NextImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        temporal_patch_size=2,
        patch_size=2,
        merge_size=2,
        patch_expand_factor=2,
        min_image_tokens=1,
        max_image_tokens=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens

    def prepare_image_processor_dict(self):
        return {
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "patch_expand_factor": self.patch_expand_factor,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
        }

    def expected_output_image_shape(self, images):
        hidden_dim = self.num_channels * self.temporal_patch_size * self.patch_size**2
        pixels_per_token = self.temporal_patch_size * (self.patch_size * self.merge_size) ** 2
        min_pixels = self.min_image_tokens * pixels_per_token
        max_pixels = self.max_image_tokens * pixels_per_token
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        seq_len = 0
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                height, width = image.shape[-2:]
            resized_height, resized_width = smart_resize(
                self.temporal_patch_size,
                height,
                width,
                temporal_factor=self.temporal_patch_size,
                height_factor=factor,
                width_factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            seq_len += (resized_height // self.patch_size) * (resized_width // self.patch_size)
        return [seq_len, hidden_dim]

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
class Glm5NextImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Glm5NextImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # images are flattened into (seq_len, hidden_dim), so batched calls concatenate instead of stacking
    def _test_call(self, image_inputs, **preprocess_kwargs):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            encoded_images = image_processing(image_inputs[0], return_tensors="pt", **preprocess_kwargs).pixel_values
            self.assertEqual(
                list(encoded_images.shape),
                self.image_processor_tester.expected_output_image_shape([image_inputs[0]]),
            )
            encoded_images = image_processing(image_inputs, return_tensors="pt", **preprocess_kwargs).pixel_values
            self.assertEqual(
                list(encoded_images.shape),
                self.image_processor_tester.expected_output_image_shape(image_inputs),
            )

    def test_call_pil(self):
        self._test_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False))

    def test_call_numpy(self):
        self._test_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True))

    def test_call_pytorch(self):
        self._test_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True))

    def test_call_numpy_4_channels(self):
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        self._test_call(
            image_inputs,
            input_data_format="channels_last",
            image_mean=(0.0, 0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0, 1.0),
        )

        # per-channel normalization: each channel block in the patch dim must be scaled by its own std
        tester = self.image_processor_tester
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(
                **self.image_processor_dict, do_convert_rgb=False, do_resize=False
            )
            image = np.zeros((16, 16, 4), dtype=np.uint8)
            for channel, value in enumerate((255, 128, 51, 0)):
                image[..., channel] = value
            output = image_processing(
                image,
                input_data_format="channels_last",
                image_mean=(0.0, 0.0, 0.0, 0.0),
                image_std=(1.0, 2.0, 4.0, 8.0),
                return_tensors="pt",
            ).pixel_values
            patch_dim = tester.temporal_patch_size * tester.patch_size**2
            for channel, expected in enumerate((1.0, 128 / 255 / 2, 51 / 255 / 4, 0.0)):
                block = output[:, channel * patch_dim : (channel + 1) * patch_dim]
                torch.testing.assert_close(block, torch.full_like(block, expected))

    def test_smart_resize_reference_cases(self):
        cases = [
            ((2, 5, 9, 2, 8, 8, 32, 2048), (8, 16)),
            ((2, 5, 9, 2, 8, 8, 512, 2048), (16, 24)),
            ((2, 37, 53, 2, 8, 8, 32, 256), (8, 16)),
        ]
        for args, expected in cases:
            self.assertEqual(smart_resize(*args), expected)

    # the padded canvas is always a multiple of patch_size * merge_size in each dimension
    def test_canvas_is_aligned_to_token_grid(self):
        processor = Glm5NextImageProcessor(**self.image_processor_dict)
        image = torch.randint(0, 256, (3, 60, 100), dtype=torch.uint8)
        output = processor(image, return_tensors="pt")
        _, grid_h, grid_w = output.image_grid_thw[0].tolist()
        factor = self.image_processor_tester.patch_size * self.image_processor_tester.merge_size
        self.assertEqual(grid_h * self.image_processor_tester.patch_size % factor, 0)
        self.assertEqual(grid_w * self.image_processor_tester.patch_size % factor, 0)
        self.assertEqual(
            grid_h * grid_w,
            processor.get_number_of_image_patches(60, 100),
        )

    def test_min_image_tokens_upscales_small_images(self):
        kwargs = {**self.image_processor_dict, "min_image_tokens": 64, "max_image_tokens": 256}
        processor = Glm5NextImageProcessor(**kwargs)
        output = processor(torch.zeros(3, 5, 9, dtype=torch.uint8), return_tensors="pt")
        tokens = output.image_grid_thw[0].prod().item() // self.image_processor_tester.merge_size**2
        self.assertGreaterEqual(tokens, 64)

    def test_per_call_overrides(self):
        kwargs = self.image_processor_dict
        processor = Glm5NextImageProcessor(**kwargs)
        image = torch.arange(3 * 37 * 53, dtype=torch.uint8).reshape(3, 37, 53)
        overrides = {"max_image_tokens": 8, "patch_expand_factor": 1}
        actual = processor(image, return_tensors="pt", **overrides)
        expected = Glm5NextImageProcessor(**(kwargs | overrides))(image, return_tensors="pt")
        self.assertTrue(torch.equal(actual.pixel_values, expected.pixel_values))
        self.assertTrue(torch.equal(actual.image_grid_thw, expected.image_grid_thw))
