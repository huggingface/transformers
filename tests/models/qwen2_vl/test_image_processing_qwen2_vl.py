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

import itertools
import tempfile
import unittest

import numpy as np
import requests

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import Qwen2VLImageProcessor

    # if is_torchvision_available():
    #     from transformers import Qwen2VLImageProcessorFast


class Qwen2VLImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=10,
        min_resolution=56,
        max_resolution=1024,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_resize=True,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_mean = OPENAI_CLIP_MEAN
        self.image_std = OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
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

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class Qwen2VLImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Qwen2VLImageProcessor if is_vision_available() else None
    # fast_image_processing_class = Qwen2VLImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Qwen2VLImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "min_pixels"))
            self.assertTrue(hasattr(image_processing, "max_pixels"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "temporal_patch_size"))
            self.assertTrue(hasattr(image_processing, "merge_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.min_pixels, 56 * 56)
            self.assertEqual(image_processor.max_pixels, 28 * 28 * 1280)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, min_pixels=256 * 256, max_pixels=640 * 640
            )
            self.assertEqual(image_processor.min_pixels, 256 * 256)
            self.assertEqual(image_processor.max_pixels, 640 * 640)

    def test_select_best_resolution(self):
        # Test with a final resize resolution
        best_resolution = smart_resize(561, 278, factor=28)
        self.assertEqual(best_resolution, (560, 280))

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], Image.Image)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], np.ndarray)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image[0], torch.Tensor)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    @unittest.skip(reason="Qwen2VLImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_input(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched as a nested list of images, where each sublist is one batch
            image_inputs_nested = image_inputs[:3] + image_inputs[3:]
            process_out = image_processing(image_inputs_nested, return_tensors="pt")
            encoded_images_nested = process_out.pixel_values
            image_grid_thws_nested = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Image processor should return same pixel values, independently of ipnut format
            self.assertTrue((encoded_images_nested == encoded_images).all())
            self.assertTrue((image_grid_thws_nested == expected_image_grid_thws).all())

    def test_video_inputs(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            expected_dims_by_frames = {1: 34300, 2: 34300, 3: 68600, 4: 68600, 5: 102900, 6: 102900}

            for num_frames, expected_dims in expected_dims_by_frames.items():
                image_processor_tester = Qwen2VLImageProcessingTester(self, num_frames=num_frames)
                video_inputs = image_processor_tester.prepare_video_inputs(equal_resolution=True)
                process_out = image_processing(None, videos=video_inputs, return_tensors="pt")
                encoded_video = process_out.pixel_values_videos
                expected_output_video_shape = (expected_dims, 1176)
                self.assertEqual(tuple(encoded_video.shape), expected_output_video_shape)

    def test_custom_patch_size(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)

            for patch_size in (1, 3, 5, 7):
                image_processor_tester = Qwen2VLImageProcessingTester(self, patch_size=patch_size)
                video_inputs = image_processor_tester.prepare_video_inputs(equal_resolution=True)
                process_out = image_processing(None, videos=video_inputs, return_tensors="pt")
                encoded_video = process_out.pixel_values_videos
                expected_output_video_shape = (171500, 1176)
                self.assertEqual(tuple(encoded_video.shape), expected_output_video_shape)

    def test_custom_image_size(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                image_processing.save_pretrained(tmpdirname)
                image_processor_loaded = image_processing_class.from_pretrained(
                    tmpdirname, max_pixels=56 * 56, min_pixels=28 * 28
                )

            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            process_out = image_processor_loaded(image_inputs, return_tensors="pt")
            expected_output_video_shape = [112, 1176]
            self.assertListEqual(list(process_out.pixel_values.shape), expected_output_video_shape)

    def test_custom_pixels(self):
        pixel_choices = frozenset(itertools.product((100, 150, 200, 20000), (100, 150, 200, 20000)))
        for image_processing_class in self.image_processor_list:
            image_processor_dict = self.image_processor_dict.copy()
            for a_pixels, b_pixels in pixel_choices:
                image_processor_dict["min_pixels"] = min(a_pixels, b_pixels)
                image_processor_dict["max_pixels"] = max(a_pixels, b_pixels)
                image_processor = image_processing_class(**image_processor_dict)
                image_inputs = self.image_processor_tester.prepare_image_inputs()
                # Just checking that it doesn't raise an error
                image_processor(image_inputs, return_tensors="pt")

    def test_temporal_padding(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # Create random video inputs with a number of frames not divisible by temporal_patch_size
            image_processor_tester = Qwen2VLImageProcessingTester(self, num_frames=5, temporal_patch_size=4)
            video_inputs = image_processor_tester.prepare_video_inputs(equal_resolution=True)

            # Process the video inputs
            process_out = image_processing(None, videos=video_inputs, return_tensors="pt")
            encoded_video = process_out.pixel_values_videos

            # Check the shape after padding
            expected_output_video_shape = (102900, 1176)  # Adjusted based on padding
            self.assertEqual(tuple(encoded_video.shape), expected_output_video_shape)
            # Check divisibility by temporal_patch_size
            self.assertEqual(encoded_video.shape[0] % 4, 0)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )

        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
