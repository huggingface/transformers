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
from packaging import version
from parameterized import parameterized

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_torch_gpu, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import VideoLlavaImageProcessor

    if is_torchvision_available():
        from transformers import VideoLlavaImageProcessorFast


class VideoLlavaImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"shortest_edge": 20}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.expected_output_image_shape
    def expected_output_image_shape(self, images):
        return self.num_channels, self.crop_size["height"], self.crop_size["width"]

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.prepare_image_inputs
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

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        # let's simply copy the frames to fake a long video-clip
        if numpify or torchify:
            videos = []
            for image in images:
                if numpify:
                    video = image[None, ...].repeat(8, 0)
                else:
                    video = image[None, ...].repeat(8, 1, 1, 1)
                videos.append(video)
        else:
            videos = []
            for pil_image in images:
                videos.append([pil_image] * 8)

        return videos


@require_torch
@require_vision
class VideoLlavaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = VideoLlavaImageProcessor if is_vision_available() else None
    fast_image_processing_class = VideoLlavaImageProcessorFast if is_torchvision_available() else None

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.setUp with CLIP->VideoLlava
    def setUp(self):
        super().setUp()
        self.image_processor_tester = VideoLlavaImageProcessingTester(self)

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_center_crop"))
            self.assertTrue(hasattr(image_processing, "center_crop"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.test_image_processor_from_dict_with_kwargs
    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 20})
            self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    def test_call_pil(self):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values_images
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values_images
            expected_output_image_shape = (5, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy(self):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(images=image_inputs[0], return_tensors="pt").pixel_values_images
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(images=image_inputs, return_tensors="pt").pixel_values_images
            expected_output_image_shape = (5, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_numpy_videos(self):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            video_inputs = self.image_processor_tester.prepare_video_inputs(numpify=True, equal_resolution=True)
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            # Test not batched input
            encoded_videos = image_processing(
                images=None, videos=video_inputs[0], return_tensors="pt"
            ).pixel_values_videos
            expected_output_video_shape = (1, 8, 3, 18, 18)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = image_processing(
                images=None, videos=video_inputs, return_tensors="pt"
            ).pixel_values_videos
            expected_output_video_shape = (5, 8, 3, 18, 18)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_pil_videos(self):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # the inputs come in list of lists batched format
            video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True)
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = image_processing(
                images=None, videos=video_inputs[0], return_tensors="pt"
            ).pixel_values_videos
            expected_output_video_shape = (1, 8, 3, 18, 18)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = image_processing(
                images=None, videos=video_inputs, return_tensors="pt"
            ).pixel_values_videos
            expected_output_video_shape = (5, 8, 3, 18, 18)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch(self):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values_images
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values_images
            expected_output_image_shape = (5, 3, 18, 18)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

    def test_call_pytorch_videos(self):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, torchify=True)
            for video in video_inputs:
                self.assertIsInstance(video, torch.Tensor)

            # Test not batched input
            encoded_videos = image_processing(
                images=None, videos=video_inputs[0], return_tensors="pt"
            ).pixel_values_videos
            expected_output_video_shape = (1, 8, 3, 18, 18)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

            # Test batched
            encoded_videos = image_processing(
                images=None, videos=video_inputs, return_tensors="pt"
            ).pixel_values_videos
            expected_output_video_shape = (5, 8, 3, 18, 18)
            self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    @parameterized.expand([(True, False), (False, True)])
    def test_call_mixed(self, numpify, torchify):
        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(
                equal_resolution=True, numpify=numpify, torchify=torchify
            )
            video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, torchify=torchify)

            # Test not batched input
            encoded = image_processing(images=image_inputs[0], videos=video_inputs[0], return_tensors="pt")
            expected_output_video_shape = (1, 8, 3, 18, 18)
            expected_output_image_shape = (1, 3, 18, 18)
            self.assertEqual(tuple(encoded.pixel_values_videos.shape), expected_output_video_shape)
            self.assertEqual(tuple(encoded.pixel_values_images.shape), expected_output_image_shape)

            # Test batched
            encoded = image_processing(images=image_inputs, videos=video_inputs, return_tensors="pt")
            expected_output_video_shape = (5, 8, 3, 18, 18)
            expected_output_image_shape = (5, 3, 18, 18)
            self.assertEqual(tuple(encoded.pixel_values_videos.shape), expected_output_video_shape)
            self.assertEqual(tuple(encoded.pixel_values_images.shape), expected_output_image_shape)

    def test_call_numpy_4_channels(self):
        # Test that can process images which have an arbitrary number of channels
        # Initialize image_processing
        image_processor = self.image_processing_class(**self.image_processor_dict)

        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        # Test not batched input
        encoded_images = image_processor(
            image_inputs[0],
            return_tensors="pt",
            input_data_format="channels_last",
            image_mean=0,
            image_std=1,
        ).pixel_values_images
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processor(
            image_inputs,
            return_tensors="pt",
            input_data_format="channels_last",
            image_mean=0,
            image_std=1,
        ).pixel_values_images
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(
            tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        """
        Use the correct attribute for testing `pixel_values_images`
        """
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")
        self.assertTrue(
            torch.allclose(encoding_slow.pixel_values_images, encoding_fast.pixel_values_images, atol=1e-1)
        )
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow.pixel_values_images - encoding_fast.pixel_values_images)).item(), 1e-3
        )

    @slow
    @require_torch_gpu
    @require_vision
    def test_can_compile_fast_image_processor(self):
        if self.fast_image_processing_class is None:
            self.skipTest("Skipping compilation test as fast image processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.fast_image_processing_class(**self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")

        torch.testing.assert_close(
            output_eager.pixel_values_images, output_compiled.pixel_values_images, rtol=1e-4, atol=1e-4
        )
