# coding=utf-8
# Copyright 2023 The Intel Team Authors, The HuggingFace Inc. team. All rights reserved.
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
from typing import Dict, List, Optional, Union

import numpy as np

from transformers.image_transforms import PaddingMode
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import TvpImageProcessor


class TvpImageProcessingTester:
    def __init__(
        self,
        parent,
        do_resize: bool = True,
        size: Dict[str, int] = {"longest_edge": 40},
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = False,
        rescale_factor: Union[int, float] = 1 / 255,
        do_pad: bool = True,
        pad_size: Dict[str, int] = {"height": 80, "width": 80},
        fill: int = None,
        pad_mode: PaddingMode = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = [0.48145466, 0.4578275, 0.40821073],
        image_std: Optional[Union[float, List[float]]] = [0.26862954, 0.26130258, 0.27577711],
        batch_size=2,
        min_resolution=40,
        max_resolution=80,
        num_channels=3,
        num_frames=2,
    ):
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.fill = fill
        self.pad_mode = pad_mode
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_frames = num_frames

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "do_center_crop": self.do_center_crop,
            "do_pad": self.do_pad,
            "pad_size": self.pad_size,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to TvpImageProcessor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            return (int(self.pad_size["height"]), int(self.pad_size["width"]))

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class TvpImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = TvpImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = TvpImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "do_pad"))
        self.assertTrue(hasattr(image_processing, "pad_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"longest_edge": 40})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size={"longest_edge": 12})
        self.assertEqual(image_processor.size, {"longest_edge": 12})

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL videos
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], Image.Image)

        # Test not batched input
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs)
        encoded_videos = image_processing(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        # Test batched
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs, batched=True)
        encoded_videos = image_processing(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)

        # Test not batched input
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs)
        encoded_videos = image_processing(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        # Test batched
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs, batched=True)
        encoded_videos = image_processing(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_call_numpy_4_channels(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)

        # Test not batched input
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs)
        encoded_videos = image_processing(
            video_inputs[0], return_tensors="pt", image_mean=0, image_std=1, input_data_format="channels_first"
        ).pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        # Test batched
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs, batched=True)
        encoded_videos = image_processing(
            video_inputs, return_tensors="pt", image_mean=0, image_std=1, input_data_format="channels_first"
        ).pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )
        self.image_processor_tester.num_channels = 3

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False, torchify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], torch.Tensor)

        # Test not batched input
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs)
        encoded_videos = image_processing(video_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                1,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        # Test batched
        expected_height, expected_width = self.image_processor_tester.get_expected_values(video_inputs, batched=True)
        encoded_videos = image_processing(video_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_videos.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_frames,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )
