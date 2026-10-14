# Copyright 2026 HuggingFace Inc.
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

from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    if is_torchvision_available():
        from transformers import VideomtVideoProcessor

    if is_torch_available():
        from transformers.models.videomt.modeling_videomt import VideomtForUniversalSegmentationOutput


class VideomtVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_center_crop=False,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        do_convert_rgb=True,
        num_queries=3,
        num_classes=2,
    ):
        super().__init__()
        size = size if size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.num_queries = num_queries
        self.num_classes = num_classes

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_video_shape(self, videos):
        return self.num_frames, self.num_channels, self.size["height"], self.size["width"]

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )

    def prepare_fake_videomt_outputs(self, num_frames):
        height, width = self.size["height"], self.size["width"]
        return VideomtForUniversalSegmentationOutput(
            masks_queries_logits=torch.randn((num_frames, self.num_queries, height, width)),
            class_queries_logits=torch.randn((num_frames, self.num_queries, self.num_classes + 1)),
        )


@require_torch
@require_vision
@require_torchvision
class VideomtVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = VideomtVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = VideomtVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_properties(self):
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        self.assertTrue(hasattr(video_processing, "do_resize"))
        self.assertTrue(hasattr(video_processing, "size"))
        self.assertTrue(hasattr(video_processing, "do_center_crop"))
        self.assertTrue(hasattr(video_processing, "do_normalize"))
        self.assertTrue(hasattr(video_processing, "image_mean"))
        self.assertTrue(hasattr(video_processing, "image_std"))
        self.assertTrue(hasattr(video_processing, "do_convert_rgb"))
        self.assertTrue(hasattr(video_processing, "model_input_names"))
        self.assertIn("pixel_values_videos", video_processing.model_input_names)

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"height": 20, "width": 20})

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, size=42)
        self.assertEqual(video_processor.size, {"height": 42, "width": 42})

    def test_post_process_semantic_segmentation(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)

        num_frames = 4
        target_sizes = [(32, 32)] * num_frames
        outputs = self.video_processor_tester.prepare_fake_videomt_outputs(num_frames)

        segmentation = video_processor.post_process_semantic_segmentation(outputs, target_sizes)

        self.assertEqual(len(segmentation), num_frames)
        for seg_map in segmentation:
            self.assertIsInstance(seg_map, torch.Tensor)
            self.assertEqual(seg_map.shape, (32, 32))

    def test_post_process_instance_segmentation(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)

        num_frames = 4
        target_sizes = [(32, 32)] * num_frames
        outputs = self.video_processor_tester.prepare_fake_videomt_outputs(num_frames)

        results = video_processor.post_process_instance_segmentation(outputs, target_sizes)

        self.assertEqual(len(results), num_frames)
        for el in results:
            self.assertIn("segmentation", el)
            self.assertIn("segments_info", el)
            self.assertIsInstance(el["segments_info"], list)
            self.assertEqual(el["segmentation"].shape, (32, 32))

    def test_post_process_panoptic_segmentation(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)

        num_frames = 4
        target_sizes = [(32, 32)] * num_frames
        outputs = self.video_processor_tester.prepare_fake_videomt_outputs(num_frames)

        results = video_processor.post_process_panoptic_segmentation(outputs, target_sizes)

        self.assertEqual(len(results), num_frames)
        for el in results:
            self.assertIn("segmentation", el)
            self.assertIn("segments_info", el)
            self.assertIsInstance(el["segments_info"], list)
            self.assertEqual(el["segmentation"].shape, (32, 32))
