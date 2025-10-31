# coding=utf-8
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
import requests

from transformers.models.fast.image_processing_fast import compute_min_area_rect, get_box_points
from transformers.testing_utils import require_cv2, require_torch, require_vision
from transformers.utils import is_cv2_available, is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import FastForSceneTextRecognition, FastImageProcessor

if is_cv2_available():
    import cv2


class FastImageProcessingTester(unittest.TestCase):
    # satisfy error of unittest.TestCase.__hash__ for pytest collection
    _testMethodName = None

    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        min_area: int = 200,
        min_score: float = 0.88,
        bounding_box_type: str = "boxes",
        pooling_size: int = 9,
    ):
        size = size if size is not None else {"height": 20, "width": 20}
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
        self.min_area = min_area
        self.min_score = min_score
        self.bounding_box_type = bounding_box_type
        self.pooling_size = pooling_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_area": self.min_area,
            "min_score": self.min_score,
            "bounding_box_type": self.bounding_box_type,
            "pooling_size": self.pooling_size,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.crop_size["height"], self.crop_size["width"]

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
class FastImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = FastImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = FastImageProcessingTester(self)
        self.image_processor_list = []

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "center_crop"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 20, "width": 20})
        self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size=42, crop_size=84, reduce_labels=True
        )
        self.assertEqual(image_processor.size, {"shortest_edge": 42})
        self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    # @slow
    def test_post_process_text_detection(self):
        model = FastForSceneTextRecognition.from_pretrained("jadechoghari/fast_tiny_ckpt")

        image_processor = FastImageProcessor.from_pretrained("jadechoghari/fast_tiny_ckpt")

        def prepare_image():
            image_url = "https://huggingface.co/datasets/Raghavan/fast_model_samples/resolve/main/img657.jpg"
            raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            return raw_image

        image = prepare_image()
        inputs = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            output = model(pixel_values=inputs["pixel_values"])
        target_sizes = [(image.height, image.width)]
        threshold = 0.88
        final_out = image_processor.post_process_text_detection(output, target_sizes, threshold, output_type="boxes")
        self.assertListEqual(final_out[0]["boxes"][0], [(148, 151), (157, 53), (357, 72), (347, 170)])

        self.assertAlmostEqual(float(final_out[0]["scores"][0]), 0.92573, places=5)

    @require_cv2
    def test_get_box_points_against_cv2(self):
        for _ in range(10):
            np.random.seed(42)
            points = (np.random.rand(10, 2) * 100).astype(np.float32)

            my_rect = compute_min_area_rect(points)
            my_box = get_box_points(my_rect)

            cv2_rect = cv2.minAreaRect(points)
            cv2_box = cv2.boxPoints(cv2_rect)

            my_box_sorted = np.array(sorted(my_box, key=lambda x: (x[0], x[1])))
            cv2_box_sorted = np.array(sorted(cv2_box, key=lambda x: (x[0], x[1])))

            assert np.allclose(my_box_sorted, cv2_box_sorted, atol=5), (
                f"Box points mismatch:\n{my_box_sorted}\nvs\n{cv2_box_sorted}"
            )
