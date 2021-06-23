# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import AutoFeatureExtractor, DetrForSegmentation, is_vision_available
from transformers.pipelines import ImageSegmentationPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, require_timm, require_torch, require_vision


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_timm
@require_vision
@require_torch
@is_pipeline_test
class ImageSegmentationPipelineTests(unittest.TestCase):
    pipeline_task = "image-segmentation"
    small_models = ["facebook/detr-resnet-50-panoptic"]  # Models tested without the @slow decorator

    def test_small_model_from_factory(self):
        for small_model in self.small_models:

            image_classifier = pipeline("image-segmentation", model=small_model)

            output = image_classifier("http://images.cocodataset.org/val2017/000000039769.jpg")
            self.assertEqual(len(output), 6)
            self.assertEqual(set(output[0].keys()), {"mask", "score", "label"})
            self.assertEqual(type(output[0]["score"]), float)
            self.assertEqual(type(output[0]["label"]), str)
            self.assertEqual(type(output[0]["mask"]), np.ndarray)

            output = image_classifier(
                images=[
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                    Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                ]
            )
            self.assertEqual(len(output), 2)
            self.assertEqual(len(output[0]), 6)
            self.assertEqual(set(output[0][0].keys()), {"mask", "score", "label"})
            self.assertEqual(type(output[0][0]["score"]), float)
            self.assertEqual(type(output[0][0]["label"]), str)
            self.assertEqual(type(output[0][0]["mask"]), np.ndarray)

    def test_small_model_from_pipeline(self):
        for small_model in self.small_models:

            model = DetrForSegmentation.from_pretrained(small_model)
            feature_extractor = AutoFeatureExtractor.from_pretrained(small_model)
            image_classifier = ImageSegmentationPipeline(model=model, feature_extractor=feature_extractor)

            output = image_classifier("http://images.cocodataset.org/val2017/000000039769.jpg")
            self.assertEqual(len(output), 6)
            self.assertEqual(set(output[0].keys()), {"mask", "score", "label"})
            self.assertEqual(type(output[0]["score"]), float)
            self.assertEqual(type(output[0]["label"]), str)
            self.assertEqual(type(output[0]["mask"]), np.ndarray)

            output = image_classifier(
                images=[
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                    Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                ]
            )
            self.assertEqual(len(output), 2)
            self.assertEqual(len(output[0]), 6)
            self.assertEqual(set(output[0][0].keys()), {"mask", "score", "label"})
            self.assertEqual(type(output[0][0]["score"]), float)
            self.assertEqual(type(output[0][0]["label"]), str)
            self.assertEqual(type(output[0][0]["mask"]), np.ndarray)
