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

from tests.test_pipelines_common import CustomInputPipelineCommonMixin
from transformers import Pipeline, is_torch_available, is_vision_available
from transformers.pipelines import pipeline
from transformers.testing_utils import _run_slow_tests, require_torch, require_vision


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_vision
@require_torch
class ImageClassificationPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "image-classification"
    small_models = ["lysandre/tiny-vit-random"]  # Models tested without the @slow decorator
    valid_inputs = [
        {"images": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        {
            "images": [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]
        },
        {"images": "tests/fixtures/coco.jpg"},
        {"images": ["tests/fixtures/coco.jpg", "tests/fixtures/coco.jpg"]},
        {"images": Image.open("tests/fixtures/coco.jpg")},
        {"images": [Image.open("tests/fixtures/coco.jpg"), Image.open("tests/fixtures/coco.jpg")]},
    ]

    def setUp(self) -> None:
        if not is_torch_available():
            return  # Currently not available in JAX or TF

        # Download needed checkpoints
        models = self.small_models
        if _run_slow_tests:
            models = models + self.large_models

        for model_name in models:
            if is_torch_available():
                pipeline(
                    self.pipeline_task,
                    model=model_name,
                    feature_extractor=model_name,
                    framework="pt",
                    **self.pipeline_loading_kwargs,
                )

    def _test_pipeline(self, image_classifier: Pipeline):
        image_classifier(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://farm2.staticflickr.com/1319/1165793046_3484e87167_z.jpg",
            ]
        )
