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

from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, is_vision_available
from transformers.pipelines import ZeroShotImageClassificationPipeline, pipeline
from transformers.testing_utils import require_torch, require_vision


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_vision
@require_torch
class ZeroShotImageClassificationPipelineTests(unittest.TestCase):
    pipeline_task = "zero-shot-image-classification"
    small_models = ["openai/clip-vit-base-patch32"]  # Models tested without the @slow decorator
    simple_inputs = [
        {"images": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        {"images": "./tests/fixtures/tests_samples/COCO/000000039769.png"},
        {"images": Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")},
    ]
    batched_inputs = [
        {
            "images": [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]
        },
        {
            "images": [
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
            ]
        },
        {
            "images": [
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
            ]
        },
        {
            "images": [
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
            ]
        },
    ]

    def test_small_model_from_factory(self):
        for small_model in self.small_models:
            image_classifier = pipeline("zero-shot-image-classification", model=small_model)

            candidate_labels = ["a dog", "a cat"]

            for valid_input in self.simple_inputs:
                output = image_classifier(**valid_input, candidate_labels=candidate_labels)
                self.assertTrue(isinstance(output, list))
                self.assertEqual(len(output), 2)
                for label_result in output:
                    self.assertTrue(isinstance(label_result, dict))
                    self.assertEqual(set(label_result.keys()), {"label", "score"})

            for valid_input in self.batched_inputs:
                output = image_classifier(**valid_input, candidate_labels=candidate_labels)
                self.assertTrue(isinstance(output, list))
                self.assertEqual(len(output), 2)
                for item in output:
                    for label_result in item:
                        self.assertTrue(isinstance(label_result, dict))
                        self.assertEqual(set(label_result.keys()), {"label", "score"})

    def test_small_model_from_pipeline(self):
        for small_model in self.small_models:
            model = AutoModel.from_pretrained(small_model)
            feature_extractor = AutoFeatureExtractor.from_pretrained(small_model)
            tokenizer = AutoTokenizer.from_pretrained(small_model)
            image_classifier = ZeroShotImageClassificationPipeline(
                model=model, feature_extractor=feature_extractor, tokenizer=tokenizer
            )

            candidate_labels = ["a dog", "a cat"]

            for valid_input in self.simple_inputs:
                output = image_classifier(**valid_input, candidate_labels=candidate_labels)
                self.assertTrue(isinstance(output, list))
                self.assertEqual(len(output), 2)
                for label_result in output:
                    self.assertTrue(isinstance(label_result, dict))
                    self.assertEqual(set(label_result.keys()), {"label", "score"})

            for valid_input in self.batched_inputs:
                output = image_classifier(**valid_input, candidate_labels=candidate_labels)
                self.assertTrue(isinstance(output, list))
                self.assertEqual(len(output), 2)
                for item in output:
                    for label_result in item:
                        self.assertTrue(isinstance(label_result, dict))
                        self.assertEqual(set(label_result.keys()), {"label", "score"})
