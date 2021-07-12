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

from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    PreTrainedTokenizer,
    is_vision_available,
)
from transformers.pipelines import ImageClassificationPipeline, pipeline
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
class ImageClassificationPipelineTests(unittest.TestCase):
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
        {"images": [Image.open("tests/fixtures/coco.jpg"), "tests/fixtures/coco.jpg"]},
    ]

    def test_small_model_from_factory(self):
        for small_model in self.small_models:

            image_classifier = pipeline("image-classification", model=small_model)

            for valid_input in self.valid_inputs:
                output = image_classifier(**valid_input)
                top_k = valid_input.get("top_k", 5)

                def assert_valid_pipeline_output(pipeline_output):
                    self.assertTrue(isinstance(pipeline_output, list))
                    self.assertEqual(len(pipeline_output), top_k)
                    for label_result in pipeline_output:
                        self.assertTrue(isinstance(label_result, dict))
                        self.assertIn("label", label_result)
                        self.assertIn("score", label_result)

                if isinstance(valid_input["images"], list):
                    self.assertEqual(len(valid_input["images"]), len(output))
                    for individual_output in output:
                        assert_valid_pipeline_output(individual_output)
                else:
                    assert_valid_pipeline_output(output)

    def test_small_model_from_pipeline(self):
        for small_model in self.small_models:

            model = AutoModelForImageClassification.from_pretrained(small_model)
            feature_extractor = AutoFeatureExtractor.from_pretrained(small_model)
            image_classifier = ImageClassificationPipeline(model=model, feature_extractor=feature_extractor)

            for valid_input in self.valid_inputs:
                output = image_classifier(**valid_input)
                top_k = valid_input.get("top_k", 5)

                def assert_valid_pipeline_output(pipeline_output):
                    self.assertTrue(isinstance(pipeline_output, list))
                    self.assertEqual(len(pipeline_output), top_k)
                    for label_result in pipeline_output:
                        self.assertTrue(isinstance(label_result, dict))
                        self.assertIn("label", label_result)
                        self.assertIn("score", label_result)

                if isinstance(valid_input["images"], list):
                    # When images are batched, pipeline output is a list of lists of dictionaries
                    self.assertEqual(len(valid_input["images"]), len(output))
                    for individual_output in output:
                        assert_valid_pipeline_output(individual_output)
                else:
                    # When images are batched, pipeline output is a list of dictionaries
                    assert_valid_pipeline_output(output)

    def test_custom_tokenizer(self):
        tokenizer = PreTrainedTokenizer()

        # Assert that the pipeline can be initialized with a feature extractor that is not in any mapping
        image_classifier = pipeline("image-classification", model=self.small_models[0], tokenizer=tokenizer)

        self.assertIs(image_classifier.tokenizer, tokenizer)
