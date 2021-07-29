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

from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, PreTrainedTokenizer, is_vision_available
from transformers.pipelines import ObjectDetectionPipeline, pipeline
from transformers.testing_utils import require_timm, require_torch, require_vision, slow


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@slow
@require_timm
@require_vision
@require_torch
class ObjectDetectionPipelineTests(unittest.TestCase):
    pipeline_task = "object-detection"
    large_models = ["facebook/detr-resnet-50"]
    small_models = ["mishig/tiny-detr-mobilenetsv3"]  # Models tested without the @slow decorator
    valid_inputs = [
        {"images": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        {
            "images": [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]
        },
        {"images": "./tests/fixtures/tests_samples/COCO/000000039769.png"},
        {
            "images": [
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
            ]
        },
        {"images": Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")},
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
        threshold = 0.0
        for small_model in self.small_models:

            object_detector = pipeline("object-detection", model=small_model)

            for valid_input in self.valid_inputs:
                output = object_detector(**valid_input, threshold=threshold)

                def assert_valid_pipeline_output(pipeline_output):
                    self.assertTrue(isinstance(pipeline_output, list))
                    for annotation_result in pipeline_output:
                        self.assertTrue(isinstance(annotation_result, dict))
                        self.assertEqual(set(annotation_result.keys()), {"score", "label", "box"})
                        self.assertEqual(type(annotation_result["score"]), float)
                        self.assertEqual(type(annotation_result["label"]), str)
                        self.assertEqual(type(annotation_result["box"]), list)

                if isinstance(valid_input["images"], list):
                    self.assertEqual(len(valid_input["images"]), len(output))
                    for individual_output in output:
                        assert_valid_pipeline_output(individual_output)
                else:
                    assert_valid_pipeline_output(output)

    def test_small_model_from_pipeline(self):
        threshold = 0.0
        for small_model in self.small_models:

            model = AutoModelForObjectDetection.from_pretrained(small_model)
            feature_extractor = AutoFeatureExtractor.from_pretrained(small_model)
            object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

            for valid_input in self.valid_inputs:
                output = object_detector(**valid_input, threshold=threshold)

                def assert_valid_pipeline_output(pipeline_output):
                    self.assertTrue(isinstance(pipeline_output, list))
                    for annotation_result in pipeline_output:
                        self.assertTrue(isinstance(annotation_result, dict))
                        self.assertEqual(set(annotation_result.keys()), {"score", "label", "box"})
                        self.assertEqual(type(annotation_result["score"]), float)
                        self.assertEqual(type(annotation_result["label"]), str)
                        self.assertEqual(type(annotation_result["box"]), list)

                if isinstance(valid_input["images"], list):
                    # When images are batched, pipeline output is a list of lists of dictionaries
                    self.assertEqual(len(valid_input["images"]), len(output))
                    for individual_output in output:
                        assert_valid_pipeline_output(individual_output)
                else:
                    # When images are batched, pipeline output is a list of dictionaries
                    assert_valid_pipeline_output(output)

    @slow
    def test_large_model_from_factory(self):
        for large_model in self.large_models:

            object_detector = pipeline("object-detection", model=large_model)

            for valid_input in self.valid_inputs:
                output = object_detector(**valid_input)

                def assert_valid_pipeline_output(pipeline_output):
                    self.assertTrue(isinstance(pipeline_output, list))
                    for annotation_result in pipeline_output:
                        self.assertTrue(isinstance(annotation_result, dict))
                        self.assertEqual(set(annotation_result.keys()), {"score", "label", "box"})
                        self.assertEqual(type(annotation_result["score"]), float)
                        self.assertEqual(type(annotation_result["label"]), str)
                        self.assertEqual(type(annotation_result["box"]), list)

                if isinstance(valid_input["images"], list):
                    self.assertEqual(len(valid_input["images"]), len(output))
                    for individual_output in output:
                        assert_valid_pipeline_output(individual_output)
                else:
                    assert_valid_pipeline_output(output)

    @slow
    def test_large_model_from_pipeline(self):
        for large_model in self.large_models:

            model = AutoModelForObjectDetection.from_pretrained(large_model)
            feature_extractor = AutoFeatureExtractor.from_pretrained(large_model)
            object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

            for valid_input in self.valid_inputs:
                output = object_detector(**valid_input)

                def assert_valid_pipeline_output(pipeline_output):
                    self.assertTrue(isinstance(pipeline_output, list))
                    for annotation_result in pipeline_output:
                        self.assertTrue(isinstance(annotation_result, dict))
                        self.assertEqual(set(annotation_result.keys()), {"score", "label", "box"})
                        self.assertEqual(type(annotation_result["score"]), float)
                        self.assertEqual(type(annotation_result["label"]), str)
                        self.assertEqual(type(annotation_result["box"]), list)

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
        object_detector = pipeline("object-detection", model=self.large_models[0], tokenizer=tokenizer)

        self.assertIs(object_detector.tokenizer, tokenizer)

    def test_annotation_box(self):
        threshold = 0.0
        model_id = "mishig/tiny-detr-mobilenetsv3"
        object_detector = pipeline("object-detection", model=model_id)

        output = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=threshold)

        self.assertTrue(isinstance(output, list))

        for annotation_result in output:
            self.assertTrue(isinstance(annotation_result, dict))
            self.assertIn("box", annotation_result)
            self.assertEqual(type(annotation_result["box"]), list)
            self.assertEqual(len(annotation_result["box"]), 4)
            for vertex in annotation_result["box"]:
                self.assertEqual(set(vertex.keys()), {"x", "y"})
                self.assertEqual(type(vertex["x"]), int)
                self.assertEqual(type(vertex["y"]), int)

    @slow
    def test_low_threshold(self):
        threshold = 0.0
        model_id = "facebook/detr-resnet-50"
        object_detector = pipeline("object-detection", model=model_id)
        valid_input = {"images": "http://images.cocodataset.org/val2017/000000039769.jpg", "threshold": threshold}

        output = object_detector(**valid_input)

        self.assertTrue(isinstance(output, list))
        self.assertEqual(len(output), 100)

    @slow
    def test_high_threshold(self):
        threshold = 1.0
        model_id = "facebook/detr-resnet-50"
        object_detector = pipeline("object-detection", model=model_id)
        valid_input = {"images": "http://images.cocodataset.org/val2017/000000039769.jpg", "threshold": threshold}

        output = object_detector(**valid_input)

        self.assertTrue(isinstance(output, list))
        self.assertEqual(len(output), 0)
