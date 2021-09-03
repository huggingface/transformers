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
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    AutoFeatureExtractor,
    AutoModelForObjectDetection,
    ObjectDetectionPipeline,
    PreTrainedTokenizer,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_datasets,
    require_tf,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_vision
@require_torch
@is_pipeline_test
class ObjectDetectionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING

    @require_datasets
    def run_pipeline_test(self, model, tokenizer, feature_extractor):
        threshold = 0.0
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)
        outputs = object_detector("./tests/fixtures/tests_samples/COCO/000000039769.png", threshold=threshold)

        self.assertGreater(len(outputs), 0)
        for detected_object in outputs:
            self.assertEqual(
                detected_object,
                {
                    "score": ANY(float),
                    "label": ANY(str),
                    "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                },
            )

        import datasets

        dataset = datasets.load_dataset("Narsil/image_dummy", "image", split="test")

        batch = [
            Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            # RGBA
            dataset[0]["file"],
            # LA
            dataset[1]["file"],
            # L
            dataset[2]["file"],
        ]
        batch_outputs = object_detector(batch, threshold=threshold)

        self.assertEqual(len(batch), len(batch_outputs))
        for outputs in batch_outputs:
            self.assertGreater(len(outputs), 0)
            for detected_object in outputs:
                self.assertEqual(
                    detected_object,
                    {
                        "score": ANY(float),
                        "label": ANY(str),
                        "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                    },
                )

    @require_tf
    @unittest.skip("Object detection not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        threshold = 0.0
        model_id = "mishig/tiny-detr-mobilenetsv3"

        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=threshold)

        self.assertGreater(len(outputs), 0)
        for detected_object in outputs:
            self.assertEqual(
                detected_object,
                {
                    "score": ANY(float),
                    "label": ANY(str),
                    "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                },
            )

        batch = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        ]
        batch_outputs = object_detector(batch, threshold=threshold)

        self.assertEqual(len(batch), len(batch_outputs))
        for outputs in batch_outputs:
            self.assertGreater(len(outputs), 0)
            for detected_object in outputs:
                self.assertEqual(
                    detected_object,
                    {
                        "score": ANY(float),
                        "label": ANY(str),
                        "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                    },
                )

    @require_torch
    @slow
    def test_large_model_pt(self):
        model_id = "facebook/detr-resnet-50"

        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg")

        self.assertGreater(len(outputs), 0)
        for detected_object in outputs:
            self.assertEqual(
                detected_object,
                {
                    "score": ANY(float),
                    "label": ANY(str),
                    "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                },
            )

        batch = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        ]
        batch_outputs = object_detector(batch)

        self.assertEqual(len(batch), len(batch_outputs))
        for outputs in batch_outputs:
            self.assertGreater(len(outputs), 0)
            for detected_object in outputs:
                self.assertEqual(
                    detected_object,
                    {
                        "score": ANY(float),
                        "label": ANY(str),
                        "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                    },
                )

    @require_torch
    @slow
    def test_ntegration_torch_object_detection(self):
        model_id = "facebook/detr-resnet-50"

        object_detector = pipeline("object-detection", model=model_id)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg")

        self.assertGreater(len(outputs), 0)
        for detected_object in outputs:
            self.assertEqual(
                detected_object,
                {
                    "score": ANY(float),
                    "label": ANY(str),
                    "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                },
            )

        batch = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        ]
        batch_outputs = object_detector(batch)

        self.assertEqual(len(batch), len(batch_outputs))
        for outputs in batch_outputs:
            self.assertGreater(len(outputs), 0)
            for detected_object in outputs:
                self.assertEqual(
                    detected_object,
                    {
                        "score": ANY(float),
                        "label": ANY(str),
                        "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
                    },
                )

    def test_custom_tokenizer(self):
        model_id = "facebook/detr-resnet-50"
        tokenizer = PreTrainedTokenizer()

        # Assert that the pipeline can be initialized with a feature extractor that is not in any mapping
        object_detector = pipeline("object-detection", model=model_id, tokenizer=tokenizer)

        self.assertIs(object_detector.tokenizer, tokenizer)

    def test_low_threshold(self):
        threshold = 0.0
        model_id = "mishig/tiny-detr-mobilenetsv3"

        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

        valid_input = {"images": "http://images.cocodataset.org/val2017/000000039769.jpg", "threshold": threshold}

        output = object_detector(**valid_input)

        self.assertTrue(isinstance(output, list))
        self.assertEqual(len(output), 5)

    def test_high_threshold(self):
        threshold = 1.0
        model_id = "mishig/tiny-detr-mobilenetsv3"

        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

        valid_input = {"images": "http://images.cocodataset.org/val2017/000000039769.jpg", "threshold": threshold}

        output = object_detector(**valid_input)

        self.assertTrue(isinstance(output, list))
        self.assertEqual(len(output), 0)
