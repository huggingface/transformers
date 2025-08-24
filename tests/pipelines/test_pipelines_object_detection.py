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

import datasets
from huggingface_hub import ObjectDetectionOutputElement

from transformers import (
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    AutoFeatureExtractor,
    AutoModelForObjectDetection,
    ObjectDetectionPipeline,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import (  #
    _run_pipeline_tests,
    compare_pipeline_output_to_hub_spec,
    is_pipeline_test,
    nested_simplify,
    require_pytesseract,
    require_tf,
    require_timm,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@is_pipeline_test
@require_vision
@require_timm
@require_torch
class ObjectDetectionPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING

    if _run_pipeline_tests:
        # we use revision="refs/pr/1" until the PR is merged
        # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1
        _dataset = datasets.load_dataset(
            "hf-internal-testing/fixtures_image_utils", split="test", revision="refs/pr/1"
        )

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        object_detector = ObjectDetectionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
        return object_detector, ["./tests/fixtures/tests_samples/COCO/000000039769.png"]

    def run_pipeline_test(self, object_detector, examples):
        outputs = object_detector("./tests/fixtures/tests_samples/COCO/000000039769.png", threshold=0.0)

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
            Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            # RGBA
            self._dataset[0]["image"],
            # LA
            self._dataset[1]["image"],
            # L
            self._dataset[2]["image"],
        ]
        batch_outputs = object_detector(batch, threshold=0.0)

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
                compare_pipeline_output_to_hub_spec(detected_object, ObjectDetectionOutputElement)

    @require_tf
    @unittest.skip(reason="Object detection not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        model_id = "hf-internal-testing/tiny-detr-mobilenetsv3"

        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=0.0)

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
                {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
            ],
        )

        outputs = object_detector(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            threshold=0.0,
        )

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
                    {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
                ],
                [
                    {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
                    {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
                ],
            ],
        )

    @require_torch
    @slow
    def test_large_model_pt(self):
        model_id = "facebook/detr-resnet-50"

        model = AutoModelForObjectDetection.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        object_detector = ObjectDetectionPipeline(model=model, feature_extractor=feature_extractor)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9982, "label": "remote", "box": {"xmin": 40, "ymin": 70, "xmax": 175, "ymax": 117}},
                {"score": 0.9960, "label": "remote", "box": {"xmin": 333, "ymin": 72, "xmax": 368, "ymax": 187}},
                {"score": 0.9955, "label": "couch", "box": {"xmin": 0, "ymin": 1, "xmax": 639, "ymax": 473}},
                {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
            ],
        )

        outputs = object_detector(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9982, "label": "remote", "box": {"xmin": 40, "ymin": 70, "xmax": 175, "ymax": 117}},
                    {"score": 0.9960, "label": "remote", "box": {"xmin": 333, "ymin": 72, "xmax": 368, "ymax": 187}},
                    {"score": 0.9955, "label": "couch", "box": {"xmin": 0, "ymin": 1, "xmax": 639, "ymax": 473}},
                    {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                    {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
                ],
                [
                    {"score": 0.9982, "label": "remote", "box": {"xmin": 40, "ymin": 70, "xmax": 175, "ymax": 117}},
                    {"score": 0.9960, "label": "remote", "box": {"xmin": 333, "ymin": 72, "xmax": 368, "ymax": 187}},
                    {"score": 0.9955, "label": "couch", "box": {"xmin": 0, "ymin": 1, "xmax": 639, "ymax": 473}},
                    {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                    {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
                ],
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_object_detection(self):
        model_id = "facebook/detr-resnet-50"

        object_detector = pipeline("object-detection", model=model_id)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9982, "label": "remote", "box": {"xmin": 40, "ymin": 70, "xmax": 175, "ymax": 117}},
                {"score": 0.9960, "label": "remote", "box": {"xmin": 333, "ymin": 72, "xmax": 368, "ymax": 187}},
                {"score": 0.9955, "label": "couch", "box": {"xmin": 0, "ymin": 1, "xmax": 639, "ymax": 473}},
                {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
            ],
        )

        outputs = object_detector(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9982, "label": "remote", "box": {"xmin": 40, "ymin": 70, "xmax": 175, "ymax": 117}},
                    {"score": 0.9960, "label": "remote", "box": {"xmin": 333, "ymin": 72, "xmax": 368, "ymax": 187}},
                    {"score": 0.9955, "label": "couch", "box": {"xmin": 0, "ymin": 1, "xmax": 639, "ymax": 473}},
                    {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                    {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
                ],
                [
                    {"score": 0.9982, "label": "remote", "box": {"xmin": 40, "ymin": 70, "xmax": 175, "ymax": 117}},
                    {"score": 0.9960, "label": "remote", "box": {"xmin": 333, "ymin": 72, "xmax": 368, "ymax": 187}},
                    {"score": 0.9955, "label": "couch", "box": {"xmin": 0, "ymin": 1, "xmax": 639, "ymax": 473}},
                    {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                    {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
                ],
            ],
        )

    @require_torch
    @slow
    def test_threshold(self):
        threshold = 0.9985
        model_id = "facebook/detr-resnet-50"

        object_detector = pipeline("object-detection", model=model_id)

        outputs = object_detector("http://images.cocodataset.org/val2017/000000039769.jpg", threshold=threshold)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9988, "label": "cat", "box": {"xmin": 13, "ymin": 52, "xmax": 314, "ymax": 470}},
                {"score": 0.9987, "label": "cat", "box": {"xmin": 345, "ymin": 23, "xmax": 640, "ymax": 368}},
            ],
        )

    @require_torch
    @require_pytesseract
    @slow
    def test_layoutlm(self):
        model_id = "Narsil/layoutlmv3-finetuned-funsd"
        threshold = 0.9993

        object_detector = pipeline("object-detection", model=model_id, threshold=threshold)

        outputs = object_detector(
            "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9993, "label": "I-ANSWER", "box": {"xmin": 294, "ymin": 254, "xmax": 343, "ymax": 264}},
                {"score": 0.9993, "label": "I-ANSWER", "box": {"xmin": 294, "ymin": 254, "xmax": 343, "ymax": 264}},
            ],
        )
