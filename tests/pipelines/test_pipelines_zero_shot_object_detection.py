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

from transformers import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING, is_vision_available, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
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
class ZeroShotObjectDetectionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):

    model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        object_detector = pipeline(
            "zero-shot-object-detection", model="hf-internal-testing/tiny-random-owlvit-object-detection"
        )

        examples = [
            {
                "images": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "text_queries": ["cat", "remote", "couch"],
            }
        ]
        return object_detector, examples

    def run_pipeline_test(self, object_detector, examples):
        batch_outputs = object_detector(
            examples,
            threshold=0.0,
        )

        self.assertEqual(len(examples), len(batch_outputs))
        for outputs in batch_outputs:
            for output_per_image in outputs:
                scores = output_per_image["scores"]
                labels = output_per_image["labels"]
                boxes = output_per_image["boxes"]

                assert len(scores) == len(boxes) and len(boxes) == len(labels)

                for score in scores:
                    self.assertEqual(score, ANY(float))
                for label in labels:
                    self.assertEqual(label, ANY(str))
                for box in boxes:
                    self.assertEqual(box, {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)})

    @require_tf
    @unittest.skip("Zero Shot Object Detection not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    def test_small_model_pt(self):
        object_detector = pipeline(
            "zero-shot-object-detection", model="hf-internal-testing/tiny-random-owlvit-object-detection"
        )

        outputs = object_detector(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text_queries=["cat", "remote", "couch"],
            threshold=0.64,
        )

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.6748, 0.6456, 0.7235, 0.642],
                    "labels": ["remote", "remote", "cat", "remote"],
                    "boxes": [
                        {"xmin": 571, "ymin": 83, "xmax": 598, "ymax": 103},
                        {"xmin": 494, "ymin": 105, "xmax": 521, "ymax": 127},
                        {"xmin": 204, "ymin": 167, "xmax": 232, "ymax": 190},
                        {"xmin": 67, "ymin": 274, "xmax": 93, "ymax": 297},
                    ],
                }
            ],
        )

        outputs = object_detector(
            ["./tests/fixtures/tests_samples/COCO/000000039769.png"],
            text_queries=["cat", "remote", "couch"],
            threshold=0.64,
        )

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.6748, 0.6456, 0.7235, 0.642],
                    "labels": ["remote", "remote", "cat", "remote"],
                    "boxes": [
                        {"xmin": 571, "ymin": 83, "xmax": 598, "ymax": 103},
                        {"xmin": 494, "ymin": 105, "xmax": 521, "ymax": 127},
                        {"xmin": 204, "ymin": 167, "xmax": 232, "ymax": 190},
                        {"xmin": 67, "ymin": 274, "xmax": 93, "ymax": 297},
                    ],
                }
            ],
        )

        outputs = object_detector(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            text_queries=[["cat", "remote", "couch"]],
            threshold=0.64,
        )

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.6748, 0.6456, 0.7235, 0.642],
                    "labels": ["remote", "remote", "cat", "remote"],
                    "boxes": [
                        {"xmin": 571, "ymin": 83, "xmax": 598, "ymax": 103},
                        {"xmin": 494, "ymin": 105, "xmax": 521, "ymax": 127},
                        {"xmin": 204, "ymin": 167, "xmax": 232, "ymax": 190},
                        {"xmin": 67, "ymin": 274, "xmax": 93, "ymax": 297},
                    ],
                }
            ],
        )

        outputs = object_detector(
            [
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            text_queries=[["cat", "remote", "couch"], ["cat", "remote", "couch"]],
            threshold=0.64,
        )

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.6748, 0.6456, 0.7235, 0.642],
                    "labels": ["remote", "remote", "cat", "remote"],
                    "boxes": [
                        {"xmin": 571, "ymin": 83, "xmax": 598, "ymax": 103},
                        {"xmin": 494, "ymin": 105, "xmax": 521, "ymax": 127},
                        {"xmin": 204, "ymin": 167, "xmax": 232, "ymax": 190},
                        {"xmin": 67, "ymin": 274, "xmax": 93, "ymax": 297},
                    ],
                },
                {
                    "scores": [0.6748, 0.6456, 0.7235, 0.642],
                    "labels": ["remote", "remote", "cat", "remote"],
                    "boxes": [
                        {"xmin": 571, "ymin": 83, "xmax": 598, "ymax": 103},
                        {"xmin": 494, "ymin": 105, "xmax": 521, "ymax": 127},
                        {"xmin": 204, "ymin": 167, "xmax": 232, "ymax": 190},
                        {"xmin": 67, "ymin": 274, "xmax": 93, "ymax": 297},
                    ],
                },
            ],
        )

    @require_torch
    @slow
    def test_large_model_pt(self):
        object_detector = pipeline("zero-shot-object-detection")

        outputs = object_detector(
            "http://images.cocodataset.org/val2017/000000039769.jpg", text_queries=["cat", "remote", "couch"]
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.277, 0.1474, 0.2868, 0.2537, 0.1208],
                    "labels": ["remote", "remote", "cat", "cat", "couch"],
                    "boxes": [
                        {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115},
                        {"xmin": 335, "ymin": 74, "xmax": 371, "ymax": 187},
                        {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373},
                        {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472},
                        {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476},
                    ],
                }
            ],
        )

        outputs = object_detector(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            text_queries=[["cat", "remote", "couch"], ["cat", "remote", "couch"]],
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.277, 0.1474, 0.2868, 0.2537, 0.1208],
                    "labels": ["remote", "remote", "cat", "cat", "couch"],
                    "boxes": [
                        {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115},
                        {"xmin": 335, "ymin": 74, "xmax": 371, "ymax": 187},
                        {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373},
                        {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472},
                        {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476},
                    ],
                },
                {
                    "scores": [0.277, 0.1474, 0.2868, 0.2537, 0.1208],
                    "labels": ["remote", "remote", "cat", "cat", "couch"],
                    "boxes": [
                        {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115},
                        {"xmin": 335, "ymin": 74, "xmax": 371, "ymax": 187},
                        {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373},
                        {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472},
                        {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476},
                    ],
                },
            ],
        )

    @require_tf
    @unittest.skip("Zero Shot Object Detection not implemented in TF")
    def test_large_model_tf(self):
        pass

    @require_torch
    @slow
    def test_threshold(self):
        threshold = 0.2
        object_detector = pipeline("zero-shot-object-detection")

        outputs = object_detector(
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            text_queries=["cat", "remote", "couch"],
            threshold=threshold,
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {
                    "scores": [0.277, 0.2868, 0.2537],
                    "labels": ["remote", "cat", "cat"],
                    "boxes": [
                        {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115},
                        {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373},
                        {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472},
                    ],
                }
            ],
        )
