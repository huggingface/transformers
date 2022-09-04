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

from transformers import is_vision_available, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_tf,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import PipelineTestCaseMeta


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
    # Deactivating auto tests since we don't have a good MODEL_FOR_XX mapping,
    # and only OwlViTForObjectDetection would be there for now.
    # model_mapping = {OwlViTConfig: OwlViTForObjectDetection}

    # def get_test_pipeline(self, model, tokenizer, feature_extractor):
    #     object_detector = pipeline("zero-shot-object-detection", model="hf-internal-testing/tiny-random-owlvit")

    #     return object_detector, ["./tests/fixtures/tests_samples/COCO/000000039769.png"]

    # def run_pipeline_test(self, object_detector, examples):
    #     outputs = object_detector(
    #         Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
    #         candidate_labels=["cat", "remote", "couch"],
    #         threshold=0.0,
    #     )

    #     self.assertGreater(len(outputs), 0)
    #     for detected_object in outputs:
    #         self.assertEqual(
    #             detected_object,
    #             {
    #                 "score": ANY(float),
    #                 "label": ANY(str),
    #                 "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
    #             },
    #         )

    #     batch = [
    #         Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
    #         "http://images.cocodataset.org/val2017/000000039769.jpg",
    #     ] * 4
    #     batch_outputs = object_detector(batch, candidate_labels=["cat", "remote", "couch"], threshold=0.0)

    #     self.assertEqual(len(batch), len(batch_outputs))

    #     for outputs in batch_outputs:
    #         self.assertGreater(len(outputs), 0)
    #         for detected_object in outputs:
    #             self.assertEqual(
    #                 detected_object,
    #                 {
    #                     "score": ANY(float),
    #                     "label": ANY(str),
    #                     "box": {"xmin": ANY(int), "ymin": ANY(int), "xmax": ANY(int), "ymax": ANY(int)},
    #                 },
    #             )

    @require_tf
    @unittest.skip("Zero Shot Object Detection not implemented in TF")
    def test_small_model_tf(self):
        pass

    @require_torch
    @unittest.skip("Using hf-internal-testing/tiny-random-owlvit throws error")
    def test_small_model_pt(self):
        object_detector = pipeline("zero-shot-object-detection", model="hf-internal-testing/tiny-random-owlvit")

        outputs = object_detector(
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            candidate_labels=["cat", "remote", "couch"],
            threshold=0.1,
        )

        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
                {"score": 0.3376, "label": "LABEL_0", "box": {"xmin": 159, "ymin": 120, "xmax": 480, "ymax": 359}},
            ],
        )

        outputs = object_detector(
            [
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            candidate_labels=["cat", "remote", "couch"],
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
        object_detector = pipeline("zero-shot-object-detection")

        outputs = object_detector(
            "http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=["cat", "remote", "couch"]
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.2754, "label": "remote", "box": {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115}},
                {"score": 0.1969, "label": "remote", "box": {"xmin": 335, "ymin": 74, "xmax": 371, "ymax": 187}},
                {"score": 0.4931, "label": "cat", "box": {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373}},
                {"score": 0.4973, "label": "cat", "box": {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472}},
                {"score": 0.1327, "label": "couch", "box": {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476}},
            ],
        )

        outputs = object_detector(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            candidate_labels=["cat", "remote", "couch"],
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.2754, "label": "remote", "box": {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115}},
                    {"score": 0.1969, "label": "remote", "box": {"xmin": 335, "ymin": 74, "xmax": 371, "ymax": 187}},
                    {"score": 0.4931, "label": "cat", "box": {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373}},
                    {"score": 0.4973, "label": "cat", "box": {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472}},
                    {"score": 0.1327, "label": "couch", "box": {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476}},
                ],
                [
                    {"score": 0.2754, "label": "remote", "box": {"xmin": 40, "ymin": 72, "xmax": 177, "ymax": 115}},
                    {"score": 0.1969, "label": "remote", "box": {"xmin": 335, "ymin": 74, "xmax": 371, "ymax": 187}},
                    {"score": 0.4931, "label": "cat", "box": {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373}},
                    {"score": 0.4973, "label": "cat", "box": {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472}},
                    {"score": 0.1327, "label": "couch", "box": {"xmin": 4, "ymin": 0, "xmax": 642, "ymax": 476}},
                ],
            ],
        )

    @require_tf
    @unittest.skip("Zero Shot Object Detection not implemented in TF")
    def test_large_model_tf(self):
        pass

    @require_torch
    @slow
    def test_threshold(self):
        threshold = 0.4
        object_detector = pipeline("zero-shot-object-detection")

        outputs = object_detector(
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            candidate_labels=["cat", "remote", "couch"],
            threshold=threshold,
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4931, "label": "cat", "box": {"xmin": 324, "ymin": 20, "xmax": 640, "ymax": 373}},
                {"score": 0.4973, "label": "cat", "box": {"xmin": 1, "ymin": 55, "xmax": 315, "ymax": 472}},
            ],
        )
