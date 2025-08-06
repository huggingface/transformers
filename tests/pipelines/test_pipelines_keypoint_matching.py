# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.models.auto.modeling_auto import MODEL_FOR_KEYPOINT_MATCHING_MAPPING
from transformers.pipelines import KeypointMatchingPipeline, pipeline
from transformers.testing_utils import (
    compare_pipeline_output_to_hub_spec,
    is_pipeline_test,
    is_vision_available,
    nested_simplify,
    require_torch,
    require_vision,
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
@require_torch
@require_vision
class KeypointMatchingPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_KEYPOINT_MATCHING_MAPPING
    _dataset = None

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls._dataset is None:
            cls._dataset = datasets.load_dataset("hf-internal-testing/image-matching-dataset", split="train")

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        image_matcher = KeypointMatchingPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
        examples = [
            Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        ]
        return image_matcher, examples

    def run_pipeline_test(self, image_matcher, examples):
        self._load_dataset()
        outputs = image_matcher(
            [
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]
        )

        self.assertEqual(
            outputs,
            [
                {"score": ANY(float), "label": ANY(str)},
                {"score": ANY(float), "label": ANY(str)},
            ],
        )

        # Accepts URL + PIL.Image + lists
        outputs = image_matcher(
            [
                [
                    Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                    "http://images.cocodataset.org/val2017/000000039769.jpg",
                ],
                [self._dataset[0]["image"], self._dataset[1]["image"]],
                [self._dataset[1]["image"], self._dataset[2]["image"]],
                [self._dataset[2]["image"], self._dataset[0]["image"]],
            ]
        )
        self.assertEqual(
            outputs,
            [
                [
                    {"score": ANY(float), "label": ANY(str)},
                    {"score": ANY(float), "label": ANY(str)},
                ],
                [
                    {"score": ANY(float), "label": ANY(str)},
                    {"score": ANY(float), "label": ANY(str)},
                ],
                [
                    {"score": ANY(float), "label": ANY(str)},
                    {"score": ANY(float), "label": ANY(str)},
                ],
                [
                    {"score": ANY(float), "label": ANY(str)},
                    {"score": ANY(float), "label": ANY(str)},
                ],
            ],
        )

        # for single_output in outputs:
        #     for output_element in single_output:
        #         compare_pipeline_output_to_hub_spec(output_element, KeypointMatchingOutputElement)

    @require_torch
    def test_single_image(self):
        self._load_dataset()
        small_model = "magic-leap-community/superglue_outdoor"
        image_matcher = pipeline("keypoint-matching", model=small_model)

        with self.assertRaises(ValueError):
            image_matcher(
                self._dataset[0]["image"],
                threshold=0.0,
            )
        with self.assertRaises(ValueError):
            image_matcher(
                [self._dataset[0]["image"]],
                threshold=0.0,
            )

    @require_torch
    def test_single_pair(self):
        self._load_dataset()
        small_model = "magic-leap-community/superglue_outdoor"
        image_matcher = pipeline("keypoint-matching", model=small_model)

        outputs = image_matcher(
            [self._dataset[0]["image"], self._dataset[1]["image"]],
            threshold=0.0,
        )
        simplified_paired_outputs = nested_simplify(outputs, decimals=4)
        truncated_outputs = [{key: value[0] for key, value in output.items()} for output in simplified_paired_outputs]
        self.assertEqual(
            truncated_outputs,
            [{"keypoints0": [551, 164], "keypoints1": [360, 171], "matching_scores": 0.9899}],
        )

    @require_torch
    def test_multiple_pairs(self):
        self._load_dataset()
        small_model = "magic-leap-community/superglue_outdoor"
        image_matcher = pipeline("keypoint-matching", model=small_model)

        outputs = image_matcher(
            [
                [self._dataset[0]["image"], self._dataset[1]["image"]],
                [self._dataset[1]["image"], self._dataset[2]["image"]],
                [self._dataset[2]["image"], self._dataset[0]["image"]],
            ],
            threshold=1e-4,
        )
        simplified_paired_outputs = nested_simplify(outputs, decimals=4)
        truncated_outputs = [{key: value[0] for key, value in output.items()} for output in simplified_paired_outputs]
        self.assertEqual(
            truncated_outputs,
            [
                {"keypoints0": [551, 164], "keypoints1": [360, 171], "matching_scores": 0.9899},
                {"keypoints0": [879, 30], "keypoints1": [748, 382], "matching_scores": 0.0166},
                {"keypoints0": [306, 81], "keypoints1": [551, 164], "matching_scores": 0.9809},
            ],
        )
