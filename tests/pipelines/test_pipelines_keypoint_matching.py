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
    is_pipeline_test,
    is_vision_available,
    require_torch,
    require_vision,
)

from .test_pipelines_common import ANY


if is_vision_available():
    from PIL import Image


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
                {
                    "keypoint_image_0": {"x": ANY(float), "y": ANY(float)},
                    "keypoint_image_1": {"x": ANY(float), "y": ANY(float)},
                    "score": ANY(float),
                }
            ]
            * 2,  # 2 matches per image pair
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
                    {
                        "keypoint_image_0": {"x": ANY(float), "y": ANY(float)},
                        "keypoint_image_1": {"x": ANY(float), "y": ANY(float)},
                        "score": ANY(float),
                    }
                ]
                * 2  # 2 matches per image pair
            ]
            * 4,  # 4 image pairs
        )

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

        image_0: Image.Image = self._dataset[0]["image"]
        image_1: Image.Image = self._dataset[1]["image"]
        outputs = image_matcher((image_0, image_1), threshold=0.0)

        output = outputs[0]  # first match from image pair
        self.assertAlmostEqual(output["keypoint_image_0"]["x"], 698, places=1)
        self.assertAlmostEqual(output["keypoint_image_0"]["y"], 469, places=1)
        self.assertAlmostEqual(output["keypoint_image_1"]["x"], 434, places=1)
        self.assertAlmostEqual(output["keypoint_image_1"]["y"], 440, places=1)
        self.assertAlmostEqual(output["score"], 0.9905, places=3)

    @require_torch
    def test_multiple_pairs(self):
        self._load_dataset()
        small_model = "magic-leap-community/superglue_outdoor"
        image_matcher = pipeline("keypoint-matching", model=small_model)

        image_0: Image.Image = self._dataset[0]["image"]
        image_1: Image.Image = self._dataset[1]["image"]
        image_2: Image.Image = self._dataset[2]["image"]

        outputs = image_matcher(
            [
                (image_0, image_1),
                (image_1, image_2),
                (image_2, image_0),
            ],
            threshold=1e-4,
        )

        # Test first pair (image_0, image_1)
        output_0 = outputs[0][0]  # First match from first pair
        self.assertAlmostEqual(output_0["keypoint_image_0"]["x"], 698, places=1)
        self.assertAlmostEqual(output_0["keypoint_image_0"]["y"], 469, places=1)
        self.assertAlmostEqual(output_0["keypoint_image_1"]["x"], 434, places=1)
        self.assertAlmostEqual(output_0["keypoint_image_1"]["y"], 440, places=1)
        self.assertAlmostEqual(output_0["score"], 0.9905, places=3)

        # Test second pair (image_1, image_2)
        output_1 = outputs[1][0]  # First match from second pair
        self.assertAlmostEqual(output_1["keypoint_image_0"]["x"], 272, places=1)
        self.assertAlmostEqual(output_1["keypoint_image_0"]["y"], 310, places=1)
        self.assertAlmostEqual(output_1["keypoint_image_1"]["x"], 228, places=1)
        self.assertAlmostEqual(output_1["keypoint_image_1"]["y"], 568, places=1)
        self.assertAlmostEqual(output_1["score"], 0.9890, places=3)

        # Test third pair (image_2, image_0)
        output_2 = outputs[2][0]  # First match from third pair
        self.assertAlmostEqual(output_2["keypoint_image_0"]["x"], 385, places=1)
        self.assertAlmostEqual(output_2["keypoint_image_0"]["y"], 677, places=1)
        self.assertAlmostEqual(output_2["keypoint_image_1"]["x"], 689, places=1)
        self.assertAlmostEqual(output_2["keypoint_image_1"]["y"], 351, places=1)
        self.assertAlmostEqual(output_2["score"], 0.9900, places=3)
