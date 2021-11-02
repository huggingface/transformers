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

from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, PreTrainedTokenizer, is_vision_available
from transformers.pipelines import ImageClassificationPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_datasets,
    require_tf,
    require_torch,
    require_vision,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@is_pipeline_test
@require_vision
@require_torch
class ImageClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

    @require_datasets
    def run_pipeline_test(self, model, tokenizer, feature_extractor):
        image_classifier = ImageClassificationPipeline(model=model, feature_extractor=feature_extractor)
        outputs = image_classifier("./tests/fixtures/tests_samples/COCO/000000039769.png")

        self.assertEqual(
            outputs,
            [
                {"score": ANY(float), "label": ANY(str)},
                {"score": ANY(float), "label": ANY(str)},
            ],
        )

        import datasets

        dataset = datasets.load_dataset("Narsil/image_dummy", "image", split="test")

        # Accepts URL + PIL.Image + lists
        outputs = image_classifier(
            [
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                # RGBA
                dataset[0]["file"],
                # LA
                dataset[1]["file"],
                # L
                dataset[2]["file"],
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
                [
                    {"score": ANY(float), "label": ANY(str)},
                    {"score": ANY(float), "label": ANY(str)},
                ],
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        small_model = "lysandre/tiny-vit-random"
        image_classifier = pipeline("image-classification", model=small_model)

        outputs = image_classifier("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.0015, "label": "chambered nautilus, pearly nautilus, nautilus"},
                {"score": 0.0015, "label": "pajama, pyjama, pj's, jammies"},
                {"score": 0.0014, "label": "trench coat"},
                {"score": 0.0014, "label": "handkerchief, hankie, hanky, hankey"},
                {"score": 0.0014, "label": "baboon"},
            ],
        )

        outputs = image_classifier(
            [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ],
            top_k=2,
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.0015, "label": "chambered nautilus, pearly nautilus, nautilus"},
                    {"score": 0.0015, "label": "pajama, pyjama, pj's, jammies"},
                ],
                [
                    {"score": 0.0015, "label": "chambered nautilus, pearly nautilus, nautilus"},
                    {"score": 0.0015, "label": "pajama, pyjama, pj's, jammies"},
                ],
            ],
        )

    @require_tf
    @unittest.skip("Image classification is not implemented for TF")
    def test_small_model_tf(self):
        pass

    def test_custom_tokenizer(self):
        tokenizer = PreTrainedTokenizer()

        # Assert that the pipeline can be initialized with a feature extractor that is not in any mapping
        image_classifier = pipeline("image-classification", model="lysandre/tiny-vit-random", tokenizer=tokenizer)

        self.assertIs(image_classifier.tokenizer, tokenizer)
