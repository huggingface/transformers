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

from transformers import is_vision_available
from transformers.pipelines import pipeline
from transformers.testing_utils import nested_simplify, require_tf, require_torch, require_vision, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@require_vision
class ZeroShotImageClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    # Deactivating auto tests since we don't have a good MODEL_FOR_XX mapping,
    # and only CLIP would be there for now.
    # model_mapping = {CLIPConfig: CLIPModel}

    # def get_test_pipeline(self, model, tokenizer, processor):
    #     if tokenizer is None:
    #         # Side effect of no Fast Tokenizer class for these model, so skipping
    #         # But the slow tokenizer test should still run as they're quite small
    #         self.skipTest("No tokenizer available")
    #         return
    #         # return None, None

    #     image_classifier = ZeroShotImageClassificationPipeline(
    #         model=model, tokenizer=tokenizer, feature_extractor=processor
    #     )

    #     # test with a raw waveform
    #     image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     image2 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     return image_classifier, [image, image2]

    # def run_pipeline_test(self, pipe, examples):
    #     image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    #     outputs = pipe(image, candidate_labels=["A", "B"])
    #     self.assertEqual(outputs, {"text": ANY(str)})

    #     # Batching
    #     outputs = pipe([image] * 3, batch_size=2, candidate_labels=["A", "B"])

    @require_torch
    def test_small_model_pt(self):
        image_classifier = pipeline(
            model="hf-internal-testing/tiny-random-clip-zero-shot-image-classification",
        )
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        output = image_classifier(image, candidate_labels=["a", "b", "c"])

        self.assertEqual(
            nested_simplify(output),
            [{"score": 0.333, "label": "a"}, {"score": 0.333, "label": "b"}, {"score": 0.333, "label": "c"}],
        )

        output = image_classifier([image] * 5, candidate_labels=["A", "B", "C"], batch_size=2)
        self.assertEqual(
            nested_simplify(output),
            # Pipeline outputs are supposed to be deterministic and
            # So we could in theory have real values "A", "B", "C" instead
            # of ANY(str).
            # However it seems that in this particular case, the floating
            # scores are so close, we enter floating error approximation
            # and the order is not guaranteed anymore with batching.
            [
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        image_classifier = pipeline(
            model="hf-internal-testing/tiny-random-clip-zero-shot-image-classification", framework="tf"
        )
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        output = image_classifier(image, candidate_labels=["a", "b", "c"])

        self.assertEqual(
            nested_simplify(output),
            [{"score": 0.333, "label": "a"}, {"score": 0.333, "label": "b"}, {"score": 0.333, "label": "c"}],
        )

        output = image_classifier([image] * 5, candidate_labels=["A", "B", "C"], batch_size=2)
        self.assertEqual(
            nested_simplify(output),
            # Pipeline outputs are supposed to be deterministic and
            # So we could in theory have real values "A", "B", "C" instead
            # of ANY(str).
            # However it seems that in this particular case, the floating
            # scores are so close, we enter floating error approximation
            # and the order is not guaranteed anymore with batching.
            [
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
                [
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                    {"score": 0.333, "label": ANY(str)},
                ],
            ],
        )

    @slow
    @require_torch
    def test_large_model_pt(self):
        image_classifier = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
        )
        # This is an image of 2 cats with remotes and no planes
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        output = image_classifier(image, candidate_labels=["cat", "plane", "remote"])

        self.assertEqual(
            nested_simplify(output),
            [
                {"score": 0.511, "label": "remote"},
                {"score": 0.485, "label": "cat"},
                {"score": 0.004, "label": "plane"},
            ],
        )

        output = image_classifier([image] * 5, candidate_labels=["cat", "plane", "remote"], batch_size=2)
        self.assertEqual(
            nested_simplify(output),
            [
                [
                    {"score": 0.511, "label": "remote"},
                    {"score": 0.485, "label": "cat"},
                    {"score": 0.004, "label": "plane"},
                ],
            ]
            * 5,
        )

    @slow
    @require_tf
    def test_large_model_tf(self):
        image_classifier = pipeline(
            task="zero-shot-image-classification", model="openai/clip-vit-base-patch32", framework="tf"
        )
        # This is an image of 2 cats with remotes and no planes
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        output = image_classifier(image, candidate_labels=["cat", "plane", "remote"])
        self.assertEqual(
            nested_simplify(output),
            [
                {"score": 0.511, "label": "remote"},
                {"score": 0.485, "label": "cat"},
                {"score": 0.004, "label": "plane"},
            ],
        )

        output = image_classifier([image] * 5, candidate_labels=["cat", "plane", "remote"], batch_size=2)
        self.assertEqual(
            nested_simplify(output),
            [
                [
                    {"score": 0.511, "label": "remote"},
                    {"score": 0.485, "label": "cat"},
                    {"score": 0.004, "label": "plane"},
                ],
            ]
            * 5,
        )
