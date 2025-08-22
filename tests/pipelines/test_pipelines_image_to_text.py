# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import requests

from transformers import MODEL_FOR_VISION_2_SEQ_MAPPING, TF_MODEL_FOR_VISION_2_SEQ_MAPPING, is_vision_available
from transformers.pipelines import ImageToTextPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
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
class ImageToTextPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING
    tf_model_mapping = TF_MODEL_FOR_VISION_2_SEQ_MAPPING

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        pipe = ImageToTextPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
            max_new_tokens=20,
        )
        examples = [
            Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]
        return pipe, examples

    def run_pipeline_test(self, pipe, examples):
        outputs = pipe(examples)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": ANY(str)}],
                [{"generated_text": ANY(str)}],
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        pipe = pipeline("image-to-text", model="hf-internal-testing/tiny-random-vit-gpt2", max_new_tokens=19)
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"

        outputs = pipe(image)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                },
            ],
        )

        outputs = pipe([image, image])
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                    }
                ],
                [
                    {
                        "generated_text": "growthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthgrowthGOGO"
                    }
                ],
            ],
        )

    @slow
    @require_torch
    def test_large_model_pt(self):
        pipe = pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"

        outputs = pipe(image)
        self.assertEqual(outputs, [{"generated_text": "a cat laying on a blanket next to a cat laying on a bed "}])

        outputs = pipe([image, image])
        self.assertEqual(
            outputs,
            [
                [{"generated_text": "a cat laying on a blanket next to a cat laying on a bed "}],
                [{"generated_text": "a cat laying on a blanket next to a cat laying on a bed "}],
            ],
        )

    @slow
    @require_torch
    def test_generation_pt_blip(self):
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
        image = Image.open(requests.get(url, stream=True).raw)

        outputs = pipe(image)
        self.assertEqual(outputs, [{"generated_text": "a pink pokemon pokemon with a blue shirt and a blue shirt"}])

    @slow
    @require_torch
    def test_generation_pt_git(self):
        pipe = pipeline("image-to-text", model="microsoft/git-base-coco")
        url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
        image = Image.open(requests.get(url, stream=True).raw)

        outputs = pipe(image)
        self.assertEqual(outputs, [{"generated_text": "a cartoon of a purple character."}])

    @slow
    @require_torch
    def test_nougat(self):
        pipe = pipeline("image-to-text", "facebook/nougat-base", max_new_tokens=19)

        outputs = pipe("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/nougat_paper.png")

        self.assertEqual(
            outputs,
            [{"generated_text": "# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blec"}],
        )
