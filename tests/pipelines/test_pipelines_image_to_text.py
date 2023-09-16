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
from transformers.pipelines import pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    is_torch_available,
    require_tf,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY


if is_torch_available():
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_11
else:
    is_torch_greater_or_equal_than_1_11 = False


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

    def get_test_pipeline(self, model, tokenizer, processor):
        pipe = pipeline("image-to-text", model=model, tokenizer=tokenizer, image_processor=processor)
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

    @require_tf
    def test_small_model_tf(self):
        pipe = pipeline("image-to-text", model="hf-internal-testing/tiny-random-vit-gpt2", framework="tf")
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

        outputs = pipe(image, max_new_tokens=1)
        self.assertEqual(
            outputs,
            [{"generated_text": "growth"}],
        )

    @require_torch
    def test_small_model_pt(self):
        pipe = pipeline("image-to-text", model="hf-internal-testing/tiny-random-vit-gpt2")
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

    @require_torch
    def test_small_model_pt_conditional(self):
        pipe = pipeline("image-to-text", model="hf-internal-testing/tiny-random-BlipForConditionalGeneration")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        prompt = "a photo of"

        outputs = pipe(image, prompt=prompt)
        self.assertTrue(outputs[0]["generated_text"].startswith(prompt))

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
    def test_conditional_generation_pt_blip(self):
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        prompt = "a photography of"

        outputs = pipe(image, prompt=prompt)
        self.assertEqual(outputs, [{"generated_text": "a photography of a volcano"}])

        with self.assertRaises(ValueError):
            outputs = pipe([image, image], prompt=[prompt, prompt])

    @slow
    @require_torch
    def test_conditional_generation_pt_git(self):
        pipe = pipeline("image-to-text", model="microsoft/git-base-coco")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        prompt = "a photo of a"

        outputs = pipe(image, prompt=prompt)
        self.assertEqual(outputs, [{"generated_text": "a photo of a tent with a tent and a tent in the background."}])

        with self.assertRaises(ValueError):
            outputs = pipe([image, image], prompt=[prompt, prompt])

    @unittest.skipIf(
        not is_torch_greater_or_equal_than_1_11, reason="`Pix2StructImageProcessor` requires `torch>=1.11.0`."
    )
    @slow
    @require_torch
    def test_conditional_generation_pt_pix2struct(self):
        pipe = pipeline("image-to-text", model="google/pix2struct-ai2d-base")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        prompt = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

        outputs = pipe(image, prompt=prompt)
        self.assertEqual(outputs, [{"generated_text": "ash cloud"}])

        with self.assertRaises(ValueError):
            outputs = pipe([image, image], prompt=[prompt, prompt])

    @slow
    @require_tf
    def test_large_model_tf(self):
        pipe = pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en", framework="tf")
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
