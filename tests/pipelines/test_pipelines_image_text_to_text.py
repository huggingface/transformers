# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING, is_vision_available
from transformers.pipelines import pipeline
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
    model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

    def get_test_pipeline(self, model, processor):
        pipe = pipeline("image-text-to-text", model=model, processor=processor)
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
        pipe = pipeline(
            "image-text-to-text",
            model="hf-internal-testing/tiny-random-BlipForConditionalGeneration",
            processor="hf-internal-testing/tiny-random-BlipForConditionalGeneration",
        )
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        text = "hello world"

        outputs = pipe(image, text=text)
        self.assertEqual(
            outputs,
            [{"generated_text": "望とe ा ჹ 辛ɔง 2ɳоল 0 ψɔง 藥 ਾ"}],
        )

        outputs = pipe([image, image], text=text)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": "望とe ा ჹ 辛ɔง 2ɳоল 0 ψɔง 藥 ਾ"}],
                [{"generated_text": "望とe ा ჹ 辛ɔง 2ɳоল 0 ψɔง 藥 ਾ"}],
            ],
        )

    @require_torch
    def test_small_model_pt_conditional(self):
        pipe = pipeline("image-text-to-text", model="hf-internal-testing/tiny-random-BlipForConditionalGeneration")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        text = "a photo of"

        outputs = pipe(image, text=text)
        self.assertTrue(outputs[0]["generated_text"].startswith(text))

    @require_torch
    def test_consistent_batching_behaviour(self):
        pipe = pipeline("image-text-to-text", model="hf-internal-testing/tiny-random-BlipForConditionalGeneration")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        text = "a photo of"

        outputs = pipe([image, image], text=text)
        self.assertTrue(outputs[0][0]["generated_text"].startswith(text))
        self.assertTrue(outputs[1][0]["generated_text"].startswith(text))

        outputs = pipe([image, image], text=text, batch_size=2)
        self.assertTrue(outputs[0][0]["generated_text"].startswith(text))
        self.assertTrue(outputs[1][0]["generated_text"].startswith(text))

        from torch.utils.data import Dataset

        class MyDataset(Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, i):
                return "./tests/fixtures/tests_samples/COCO/000000039769.png"

        dataset = MyDataset()
        for batch_size in (1, 2, 4):
            outputs = pipe(dataset, text=text, batch_size=batch_size if batch_size > 1 else None)
            self.assertTrue(list(outputs)[0][0]["generated_text"].startswith(text))
            self.assertTrue(list(outputs)[1][0]["generated_text"].startswith(text))

    @slow
    @require_torch
    def test_large_model_pt(self):
        pipe = pipeline("image-text-to-text", model="ydshieh/vit-gpt2-coco-en")
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
        pipe = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
        url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
        image = Image.open(requests.get(url, stream=True).raw)

        outputs = pipe(image)
        self.assertEqual(outputs, [{"generated_text": "a pink pokemon pokemon with a blue shirt and a blue shirt"}])

    @slow
    @require_torch
    def test_generation_pt_git(self):
        pipe = pipeline("image-text-to-text", model="microsoft/git-base-coco")
        url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
        image = Image.open(requests.get(url, stream=True).raw)

        outputs = pipe(image)
        self.assertEqual(outputs, [{"generated_text": "a cartoon of a purple character."}])

    @slow
    @require_torch
    def test_conditional_generation_pt_blip(self):
        pipe = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        text = "a photography of"

        outputs = pipe(image, text=text)
        self.assertEqual(outputs, [{"generated_text": "a photography of a volcano"}])

        with self.assertRaises(ValueError):
            outputs = pipe([image, image], text=[text, text])

    @slow
    @require_torch
    def test_conditional_generation_pt_git(self):
        pipe = pipeline("image-text-to-text", model="microsoft/git-base-coco")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        text = "a photo of a"

        outputs = pipe(image, text=text)
        self.assertEqual(outputs, [{"generated_text": "a photo of a tent with a tent and a tent in the background."}])

        with self.assertRaises(ValueError):
            outputs = pipe([image, image], text=[text, text])

    @slow
    @require_torch
    def test_conditional_generation_pt_pix2struct(self):
        pipe = pipeline("image-text-to-text", model="google/pix2struct-ai2d-base")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        text = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

        outputs = pipe(image, text=text)
        self.assertEqual(outputs, [{"generated_text": "ash cloud"}])

        with self.assertRaises(ValueError):
            outputs = pipe([image, image], text=[text, text])

    @slow
    @require_torch
    def test_conditional_generation_llava(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/bakLlava-v1-hf")

        text = (
            "<image>\nUSER: What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud?\nASSISTANT:"
        )

        outputs = pipe(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
            text=text,
            generate_kwargs={"max_new_tokens": 200},
        )
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": "<image> \nUSER: What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud?\nASSISTANT: Lava"
                }
            ],
        )
