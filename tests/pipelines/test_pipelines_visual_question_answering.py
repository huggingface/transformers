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

import tempfile
import unittest

from datasets import load_dataset

from transformers import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING, is_vision_available
from transformers.pipelines import pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    is_torch_available,
    nested_simplify,
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)

from .test_pipelines_common import ANY


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, CLIPImageProcessor, GitConfig, GitForCausalLM
    from transformers.pipelines.pt_utils import KeyDataset


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
class VisualQuestionAnsweringPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        dtype="float32",
    ):
        vqa_pipeline = pipeline(
            "visual-question-answering",
            model="hf-internal-testing/tiny-vilt-random-vqa",
            dtype=dtype,
        )
        examples = [
            {
                "image": Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "question": "How many cats are there?",
            },
            {
                "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "question": "How many cats are there?",
            },
        ]
        return vqa_pipeline, examples

    def run_pipeline_test(self, vqa_pipeline, examples):
        outputs = vqa_pipeline(examples, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question="How many cats are there?", top_k=2)
        self.assertEqual(
            outputs, [{"score": ANY(float), "answer": ANY(str)}, {"score": ANY(float), "answer": ANY(str)}]
        )

        outputs = vqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            outputs, [{"score": ANY(float), "answer": ANY(str)}, {"score": ANY(float), "answer": ANY(str)}]
        )

    @require_torch
    @require_torch_accelerator
    def test_small_model_pt_blip2(self):
        vqa_pipeline = pipeline(
            "visual-question-answering", model="hf-internal-testing/tiny-random-Blip2ForConditionalGeneration"
        )
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline({"image": image, "question": question})
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline([{"image": image, "question": question}, {"image": image, "question": question}])
        self.assertEqual(outputs, [[{"answer": ANY(str)}]] * 2)

        vqa_pipeline = pipeline(
            "visual-question-answering",
            model="hf-internal-testing/tiny-random-Blip2ForConditionalGeneration",
            model_kwargs={"dtype": torch.float16},
            device=torch_device,
        )
        self.assertEqual(vqa_pipeline.model.device, torch.device(f"{torch_device}:0"))
        self.assertEqual(vqa_pipeline.model.language_model.dtype, torch.float16)
        self.assertEqual(vqa_pipeline.model.vision_model.dtype, torch.float16)

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": ANY(str)}])

    @require_torch
    def test_small_model_pt_git(self):
        # GIT uses GitForCausalLM for VQA (generative approach).
        # Create a tiny GIT model from config since there is no public tiny-random model on the hub.
        config = GitConfig(
            vision_config={
                "num_channels": 3,
                "image_size": 30,
                "patch_size": 15,
                "hidden_size": 32,
                "projection_dim": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
            },
            vocab_size=30522,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
            max_position_embeddings=512,
        )
        model = GitForCausalLM(config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            # GIT uses a BertTokenizer and CLIPImageProcessor
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-bert")
            tokenizer.save_pretrained(tmp_dir)
            image_processor = CLIPImageProcessor(size={"height": 30, "width": 30}, crop_size={"height": 30, "width": 30})
            image_processor.save_pretrained(tmp_dir)

            vqa_pipeline = pipeline(
                "visual-question-answering",
                model=tmp_dir,
            )

        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline({"image": image, "question": question})
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline([{"image": image, "question": question}, {"image": image, "question": question}])
        self.assertEqual(outputs, [[{"answer": ANY(str)}]] * 2)

    @slow
    @require_torch
    def test_large_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4), [{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]
        )

        outputs = vqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4), [{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]
        )

        outputs = vqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [[{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]] * 2,
        )

    @slow
    @require_torch
    @require_torch_accelerator
    def test_large_model_pt_blip2(self):
        vqa_pipeline = pipeline(
            "visual-question-answering",
            model="Salesforce/blip2-opt-2.7b",
            model_kwargs={"dtype": torch.float16},
            device=torch_device,
        )
        self.assertEqual(vqa_pipeline.model.device, torch.device(f"{torch_device}:0"))
        self.assertEqual(vqa_pipeline.model.language_model.dtype, torch.float16)

        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "Question: how many cats are there? Answer:"

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": "two"}])

        outputs = vqa_pipeline({"image": image, "question": question})
        self.assertEqual(outputs, [{"answer": "two"}])

        outputs = vqa_pipeline([{"image": image, "question": question}, {"image": image, "question": question}])
        self.assertEqual(outputs, [[{"answer": "two"}]] * 2)

    @slow
    @require_torch
    def test_large_model_pt_git(self):
        vqa_pipeline = pipeline("visual-question-answering", model="microsoft/git-base-textvqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline({"image": image, "question": question})
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline([{"image": image, "question": question}, {"image": image, "question": question}])
        self.assertEqual(outputs, [[{"answer": ANY(str)}]] * 2)

    @require_torch
    def test_small_model_pt_image_list(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        images = [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000004016.png",
        ]

        outputs = vqa_pipeline(image=images, question="How many cats are there?", top_k=1)
        self.assertEqual(
            outputs, [[{"score": ANY(float), "answer": ANY(str)}], [{"score": ANY(float), "answer": ANY(str)}]]
        )

    @require_torch
    def test_small_model_pt_question_list(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        questions = ["How many cats are there?", "Are there any dogs?"]

        outputs = vqa_pipeline(image=image, question=questions, top_k=1)
        self.assertEqual(
            outputs, [[{"score": ANY(float), "answer": ANY(str)}], [{"score": ANY(float), "answer": ANY(str)}]]
        )

    @require_torch
    def test_small_model_pt_both_list(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        images = [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000004016.png",
        ]
        questions = ["How many cats are there?", "Are there any dogs?"]

        outputs = vqa_pipeline(image=images, question=questions, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )

    @require_torch
    def test_small_model_pt_dataset(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        dataset = load_dataset("hf-internal-testing/dummy_image_text_data", split="train[:2]")
        question = "What's in the image?"

        outputs = vqa_pipeline(image=KeyDataset(dataset, "image"), question=question, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )
