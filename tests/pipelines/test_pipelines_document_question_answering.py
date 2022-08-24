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

import json
import unittest

from transformers import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING, AutoTokenizer, is_vision_available
from transformers.pipelines import pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_detectron2,
    require_pytesseract,
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


@is_pipeline_test
@require_torch
@require_vision
class DocumentQuestionAnsweringPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        dqa_pipeline = pipeline(
            "document-question-answering", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
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
            {
                "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "question": "How many cats are there?",
                "word_boxes": json.load(open("./tests/fixtures/tests_samples/COCO/000000039769.json", "r")),
            },
        ]
        return dqa_pipeline, examples

    def run_pipeline_test(self, dqa_pipeline, examples):
        outputs = dqa_pipeline(examples, top_k=2)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )

    # TODO: Add layoutlmv1 once PR #18407 lands

    @require_torch
    @require_detectron2
    @require_pytesseract
    def test_small_model_pt_layoutlmv2(self):
        dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-layoutlmv2")
        image = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
        question = "How many cats are there?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(outputs, [{"score": 2.0, "answer": "te"}, {"score": 2.0, "answer": "te"}])

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(outputs, [{"score": 2.0, "answer": "te"}, {"score": 2.0, "answer": "te"}])

        # This image does not detect ANY text in it, meaning layoutlmv2 should fail.
        # Empty answer probably
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(outputs, [])

        # We can optionnally pass directly the words and bounding boxes
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        words = []
        boxes = []
        outputs = dqa_pipeline(image=image, question=question, words=words, boxes=boxes, top_k=2)
        self.assertEqual(outputs, [])

    def test_small_model_pt_donut(self):
        dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-donut")
        # dqa_pipeline = pipeline("document-question-answering", model="../tiny-random-donut")
        image = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
        question = "How many cats are there?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4), [{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]
        )

    @slow
    @require_torch
    @require_detectron2
    @require_pytesseract
    def test_large_model_pt_layoutlmv2(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
            revision="9977165",
        )
        image = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9966, "answer": "us-001", "start": 15, "end": 15},
                {"score": 0.0009, "answer": "us-001", "start": 15, "end": 15},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9966, "answer": "us-001", "start": 15, "end": 15},
                {"score": 0.0009, "answer": "us-001", "start": 15, "end": 15},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9966, "answer": "us-001", "start": 15, "end": 15},
                    {"score": 0.0009, "answer": "us-001", "start": 15, "end": 15},
                ],
            ]
            * 2,
        )

    @slow
    @require_torch
    def test_large_model_pt_donut(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="naver-clova-ix/donut-base-finetuned-docvqa",
            tokenizer=AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa"),
            feature_extractor="naver-clova-ix/donut-base-finetuned-docvqa",
        )

        image = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
        question = "What is the invoice number?"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), {"score": 0.5, "answer": "us-001"})

    @require_tf
    @unittest.skip("Document question answering not implemented in TF")
    def test_small_model_tf(self):
        pass
