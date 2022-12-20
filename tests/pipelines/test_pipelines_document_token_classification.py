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

from transformers import MODEL_FOR_DOCUMENT_TOKEN_CLASSIFICATION_MAPPING, AutoTokenizer, AutoFeatureExtractor, is_vision_available
from transformers.pipelines import pipeline
from transformers.models.layoutlmv3.image_processing_layoutlmv3 import apply_tesseract as apply_ocr
from transformers.testing_utils import (
    nested_simplify,
    require_pytesseract,
    require_tf,
    require_torch,
    require_vision,
    require_detectron2,
    slow,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_vision_available():
    from PIL import Image

    from transformers.image_utils import load_image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass

    def load_image(_):
        return None


# This is a pinned image from a specific revision of a document question answering space, hosted by HuggingFace,
# so we can expect it to be available.
INVOICE_URL = (
    "https://huggingface.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/invoice.png"
)


@require_torch
@require_vision
class DocumentTokenClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_DOCUMENT_TOKEN_CLASSIFICATION_MAPPING

    @require_pytesseract
    @require_vision
    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        dtc_pipeline = pipeline(
            "document-token-classification", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )

        image = INVOICE_URL
        word_boxes = list(zip(*apply_ocr(load_image(image), None, "")))
        question = "What is the placebo?"
        examples = [
            {
                "image": load_image(image),
            },
            {
                "image": image,
            },
            {
                "image": image,
                "word_boxes": word_boxes,
            },
            {
                "image": None,
                "word_boxes": word_boxes,
            },
        ]
        return dtc_pipeline, examples

    def run_pipeline_test(self, dtc_pipeline, examples):
        outputs = dtc_pipeline(examples)
        self.assertEqual(
            outputs,
            [
                [
                    {"score": ANY(float), "answer": ANY(str), "start": ANY(int), "end": ANY(int)},
                    {"score": ANY(float), "answer": ANY(str), "start": ANY(int), "end": ANY(int)},
                ]
            ]
            * 4,
        )

    @require_torch
    @require_pytesseract
   # @require_detectron2
    def test_small_model_pt(self):
        dtc_pipeline = pipeline("document-token-classification", model="hf-internal-testing/tiny-random-LayoutLMv3ForTokenClassification")
        image = INVOICE_URL

        expected_output = [
            {"score": 0.0001, "answer": "oy 2312/2019", "start": 38, "end": 39},
            {"score": 0.0001, "answer": "oy 2312/2019 DUE", "start": 38, "end": 40},
        ]
        outputs = dtc_pipeline(image=image)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        outputs = dtc_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        # This image does not detect ANY text in it, meaning layoutlmv2 should fail.
        # Empty answer probably
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        outputs = dtc_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(outputs, [1])

        # We can optionnally pass directly the words and bounding boxes
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        words = []
        boxes = []
        outputs = dqa_pipeline(image=image, question=question, words=words, boxes=boxes, top_k=2)
        self.assertEqual(outputs, [])


    @slow
    @require_torch
    @require_pytesseract
    @require_vision
    def test_large_model_pt_layoutlm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "impira/layoutlm-document-qa", revision="3dc6de3", add_prefix_space=True
        )
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            tokenizer=tokenizer,
            revision="3dc6de3",
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
                ]
            ]
            * 2,
        )

        word_boxes = list(zip(*apply_ocr(load_image(image), None, "")))

        # This model should also work if `image` is set to None
        outputs = dqa_pipeline({"image": None, "word_boxes": word_boxes, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

#    @slow
    @require_torch
    @require_pytesseract
    @require_vision
    def test_large_model_pt_layoutlm_chunk(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlmv3-base", revision="07c9b08", add_prefix_space=True
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/layoutlmv3-base", revision="07c9b08"
        )
        dtc_pipeline = pipeline(
            "document-token-classification",
            model="microsoft/layoutlmv3-base",
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            revision="07c9b08",
            max_seq_len=50,
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dtc_pipeline(image=image)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dtc_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}],
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
                ]
            ]
            * 2,
        )

        word_boxes = list(zip(*apply_ocr(load_image(image), None, "")))

        # This model should also work if `image` is set to None
        outputs = dtc_pipeline({"image": None, "word_boxes": word_boxes})
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

    @require_tf
    @unittest.skip("Document Token Classification not implemented in TF")
    def test_small_model_tf(self):
        pass
