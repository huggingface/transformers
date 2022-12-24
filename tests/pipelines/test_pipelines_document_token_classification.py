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

from transformers import MODEL_FOR_DOCUMENT_TOKEN_CLASSIFICATION_MAPPING, AutoTokenizer, AutoFeatureExtractor, is_vision_available, AutoConfig, AutoModelForDocumentTokenClassification
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
    def test_small_model_pt(self):
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LayoutLMv3ForTokenClassification")
        config_ms= AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
        config.update(config_ms.to_dict())
        model = AutoModelForDocumentTokenClassification.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlmv3-base", revision="07c9b08", add_prefix_space=True
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/layoutlmv3-base", revision="07c9b08"
        )
        dtc_pipeline = pipeline("document-token-classification", 
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )
        image = INVOICE_URL
        outputs = dtc_pipeline(image=image)
        self.assertEqual(len(outputs["words"]), 95)
        self.assertEqual(len(outputs["words"]), len(outputs["word_labels"]))
        self.assertEqual(set(outputs["word_labels"]), set(['LABEL_0', 'LABEL_1']))

        outputs = dtc_pipeline({"image": image})
        self.assertEqual(len(outputs["words"]), 95)
        self.assertEqual(len(outputs["words"]), len(outputs["word_labels"]))
        self.assertEqual(set(outputs["word_labels"]), set(['LABEL_0', 'LABEL_1']))

        # No text detected -> empty list
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        outputs = dtc_pipeline(image=image)
        self.assertEqual(outputs["words"], [])
        self.assertEqual(outputs["boxes"], [])
        self.assertEqual(outputs["word_labels"], [])

        # We can pass the words and bounding boxes directly
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        words = []
        boxes = []
        outputs = dtc_pipeline(image=image, words=words, boxes=boxes)
        self.assertEqual(outputs["words"], [])
        self.assertEqual(outputs["boxes"], [])
        self.assertEqual(outputs["word_labels"], [])


#    @slow
    @require_torch
    @require_pytesseract
    @require_vision
    def test_large_model_pt_layoutlm(self):
        dtc_pipeline = pipeline(
            "document-token-classification",
            model="Theivaprakasham/layoutlmv3-finetuned-invoice",
        )
        image = INVOICE_URL

        outputs = dtc_pipeline(image=image)
        self.assertEqual(len(outputs["words"]), 95)
        self.assertEqual(len(outputs["words"]), len(outputs["word_labels"]))
        self.assertEqual(set(outputs["word_labels"]), {'B-BILLER_POST_CODE', 'B-BILLER', 'B-GST', 'O', 'B-TOTAL'})
        self.assertEqual(outputs["word_labels"].count("B-BILLER_POST_CODE"), 2)
        self.assertEqual(outputs["word_labels"].count("B-BILLER"), 2)
        self.assertEqual(outputs["word_labels"].count("B-GST"), 7)
        self.assertEqual(outputs["word_labels"].count("O"), 80)
        self.assertEqual(outputs["word_labels"].count("B-TOTAL"), 4)


        outputs = dtc_pipeline({"image": image})
        self.assertEqual(len(outputs["words"]), 95)
        self.assertEqual(len(outputs["words"]), len(outputs["word_labels"]))
        self.assertEqual(set(outputs["word_labels"]), {'B-BILLER_POST_CODE', 'B-BILLER', 'B-GST', 'O', 'B-TOTAL'})
        self.assertEqual(outputs["word_labels"].count("B-BILLER_POST_CODE"), 2)
        self.assertEqual(outputs["word_labels"].count("B-BILLER"), 2)
        self.assertEqual(outputs["word_labels"].count("B-GST"), 7)
        self.assertEqual(outputs["word_labels"].count("O"), 80)
        self.assertEqual(outputs["word_labels"].count("B-TOTAL"), 4)

        outputs = dtc_pipeline(
            [{"image": image}, {"image": image}]
        )
        self.assertEqual(len(outputs[0]["words"]), 95)
        self.assertEqual(len(outputs[0]["words"]), len(outputs[0]["word_labels"]))
        self.assertEqual(set(outputs[0]["word_labels"]), {'B-BILLER_POST_CODE', 'B-BILLER', 'B-GST', 'O', 'B-TOTAL'})
        self.assertEqual(outputs[0]["word_labels"].count("B-BILLER_POST_CODE"), 2)
        self.assertEqual(outputs[0]["word_labels"].count("B-BILLER"), 2)
        self.assertEqual(outputs[0]["word_labels"].count("B-GST"), 7)
        self.assertEqual(outputs[0]["word_labels"].count("O"), 80)
        self.assertEqual(outputs[0]["word_labels"].count("B-TOTAL"), 4)

        self.assertEqual(len(outputs[1]["words"]), 95)
        self.assertEqual(len(outputs[1]["words"]), len(outputs[1]["word_labels"]))
        self.assertEqual(set(outputs[1]["word_labels"]), {'B-BILLER_POST_CODE', 'B-BILLER', 'B-GST', 'O', 'B-TOTAL'})
        self.assertEqual(outputs[1]["word_labels"].count("B-BILLER_POST_CODE"), 2)
        self.assertEqual(outputs[1]["word_labels"].count("B-BILLER"), 2)
        self.assertEqual(outputs[1]["word_labels"].count("B-GST"), 7)
        self.assertEqual(outputs[1]["word_labels"].count("O"), 80)
        self.assertEqual(outputs[1]["word_labels"].count("B-TOTAL"), 4)

    @require_tf
    @unittest.skip("Document Token Classification not implemented in TF")
    def test_small_model_tf(self):
        pass
