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
import shutil
import tempfile
import unittest

from transformers import AutoProcessor, BartTokenizerFast, Florence2Processor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import CLIPImageProcessor


@require_torch
@require_vision
class Florence2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Florence2Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        image_processor = CLIPImageProcessor.from_pretrained("ducviet00/Florence-2-base-hf")
        image_processor.image_seq_length = 0
        tokenizer = BartTokenizerFast.from_pretrained("ducviet00/Florence-2-base-hf")
        tokenizer.image_token = "<image>"
        tokenizer.image_token_id = tokenizer.encode(tokenizer.image_token, add_special_tokens=False)[0]
        tokenizer.extra_special_tokens = {"image_token": "<image>"}
        processor_kwargs = cls.prepare_processor_dict()
        processor = Florence2Processor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

    @staticmethod
    def prepare_processor_dict():
        return {
            "post_processor_config": {
                "ocr": {
                    "pattern": r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
                    "area_threshold": 0.0,
                },
                "phrase_grounding": {"banned_grounding_tokens": ["the image"]},
                "pure_text": {},
                "description_with_bboxes": {},
                "description_with_polygons": {},
                "polygons": {},
                "bboxes": {},
                "description_with_bboxes_or_polygons": {},
            }
        }

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_construct_prompts(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)

        # Test single text without task token
        text = "This is a simple text."
        prompts = processor._construct_prompts(text)
        self.assertEqual(prompts, [text])

        # Test list of texts with task without input
        texts = ["<OCR>", "<CAPTION>"]
        prompts = processor._construct_prompts(texts)
        EXPECTED_PROMPTS_WITHOUT_INPUT = ["What is the text in the image?", "What does the image describe?"]
        self.assertEqual(prompts, EXPECTED_PROMPTS_WITHOUT_INPUT)

        # Test task with input
        texts = ["<CAPTION_TO_PHRASE_GROUNDING> a red car"]
        prompts = processor._construct_prompts(texts)
        EXPECTED_PROMPTS_WITH_INPUT = ["Locate the phrases in the caption: a red car"]
        self.assertEqual(prompts, EXPECTED_PROMPTS_WITH_INPUT)

        # Test invalid prompt with task token not alone
        with self.assertRaises(ValueError):
            processor._construct_prompts("<OCR> extra text")

    def test_quantizer_quantize_dequantize(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)

        # Test bounding box quantization and dequantization
        boxes = torch.tensor([[0, 0, 30, 40], [500, 550, 600, 690], [750, 1121, 851, 1239]], dtype=torch.int32)
        size = (800, 1200)
        quantized_boxes = processor.post_processor.quantize(boxes, size)
        dequantized_boxes = processor.post_processor.dequantize(quantized_boxes, size)
        EXPECTED_DEQUANTIZED_BBOX = torch.tensor(
            [[0, 0, 30, 40], [500, 550, 600, 690], [750, 1121, 799, 1199]], dtype=torch.int32
        )
        self.assertTrue(torch.allclose(dequantized_boxes, EXPECTED_DEQUANTIZED_BBOX))

        # Test points quantization and dequantization
        points = torch.tensor([[0, 0], [300, 400], [850, 1250]], dtype=torch.int32)
        quantized_points = processor.post_processor.quantize(points, size)
        dequantized_points = processor.post_processor.dequantize(quantized_points, size)
        EXPECTED_DEQUANTIZED_POINTS = torch.tensor([[0, 0], [300, 400], [799, 1199]], dtype=torch.int32)
        self.assertTrue(torch.allclose(dequantized_points, EXPECTED_DEQUANTIZED_POINTS))

        # Test invalid shape
        with self.assertRaises(ValueError):
            processor.post_processor.quantize(torch.tensor([[1, 2, 3]]), size)

    def test_post_process_parse_description_with_bboxes_from_text_and_spans(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        text_without_phrase = "</s><s><loc_53><loc_334><loc_933><loc_775><loc_711><loc_203><loc_906><loc_546><loc_585><loc_309><loc_774><loc_709><loc_577></s><pad>"
        image_size = (1000, 1000)
        parsed_text_without_phrase = processor.post_processor.parse_description_with_bboxes_from_text_and_spans(
            text_without_phrase, image_size=image_size, allow_empty_phrase=True
        )
        EXPECTED_PARSED_TEXT_WITHOUT_PHRASE = [
            {"bbox": [53, 334, 933, 775], "cat_name": ""},
            {"bbox": [711, 203, 906, 546], "cat_name": ""},
            {"bbox": [585, 309, 774, 709], "cat_name": ""},
        ]
        self.assertEqual(parsed_text_without_phrase, EXPECTED_PARSED_TEXT_WITHOUT_PHRASE)

        text_with_phrase = (
            "</s><s>car<loc_53><loc_334><loc_933><loc_775>door handle<loc_425><loc_504><loc_474><loc_516></s><pad>"
        )
        image_size = (1000, 1000)
        parsed_text_with_phrase = processor.post_processor.parse_description_with_bboxes_from_text_and_spans(
            text_with_phrase, image_size=image_size, allow_empty_phrase=False
        )
        EXPECTED_PARSED_TEXT_WITH_PHRASE = [
            {"bbox": [53, 334, 933, 775], "cat_name": "car"},
            {"bbox": [425, 504, 474, 516], "cat_name": "door handle"},
        ]
        self.assertEqual(parsed_text_with_phrase, EXPECTED_PARSED_TEXT_WITH_PHRASE)

    def test_post_process_parse_description_with_polygons_from_text_and_spans(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        text_without_phrase = "<loc_279><loc_379><loc_282><loc_379><loc_290><loc_373><loc_293><loc_373><loc_298><loc_369><loc_301><loc_369>"
        image_size = (1000, 1000)
        parsed_text_without_phrase = processor.post_processor.parse_description_with_polygons_from_text_and_spans(
            text_without_phrase, image_size=image_size, allow_empty_phrase=True
        )
        EXPECTED_PARSED_TEXT_WITHOUT_PHRASE = [
            {
                "cat_name": "",
                "polygons": [[279, 379, 282, 379, 290, 373, 293, 373, 298, 369, 301, 369]],
            }
        ]
        self.assertEqual(parsed_text_without_phrase, EXPECTED_PARSED_TEXT_WITHOUT_PHRASE)

        text_with_phrase = (
            "Hello<loc_769><loc_248><loc_771><loc_234><loc_773><loc_206><loc_773><loc_198><loc_771><loc_193>"
        )
        image_size = (1000, 1000)
        parsed_text_with_phrase = processor.post_processor.parse_description_with_polygons_from_text_and_spans(
            text_with_phrase, image_size=image_size, allow_empty_phrase=False
        )
        EXPECTED_PARSED_TEXT_WITH_PHRASE = [
            {
                "cat_name": "Hello",
                "polygons": [[769, 248, 771, 234, 773, 206, 773, 198, 771, 193]],
            }
        ]
        self.assertEqual(parsed_text_with_phrase, EXPECTED_PARSED_TEXT_WITH_PHRASE)

    def test_post_process_parse_ocr_from_text_and_spans(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        text = "</s><s>Hello<loc_100><loc_100><loc_200><loc_100><loc_200><loc_200><loc_100><loc_200>World<loc_300><loc_300><loc_400><loc_300><loc_400><loc_400><loc_300><loc_400></s>"
        image_size = (1000, 1000)
        parsed = processor.post_processor.parse_ocr_from_text_and_spans(
            text, pattern=None, image_size=image_size, area_threshold=0.0
        )
        EXPECTED_PARSED_OCR = [
            {"quad_box": [100, 100, 200, 100, 200, 200, 100, 200], "text": "Hello"},
            {"quad_box": [300, 300, 400, 300, 400, 400, 300, 400], "text": "World"},
        ]
        self.assertEqual(parsed, EXPECTED_PARSED_OCR)

        # Test with area threshold filtering
        small_text = "Small<loc_1><loc_1><loc_2><loc_2><loc_2><loc_2><loc_1><loc_1>"
        parsed_small = processor.post_processor.parse_ocr_from_text_and_spans(
            small_text, pattern=None, image_size=image_size, area_threshold=0.01
        )
        EXPECTED_PARSED_OCR_SMALL = []
        self.assertEqual(parsed_small, EXPECTED_PARSED_OCR_SMALL)

    def test_post_process_parse_phrase_grounding_from_text_and_spans(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        text = "</s><s>red car<loc_53><loc_334><loc_933><loc_775><loc_711><loc_203><loc_906><loc_546>sky<loc_0><loc_0><loc_1000><loc_300></s>"
        image_size = (1000, 1000)
        parsed = processor.post_processor.parse_phrase_grounding_from_text_and_spans(text, image_size=image_size)
        EXPECTED_PARSED_PHRASE_GROUNDING = [
            {"bbox": [[53, 334, 933, 775], [711, 203, 906, 546]], "cat_name": "red car"},
            {"bbox": [[0, 0, 1000, 300]], "cat_name": "sky"},
        ]
        self.assertEqual(parsed, EXPECTED_PARSED_PHRASE_GROUNDING)

        # Test with blacklisted phrase
        blacklisted_text = "the image<loc_100><loc_100><loc_200><loc_200>"
        parsed_blacklisted = processor.post_processor.parse_phrase_grounding_from_text_and_spans(
            blacklisted_text, image_size=image_size
        )
        EXPECTED_PARSED_BLACKLISTED = []
        self.assertEqual(parsed_blacklisted, EXPECTED_PARSED_BLACKLISTED)

    def test_post_process_generation(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)

        # Test pure_text task
        text = "<s>Hello world</s>"
        cap_result = processor.post_process_generation(text=text, task="<CAPTION>", image_size=None)
        EXPECTED_PURE_TEXT_RESULT = {"<CAPTION>": "Hello world"}
        self.assertEqual(cap_result, EXPECTED_PURE_TEXT_RESULT)

        # Test description_with_bboxes task
        text = "car<loc_53><loc_334><loc_933><loc_775>"
        od_result = processor.post_process_generation(text=text, task="<OD>", image_size=(1000, 1000))
        EXPECTED_BBOXES_RESULT = {"<OD>": {"bboxes": [[53, 334, 933, 775]], "labels": ["car"]}}
        self.assertEqual(od_result, EXPECTED_BBOXES_RESULT)

        # Test OCR task
        text = "Hello<loc_100><loc_100><loc_200><loc_100><loc_200><loc_200><loc_100><loc_200>"
        ocr_result = processor.post_process_generation(text=text, task="<OCR_WITH_REGION>", image_size=(1000, 1000))
        EXPECTED_OCR_RESULT = {
            "<OCR_WITH_REGION>": {"quad_boxes": [[100, 100, 200, 100, 200, 200, 100, 200]], "labels": ["Hello"]}
        }
        self.assertEqual(ocr_result, EXPECTED_OCR_RESULT)
