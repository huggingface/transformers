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

import json
import os
import shutil
import tempfile
import unittest

from PIL import Image

from transformers.file_utils import FEATURE_EXTRACTOR_NAME
from transformers.models.layoutlmv2 import LayoutLMv2FeatureExtractor, LayoutLMv2Processor, LayoutLMv2Tokenizer
from transformers.models.layoutlmv2.tokenization_layoutlmv2 import VOCAB_FILES_NAMES
from transformers.testing_utils import slow


class LayoutLMv2ProcessorTest(unittest.TestCase):
    def setUp(self):
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]

        feature_extractor_map = {
            "do_resize": True,
            "size": 224,
            "apply_ocr": True,
        }

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        self.feature_extraction_file = os.path.join(self.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(self.feature_extraction_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(feature_extractor_map) + "\n")

    def get_tokenizer(self, **kwargs):
        return LayoutLMv2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return LayoutLMv2FeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        processor.save_pretrained(self.tmpdirname)
        processor = LayoutLMv2Processor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, LayoutLMv2Tokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, LayoutLMv2FeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = LayoutLMv2Processor(feature_extractor=self.get_feature_extractor(), tokenizer=self.get_tokenizer())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_resize=False, size=30)

        processor = LayoutLMv2Processor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_resize=False, size=30
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, LayoutLMv2Tokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, LayoutLMv2FeatureExtractor)

    # integration tests (7 cases)
    @slow
    def test_processor_case_1(self):
        # case 1: document image classification (training, inference) + token classification (inference), apply_ocr = True

        feature_extractor = self.get_feature_extractor()
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        image = Image.open("tests/fixtures/tests_samples/DocVQA/document.png").convert("RGB")

        input_feat_extract = feature_extractor(image, return_tensors="pt")
        input_processor = processor(image, return_tensors="pt")

        # verify keys
        expected_keys = ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]
        self.assertListEqual(list(input_processor.keys()), expected_keys)

        # verify image
        self.assertAlmostEqual(input_feat_extract["pixel_values"].sum(), input_processor["image"].sum(), delta=1e-2)

        # verify input_ids
        # fmt: off
        expected_decoding = "[CLS] 11 : 14 to 11 : 39 a. m 11 : 39 to 11 : 44 a. m. 11 : 44 a. m. to 12 : 25 p. m. 12 : 25 to 12 : 58 p. m. 12 : 58 to 4 : 00 p. m. 2 : 00 to 5 : 00 p. m. coffee break coffee will be served for men and women in the lobby adjacent to exhibit area. please move into exhibit area. ( exhibits open ) trrf general session ( part | ) presiding : lee a. waller trrf vice president “ introductory remarks ” lee a. waller, trrf vice presi - dent individual interviews with trrf public board members and sci - entific advisory council mem - bers conducted by trrf treasurer philip g. kuehn to get answers which the public refrigerated warehousing industry is looking for. plus questions from the floor. dr. emil m. mrak, university of cal - ifornia, chairman, trrf board ; sam r. cecil, university of georgia college of agriculture ; dr. stanley charm, tufts university school of medicine ; dr. robert h. cotton, itt continental baking company ; dr. owen fennema, university of wis - consin ; dr. robert e. hardenburg, usda. questions and answers exhibits open capt. jack stoney room trrf scientific advisory council meeting ballroom foyer [SEP]" # noqa: E231
        # fmt: on
        decoding = tokenizer.decode(input_processor.input_ids.squeeze().tolist())
        self.assertSequenceEqual(decoding, expected_decoding)

    def test_processor_case_2(self):
        # case 2: document image classification (training, inference) + token classification (inference), apply_ocr=False

        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        image = Image.open("tests/fixtures/tests_samples/DocVQA/document.png").convert("RGB")

        words = ["hello", "world"]
        boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        input_processor = processor(image, words, boxes=boxes, return_tensors="pt")

        # verify keys
        expected_keys = ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]
        self.assertListEqual(list(input_processor.keys()), expected_keys)

        # verify input_ids
        expected_decoding = "[CLS] hello world [SEP]"
        decoding = tokenizer.decode(input_processor.input_ids.squeeze().tolist())
        self.assertSequenceEqual(decoding, expected_decoding)

    def test_processor_case_3(self):
        # case 3: token classification (training), apply_ocr=False

        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        image = Image.open("tests/fixtures/tests_samples/DocVQA/document.png").convert("RGB")

        words = ["weirdly", "world"]
        boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        word_labels = [1, 2]
        input_processor = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

        # verify keys
        expected_keys = ["input_ids", "bbox", "token_type_ids", "labels", "attention_mask", "image"]
        self.assertListEqual(list(input_processor.keys()), expected_keys)

        # verify input_ids
        expected_decoding = "[CLS] weirdly world [SEP]"
        decoding = tokenizer.decode(input_processor.input_ids.squeeze().tolist())
        self.assertSequenceEqual(decoding, expected_decoding)

        # verify labels
        expected_labels = [-100, 1, -100, 2, -100]
        self.assertListEqual(input_processor.labels.squeeze().tolist(), expected_labels)

    def test_processor_case_4(self):
        # case 4: visual question answering (inference), apply_ocr=True

        feature_extractor = LayoutLMv2FeatureExtractor()
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        image = Image.open("tests/fixtures/tests_samples/DocVQA/document.png").convert("RGB")

        question = "What's his name?"
        input_processor = processor(image, question, return_tensors="pt")

        # verify keys
        expected_keys = ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]
        self.assertListEqual(list(input_processor.keys()), expected_keys)

        # verify input_ids
        # fmt: off
        expected_decoding = "[CLS] what's his name? [SEP] 11 : 14 to 11 : 39 a. m 11 : 39 to 11 : 44 a. m. 11 : 44 a. m. to 12 : 25 p. m. 12 : 25 to 12 : 58 p. m. 12 : 58 to 4 : 00 p. m. 2 : 00 to 5 : 00 p. m. coffee break coffee will be served for men and women in the lobby adjacent to exhibit area. please move into exhibit area. ( exhibits open ) trrf general session ( part | ) presiding : lee a. waller trrf vice president “ introductory remarks ” lee a. waller, trrf vice presi - dent individual interviews with trrf public board members and sci - entific advisory council mem - bers conducted by trrf treasurer philip g. kuehn to get answers which the public refrigerated warehousing industry is looking for. plus questions from the floor. dr. emil m. mrak, university of cal - ifornia, chairman, trrf board ; sam r. cecil, university of georgia college of agriculture ; dr. stanley charm, tufts university school of medicine ; dr. robert h. cotton, itt continental baking company ; dr. owen fennema, university of wis - consin ; dr. robert e. hardenburg, usda. questions and answers exhibits open capt. jack stoney room trrf scientific advisory council meeting ballroom foyer [SEP]" #noqa: E231
        # fmt: on
        decoding = tokenizer.decode(input_processor.input_ids.squeeze().tolist())
        self.assertSequenceEqual(decoding, expected_decoding)

    def test_processor_case_5(self):
        # case 4: visual question answering (inference), apply_ocr=False

        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        image = Image.open("tests/fixtures/tests_samples/DocVQA/document.png").convert("RGB")

        question = "What's his name?"
        words = ["hello", "world"]
        boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        input_processor = processor(image, question, words, boxes, return_tensors="pt")

        # verify keys
        expected_keys = ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]
        self.assertListEqual(list(input_processor.keys()), expected_keys)

        # verify input_ids
        expected_decoding = "[CLS] what's his name? [SEP] hello world [SEP]"
        decoding = tokenizer.decode(input_processor.input_ids.squeeze().tolist())
        self.assertSequenceEqual(decoding, expected_decoding)
