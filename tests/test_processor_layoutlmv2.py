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

    # integration tests (3 cases)

    def test_processor_case_1(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        image = Image.open("tests/fixtures/tests_samples/DocVQA/document.png").convert("RGB")

        input_feat_extract = feature_extractor(image, return_tensors="np")
        input_processor = processor(image, return_tensors="np")

        self.assertAlmostEqual(input_feat_extract["image"].sum(), input_processor["image"].sum(), delta=1e-2)
