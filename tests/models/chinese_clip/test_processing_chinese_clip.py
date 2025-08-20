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

import pytest

from transformers import BertTokenizer, BertTokenizerFast
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.testing_utils import require_vision
from transformers.utils import FEATURE_EXTRACTOR_NAME, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import ChineseCLIPImageProcessor, ChineseCLIPProcessor


@require_vision
class ChineseCLIPProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ChineseCLIPProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "的",
            "价",
            "格",
            "是",
            "15",
            "便",
            "alex",
            "##andra",
            "，",
            "。",
            "-",
            "t",
            "shirt",
        ]
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        image_processor_map = {
            "do_resize": True,
            "size": {"height": 224, "width": 224},
            "do_center_crop": True,
            "crop_size": {"height": 18, "width": 18},
            "do_normalize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "do_convert_rgb": True,
        }
        cls.image_processor_file = os.path.join(cls.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(cls.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

        tokenizer = cls.get_tokenizer()
        image_processor = cls.get_image_processor()
        processor = ChineseCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, **kwargs):
        return BertTokenizer.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def get_rust_tokenizer(cls, **kwargs):
        return BertTokenizerFast.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def get_image_processor(cls, **kwargs):
        return ChineseCLIPImageProcessor.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        with tempfile.TemporaryDirectory() as tmpdir:
            processor_slow = ChineseCLIPProcessor(tokenizer=tokenizer_slow, image_processor=image_processor)
            processor_slow.save_pretrained(tmpdir)
            processor_slow = ChineseCLIPProcessor.from_pretrained(self.tmpdirname, use_fast=False)

            processor_fast = ChineseCLIPProcessor(tokenizer=tokenizer_fast, image_processor=image_processor)
            processor_fast.save_pretrained(tmpdir)
            processor_fast = ChineseCLIPProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, BertTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, BertTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, ChineseCLIPImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, ChineseCLIPImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = ChineseCLIPProcessor(
                tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor()
            )
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = self.get_tokenizer(cls_token="(CLS)", sep_token="(SEP)")
            image_processor_add_kwargs = self.get_image_processor(do_normalize=False)

            processor = ChineseCLIPProcessor.from_pretrained(
                tmpdir, cls_token="(CLS)", sep_token="(SEP)", do_normalize=False
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, BertTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, ChineseCLIPImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ChineseCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ChineseCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "Alexandra，T-shirt的价格是15便士。"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ChineseCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "Alexandra，T-shirt的价格是15便士。"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "token_type_ids", "attention_mask", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ChineseCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = ChineseCLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "Alexandra，T-shirt的价格是15便士。"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)
