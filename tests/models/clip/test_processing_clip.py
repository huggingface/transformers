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

import shutil
import tempfile
import unittest

import pytest

from transformers import AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CLIPImageProcessor, CLIPProcessor


TEST_MODEL_PATH = "openai/clip-vit-base-patch32"


@require_vision
class CLIPProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CLIPProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_PATH)
        image_processor = CLIPImageProcessor.from_pretrained(TEST_MODEL_PATH)
        processor = CLIPProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, **kwargs):
        return CLIPTokenizer.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def get_rust_tokenizer(cls, **kwargs):
        return CLIPTokenizerFast.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def get_image_processor(cls, **kwargs):
        return CLIPImageProcessor.from_pretrained(cls.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        with tempfile.TemporaryDirectory() as tmpdir:
            processor_slow = CLIPProcessor(tokenizer=tokenizer_slow, image_processor=image_processor)
            processor_slow.save_pretrained(tmpdir)
            processor_slow = CLIPProcessor.from_pretrained(tmpdir, use_fast=False)

            processor_fast = CLIPProcessor(tokenizer=tokenizer_fast, image_processor=image_processor)
            processor_fast.save_pretrained(tmpdir)
            processor_fast = CLIPProcessor.from_pretrained(tmpdir)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, CLIPTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, CLIPTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, CLIPImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, CLIPImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = CLIPProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = CLIPTokenizer.from_pretrained(tmpdir, bos_token="(BOS)", eos_token="(EOS)")
            image_processor_add_kwargs = CLIPImageProcessor.from_pretrained(
                tmpdir, do_normalize=False, padding_value=1.0
            )

            processor = CLIPProcessor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, CLIPTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, CLIPImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertSetEqual(set(inputs.keys()), {"input_ids", "attention_mask", "pixel_values"})

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)
