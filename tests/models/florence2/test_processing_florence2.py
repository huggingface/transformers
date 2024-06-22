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

import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers import BartTokenizer, BartTokenizerFast
from transformers.models.bart.tokenization_bart import VOCAB_FILES_NAMES
from transformers.testing_utils import require_vision
from transformers.utils import IMAGE_PROCESSOR_NAME, is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import CLIPImageProcessor, Florence2Processor


@require_vision
class Florence2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = []  # TODO: add vocab tokens
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write("".join([x + "\n" for x in vocab_tokens]))

        self.processor_file = os.path.join(self.tmpdirname, IMAGE_PROCESSOR_NAME)

    def get_tokenizer(self, **kwargs):
        return BartTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return BartTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return CLIPImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images"""
        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        processor_slow = Florence2Processor(tokenizer=tokenizer_slow, image_processor=image_processor)
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = Florence2Processor.from_pretrained(self.tmpdirname)

        processor_fast = Florence2Processor(tokenizer=tokenizer_fast, image_processor=image_processor)
        processor_fast.save_pretrained(self.tmpdirname)
        processor_fast = Florence2Processor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, BartTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, BartTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, CLIPImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, CLIPImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        processor = Florence2Processor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = Florence2Processor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, BartTokenizer)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, CLIPImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Florence2Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_inputs = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_inputs, return_tensors="np")
        input_processor = processor(images=image_inputs, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Florence2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "huggingface is cool"

        encoded_processor = processor(text=input_str)
        encoded_tok = tokenizer(input_str)
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_processor[key], encoded_tok[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Florence2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "huggingface is cool"
        image_inputs = self.prepare_image_inputs()

        encoded_processor = processor(text=input_str, images=image_inputs)

        self.assertListEqual(
            list(encoded_processor.keys()), ["input_ids", "token_type_ids", "attention_mask", "pixel_values"]
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()
