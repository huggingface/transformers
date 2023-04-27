# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import numpy as np
import pytest

from transformers import BertTokenizer, BertTokenizerFast
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.testing_utils import require_vision
from transformers.utils import IMAGE_PROCESSOR_NAME, is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import AlignProcessor, EfficientNetImageProcessor


@require_vision
class AlignProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

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
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        image_processor_map = {
            "do_resize": True,
            "size": 20,
            "do_center_crop": True,
            "crop_size": 18,
            "do_normalize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        self.image_processor_file = os.path.join(self.tmpdirname, IMAGE_PROCESSOR_NAME)
        with open(self.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

    def get_tokenizer(self, **kwargs):
        return BertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return BertTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return EfficientNetImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
        return image_inputs

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_rust_tokenizer()
        image_processor = self.get_image_processor()

        processor_slow = AlignProcessor(tokenizer=tokenizer_slow, image_processor=image_processor)
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = AlignProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        processor_fast = AlignProcessor(tokenizer=tokenizer_fast, image_processor=image_processor)
        processor_fast.save_pretrained(self.tmpdirname)
        processor_fast = AlignProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, BertTokenizer)
        self.assertIsInstance(processor_fast.tokenizer, BertTokenizerFast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, EfficientNetImageProcessor)
        self.assertIsInstance(processor_fast.image_processor, EfficientNetImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        processor = AlignProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = AlignProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, BertTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, EfficientNetImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = AlignProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = AlignProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str, padding="max_length", max_length=64)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = AlignProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "token_type_ids", "attention_mask", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = AlignProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = AlignProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)
