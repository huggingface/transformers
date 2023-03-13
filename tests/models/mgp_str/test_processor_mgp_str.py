# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the MgpstrProcessor. """

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers import MgpstrTokenizer
from transformers.models.mgp_str.tokenization_mgp_str import VOCAB_FILES_NAMES
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import IMAGE_PROCESSOR_NAME, is_torch_available, is_vision_available


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image

    from transformers import MgpstrProcessor, ViTImageProcessor


@require_torch
@require_vision
class MgpstrProcessorTest(unittest.TestCase):
    image_processing_class = ViTImageProcessor if is_vision_available() else None

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def setUp(self):
        self.image_size = (3, 32, 128)
        self.tmpdirname = tempfile.mkdtemp()

        # fmt: off
        vocab = ['[GO]', '[s]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        # fmt: on
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

        image_processor_map = {
            "do_normalize": False,
            "do_resize": True,
            "feature_extractor_type": "ViTFeatureExtractor",
            "resample": 3,
            "size": {"height": 32, "width": 128},
        }
        self.image_processor_file = os.path.join(self.tmpdirname, IMAGE_PROCESSOR_NAME)
        with open(self.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

    def get_tokenizer(self, **kwargs):
        return MgpstrTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return ViTImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images."""

        image_input = np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)

        image_input = Image.fromarray(np.moveaxis(image_input, 0, -1))

        return image_input

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)
        processor = MgpstrProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.char_tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.char_tokenizer, MgpstrTokenizer)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.image_processor, ViTImageProcessor)

    def test_save_load_pretrained_additional_features(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = MgpstrProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.char_tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.char_tokenizer, MgpstrTokenizer)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, ViTImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "test"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "test"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "labels"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.char_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)
        decode_strs = [seq.replace(" ", "") for seq in decoded_tok]

        self.assertListEqual(decode_strs, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = None
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)

    def test_processor_batch_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MgpstrProcessor(tokenizer=tokenizer, image_processor=image_processor)

        char_input = torch.randn(1, 27, 38)
        bpe_input = torch.randn(1, 27, 50257)
        wp_input = torch.randn(1, 27, 30522)

        results = processor.batch_decode([char_input, bpe_input, wp_input])

        self.assertListEqual(list(results.keys()), ["generated_text", "scores", "char_preds", "bpe_preds", "wp_preds"])
