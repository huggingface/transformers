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
import shutil
import tempfile
import unittest

import pytest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        AutoProcessor,
        BertTokenizerFast,
        GPT2Tokenizer,
        InstructBlipVideoProcessor,
        PreTrainedTokenizerFast,
    )

    if is_torchvision_available():
        from transformers import InstructBlipVideoVideoProcessor


@require_vision
@require_torch
class InstructBlipVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InstructBlipVideoProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        video_processor = InstructBlipVideoVideoProcessor()
        tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        qformer_tokenizer = BertTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-bert")

        processor = InstructBlipVideoProcessor(video_processor, tokenizer, qformer_tokenizer)

        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_qformer_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).qformer_tokenizer

    def prepare_processor_dict(self):
        return {"num_query_tokens": 1}

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_additional_features(self):
        processor = InstructBlipVideoProcessor(
            tokenizer=self.get_tokenizer(),
            video_processor=self.get_video_processor(),
            qformer_tokenizer=self.get_qformer_tokenizer(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
            video_processor_add_kwargs = self.get_video_processor(do_normalize=False, padding_value=1.0)

            processor = InstructBlipVideoProcessor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.video_processor.to_json_string(), video_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.video_processor, InstructBlipVideoVideoProcessor)
        self.assertIsInstance(processor.qformer_tokenizer, BertTokenizerFast)

    def test_video_processor(self):
        video_processor = self.get_video_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()
        processor_kwargs = self.prepare_processor_dict()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer,
            video_processor=video_processor,
            qformer_tokenizer=qformer_tokenizer,
            **processor_kwargs,
        )

        image_input = self.prepare_image_inputs()

        input_feat_extract = video_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        video_processor = self.get_video_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()
        processor_kwargs = self.prepare_processor_dict()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer,
            video_processor=video_processor,
            qformer_tokenizer=qformer_tokenizer,
            **processor_kwargs,
        )

        input_str = ["lower newer"]
        encoded_processor = processor(text=input_str)
        encoded_tokens = tokenizer(input_str, return_token_type_ids=False)
        encoded_tokens_qformer = qformer_tokenizer(input_str, return_token_type_ids=False)

        for key in encoded_tokens:
            self.assertListEqual(encoded_tokens[key], encoded_processor[key])

        for key in encoded_tokens_qformer:
            self.assertListEqual(encoded_tokens_qformer[key], encoded_processor["qformer_" + key])

    def test_processor(self):
        video_processor = self.get_video_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()
        processor_kwargs = self.prepare_processor_dict()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer,
            video_processor=video_processor,
            qformer_tokenizer=qformer_tokenizer,
            **processor_kwargs,
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["qformer_input_ids", "qformer_attention_mask", "input_ids", "attention_mask", "pixel_values"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        video_processor = self.get_video_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()
        processor_kwargs = self.prepare_processor_dict()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer,
            video_processor=video_processor,
            qformer_tokenizer=qformer_tokenizer,
            **processor_kwargs,
        )

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)
