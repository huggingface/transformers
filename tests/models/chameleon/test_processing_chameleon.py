# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch chameleon model."""

import unittest

from transformers import ChameleonProcessor
from transformers.testing_utils import get_tests_dir

from ...test_processing_common import ProcessorTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


class ChameleonProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ChameleonProcessor

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class(vocab_file=SAMPLE_VOCAB)
        tokenizer.pad_token_id = 0
        tokenizer.sep_token_id = 1
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        return tokenizer

    @unittest.skip("Chameleon processor add a sep_token at the end of each sample")
    def test_tokenizer_defaults(self):
        pass

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)

        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2,
            )

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 2}  # fmt: skip

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)
