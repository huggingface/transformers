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
"""Testing suite for the MgpstrProcessor."""

import json
import os
import unittest

from transformers.models.mgp_str.tokenization_mgp_str import VOCAB_FILES_NAMES
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from transformers import MgpstrProcessor


@require_torch
@require_vision
class MgpstrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MgpstrProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab = ['[GO]', '[s]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']  # fmt: skip
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

        return tokenizer_class.from_pretrained(cls.tmpdirname)

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_map = {
            "do_normalize": False,
            "do_resize": True,
            "resample": 3,
            "size": {"height": 32, "width": 128},
        }
        return image_processor_class(**image_processor_map)

    # override as MgpstrProcessor returns "labels" and not "input_ids"
    def test_processor_with_multiple_inputs(self):
        processor = self.get_processor()

        input_str = "test"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "labels"])

        # Test that it raises error when no input is passed
        with self.assertRaises((TypeError, ValueError)):
            processor()

    # override as MgpstrTokenizer uses char_decode
    def test_tokenizer_decode_defaults(self):
        """
        Tests that tokenizer is called correctly when passing text to the processor.
        This test verifies that processor(text=X) produces the same output as tokenizer(X).
        """
        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)
        tokenizer = components["tokenizer"]

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.char_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)
        decode_strs = [seq.replace(" ", "") for seq in decoded_tok]

        self.assertListEqual(decode_strs, decoded_processor)

        char_input = torch.randn(1, 27, 38)
        bpe_input = torch.randn(1, 27, 50257)
        wp_input = torch.randn(1, 27, 30522)

        results = processor.batch_decode([char_input, bpe_input, wp_input])

        self.assertListEqual(list(results.keys()), ["generated_text", "scores", "char_preds", "bpe_preds", "wp_preds"])
