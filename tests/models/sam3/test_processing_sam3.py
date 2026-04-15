# Copyright 2025 HuggingFace Inc.
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

import unittest

from transformers import AutoTokenizer
from transformers.models.sam3.processing_sam3 import Sam3Processor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_vision
class Sam3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Sam3Processor

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", max_length=32, model_max_length=32)

    # Sam3Processor has a custom non-standard __call__ signature (no chat template, extra
    # prompting args like input_boxes). Skip mixin tests that assume a standard VLM interface.
    def test_chat_template_save_loading(self):
        self.skipTest("Sam3Processor does not use a chat template")

    def test_model_input_names(self):
        self.skipTest("Sam3Processor outputs extra keys (e.g. original_sizes) beyond model_input_names")

    def test_tokenizer_defaults(self):
        self.skipTest("Sam3Processor always pads tokenizer output to max_length=32")

    def test_processor_text_has_no_visual(self):
        self.skipTest("Sam3Processor has a custom interface, not a standard VLM text+image interface")

    def test_processor_with_multiple_inputs(self):
        self.skipTest("Sam3Processor has a custom interface, not a standard VLM text+image interface")

    # --- Sam3-specific tests ---

    def test_input_boxes_default_labels_mixed_batch(self):
        # Regression test for https://github.com/huggingface/transformers/issues/45059:
        # None entries should get pad label (-10), real entries should get positive label (1).
        processor = self.get_processor()
        images = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            images=images,
            text=["cat", None],
            input_boxes=[None, [[100, 100, 200, 200]]],
            return_tensors="pt",
        )
        self.assertIn("input_boxes", inputs)
        self.assertIn("input_boxes_labels", inputs)

        # The None entry (index 0) should have label -10 (pad value)
        self.assertEqual(inputs["input_boxes_labels"][0, 0].item(), -10)
        # The real entry (index 1) should have label 1 (positive)
        self.assertEqual(inputs["input_boxes_labels"][1, 0].item(), 1)

    def test_input_boxes_default_labels_all_real(self):
        processor = self.get_processor()
        images = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            images=images,
            text=["cat", "dog"],
            input_boxes=[[[50, 50, 150, 150]], [[200, 200, 300, 300]]],
            return_tensors="pt",
        )
        self.assertIn("input_boxes_labels", inputs)
        self.assertTrue((inputs["input_boxes_labels"] == 1).all())

    def test_no_input_boxes_omits_labels(self):
        processor = self.get_processor()
        images = self.prepare_image_inputs(batch_size=1)
        inputs = processor(
            images=images,
            text=["cat"],
            return_tensors="pt",
        )
        self.assertNotIn("input_boxes", inputs)
        self.assertNotIn("input_boxes_labels", inputs)

    def test_user_provided_labels_preserved(self):
        processor = self.get_processor()
        images = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            images=images,
            text=["cat", "dog"],
            input_boxes=[[[50, 50, 150, 150]], [[200, 200, 300, 300]]],
            input_boxes_labels=[[1], [0]],
            return_tensors="pt",
        )
        self.assertEqual(inputs["input_boxes_labels"][0, 0].item(), 1)
        self.assertEqual(inputs["input_boxes_labels"][1, 0].item(), 0)
