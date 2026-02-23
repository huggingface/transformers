# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch

from transformers import PI0Processor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_vision_available():
    from transformers import GemmaTokenizer, SiglipImageProcessor


@require_vision
class PI0ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PI0Processor

    @classmethod
    def _setup_image_processor(cls):
        image_processor = SiglipImageProcessor()
        image_processor.image_seq_length = 0
        return image_processor

    @classmethod
    def _setup_tokenizer(cls):
        return GemmaTokenizer.from_pretrained(SAMPLE_VOCAB, keep_accents=True)

    def test_get_num_vision_tokens(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100)])
        self.assertIn("num_image_tokens", output)
        self.assertEqual(len(output["num_image_tokens"]), 2)

    def test_image_processor_defaults(self):
        image_processor = self.get_component("image_processor")
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, text="", return_tensors="pt")

        for key in input_image_proc:
            torch.testing.assert_close(input_image_proc[key], input_processor[key][:, 0])
        self.assertTrue(torch.equal(input_processor["image_masks"], torch.tensor([[True]])))

    @require_torch
    def test_single_camera_output_is_5d(self):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        outputs = processor(images=image, text="task", return_tensors="pt")
        self.assertEqual(outputs["pixel_values"].ndim, 5)
        self.assertEqual(outputs["pixel_values"].shape[0], 1)
        self.assertEqual(outputs["pixel_values"].shape[1], 1)
        self.assertTrue(torch.equal(outputs["image_masks"], torch.tensor([[True]])))

    @require_torch
    def test_multi_camera_padding_and_masks(self):
        processor = self.get_processor()
        image_a = self.prepare_image_inputs()
        image_b = self.prepare_image_inputs()
        image_c = self.prepare_image_inputs()

        outputs = processor(
            images=[[image_a, image_b], [image_c]],
            text=["task a", "task b"],
            return_tensors="pt",
        )

        self.assertEqual(outputs["pixel_values"].ndim, 5)
        self.assertEqual(outputs["pixel_values"].shape[:2], torch.Size([2, 2]))
        self.assertTrue(torch.equal(outputs["image_masks"], torch.tensor([[True, True], [True, False]])))

    @require_torch
    def test_newline_normalization(self):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        out_no_newline = processor(images=image, text="pick object", return_tensors="pt")
        out_with_newline = processor(images=image, text="pick object\n", return_tensors="pt")
        self.assertTrue(torch.equal(out_no_newline["input_ids"], out_with_newline["input_ids"]))
        self.assertTrue(torch.equal(out_no_newline["attention_mask"], out_with_newline["attention_mask"]))
