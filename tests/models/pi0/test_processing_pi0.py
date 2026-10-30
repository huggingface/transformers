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

import tempfile
import unittest

import torch

from transformers import PI0Processor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_vision_available():
    from transformers import GemmaTokenizer, PI0ImageProcessor, SiglipImageProcessor


@require_vision
class PI0ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PI0Processor

    @classmethod
    def _setup_image_processor(cls):
        # Use 64×64 instead of the default 224×224 to avoid large tensors.
        # image_seq_length=0 matches the processor attribute so token-count tests pass.
        image_processor = SiglipImageProcessor(size={"height": 64, "width": 64})
        image_processor.image_seq_length = 0
        return image_processor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer = GemmaTokenizer.from_pretrained(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        return tokenizer

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @require_vision
    def prepare_images_inputs(self, batch_size: int | None = None, nested: bool = True):
        return super().prepare_images_inputs(batch_size, nested=nested)

    def test_subprocessor_defaults_1_images(self):
        image_processor = self.get_component("image_processor")
        processor = self.get_processor()
        image_input = self.prepare_images_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, text="", return_tensors="pt")

        for key in input_image_proc:
            torch.testing.assert_close(input_image_proc[key], input_processor[key][:, 0])
        self.assertTrue(torch.equal(input_processor["pixel_attention_mask"], torch.tensor([[True]])))

    @require_torch
    def test_single_camera_output_is_5d(self):
        processor = self.get_processor()
        image = self.prepare_images_inputs()
        outputs = processor(images=image, text="task", return_tensors="pt")
        self.assertEqual(outputs["pixel_values"].ndim, 5)
        self.assertEqual(outputs["pixel_values"].shape[0], 1)
        self.assertEqual(outputs["pixel_values"].shape[1], 1)
        self.assertTrue(torch.equal(outputs["pixel_attention_mask"], torch.tensor([[True]])))

    @require_torch
    def test_multi_camera_padding_and_masks(self):
        processor = self.get_processor()
        image_a = self.prepare_images_inputs()
        image_b = self.prepare_images_inputs()
        image_c = self.prepare_images_inputs()

        outputs = processor(
            images=[[image_a, image_b], [image_c]],
            text=["task a", "task b"],
            return_tensors="pt",
        )

        self.assertEqual(outputs["pixel_values"].ndim, 5)
        self.assertEqual(outputs["pixel_values"].shape[:2], torch.Size([2, 2]))
        self.assertTrue(torch.equal(outputs["pixel_attention_mask"], torch.tensor([[True, True], [True, False]])))

    @require_torch
    def test_newline_normalization(self):
        processor = self.get_processor()
        image = self.prepare_images_inputs()
        out_no_newline = processor(images=image, text="pick object", return_tensors="pt")
        out_with_newline = processor(images=image, text="pick object\n", return_tensors="pt")
        self.assertTrue(torch.equal(out_no_newline["input_ids"], out_with_newline["input_ids"]))
        self.assertTrue(torch.equal(out_no_newline["attention_mask"], out_with_newline["attention_mask"]))

    @require_torch
    def test_processor_with_pi0_image_processor(self):
        processor = self.processor_class(
            image_processor=PI0ImageProcessor(), tokenizer=self.get_component("tokenizer")
        )
        outputs = processor(images=self.prepare_images_inputs(), text="pick", padding="longest")
        self.assertEqual(outputs["pixel_values"].shape, torch.Size([1, 1, 3, 224, 224]))
        self.assertEqual((outputs["input_ids"] == processor.image_token_id).sum().item(), 256)

    def test_normalization_statistics_survive_save_and_load(self):
        processor = self.processor_class(**self.prepare_components(), state_mean=[1.0] * 8, chunk_size=10)
        with tempfile.TemporaryDirectory() as saved_directory:
            processor.save_pretrained(saved_directory)
            reloaded = self.processor_class.from_pretrained(saved_directory)
        self.assertEqual(reloaded.state_mean, [1.0] * 8)
        self.assertEqual(reloaded.chunk_size, 10)

    @unittest.skip("PI0 doesn't need vLLM integration")
    def test_get_num_multimodal_tokens_matches_processor_call(self):
        pass
