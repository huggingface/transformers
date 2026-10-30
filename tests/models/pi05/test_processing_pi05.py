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

import numpy as np
import torch

from transformers import PI05Processor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_vision_available():
    from transformers import GemmaTokenizer, SiglipImageProcessor


@require_vision
class PI05ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PI05Processor

    @classmethod
    def _setup_image_processor(cls):
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
    def test_state_is_discretized_into_the_prompt(self):
        processor = self.get_processor()
        state = [[-1.0, -0.5, 0.0, 0.25, 0.999, 1.0, -0.9921875, 0.5]]
        outputs = processor(images=self.prepare_images_inputs(), text="pick_up the\nobject", state=state)

        expected_bins = [0, 64, 128, 160, 255, 255, 1, 192]
        lerobot_bins = np.digitize(np.array(state), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        self.assertEqual(lerobot_bins[0].tolist(), expected_bins)

        expected_prompt = f"{processor.tokenizer.bos_token}Task: pick up the object, State: {' '.join(map(str, expected_bins))};\nAction: "
        expected_inputs = processor.tokenizer(
            expected_prompt, padding="max_length", max_length=200, padding_side="right", return_tensors="pt"
        )
        self.assertTrue(torch.equal(outputs["input_ids"], expected_inputs["input_ids"]))
        self.assertNotIn("state", outputs)

    @require_torch
    def test_prompt_without_state(self):
        processor = self.get_processor()
        image = self.prepare_images_inputs()
        outputs = processor(images=image, text="pick object", return_tensors="pt")
        expected_inputs = processor.tokenizer(
            f"{processor.tokenizer.bos_token}Task: pick object;\nAction: ",
            padding="max_length",
            max_length=200,
            padding_side="right",
            return_tensors="pt",
        )
        self.assertTrue(torch.equal(outputs["input_ids"], expected_inputs["input_ids"]))

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

        self.assertEqual(outputs["pixel_values"].shape[:2], torch.Size([2, 2]))
        self.assertTrue(torch.equal(outputs["pixel_attention_mask"], torch.tensor([[True, True], [True, False]])))

    @unittest.skip("PI0.5 doesn't need vLLM integration")
    def test_get_num_multimodal_tokens_matches_processor_call(self):
        pass
