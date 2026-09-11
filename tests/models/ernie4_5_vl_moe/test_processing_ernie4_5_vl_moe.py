# Copyright 2025 HuggingFace Inc team. All rights reserved.
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

import pytest

from transformers import AutoProcessor, TokenizersBackend
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Ernie4_5_VLMoeImageProcessor, Ernie4_5_VLMoeProcessor


@require_vision
@require_torch
@require_torchvision
class Ernie4_5_VLMoeProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ernie4_5_VLMoeProcessor
    # Use tiny repos to avoid loading the full 100k-vocab tokenizer (~342 MB)
    # Tiny processor created with make_tiny_processor.py from "hf-internal-testing/Ernie-VL-Moe-Small"
    tiny_model_id = "hf-internal-testing/tiny-processor-ernie4_5_vl_moe"

    @classmethod
    def _setup_video_processor(cls):
        component = AutoProcessor.from_pretrained(cls.tiny_model_id, min_frames=1).video_processor
        return component

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": None, "fps": 3, "expected_dim": 0, "output_length": 384},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 2304},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 2304},
            {"expected_dim": 0, "output_length": 2304},
        ]

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_component("tokenizer")
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")

        processor = Ernie4_5_VLMoeProcessor(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        processor.save_pretrained(self.tmpdirname)
        processor = Ernie4_5_VLMoeProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.tokenizer, TokenizersBackend)
        self.assertIsInstance(processor.image_processor, Ernie4_5_VLMoeImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        video_processor = self.get_component("video_processor")

        processor = Ernie4_5_VLMoeProcessor(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )

        image_input = self.prepare_images_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, text="dummy", return_tensors="pt")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        video_processor = self.get_component("video_processor")

        processor = Ernie4_5_VLMoeProcessor(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )

        input_str = "lower newer"
        image_input = self.prepare_images_inputs()
        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            [
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
                "moe_mm_token_type_ids",
            ],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

        # test if it raises when no text is passed
        with pytest.raises(KeyError):
            processor(images=image_input)

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_images_inputs()

        size = {"shortest_edge": processor.image_processor.size["shortest_edge"], "longest_edge": 56 * 56 * 4}
        inputs = processor(text=input_str, images=image_input, size=size, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 612)
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 100)
