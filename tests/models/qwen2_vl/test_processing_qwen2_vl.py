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

import unittest

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Qwen2VLProcessor

if is_torch_available():
    import torch

    from transformers import Qwen2VLConfig, Qwen2VLModel


@require_vision
@require_torch
@require_torchvision
class Qwen2VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen2VLProcessor
    # Tiny processor created with make_tiny_processor.py from "Qwen/Qwen2-VL-7B-Instruct"
    tiny_model_id = "hf-internal-testing/tiny-processor-qwen2_vl"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, patch_size=4, max_pixels=56 * 56, min_pixels=28 * 28, **kwargs)

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 384},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 576},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 1152},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 1152},
            {"expected_dim": 0, "output_length": 1152},
        ]

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_images_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 100)
        inputs = processor(text=input_str, images=image_input, max_pixels=56 * 56 * 4, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 612)

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_images_inputs(batch_size=2)

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
                max_length=20,
            )

    def test_position_ids_match_model_rope_index(self):
        """The processor's 3D RoPE `position_ids` are the ones the model computes with `get_rope_index`."""
        processor = self.get_processor()
        processor.tokenizer.padding_side = "left"
        inputs = processor(
            text=self.prepare_text_inputs(batch_size=2, modalities="image"),
            images=self.prepare_images_inputs(batch_size=2),
            padding=True,
            return_tensors="pt",
        )
        config = Qwen2VLConfig(
            text_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2},
            vision_config={"depth": 1, "embed_dim": 16, "hidden_size": 16, "num_heads": 2},
        )
        expected, _ = Qwen2VLModel(config).get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        torch.testing.assert_close(inputs["position_ids"], expected)
