# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ovis2.5 model."""

import unittest

from transformers import (
    Ovis2_5Config,
    Ovis2_5ForConditionalGeneration,
    Ovis2_5Model,
    Ovis2_5Processor,
    Ovis2_5VisionConfig,
    is_torch_available,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_image_processing_common import load_coco_image
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Ovis2_5VisionText2TextModelTester(VLMModelTester):
    base_model_class = Ovis2_5Model
    config_class = Ovis2_5Config
    text_config_class = Qwen3Config
    vision_config_class = Ovis2_5VisionConfig
    conditional_generation_class = Ovis2_5ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("seq_length", 8)
        kwargs.setdefault("vocab_size", 32)
        kwargs.setdefault("hidden_size", 16)
        kwargs.setdefault("intermediate_size", 32)
        kwargs.setdefault("num_hidden_layers", 1)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 4)
        kwargs.setdefault("max_position_embeddings", 32)
        kwargs.setdefault("attention_dropout", 0.0)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("image_size", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("image_start_token_id", 5)
        kwargs.setdefault("image_end_token_id", 6)
        kwargs.setdefault("video_start_token_id", 7)
        kwargs.setdefault("video_end_token_id", 8)
        kwargs.setdefault("visual_vocab_size", 12)
        super().__init__(parent, **kwargs)
        self.num_image_patches = (self.image_size // self.patch_size) ** 2

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
        }

    def get_text_config(self):
        return Qwen3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            attention_dropout=self.attention_dropout,
            hidden_act=self.hidden_act,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def get_vision_config(self):
        return Ovis2_5VisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            spatial_merge_size=2,
            window_size=self.image_size,
            attention_dropout=self.attention_dropout,
            vocab_size=self.visual_vocab_size,
            num_visual_indicator_tokens=4,
        )

    def get_config(self):
        return Ovis2_5Config(
            text_config=self.get_text_config(),
            vision_config=self.get_vision_config(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
        )

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids, device=torch_device)

    def create_pixel_values(self):
        return torch.randn(
            self.batch_size * self.num_image_patches,
            self.num_channels * self.patch_size**2,
            device=torch_device,
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        for token_id in self._special_token_ids:
            input_ids[input_ids == token_id] = self._safe_token_id()
        input_ids[:, 0] = config.image_start_token_id
        input_ids[:, 1] = config.image_token_id
        input_ids[:, 2] = config.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        grid_size = self.image_size // self.patch_size
        return {
            "image_grid_thw": torch.tensor(
                [[1, grid_size, grid_size]] * self.batch_size,
                dtype=torch.long,
                device=torch_device,
            )
        }

    def prepare_video_inputs(self):
        num_frames = 2
        input_ids = torch.tensor(
            [
                [
                    self.bos_token_id,
                    self.video_start_token_id,
                    self.video_token_id,
                    self.video_token_id,
                    self.video_end_token_id,
                    self._safe_token_id(),
                    self.eos_token_id,
                ]
            ]
            * self.batch_size,
            dtype=torch.long,
            device=torch_device,
        )
        grid_size = self.image_size // self.patch_size
        patches_per_video = num_frames * grid_size**2
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values_videos": torch.randn(
                self.batch_size * patches_per_video,
                self.num_channels * self.patch_size**2,
                device=torch_device,
            ),
            "video_grid_thw": torch.tensor(
                [[num_frames, grid_size, grid_size]] * self.batch_size,
                dtype=torch.long,
                device=torch_device,
            ),
        }


@require_torch
class Ovis2_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ovis2_5VisionText2TextModelTester
    # The visual-tokenizer head consumes the encoder state before post_layernorm.
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        # The official-key mapping targets the conditional model's `model.*` subtree.
        super().test_reverse_loading_mapping(skip_base_model=True)

    # Generic batch slicing would separate flattened patches from their grid metadata.
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        filtered_inputs = {}
        for key, value in inputs_dict.items():
            if key == "pixel_values":
                filtered_inputs[key] = value[: batch_size * self.model_tester.num_image_patches]
            elif key == "image_grid_thw":
                filtered_inputs[key] = value[:batch_size]
            elif isinstance(value, torch.Tensor):
                filtered_inputs[key] = value[:batch_size]
            else:
                filtered_inputs[key] = value

        text_config = config.get_text_config(decoder=True)
        text_config.eos_token_id = None
        text_config.forced_eos_token_id = None
        return config, filtered_inputs

    # One Ovis image is a complete packed patch group plus one grid row, not one pixel tensor row.
    def test_mismatching_num_image_tokens(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = self.model_tester.num_image_patches

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            model(**inputs)

            one_image_inputs = {
                "input_ids": inputs["input_ids"][:1],
                "attention_mask": inputs["attention_mask"][:1],
                "pixel_values": inputs["pixel_values"][:patches_per_image],
                "image_grid_thw": inputs["image_grid_thw"][:1],
            }
            two_prompt_inputs = {
                "input_ids": one_image_inputs["input_ids"].repeat(2, 1),
                "attention_mask": one_image_inputs["attention_mask"].repeat(2, 1),
                "pixel_values": one_image_inputs["pixel_values"],
                "image_grid_thw": one_image_inputs["image_grid_thw"],
            }
            with self.assertRaises(ValueError):
                model(**two_prompt_inputs)

            two_prompt_inputs["pixel_values"] = one_image_inputs["pixel_values"].repeat(2, 1)
            two_prompt_inputs["image_grid_thw"] = one_image_inputs["image_grid_thw"].repeat(2, 1)
            model(**two_prompt_inputs)

    def test_video_forward(self):
        config = self.model_tester.get_config()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()
        inputs = self.model_tester.prepare_video_inputs()

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(outputs.logits.shape[:2], inputs["input_ids"].shape)
        self.assertTrue(torch.isfinite(outputs.logits).all())


@slow
@require_torch_accelerator
class Ovis2_5IntegrationTest(unittest.TestCase):
    model_id = "AIDC-AI/Ovis2.5-2B"
    smoke_pixels = 448 * 448

    @classmethod
    def setUpClass(cls):
        cls.processor = Ovis2_5Processor.from_pretrained(cls.model_id)
        cls.model, cls.loading_info = Ovis2_5ForConditionalGeneration.from_pretrained(
            cls.model_id,
            dtype="auto",
            device_map=torch_device,
            output_loading_info=True,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        cleanup(torch_device, gc_collect=True)

    def test_image_generation(self):
        image = load_coco_image("000000039769.jpg").convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "What animals are visible? Answer in one short sentence.",
                    },
                ],
            }
        ]
        self.assertFalse(self.loading_info["missing_keys"])
        self.assertFalse(self.loading_info["unexpected_keys"])
        self.assertFalse(self.loading_info["mismatched_keys"])
        self.assertFalse(self.loading_info["error_msgs"])
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": self.smoke_pixels,
                        "longest_edge": self.smoke_pixels,
                    }
                }
            },
        ).to(torch_device, dtype=self.model.dtype)

        self.assertEqual(inputs.image_grid_thw.tolist(), [[1, 24, 32]])
        output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_text = self.processor.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.assertEqual(generated_text, "Two cats")

    def test_video_generation(self):
        image = load_coco_image("000000039769.jpg").convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": [image.copy() for _ in range(4)]},
                    {
                        "type": "text",
                        "text": "What remains visible throughout all four frames? Answer in one short sentence.",
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={
                "videos_kwargs": {
                    "size": {
                        "shortest_edge": self.smoke_pixels,
                        "longest_edge": self.smoke_pixels,
                    },
                    "do_sample_frames": False,
                }
            },
        ).to(torch_device, dtype=self.model.dtype)

        self.assertEqual(inputs.video_grid_thw.tolist(), [[4, 24, 32]])
        output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_text = self.processor.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.assertEqual(generated_text, "The two remote controls remain visible throughout all four frames.")
