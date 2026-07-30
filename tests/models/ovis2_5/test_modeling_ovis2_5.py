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
"""Synthetic tests for the native Ovis2.5 configuration and PyTorch model."""

import unittest

from transformers import AutoConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        AutoModelForImageTextToText,
        Ovis2_5Config,
        Ovis2_5ForConditionalGeneration,
        Ovis2_5Model,
        Ovis2_5VisionConfig,
        Ovis2_5VisionModel,
    )
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


if is_torch_available():

    class Ovis2_5VisionText2TextModelTester(VLMModelTester):
        """Standard multimodal tester using Ovis2.5's packed-patch representation."""

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
            kwargs.setdefault("visual_atom_token_id", 4)
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
                self.visual_atom_token_id,
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
                hidden_stride=2,
                window_size=self.image_size,
                attention_dropout=self.attention_dropout,
                vocab_size=self.visual_vocab_size,
                num_visual_indicator_tokens=4,
                preserve_original_pe=True,
                use_rope=True,
                fullatt_block_indexes=None,
            )

        def get_config(self):
            return Ovis2_5Config(
                text_config=self.get_text_config(),
                vision_config=self.get_vision_config(),
                visual_vocab_size=self.visual_vocab_size,
                visual_atom_token_id=self.visual_atom_token_id,
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
            input_ids[:, 1] = config.visual_atom_token_id
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


class Ovis2_5ModelTester:
    """Build a tiny Ovis2.5 model and matching packed visual inputs."""

    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.text_vocab_size = 32
        self.text_hidden_size = 16
        self.visual_vocab_size = 12
        self.num_visual_indicator_tokens = 4
        self.num_channels = 3
        self.patch_size = 2
        self.grid_height = 2
        self.grid_width = 2
        self.image_num_patches = self.grid_height * self.grid_width

        self.visual_atom_token_id = 4
        self.image_start_token_id = 5
        self.image_end_token_id = 6
        self.video_start_token_id = 7
        self.video_end_token_id = 8

    @property
    def patch_dim(self):
        return self.num_channels * self.patch_size**2

    def get_text_config_dict(self):
        return {
            "model_type": "qwen3",
            "vocab_size": self.text_vocab_size,
            "hidden_size": self.text_hidden_size,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": None,
            "pad_token_id": 0,
            "tie_word_embeddings": False,
        }

    def get_vision_config_dict(self):
        return {
            "model_type": "ovis2_5_vision",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": self.num_channels,
            "image_size": self.patch_size * self.grid_height,
            "patch_size": self.patch_size,
            "hidden_stride": 2,
            "window_size": 4,
            "attention_dropout": 0.0,
            "vocab_size": self.visual_vocab_size,
            "num_visual_indicator_tokens": self.num_visual_indicator_tokens,
            "preserve_original_pe": True,
            "fullatt_block_indexes": None,
        }

    def get_config(self, legacy_aliases=False):
        sub_configs = (
            {
                "llm_config": self.get_text_config_dict(),
                "vit_config": self.get_vision_config_dict(),
            }
            if legacy_aliases
            else {
                "text_config": self.get_text_config_dict(),
                "vision_config": self.get_vision_config_dict(),
            }
        )
        return Ovis2_5Config(
            **sub_configs,
            visual_vocab_size=self.visual_vocab_size,
            visual_atom_token_id=self.visual_atom_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            image_token_id=self.visual_atom_token_id,
            video_token_id=self.visual_atom_token_id,
        )

    def get_image_inputs(self):
        input_ids = torch.tensor(
            [[1, self.image_start_token_id, self.visual_atom_token_id, self.image_end_token_id, 9, 2]]
            * self.batch_size,
            dtype=torch.long,
            device=torch_device,
        )
        pixel_values = torch.randn(
            self.batch_size * self.image_num_patches,
            self.patch_dim,
            device=torch_device,
        )
        image_grid_thw = torch.tensor(
            [[1, self.grid_height, self.grid_width]] * self.batch_size,
            dtype=torch.long,
            device=torch_device,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def get_video_inputs(self):
        num_frames = 2
        num_visual_atoms = num_frames
        input_ids = torch.tensor(
            [
                [
                    1,
                    self.video_start_token_id,
                    *([self.visual_atom_token_id] * num_visual_atoms),
                    self.video_end_token_id,
                    9,
                    2,
                ]
            ]
            * self.batch_size,
            dtype=torch.long,
            device=torch_device,
        )
        patches_per_video = num_frames * self.image_num_patches
        pixel_values_videos = torch.randn(
            self.batch_size * patches_per_video,
            self.patch_dim,
            device=torch_device,
        )
        video_grid_thw = torch.tensor(
            [[num_frames, self.grid_height, self.grid_width]] * self.batch_size,
            dtype=torch.long,
            device=torch_device,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }


@require_torch
class Ovis2_5VisionModelTest(unittest.TestCase):
    all_model_classes = (Ovis2_5VisionModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Ovis2_5ModelTester(self)

    def test_packed_vision_forward(self):
        config = Ovis2_5VisionConfig(**self.model_tester.get_vision_config_dict())
        model = Ovis2_5VisionModel(config).to(torch_device).eval()
        pixel_values = torch.randn(
            self.model_tester.batch_size * self.model_tester.image_num_patches,
            self.model_tester.patch_dim,
            device=torch_device,
        )
        grid_thw = torch.tensor(
            [
                [1, self.model_tester.grid_height, self.model_tester.grid_width],
            ]
            * self.model_tester.batch_size,
            dtype=torch.long,
            device=torch_device,
        )

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, grid_thw=grid_thw)

        self.assertEqual(
            outputs.last_hidden_state.shape,
            (
                self.model_tester.batch_size * self.model_tester.image_num_patches,
                config.hidden_size,
            ),
        )
        self.assertEqual(
            outputs.pooler_output.shape,
            (self.model_tester.batch_size, config.vocab_size),
        )


@require_torch
class Ovis2_5ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Ovis2_5ModelTester(self)

    def test_config_composition_and_legacy_aliases(self):
        config = self.model_tester.get_config()

        self.assertIsInstance(config.text_config, Qwen3Config)
        self.assertIsInstance(config.vision_config, Ovis2_5VisionConfig)
        self.assertIs(config.llm_config, config.text_config)
        self.assertIs(config.vit_config, config.vision_config)
        self.assertEqual(config.vision_config.vocab_size, self.model_tester.visual_vocab_size)
        self.assertEqual(config.tie_word_embeddings, config.text_config.tie_word_embeddings)

        legacy_config = self.model_tester.get_config(legacy_aliases=True)
        self.assertIsInstance(legacy_config.text_config, Qwen3Config)
        self.assertIsInstance(legacy_config.vision_config, Ovis2_5VisionConfig)
        self.assertEqual(legacy_config.text_config.hidden_size, self.model_tester.text_hidden_size)
        self.assertEqual(legacy_config.vision_config.hidden_stride, 2)

        restored_config = Ovis2_5Config.from_dict(config.to_dict())
        self.assertIsInstance(restored_config.text_config, Qwen3Config)
        self.assertIsInstance(restored_config.vision_config, Ovis2_5VisionConfig)
        self.assertEqual(restored_config.to_dict(), config.to_dict())

    def test_config_rejects_a_visual_vocab_without_atom_tokens(self):
        config_kwargs = {
            "text_config": self.model_tester.get_text_config_dict(),
            "vision_config": self.model_tester.get_vision_config_dict(),
        }
        for visual_vocab_size in (4, 3, 0, -1):
            with self.subTest(visual_vocab_size=visual_vocab_size):
                with self.assertRaisesRegex(ValueError, "visual_vocab_size"):
                    Ovis2_5Config(
                        **config_kwargs,
                        visual_vocab_size=visual_vocab_size,
                    )

    def test_auto_config_and_model_mappings(self):
        auto_config = AutoConfig.for_model(
            "ovis2_5",
            text_config=self.model_tester.get_text_config_dict(),
            vision_config=self.model_tester.get_vision_config_dict(),
            visual_vocab_size=self.model_tester.visual_vocab_size,
            visual_atom_token_id=self.model_tester.visual_atom_token_id,
            image_start_token_id=self.model_tester.image_start_token_id,
            image_end_token_id=self.model_tester.image_end_token_id,
            video_start_token_id=self.model_tester.video_start_token_id,
            video_end_token_id=self.model_tester.video_end_token_id,
        )
        auto_vision_config = AutoConfig.for_model(
            "ovis2_5_vision",
            **{key: value for key, value in self.model_tester.get_vision_config_dict().items() if key != "model_type"},
        )

        self.assertIsInstance(auto_config, Ovis2_5Config)
        self.assertIsInstance(auto_vision_config, Ovis2_5VisionConfig)
        self.assertIsInstance(AutoModel.from_config(auto_config), Ovis2_5Model)
        self.assertIsInstance(AutoModel.from_config(auto_vision_config), Ovis2_5VisionModel)
        self.assertIsInstance(
            AutoModelForImageTextToText.from_config(auto_config),
            Ovis2_5ForConditionalGeneration,
        )

    def test_image_forward_output_shapes_and_loss(self):
        config = self.model_tester.get_config()
        inputs = self.model_tester.get_image_inputs()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()

        with torch.no_grad():
            vision_outputs = model.model.vision_tower(
                pixel_values=inputs["pixel_values"],
                grid_thw=inputs["image_grid_thw"],
            )
            outputs = model(**inputs, labels=inputs["input_ids"])

        num_patches = self.model_tester.batch_size * self.model_tester.image_num_patches
        num_visual_atoms = self.model_tester.batch_size
        self.assertEqual(
            vision_outputs.last_hidden_state.shape,
            (num_patches, config.vision_config.hidden_size),
        )
        self.assertEqual(
            vision_outputs.pooler_output.shape,
            (num_visual_atoms, config.visual_vocab_size),
        )
        torch.testing.assert_close(
            vision_outputs.pooler_output[:, -config.vision_config.num_visual_indicator_tokens :],
            torch.zeros(
                num_visual_atoms,
                config.vision_config.num_visual_indicator_tokens,
                device=torch_device,
            ),
        )
        torch.testing.assert_close(
            vision_outputs.pooler_output[:, : -config.vision_config.num_visual_indicator_tokens].sum(dim=-1),
            torch.ones(num_visual_atoms, device=torch_device),
        )

        self.assertEqual(
            outputs.logits.shape,
            (
                self.model_tester.batch_size,
                inputs["input_ids"].shape[1],
                config.text_config.vocab_size,
            ),
        )
        self.assertEqual(
            outputs.image_hidden_states.shape,
            (num_visual_atoms, config.text_config.hidden_size),
        )
        self.assertIsNone(outputs.video_hidden_states)
        self.assertEqual(outputs.loss.ndim, 0)
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_video_forward_output_shapes_and_loss(self):
        config = self.model_tester.get_config()
        inputs = self.model_tester.get_video_inputs()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        expected_visual_atoms = self.model_tester.batch_size * 2
        self.assertEqual(
            outputs.logits.shape,
            (
                self.model_tester.batch_size,
                inputs["input_ids"].shape[1],
                config.text_config.vocab_size,
            ),
        )
        self.assertEqual(
            outputs.video_hidden_states.shape,
            (expected_visual_atoms, config.text_config.hidden_size),
        )
        self.assertIsNone(outputs.image_hidden_states)
        self.assertEqual(outputs.loss.ndim, 0)
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_visual_token_scattering(self):
        config = self.model_tester.get_config()
        model = Ovis2_5Model(config).to(torch_device).eval()
        hidden_size = config.text_config.hidden_size
        indicator_features = torch.stack(
            [
                torch.full((hidden_size,), float(index + 10), device=torch_device)
                for index in range(config.vision_config.num_visual_indicator_tokens)
            ]
        )

        cases = (
            (
                torch.tensor(
                    [
                        [
                            1,
                            config.image_start_token_id,
                            config.visual_atom_token_id,
                            config.image_end_token_id,
                            9,
                        ]
                    ],
                    device=torch_device,
                ),
                torch.full((1, hidden_size), 3.0, device=torch_device),
                False,
                (10.0, 11.0),
            ),
            (
                torch.tensor(
                    [
                        [
                            1,
                            config.video_start_token_id,
                            config.visual_atom_token_id,
                            config.visual_atom_token_id,
                            config.video_end_token_id,
                        ]
                    ],
                    device=torch_device,
                ),
                torch.stack(
                    (
                        torch.full((hidden_size,), 4.0, device=torch_device),
                        torch.full((hidden_size,), 5.0, device=torch_device),
                    )
                ),
                True,
                (12.0, 13.0),
            ),
        )

        for input_ids, visual_features, is_video, expected_boundaries in cases:
            with self.subTest(is_video=is_video):
                inputs_embeds = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    hidden_size,
                    device=torch_device,
                )
                merged = model._merge_visual_features(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    visual_features=visual_features,
                    visual_indicator_features=indicator_features,
                    grid_thw=torch.tensor([[1, 2, 2]], device=torch_device),
                    is_video=is_video,
                )

                atom_mask = input_ids == config.visual_atom_token_id
                torch.testing.assert_close(merged[atom_mask], visual_features)
                torch.testing.assert_close(
                    merged[0, 1],
                    torch.full((hidden_size,), expected_boundaries[0], device=torch_device),
                )
                torch.testing.assert_close(
                    merged[0, -1 if is_video else 3],
                    torch.full((hidden_size,), expected_boundaries[1], device=torch_device),
                )
                torch.testing.assert_close(merged[0, 0], torch.zeros(hidden_size, device=torch_device))

        with self.assertRaisesRegex(ValueError, "Visual features and visual atom tokens do not match"):
            model._merge_visual_features(
                inputs_embeds=torch.zeros(1, 5, hidden_size, device=torch_device),
                input_ids=cases[0][0],
                visual_features=torch.zeros(2, hidden_size, device=torch_device),
                visual_indicator_features=indicator_features,
                grid_thw=torch.tensor([[1, 2, 2]], device=torch_device),
                is_video=False,
            )

    def test_beam_expansion_and_generation(self):
        config = self.model_tester.get_config()
        inputs = self.model_tester.get_image_inputs()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()

        expand_size = 3
        expanded_input_ids, expanded_kwargs = model._expand_inputs_for_generation(
            expand_size=expand_size,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )

        patches_per_image = self.model_tester.image_num_patches
        expected_pixel_values = torch.cat(
            [
                inputs["pixel_values"][:patches_per_image].repeat(expand_size, 1),
                inputs["pixel_values"][patches_per_image:].repeat(expand_size, 1),
            ]
        )
        torch.testing.assert_close(
            expanded_input_ids,
            inputs["input_ids"].repeat_interleave(expand_size, dim=0),
        )
        torch.testing.assert_close(expanded_kwargs["pixel_values"], expected_pixel_values)
        torch.testing.assert_close(
            expanded_kwargs["image_grid_thw"],
            inputs["image_grid_thw"].repeat_interleave(expand_size, dim=0),
        )
        torch.testing.assert_close(
            expanded_kwargs["attention_mask"],
            inputs["attention_mask"].repeat_interleave(expand_size, dim=0),
        )

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2,
                num_beams=2,
                do_sample=False,
                pad_token_id=0,
            )

        self.assertEqual(
            generated_ids.shape,
            (self.model_tester.batch_size, inputs["input_ids"].shape[1] + 2),
        )
        torch.testing.assert_close(
            generated_ids[:, : inputs["input_ids"].shape[1]],
            inputs["input_ids"],
        )


@require_torch
class Ovis2_5CommonModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ovis2_5VisionText2TextModelTester if is_torch_available() else None
    # The visual-tokenizer head intentionally consumes the encoder state before
    # post_layernorm, matching the released checkpoints.
    test_all_params_have_gradient = False
    # Packed patch counts and grid metadata use data-dependent Python control flow.
    test_torch_exportable = False
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    def test_reverse_loading_mapping(self):
        # The official-key mapping targets the conditional model's `model.*`
        # subtree. Base-model prefix handling covers bare-model loading.
        super().test_reverse_loading_mapping(skip_base_model=True)

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
