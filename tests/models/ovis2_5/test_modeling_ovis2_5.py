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

from parameterized import parameterized

from transformers import (
    Ovis2_5Config,
    Ovis2_5ForConditionalGeneration,
    Ovis2_5Model,
    Ovis2_5Processor,
    is_torch_available,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_image_processing_common import COCO_CATS_IMAGE_URL, load_test_image
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Ovis2_5VisionText2TextModelTester(VLMModelTester):
    base_model_class = Ovis2_5Model
    config_class = Ovis2_5Config
    conditional_generation_class = Ovis2_5ForConditionalGeneration

    small_config_overrides = {
        "batch_size": 2,
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "max_position_embeddings": 32,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "image_size": 8,
        "patch_size": 2,
        "num_image_tokens": 4,
        "num_image_patches": 16,
        "image_token_id": 4,
        "video_token_id": 9,
        "image_start_token_id": 5,
        "image_end_token_id": 6,
        "video_start_token_id": 7,
        "video_end_token_id": 8,
        "vision_vocab_size": 12,
        "vision_window_size": 8,
        "vision_hidden_act": "gelu_pytorch_tanh",
        "vision_max_position_embeddings": 1024,
    }

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
        }

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return torch.randn(
            batch_size * self.num_image_patches,
            self.num_channels * self.patch_size**2,
            device=torch_device,
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = super().place_image_tokens(input_ids, config)
        input_ids[:, 0] = config.image_start_token_id
        input_ids[:, 1 : self.num_image_tokens + 1] = config.image_token_id
        input_ids[:, self.num_image_tokens + 1] = config.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        grid_size = self.image_size // self.patch_size
        return {
            "image_grid_thw": torch.tensor(
                [[1, grid_size, grid_size]] * batch_size,
                dtype=torch.long,
                device=torch_device,
            )
        }


@require_torch
class Ovis2_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ovis2_5VisionText2TextModelTester
    # The visual-tokenizer head consumes the encoder state before post_layernorm.
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        # The official-key mapping targets the conditional model's `model.*` subtree.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_load_original_visual_projection_into_base_model(self):
        config = self.model_tester.get_config()
        state_dict = Ovis2_5Model(config).state_dict()
        key = "multi_modal_projector.visual_embeddings_table.weight"
        weight = state_dict.pop(key)
        original_weight = torch.arange(weight.numel(), dtype=weight.dtype).reshape(weight.shape[1], weight.shape[0])
        state_dict["vte.weight"] = original_weight

        model, loading_info = Ovis2_5Model.from_pretrained(
            None, config=config, state_dict=state_dict, output_loading_info=True
        )

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        torch.testing.assert_close(model.state_dict()[key], original_weight.T, rtol=0, atol=0)

    @parameterized.expand([("image",), ("video",)])
    def test_precomputed_visual_features(self, modality):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5Model(config).to(torch_device).eval()
        if modality == "video":
            inputs["pixel_values_videos"] = inputs.pop("pixel_values")
            inputs["video_grid_thw"] = inputs.pop("image_grid_thw")
            inputs["input_ids"][inputs["input_ids"] == config.image_token_id] = config.video_token_id
            inputs["input_ids"][inputs["input_ids"] == config.image_start_token_id] = config.video_start_token_id
            inputs["input_ids"][inputs["input_ids"] == config.image_end_token_id] = config.video_end_token_id

        modality_key = f"{modality}_outputs"
        with torch.no_grad():
            raw_outputs = model(**inputs)
            pooler_output = getattr(raw_outputs, modality_key).pooler_output
            feature_keys = (
                ("pixel_values", "image_grid_thw")
                if modality == "image"
                else ("pixel_values_videos", "video_grid_thw")
            )
            for key in feature_keys:
                inputs.pop(key)
            inputs["mm_encoder_outputs"] = {modality: BaseModelOutputWithPooling(pooler_output=pooler_output)}
            outputs = model(**inputs)
            inputs["inputs_embeds"] = model.get_input_embeddings()(inputs.pop("input_ids"))
            embedded_outputs = model(**inputs)

        torch.testing.assert_close(outputs.last_hidden_state, raw_outputs.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(embedded_outputs.last_hidden_state, raw_outputs.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(getattr(outputs, modality_key).pooler_output, pooler_output, rtol=0, atol=0)

    def test_mixed_visual_features_match_inputs_embeds(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5Model(config).to(torch_device).eval()
        num_frames = 2
        video_tokens = [config.video_token_id] * (num_frames * self.model_tester.num_image_tokens)
        video_ids = torch.tensor(
            [[config.video_start_token_id, *video_tokens, config.video_end_token_id]], device=torch_device
        )
        input_ids = torch.cat((inputs["input_ids"][:1], video_ids), dim=-1)
        image_grid = inputs["image_grid_thw"][:1]
        video_grid = image_grid.clone()
        video_grid[:, 0] = num_frames
        raw_features = {
            "pixel_values": inputs["pixel_values"][: self.model_tester.num_image_patches],
            "image_grid_thw": image_grid,
            "pixel_values_videos": self.model_tester.create_pixel_values(batch_size=num_frames),
            "video_grid_thw": video_grid,
        }

        with torch.no_grad():
            embeddings = model.get_input_embeddings()
            # The published checkpoint shares an embedding for these four token IDs.
            shared_ids = [config.video_token_id, config.image_start_token_id, config.image_end_token_id]
            embeddings.weight[shared_ids] = embeddings.weight[config.image_token_id].clone()
            mm_token_type_ids = (input_ids == config.image_token_id).long() + 2 * (
                input_ids == config.video_token_id
            ).long()
            expected = model(input_ids=input_ids, **raw_features)
            embed_kwargs = {"inputs_embeds": embeddings(input_ids), "mm_token_type_ids": mm_token_type_ids}
            precomputed = {"image": expected.image_outputs, "video": expected.video_outputs}
            from_embeds = model(**embed_kwargs, **raw_features)
            from_cached_embeds = model(**embed_kwargs, mm_encoder_outputs=precomputed)

        torch.testing.assert_close(from_embeds.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(from_cached_embeds.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)

    def test_video_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()
        num_frames = 2
        num_video_tokens = num_frames * self.model_tester.num_image_tokens
        inputs["input_ids"][:, : num_video_tokens + 2] = torch.tensor(
            [config.video_start_token_id] + [config.video_token_id] * num_video_tokens + [config.video_end_token_id],
            device=torch_device,
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs.pop("pixel_values")
        inputs["pixel_values_videos"] = self.model_tester.create_pixel_values(
            batch_size=self.model_tester.batch_size * num_frames
        )
        inputs["video_grid_thw"] = inputs.pop("image_grid_thw")
        inputs["video_grid_thw"][:, 0] = num_frames

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(outputs.logits.shape[:2], inputs["input_ids"].shape)
        self.assertTrue(torch.isfinite(outputs.logits).all())
        self.assertEqual(len(outputs.video_outputs.pooler_output), self.model_tester.batch_size)


@slow
@require_torch_accelerator
class Ovis2_5IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "ovis-community/Ovis2.5-2B"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.processor = Ovis2_5Processor.from_pretrained(cls.model_id)
        cls.model = Ovis2_5ForConditionalGeneration.from_pretrained(cls.model_id, device_map=torch_device)

    def test_image_generation(self):
        image = load_test_image(COCO_CATS_IMAGE_URL).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What do you see in this image?"},
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
        ).to(torch_device, dtype=self.model.dtype)

        self.assertEqual(inputs.image_grid_thw.tolist(), [[1, 30, 40]])
        output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_text = self.processor.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        expected_text = Expectations(
            {
                (None, None): (
                    "I see two cats lying on a pink surface, possibly a couch or blanket. "
                    "One cat is positioned on the left side of the image, while the other is on the right. "
                    "Both cats appear to be relaxed and are near two remote controls. "
                    "The cat on the left has a green collar around its neck, and both"
                )
            }
        ).get_expectation()
        self.assertEqual(generated_text, expected_text)

    def test_video_generation(self):
        image = load_test_image(COCO_CATS_IMAGE_URL).convert("RGB")
        video = [image.copy() for _ in range(4)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video},
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
            processor_kwargs={"videos_kwargs": {"do_sample_frames": False}},
        ).to(torch_device, dtype=self.model.dtype)

        self.assertEqual(inputs.video_grid_thw.tolist(), [[4, 30, 40]])
        output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_text = self.processor.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        expected_text = Expectations(
            {(None, None): "The two cats and the pink blanket remain visible throughout all four frames."}
        ).get_expectation()
        self.assertEqual(generated_text, expected_text)
