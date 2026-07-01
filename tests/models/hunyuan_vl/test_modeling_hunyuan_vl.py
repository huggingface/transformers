# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HunYuanVL model."""

import copy
import unittest

from transformers import (
    AutoModel,
    HunYuanVLConfig,
    HunYuanVLForConditionalGeneration,
    HunYuanVLModel,
    HunYuanVLTextConfig,
    HunYuanVLVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class HunYuanVLVisionText2TextModelTester(VLMModelTester):
    """Build a tiny HunYuanVL config plus matching multimodal inputs for unit tests."""

    base_model_class = HunYuanVLModel
    config_class = HunYuanVLConfig
    text_config_class = HunYuanVLTextConfig
    vision_config_class = HunYuanVLVisionConfig
    conditional_generation_class = HunYuanVLForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("seq_length", 32)
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 128)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 128)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("rope_theta", 10000.0)
        kwargs.setdefault(
            "rope_parameters", {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 2, 2, 2]}
        )
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("image_size", 64)
        kwargs.setdefault("image_token_id", 5)
        kwargs.setdefault("out_hidden_size", kwargs["hidden_size"])
        kwargs.setdefault("text_hidden_size", kwargs["hidden_size"])
        kwargs.setdefault("max_image_size", kwargs["image_size"])
        kwargs.setdefault("min_image_size", kwargs["image_size"])
        kwargs.setdefault("anyres_vit_max_image_size", kwargs["image_size"])
        grid_hw = kwargs["image_size"] // kwargs["patch_size"]
        # HunYuanVL inserts an extra column per row (newline) and 2 begin/end tokens.
        kwargs.setdefault("num_image_tokens", grid_hw * (grid_hw + 1) + 2)
        kwargs.setdefault("max_vit_seq_len", grid_hw**2)
        super().__init__(parent, **kwargs)
        self.device = torch_device
        self.grid_hw = self.image_size // self.patch_size
        self.num_image_patches = self.grid_hw**2
        self.num_image_placeholder_tokens = self.num_image_tokens

    def get_config(self):
        return HunYuanVLConfig(
            attn_implementation="eager",
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
        )

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids, device=torch_device)

    def create_pixel_values(self):
        return floats_tensor(
            [self.batch_size * self.num_image_patches, self.num_channels * self.patch_size * self.patch_size]
        ).to(torch_device)

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[input_ids == self.image_token_id] = config.text_config.pad_token_id
        input_ids[:, : self.num_image_placeholder_tokens] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        return {
            "image_grid_thw": torch.tensor([[1, self.grid_hw, self.grid_hw]] * self.batch_size, device=torch_device)
        }

    def prepare_config_and_inputs(self):
        config, inputs_dict = self.prepare_config_and_inputs_for_common()
        config.text_config.rope_parameters["mrope_section"] = [2, 2, 2, 2]
        # HunYuanVL uses 4 multimodal RoPE axes: position, width, height, and temporal.
        inputs_dict["position_ids"] = (
            torch.arange(self.seq_length, device=torch_device).view(1, 1, -1).expand(4, self.batch_size, -1)
        )
        return config, inputs_dict


@require_torch
class HunYuanVLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HunYuanVLVisionText2TextModelTester
    test_all_params_have_gradient = False
    # HunYuanVL packs all images into one flat patch stream; pixel_values.shape[0] is total patches, not batch size.
    skip_test_image_features_output_shape = True

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        filtered_inputs_dict = {}
        for key, value in inputs_dict.items():
            if key == "pixel_values":
                filtered_inputs_dict[key] = value[: batch_size * self.model_tester.num_image_patches]
            elif key == "image_grid_thw":
                filtered_inputs_dict[key] = value[:batch_size]
            elif isinstance(value, torch.Tensor):
                filtered_inputs_dict[key] = value[:batch_size, ...]
            else:
                filtered_inputs_dict[key] = value

        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def test_auto_model_uses_base_model(self):
        config = self.model_tester.get_config()
        model = AutoModel.from_config(config).to(self.model_tester.device)
        self.assertIsInstance(model, HunYuanVLModel)
        self.assertFalse(hasattr(model, "lm_head"))

    def test_forward_uses_text_backbone(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        self.assertIsInstance(model.model, HunYuanVLModel)
        self.assertEqual(model.model.language_model.__class__.__name__, "HunYuanVLTextModel")
        self.assertIn("HunYuanVLPreTrainedModel", [cls.__name__ for cls in model.model.language_model.__class__.mro()])
        self.assertFalse(hasattr(model.model, "lm_head"))
        self.assertIs(model.get_input_embeddings(), model.model.language_model.embed_tokens)
        self.assertIs(model.get_output_embeddings(), model.lm_head)
        self.assertIs(model.get_decoder(), model.model.language_model)
        for layer in model.model.language_model.layers:
            self.assertFalse(hasattr(layer.self_attn, "rotary_emb"))

    def test_mrope_embeddings_are_built_once_per_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        inputs_dict.pop("position_ids")
        config.text_config.rope_parameters["mrope_section"] = [2, 2, 2, 2]
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        embedding_call_count = 0
        rotary_forward = model.model.language_model.rotary_emb.forward

        def wrapped_rotary_forward(*args, **kwargs):
            nonlocal embedding_call_count
            embedding_call_count += 1
            return rotary_forward(*args, **kwargs)

        model.model.language_model.rotary_emb.forward = wrapped_rotary_forward
        with torch.no_grad():
            model(**inputs_dict)

        self.assertEqual(embedding_call_count, 1)

    def test_model_builds_mrope_position_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)

        position_ids, rope_deltas = model.model.get_rope_index(
            inputs_dict["input_ids"],
            image_grid_thw=inputs_dict["image_grid_thw"],
            attention_mask=inputs_dict["attention_mask"],
        )

        grid_tokens = self.model_tester.grid_hw * (self.model_tester.grid_hw + 1)
        self.assertEqual(position_ids.shape, (4, self.model_tester.batch_size, self.model_tester.seq_length))
        self.assertEqual(rope_deltas.shape, (self.model_tester.batch_size, 1))
        self.assertTrue(
            torch.equal(
                position_ids[1, 0, 1 : 1 + grid_tokens],
                torch.arange(self.model_tester.grid_hw + 1, device=position_ids.device).repeat(
                    self.model_tester.grid_hw
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                position_ids[2, 0, 1 : 1 + grid_tokens],
                torch.arange(self.model_tester.grid_hw, device=position_ids.device).repeat_interleave(
                    self.model_tester.grid_hw + 1
                ),
            )
        )

    def test_legacy_xdrope_section_normalizes_to_mrope_section(self):
        text_config = HunYuanVLTextConfig(
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "xdrope_section": [2.0, 2, 2, 2]},
        )

        self.assertEqual(text_config.rope_parameters["mrope_section"], [2, 2, 2, 2])
        self.assertNotIn("xdrope_section", text_config.rope_parameters)

    def test_text_backbone_records_outputs_from_pretrained_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        with torch.no_grad():
            outputs = model.model.language_model(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                position_ids=inputs_dict["position_ids"],
                output_hidden_states=True,
                output_attentions=True,
            )

        self.assertEqual(len(outputs.hidden_states), config.text_config.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), config.text_config.num_hidden_layers)

    def test_vision_tower_records_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        with torch.no_grad():
            outputs = model.model.vit(
                inputs_dict["pixel_values"],
                grid_thw=inputs_dict["image_grid_thw"],
                output_hidden_states=True,
                output_attentions=True,
            )

        expected_num_patches = self.model_tester.batch_size * self.model_tester.num_image_patches
        expected_num_image_tokens = self.model_tester.batch_size * self.model_tester.num_image_placeholder_tokens
        self.assertEqual(
            outputs.last_hidden_state.shape,
            (1, expected_num_patches, config.vision_config.hidden_size),
        )
        self.assertEqual(
            outputs.pooler_output.shape,
            (1, expected_num_image_tokens, config.text_config.hidden_size),
        )
        self.assertEqual(len(outputs.hidden_states), config.vision_config.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), config.vision_config.num_hidden_layers)

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)

            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][: -self.model_tester.num_image_patches]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][:-1]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            input_ids = input_dict["input_ids"][:1]
            attention_mask = input_dict["attention_mask"][:1]
            pixel_values = input_dict["pixel_values"][: self.model_tester.num_image_patches]
            image_grid_thw = input_dict["image_grid_thw"][:1]

            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=torch.cat([input_ids, input_ids], dim=0),
                    attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            _ = model(
                input_ids=torch.cat([input_ids, input_ids], dim=0),
                attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
                pixel_values=torch.cat([pixel_values, pixel_values], dim=0),
                image_grid_thw=torch.cat([image_grid_thw, image_grid_thw], dim=0),
            )

    def test_prepare_inputs_for_generation_drops_pixel_values_after_prefill(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        prefill_inputs = model.prepare_inputs_for_generation(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            position_ids=inputs_dict["position_ids"],
            pixel_values=inputs_dict["pixel_values"],
            image_grid_thw=inputs_dict["image_grid_thw"],
            use_cache=True,
            is_first_iteration=True,
        )
        self.assertIs(prefill_inputs["pixel_values"], inputs_dict["pixel_values"])
        self.assertIs(prefill_inputs["image_grid_thw"], inputs_dict["image_grid_thw"])
        self.assertEqual(prefill_inputs["position_ids"].shape, inputs_dict["position_ids"].shape)

        decode_inputs = model.prepare_inputs_for_generation(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            position_ids=inputs_dict["position_ids"],
            pixel_values=inputs_dict["pixel_values"],
            image_grid_thw=inputs_dict["image_grid_thw"],
            use_cache=True,
            is_first_iteration=False,
            next_sequence_length=1,
        )
        self.assertIsNone(decode_inputs["pixel_values"])
        self.assertIs(decode_inputs["image_grid_thw"], inputs_dict["image_grid_thw"])
        self.assertEqual(decode_inputs["position_ids"].shape, (4, self.model_tester.batch_size, 1))

    def test_batching_equivalence(self, atol=2e-5, rtol=1e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=True):
        super().test_reverse_loading_mapping(check_keys_were_modified, skip_base_model)
