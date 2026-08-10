# Copyright 2025 Cohere and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch CohereCompass model."""

import copy
import unittest

from parameterized import parameterized

from transformers import (
    CohereCompassConfig,
    CohereCompassTextConfig,
    CohereCompassVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        CohereCompassForCausalLM,
        CohereCompassForConditionalGeneration,
        CohereCompassModel,
        CohereCompassTextForSequenceClassification,
        CohereCompassTextModel,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast


class CohereCompassTextModelTester(CausalLMModelTester):
    base_model_class = CohereCompassTextModel
    config_class = CohereCompassTextConfig
    causal_lm_class = CohereCompassForCausalLM
    sequence_classification_class = CohereCompassTextForSequenceClassification

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("vocab_size", 64)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("layer_types", ["full_attention", "full_attention"])
        kwargs.setdefault(
            "rope_parameters",
            {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000,
                }
            },
        )
        super().__init__(parent, **kwargs)


@require_torch
class CohereCompassTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = CohereCompassTextModelTester

    @unittest.skip("TODO: compass configures RoPE per layer type")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("TODO: compass configures RoPE per layer type")
    def test_model_rope_scaling_from_config(self):
        pass

    def test_text_config_is_causal(self):
        config = self.model_tester.get_config().to_dict()
        self.assertTrue(CohereCompassTextConfig(**{**config, "is_causal": True}).is_causal)
        self.assertFalse(CohereCompassTextConfig(**{**config, "is_causal": False}).is_causal)

    def test_rope_parameters_are_per_layer_type(self):
        config = self.model_tester.get_config().to_dict()
        config = CohereCompassTextConfig(
            **{
                **config,
                "layer_types": ["full_attention", "sliding_attention"],
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 20_000},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000},
                },
            }
        )
        self.assertEqual(config.rope_parameters["full_attention"]["rope_theta"], 20_000)
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_theta"], 10_000)

        model = CohereCompassTextModel(config)
        self.assertFalse(
            torch.equal(
                model.rotary_emb.full_attention_inv_freq,
                model.rotary_emb.sliding_attention_inv_freq,
            )
        )

    def test_null_rope_parameters_disable_position_embeddings(self):
        config = self.model_tester.get_config().to_dict()
        config = CohereCompassTextConfig(
            **{
                **config,
                "layer_types": ["full_attention", "sliding_attention"],
                "sliding_window": 4,
                "rope_parameters": {
                    "full_attention": None,
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000},
                },
            }
        )
        model = CohereCompassTextModel(config).to(torch_device)

        self.assertIsNone(config.rope_parameters["full_attention"])
        self.assertFalse(hasattr(model.rotary_emb, "full_attention_inv_freq"))
        self.assertTrue(hasattr(model.rotary_emb, "sliding_attention_inv_freq"))
        outputs = model(torch.randint(0, config.vocab_size, (2, 8), device=torch_device))
        self.assertEqual(outputs.last_hidden_state.shape, (2, 8, config.hidden_size))

    def test_sequence_classification_pooling(self):
        class StaticBackbone(nn.Module):
            def __init__(self, hidden_states):
                super().__init__()
                self.hidden_states = hidden_states

            def forward(self, *args, **kwargs):
                return BaseModelOutputWithPast(last_hidden_state=self.hidden_states)

        input_ids = torch.tensor([[1, 2, 0]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 0]], device=torch_device)
        hidden_states = torch.zeros(1, 3, self.model_tester.hidden_size, device=torch_device)
        hidden_states[0, :, 0] = torch.tensor([1.0, 2.0, 10.0], device=torch_device)

        for pooling, expected_score in {"bos": 1.0, "eos": 2.0, "mean": 1.5}.items():
            config = self.model_tester.get_config()
            config.num_labels = 1
            config.pooling = pooling
            model = CohereCompassTextForSequenceClassification(config).to(torch_device).eval()
            model.model = StaticBackbone(hidden_states)
            with torch.no_grad():
                model.score.weight.zero_()
                model.score.weight[0, 0] = 1
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.testing.assert_close(output.logits, torch.tensor([[expected_score]], device=torch_device))


class CohereCompassModelTester(VLMModelTester):
    base_model_class = CohereCompassModel
    config_class = CohereCompassConfig
    text_config_class = CohereCompassTextConfig
    vision_config_class = CohereCompassVisionConfig
    conditional_generation_class = CohereCompassForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("vocab_size", 64)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("image_token_id", 5)
        kwargs.setdefault("vision_start_token_id", 6)
        kwargs.setdefault("vision_end_token_id", 7)
        kwargs.setdefault("video_token_id", 8)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("deepstack_visual_indexes", [0])
        kwargs.setdefault(
            "rope_parameters",
            {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000,
                    "mrope_section": [1, 1, 2],
                }
            },
        )
        super().__init__(parent, **kwargs)
        self.out_hidden_size = self.hidden_size

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }

    def create_pixel_values(self):
        patches_per_image = (self.image_size // self.patch_size) ** 2
        return floats_tensor(
            [
                self.batch_size * patches_per_image,
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        for token_id in self._special_token_ids:
            input_ids[input_ids == token_id] = self.pad_token_id
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, 2, 2]] * self.batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )

    def prepare_text_inputs(self):
        input_ids = torch.randint(3, self.vocab_size, (self.batch_size, self.seq_length), device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def prepare_image_inputs(self, config):
        """A single-image, single-row batch with the correct number of image placeholder tokens."""
        vision_config = config.vision_config
        grid_t, grid_h, grid_w = 1, 2, 2
        num_patches = grid_t * grid_h * grid_w
        patch_dim = (
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * vision_config.patch_size
            * vision_config.patch_size
        )
        num_image_tokens = num_patches // (vision_config.spatial_merge_size**2)
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=torch_device)
        pixel_values = torch.randn(num_patches, patch_dim, device=torch_device)
        ids = (
            [10, self.vision_start_token_id]
            + [self.image_token_id] * num_image_tokens
            + [self.vision_end_token_id, 11]
        )
        input_ids = torch.tensor([ids], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = (input_ids == self.image_token_id).int()
        return input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids


@require_torch
class CohereCompassModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = CohereCompassModelTester

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        patches_per_image = (self.model_tester.image_size // self.model_tester.patch_size) ** 2
        inputs_dict["pixel_values"] = self.model_tester.create_pixel_values()[: batch_size * patches_per_image]
        return config, inputs_dict

    @unittest.skip("CohereCompass does not support video modeling.")
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip("CohereCompass does not support video modeling.")
    def test_get_video_features_hidden_states(self):
        pass

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = (self.model_tester.image_size // self.model_tester.patch_size) ** 2

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            _ = model(**input_dict)

            one_image_inputs = copy.deepcopy(input_dict)
            one_image_inputs["pixel_values"] = one_image_inputs["pixel_values"][:patches_per_image]
            one_image_inputs["image_grid_thw"] = one_image_inputs["image_grid_thw"][:1]
            with self.assertRaises(ValueError):
                _ = model(**one_image_inputs)

            model.base_model.rope_deltas = None
            two_prompt_inputs = {
                key: torch.cat([value[:1], value[:1]], dim=0)
                for key, value in one_image_inputs.items()
                if key not in {"pixel_values", "image_grid_thw"}
            }
            two_prompt_inputs["pixel_values"] = one_image_inputs["pixel_values"]
            two_prompt_inputs["image_grid_thw"] = one_image_inputs["image_grid_thw"]
            with self.assertRaises(ValueError):
                _ = model(**two_prompt_inputs)

            model.base_model.rope_deltas = None
            two_prompt_inputs["pixel_values"] = torch.cat(
                [one_image_inputs["pixel_values"], one_image_inputs["pixel_values"]], dim=0
            )
            two_prompt_inputs["image_grid_thw"] = torch.cat(
                [one_image_inputs["image_grid_thw"], one_image_inputs["image_grid_thw"]], dim=0
            )
            _ = model(**two_prompt_inputs)

    def test_model_vl_text_input_forward(self):
        config = self.model_tester.get_config()
        model = CohereCompassModel(config).to(torch_device).eval()
        input_ids, attention_mask = self.model_tester.prepare_text_inputs()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(
            out.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.text_config.hidden_size),
        )

    def test_conditional_generation_multiple_images(self):
        config = self.model_tester.get_config()
        model = CohereCompassForConditionalGeneration(config).to(torch_device).eval()
        input_ids, _, pixel_values, image_grid_thw, mm_token_type_ids = self.model_tester.prepare_image_inputs(config)
        input_ids = torch.cat([input_ids, input_ids[:, 1:]], dim=1)
        mm_token_type_ids = torch.cat([mm_token_type_ids, mm_token_type_ids[:, 1:]], dim=1)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
        image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
        self.assertEqual(output.logits.shape[:2], input_ids.shape)


if __name__ == "__main__":
    unittest.main()
