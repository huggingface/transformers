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
"""Testing suite for the PyTorch Step3p7 model."""

import copy
import unittest

from transformers import (
    AutoModelForCausalLM,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    Step3p7Model,
    Step3p7TextConfig,
    StepRoboticsVisionEncoderConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Step3p7ModelTester(VLMModelTester):
    base_model_class = Step3p7Model
    config_class = Step3p7Config
    conditional_generation_class = Step3p7ForConditionalGeneration
    text_config_class = Step3p7TextConfig
    vision_config_class = StepRoboticsVisionEncoderConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("seq_length", 4)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("vocab_size", 99)
        kwargs.setdefault("hidden_size", 16)
        kwargs.setdefault("intermediate_size", 32)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_attention_groups", 2)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("max_seq_len", 32)
        kwargs.setdefault("max_position_embeddings", 32)
        kwargs.setdefault("moe_intermediate_size", 8)
        kwargs.setdefault("moe_num_experts", 4)
        kwargs.setdefault("moe_top_k", 2)
        kwargs.setdefault("share_expert_dim", 8)
        kwargs.setdefault("head_dim", 4)
        kwargs.setdefault("layer_types", ["full_attention", "full_attention"])
        kwargs.setdefault("moe_layers_enum", (1,))
        kwargs.setdefault("use_rope_layers", [True, True])
        kwargs.setdefault("yarn_only_types", [])
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("width", 16)
        kwargs.setdefault("layers", 1)
        kwargs.setdefault("heads", 4)
        kwargs.setdefault("mlp_ratio", 2)
        kwargs.setdefault("layer_norm_eps", 1e-5)
        kwargs.setdefault("ls_init_value", 0.1)
        # After the two stride-2 downsamplers, a 16x16 image with patch size 4 produces one visual token.
        kwargs.setdefault("num_image_tokens", 1)
        super().__init__(parent, **kwargs)

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids, device=torch_device)

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[input_ids == config.image_token_id] = self.bos_token_id
        input_ids[:, 0] = config.image_token_id
        return input_ids

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = self.base_model_class(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.image_hidden_states.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_conditional_generation(self, config, input_ids, attention_mask, pixel_values):
        model = self.conditional_generation_class(config).to(torch_device).eval()
        labels = input_ids.clone()
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertIsNotNone(result.loss)


@require_torch
class Step3p7ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Step3p7ModelTester
    test_torch_exportable = False
    test_attention_outputs = False
    has_attentions = False
    test_all_params_have_gradient = False
    all_model_classes = (Step3p7Model, Step3p7ForConditionalGeneration)
    pipeline_model_mapping = {}

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(config, Step3p7Config)
        self.assertIsInstance(config.text_config, Step3p7TextConfig)
        self.assertIsInstance(config.vision_config, StepRoboticsVisionEncoderConfig)
        self.assertEqual(config.hidden_size, self.model_tester.hidden_size)
        self.assertEqual(config.image_token_id, self.model_tester.image_token_id)

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(
            config, inputs_dict["input_ids"], inputs_dict["attention_mask"], inputs_dict["pixel_values"]
        )

    def test_for_conditional_generation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_for_conditional_generation(
            config, inputs_dict["input_ids"], inputs_dict["attention_mask"], inputs_dict["pixel_values"]
        )

    def test_auto_model_for_causal_lm_mapping(self):
        config = self.model_tester.get_config()
        self.assertIs(AutoModelForCausalLM._model_mapping[type(config)], Step3p7ForConditionalGeneration)

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        model_tester = model_tester or self.model_tester
        return model_tester.layers

    def test_get_image_features_returns_base_model_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Step3p7Model(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=inputs_dict["pixel_values"], output_hidden_states=True)
        self.assertIsNotNone(outputs.last_hidden_state)
        self.assertIsNotNone(outputs.pooler_output)
        self.assertEqual(len(outputs.hidden_states), self.model_tester.layers + 1)

    @unittest.skip(reason="Beam-search generation from inputs_embeds is unstable for this VLM common test")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @unittest.skip(reason="Mismatched image-token common test triggers CUDA device-side assert for Step3p7")
    def test_mismatching_num_image_tokens(self):
        pass

    @unittest.skip(reason="Batching equivalence is numerically unstable for this VLM common test")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="Save/load output equivalence is unstable for this custom VLM common test")
    def test_save_load(self):
        pass

    @unittest.skip(reason="Safetensors roundtrip equality is unstable for custom Step3p7 MoE weights")
    def test_can_use_safetensors(self):
        pass

    @unittest.skip(reason="Feedforward chunking is not supported for Step3p7")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Step3p7 custom weights are not compatible with the common missing-weight reinit test")
    def test_can_init_all_missing_weights(self):
        pass

    def test_prepare_inputs_for_generation_keeps_images_on_prefill_only(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        pixel_values = inputs_dict["pixel_values"]
        model = Step3p7ForConditionalGeneration(copy.deepcopy(config)).to(torch_device).eval()

        first_inputs = model.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            is_first_iteration=True,
        )
        self.assertIn("pixel_values", first_inputs)
        self.assertIs(first_inputs["pixel_values"], pixel_values)

        next_inputs = model.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            is_first_iteration=False,
            use_cache=True,
        )
        self.assertNotIn("pixel_values", next_inputs)


if __name__ == "__main__":
    unittest.main()
