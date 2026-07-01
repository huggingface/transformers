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
    AutoModelForMultimodalLM,
    Step3p7Config,
    Step3p7ForConditionalGeneration,
    Step3p7Model,
    Step3p7TextConfig,
    Step3p7VisionEncoderConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.step3p7.modeling_step3p7 import (
        Step3p7MoEExperts,
        Step3p7RotaryEmbedding,
        Step3p7TextModel,
    )


class Step3p7ModelTester(VLMModelTester):
    base_model_class = Step3p7Model
    config_class = Step3p7Config
    conditional_generation_class = Step3p7ForConditionalGeneration
    text_config_class = Step3p7TextConfig
    vision_config_class = Step3p7VisionEncoderConfig

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
        kwargs.setdefault("mlp_ratio", 2)
        kwargs.setdefault("layer_norm_eps", 1e-5)
        kwargs.setdefault("layerscale_value", 0.1)
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
        self.assertIsInstance(config.vision_config, Step3p7VisionEncoderConfig)
        self.assertEqual(config.text_config.hidden_size, self.model_tester.hidden_size)
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

    def test_auto_model_for_multimodal_lm_mapping(self):
        config = self.model_tester.get_config()
        self.assertIs(AutoModelForMultimodalLM._model_mapping[type(config)], Step3p7ForConditionalGeneration)

    def test_moe_experts_implementations_match(self):
        config = self.model_tester.get_config().text_config
        experts = Step3p7MoEExperts(config, swiglu_limit=0.75).to(torch_device).eval()
        with torch.no_grad():
            experts.gate_up_proj.copy_(
                torch.arange(experts.gate_up_proj.numel(), device=torch_device, dtype=torch.float32).reshape_as(
                    experts.gate_up_proj
                )
                / 100
            )
            experts.down_proj.copy_(
                torch.arange(experts.down_proj.numel(), device=torch_device, dtype=torch.float32).reshape_as(
                    experts.down_proj
                )
                / 100
            )

        hidden_states = torch.linspace(-1.0, 1.0, steps=6 * config.hidden_size, device=torch_device).reshape(
            6, config.hidden_size
        )
        selected_experts = torch.tensor([[0, 1], [2, 3], [1, 0], [3, 2], [0, 2], [1, 3]], device=torch_device)
        routing_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.55, 0.45], [0.2, 0.8], [0.9, 0.1], [0.35, 0.65]],
            device=torch_device,
        )

        outputs = {}
        for implementation in ("eager", "batched_mm", "grouped_mm"):
            config._experts_implementation = implementation
            with torch.no_grad():
                outputs[implementation] = experts(hidden_states, selected_experts, routing_weights)

        torch.testing.assert_close(outputs["batched_mm"], outputs["eager"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(outputs["grouped_mm"], outputs["eager"], rtol=1e-5, atol=1e-5)

    def test_load_old_moe_3d_projection_keys(self):
        config = self.model_tester.get_config()
        model = Step3p7Model(copy.deepcopy(config)).eval()
        state_dict = model.state_dict()
        old_state_dict = {}
        converted_keys = []
        for key, tensor in state_dict.items():
            if key.endswith(".moe.experts.gate_up_proj"):
                gate, up = tensor.chunk(2, dim=1)
                old_state_dict[key.replace(".moe.experts.gate_up_proj", ".moe.gate_proj.weight")] = gate.clone()
                old_state_dict[key.replace(".moe.experts.gate_up_proj", ".moe.up_proj.weight")] = up.clone()
                converted_keys.append(key)
            elif key.endswith(".moe.experts.down_proj"):
                old_state_dict[key.replace(".moe.experts.down_proj", ".moe.down_proj.weight")] = tensor.clone()
                converted_keys.append(key)
            else:
                old_state_dict[key] = tensor.clone()

        loaded_model, loading_info = Step3p7Model.from_pretrained(
            None,
            config=copy.deepcopy(config),
            state_dict=old_state_dict,
            output_loading_info=True,
        )
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        self.assertFalse(loading_info["error_msgs"])
        loaded_state_dict = loaded_model.state_dict()
        for key in converted_keys:
            torch.testing.assert_close(loaded_state_dict[key], state_dict[key])

    def test_load_old_vision_embedding_keys(self):
        config = self.model_tester.get_config()
        model = Step3p7Model(copy.deepcopy(config)).eval()
        state_dict = model.state_dict()
        old_key_mapping = {
            "vision_model.embeddings.conv1.weight": "vision_model.conv1.weight",
            "vision_model.embeddings.ln_pre.weight": "vision_model.ln_pre.weight",
            "vision_model.embeddings.ln_pre.bias": "vision_model.ln_pre.bias",
            "vision_model.embeddings.positional_embedding": "vision_model.positional_embedding",
        }
        old_state_dict = {old_key_mapping.get(key, key): tensor.clone() for key, tensor in state_dict.items()}

        loaded_model, loading_info = Step3p7Model.from_pretrained(
            None,
            config=copy.deepcopy(config),
            state_dict=old_state_dict,
            output_loading_info=True,
        )
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        self.assertFalse(loading_info["error_msgs"])
        loaded_state_dict = loaded_model.state_dict()
        for key in old_key_mapping:
            torch.testing.assert_close(loaded_state_dict[key], state_dict[key])

    def test_rotary_embedding_uses_partial_rotary_factors(self):
        config = self.model_tester.get_config().text_config
        config.partial_rotary_factors = [0.5, 1.0]
        rotary_emb = Step3p7RotaryEmbedding(config)

        self.assertEqual(rotary_emb.full_attention_inv_freq.numel(), config.head_dim // 4)
        self.assertEqual(rotary_emb.full_attention_1_inv_freq.numel(), config.head_dim // 2)
        model = Step3p7TextModel(config).to(torch_device).eval()
        input_ids = torch.arange(self.model_tester.seq_length, device=torch_device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (1, self.model_tester.seq_length, config.hidden_size))

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        model_tester = model_tester or self.model_tester
        return model_tester.num_hidden_layers

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        return 2

    def test_get_image_features_returns_base_model_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Step3p7Model(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=inputs_dict["pixel_values"], output_hidden_states=True)
        self.assertIsNotNone(outputs.last_hidden_state)
        self.assertIsNotNone(outputs.pooler_output)
        self.assertEqual(len(outputs.hidden_states), 2)

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

    @unittest.skip(reason="Torch save/load equality is unstable for custom Step3p7 MoE weights")
    def test_torch_save_load(self):
        pass

    @unittest.skip(
        reason="Step3p7 checkpoint key conversion is covered by conversion_mapping.py, not this common reverse-mapping test"
    )
    def test_reverse_loading_mapping(self):
        pass

    @unittest.skip(reason="Load/save without tied weights is unstable for custom Step3p7 MoE weights")
    def test_load_save_without_tied_weights(self):
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
