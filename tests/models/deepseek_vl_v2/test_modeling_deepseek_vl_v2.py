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
"""Testing suite for the PyTorch DeepSeek-VL2 model."""

import unittest

from transformers import DeepseekVLV2Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import DeepseekVLV2ForConditionalGeneration, DeepseekVLV2Model


class DeepseekVLV2VisionText2TextModelTester(VLMModelTester):
    base_model_class = DeepseekVLV2Model
    config_class = DeepseekVLV2Config
    conditional_generation_class = DeepseekVLV2ForConditionalGeneration

    small_config_overrides = {
        # 8 px images, 2 px patches: 4x4 patches, merged 2x2 -> 2x2 tokens per view.
        # Global 2 rows x (2 + newline) + 1 separator + one local tile 2 x (2 + newline) = 13.
        "num_image_tokens": 13,
        "image_token_id": 1,
        "image_size": 8,
        "patch_size": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 4,
        # DeepSeek-V2 MoE with MLA, as in deepseek-vl2-small
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "first_k_dense_replace": 1,
        "q_lora_rank": None,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "vision_hidden_act": "gelu_pytorch_tanh",
        "vision_use_head": False,
    }

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {
            "pixel_values_local": floats_tensor([batch_size, self.num_channels, self.image_size, self.image_size]),
            "image_spatial_crop": torch.ones((batch_size, 2), dtype=torch.long, device=torch_device),
        }


@require_torch
class DeepseekVLV2ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekVLV2VisionText2TextModelTester

    @unittest.skip(
        reason="DeepseekV2 computes RoPE in the complex domain; tests/models/deepseek_v2 overrides this check for it."
    )
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip(
        reason="DeepseekV2Config's TP plan always lists self_attn.q_b_proj, which does not exist when q_lora_rank is "
        "None (deepseek-vl2-small and deepseek-vl2)."
    )
    def test_tp_plan_matches_params(self):
        pass

    def test_image_token_layout(self):
        """A 2x1 local grid adds one newline per stitched row, so the token count follows the original formula."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DeepseekVLV2Model(config).to(torch_device).eval()
        size = self.model_tester.image_size
        pixel_values = floats_tensor([1, 3, size, size])
        pixel_values_local = floats_tensor([2, 3, size, size])
        with torch.no_grad():
            features = model.get_image_features(
                pixel_values, pixel_values_local, torch.tensor([[2, 1]], device=torch_device)
            ).pooler_output
        side = 2
        expected = side * (side + 1) + 1 + side * (2 * side + 1)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (expected, config.text_config.hidden_size))
        # Row ends are the learned newline vector; the token after the global view is the view separator.
        torch.testing.assert_close(features[0][side], model.image_newline.to(features[0].dtype))
        torch.testing.assert_close(features[0][side * (side + 1)], model.view_separator.to(features[0].dtype))
