# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Kimi2.6 model."""

import unittest

from parameterized import parameterized

from transformers import (
    DeepseekV3Config,
    Kimi_K25Config,
    Kimi_K25VisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.cache_utils import Cache
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_modeling_common import (
    floats_tensor,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import Kimi_K25ForConditionalGeneration, Kimi_K25Model


if is_vision_available():
    pass


class Kimi_K25VisionText2TextModelTester(VLMModelTester):
    base_model_class = Kimi_K25Model
    config_class = Kimi_K25Config
    text_config_class = DeepseekV3Config
    vision_config_class = Kimi_K25VisionConfig
    conditional_generation_class = Kimi_K25ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 16)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("pos_emb_height", 4)
        kwargs.setdefault("merge_kernel_size", (1, 1))
        kwargs.setdefault("pos_emb_width", 4)
        kwargs.setdefault("pos_emb_time", 1)
        kwargs.setdefault("kv_lora_rank", 16)
        kwargs.setdefault("q_lora_rank", 32)
        kwargs.setdefault("qk_rope_head_dim", 16)
        kwargs.setdefault("v_head_dim", 32)
        kwargs.setdefault("qk_nope_head_dim", 32)
        kwargs.setdefault("attention_probs_dropout_prob", 0.0)

        # MoE fields synced with DeepSeekV3Tester
        # `first_k_dense_replace` enables MoEs after `layer=0`
        kwargs.setdefault("first_k_dense_replace", 1)
        kwargs.setdefault("n_group", 2)
        kwargs.setdefault("topk_group", 1)
        kwargs.setdefault("num_experts_per_tok", 8)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("aux_loss_alpha", 0.001)
        kwargs.setdefault("routed_scaling_factor", 2.5)

        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": 10000,
            },
        )
        super().__init__(parent, **kwargs)

        # These can be inferred from existing properties and don't get separate kwargs
        self.projection_hidden_size = self.hidden_size

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels,
                self.patch_size,
                self.patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        # Place image tokens with vision_start_token_id prefix
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {
            "image_grid_thw": torch.tensor([[1, 4, 4]] * self.batch_size, device=torch_device),
        }


@require_torch
class Kimi_K25ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Kimi_K25VisionText2TextModelTester
    test_torch_exportable = False

    # Kimi has images shaped as (bs*patch_len, dim) so we can't slice to batches in generate
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "use_cache",
            "labels",
        ]

        # The diff from the general `prepare_config_and_inputs_for_generate` lies here
        patch_size = config.vision_config.patch_size
        filtered_image_length = batch_size * (self.model_tester.image_size**2) // (patch_size**2)
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }
        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][:filtered_image_length]

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
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

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as deepseek has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Deepseek-V3 backbone uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Deepseek-V3 backbone uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims on LM backbone")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Needs to update values in `grid_thw` otherwise it just gets broadcasted")
    def test_mismatching_num_image_tokens(self):
        pass
