# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HY4-VL model."""

import copy
import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    HYV4VLConfig,
    HYV4VLForConditionalGeneration,
    HYV4VLModel,
    is_torch_available,
)
from transformers.testing_utils import Expectations, require_torch, slow, torch_device

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class HYV4VLVisionText2TextModelTester(VLMModelTester):
    config_class = HYV4VLConfig
    base_model_class = HYV4VLModel
    conditional_generation_class = HYV4VLForConditionalGeneration

    small_config_overrides = {
        "batch_size": 2,
        "vocab_size": 256,
        "hidden_size": 48,
        "intermediate_size": 96,
        "moe_intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 16,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "index_topk": 16,
        "hc_mult": 2,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [4, 2, 2]},
        "video_token_id": 4,
        "image_size": 16,
        "spatial_merge_size": 2,
        "num_image_tokens": 6,
        "vision_hidden_size": 32,
        "vision_intermediate_size": 64,
        "vision_num_attention_heads": 2,
        "vision_num_key_value_heads": None,
        "vision_num_hidden_layers": 2,
        "vision_hidden_act": "gelu",
        "vision_max_image_size": 64,
        # The vision output projects into the text width (hidden_size).
        "vision_out_hidden_size": 48,
        # image_size // patch_size (VLMModelTester's 4), and its square.
        "grid_hw": 4,
        "num_image_patches": 16,
    }

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id}

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size * self.num_image_patches, self.num_channels * self.patch_size**2]).to(
            torch_device
        )

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {
            "image_grid_thw": torch.tensor([[1, self.grid_hw, self.grid_hw]] * batch_size, device=torch_device),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }

    def prepare_multimodal_inputs(self, image_grid=None, video_grid=None):
        """Text, then optionally one image and one video, each followed by a text token."""
        input_ids, token_types = [10, 11, 12], [0, 0, 0]

        inputs = {}
        for grid, token_id, modality, pixel_key, grid_key in (
            (image_grid, self.image_token_id, 1, "pixel_values", "image_grid_thw"),
            (video_grid, self.video_token_id, 2, "pixel_values_videos", "video_grid_thw"),
        ):
            if grid is None:
                continue
            frames, height, width = grid
            # One newline token per merged row
            num_tokens = frames * (height // self.spatial_merge_size) * (width // self.spatial_merge_size + 1)
            input_ids.extend([token_id] * num_tokens + [13])
            token_types.extend([modality] * num_tokens + [0])
            inputs[pixel_key] = floats_tensor([frames * height * width, self.num_channels * self.patch_size**2]).to(
                torch_device
            )
            inputs[grid_key] = torch.tensor([grid], device=torch_device)

        inputs["input_ids"] = torch.tensor([input_ids], device=torch_device)
        inputs["mm_token_type_ids"] = torch.tensor([token_types], device=torch_device)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs


@require_torch
class HYV4VLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HYV4VLVisionText2TextModelTester
    # HYV4 routes each token to a subset of experts, so not every expert receives a gradient.
    test_all_params_have_gradient = False
    # Packed vision features have a leading singleton dimension and one newline token per merged row
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True
    model_split_percents = [0.5, 0.9]

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """Override to account for the difference in MHC and the final state shapes"""
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            model_input_length = 1 if use_cache and generated_length > 0 else prompt_length + generated_length
            # We have raw MHC shapes until the final one which is collapsed
            mhc_shape = (batch_size, model_input_length, config.hc_mult, config.hidden_size)
            final_shape = (batch_size, model_input_length, config.hidden_size)
            expected_shapes = [mhc_shape] * (len(iter_hidden_states) - 1) + [final_shape]
            self.assertListEqual([state.shape for state in iter_hidden_states], expected_shapes)

    def test_hidden_states_output(self):
        """Override to account for the difference in MHC and the final state shapes"""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        text_config = config.text_config
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states

            self.assertEqual(len(hidden_states), text_config.num_hidden_layers + 1)
            batch_size, seq_len = inputs["input_ids"].shape
            for state in hidden_states[:-1]:
                self.assertEqual(state.shape, (batch_size, seq_len, text_config.hc_mult, text_config.hidden_size))
            self.assertEqual(hidden_states[-1].shape, (batch_size, seq_len, text_config.hidden_size))

    @unittest.skip("The text attention (sinks + DSA) is eager-only, so the top-level model cannot switch to SDPA")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip("Packed vision attention returns no attention matrix")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("Packed vision attention returns no attention matrix")
    def test_get_video_features_attentions(self):
        pass

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=True):
        # Source prefixes describe the complete checkpoint, including the model wrapper
        return super().test_reverse_loading_mapping(check_keys_were_modified, skip_base_model)

    @parameterized.expand([((1, 4, 6), None), (None, (2, 4, 4)), ((1, 4, 6), (2, 4, 4))])
    def test_image_video_and_mixed_forward(self, image_grid, video_grid):
        config = self.model_tester.get_config()
        inputs = self.model_tester.prepare_multimodal_inputs(image_grid, video_grid)
        model = HYV4VLForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape, (*inputs["input_ids"].shape, config.text_config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_rope_index_with_newline_and_video_frames(self):
        """Each frame is a `(H, W + 1)` block, all three axes start after the previous block."""
        model = HYV4VLModel(self.model_tester.get_config()).to(torch_device)
        inputs = self.model_tester.prepare_multimodal_inputs((1, 4, 6), (2, 4, 4))
        positions, deltas = model.get_rope_index(**inputs)
        expected = torch.tensor(
            [
                [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 7, 8, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 14],
                [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 7, 8, 8, 8, 9, 9, 9, 11, 11, 11, 12, 12, 12, 14],
                [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 13, 11, 12, 13, 14],
            ],
            device=torch_device,
        ).unsqueeze(1)
        torch.testing.assert_close(positions, expected)
        torch.testing.assert_close(deltas, torch.tensor([[-10]], device=torch_device))


@slow
@require_torch
class HYV4VLIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    # Tiny randomly initialized model with the real tokenizer, chat template and processors
    model_id = "hf-internal-testing/tiny-random-HYV4VLForCausalLM"
    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
    )
    video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def prepare_inputs(self, modality, url, **processor_kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": modality, "url": url_to_local_path(url)},
                    {"type": "text", "text": f"Describe this {modality}."},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="no_think",
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        ).to(torch_device)

    def check_logits_and_generation(self, inputs, *, num_input_tokens, expected_logits, expected_ids):
        model = HYV4VLForConditionalGeneration.from_pretrained(self.model_id).to(torch_device)
        self.assertEqual(inputs["input_ids"].shape[1], num_input_tokens)

        with torch.no_grad():
            logits = model(**inputs).logits
        torch.testing.assert_close(
            logits[0, -1, :10].float().cpu(), torch.tensor(expected_logits), atol=1e-4, rtol=1e-4
        )

        generated = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        self.assertEqual(generated[0, num_input_tokens:].tolist(), expected_ids)

    def test_small_model_integration_test_image(self):
        inputs = self.prepare_inputs("image", self.image_url)
        expected_logits = Expectations(
            {
                ("cpu", None): [-0.0391, 0.0135, 0.0313, -0.0234, 0.0115, -0.0174, 0.0262, 0.0399, 0.0332, -0.0241],
                ("cuda", None): [-0.0391, 0.0135, 0.0313, -0.0234, 0.0115, -0.0174, 0.0262, 0.0399, 0.0332, -0.0241],
                ("xpu", None): [-0.0391, 0.0135, 0.0313, -0.0234, 0.0115, -0.0174, 0.0262, 0.0399, 0.0332, -0.0241],
            }
        ).get_expectation()  # fmt: skip
        expected_ids = Expectations(
            {
                ("cpu", None): [16079, 84189, 20392, 17356, 59944, 48231, 55278, 119971, 84746, 97572],
                ("cuda", None): [16079, 84189, 20392, 17356, 59944, 8939, 92382, 9537, 2002, 109634],
                ("xpu", None): [16079, 84189, 20392, 17356, 59944, 8939, 92382, 9537, 2002, 109634],
            }
        ).get_expectation()  # fmt: skip
        self.check_logits_and_generation(
            inputs, num_input_tokens=681, expected_logits=expected_logits, expected_ids=expected_ids
        )

    def test_small_model_integration_test_video(self):
        inputs = self.prepare_inputs(
            "video", self.video_url, videos_kwargs={"do_sample_frames": True, "num_frames": 4, "fps": None}
        )
        expected_logits = [-0.0341, 0.0138, 0.0291, -0.0269, 0.0079, -0.0128, 0.0272, 0.0359, 0.0318, -0.0189]
        expected_ids = [16079, 84189, 20392, 17356, 59944, 48231, 55278, 119971, 84746, 109384]
        self.check_logits_and_generation(
            inputs, num_input_tokens=3638, expected_logits=expected_logits, expected_ids=expected_ids
        )
