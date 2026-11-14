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

import unittest

from transformers import Qwen2_5_VLConfig, Qwen2VLConfig, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.vision_utils import prepare_qwen_vl_inputs, prepare_vision_inputs


@require_torch
class PrepareVisionInputsTest(unittest.TestCase):
    def test_temporal_and_partial_window_boundaries(self):
        config = Qwen2_5_VLConfig(vision_config={"spatial_merge_size": 2, "patch_size": 14, "window_size": 56})
        grid = torch.tensor([[1, 4, 4], [2, 2, 6]])
        prepared = prepare_vision_inputs(grid, config.vision_config)
        self.assertEqual(prepared["cu_seqlens"].tolist(), [0, 16, 28, 40])
        self.assertEqual(prepared["sequence_lengths"], (16, 12, 12))
        self.assertEqual(prepared["cu_window_seqlens"].tolist(), [0, 16, 24, 28, 36, 40])
        self.assertEqual(prepared["window_sequence_lengths"], (16, 8, 4, 8, 4))
        self.assertEqual(prepared["window_index"].tolist(), list(range(10)))
        self.assertEqual(prepared["max_seqlen"], 16)
        self.assertEqual(prepared["max_window_seqlen"], 16)

    def test_position_order_and_partial_overrides(self):
        config = Qwen2VLConfig(vision_config={"spatial_merge_size": 2})
        positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]])
        prepared = prepare_vision_inputs(torch.tensor([[1, 2, 4]]), config.vision_config)
        torch.testing.assert_close(prepared["position_ids"], positions)
        overrides = {"position_ids": positions, "max_seqlen": 16}
        actual = prepare_vision_inputs(torch.tensor([[1, 2, 4]]), config.vision_config, overrides)
        self.assertIs(actual["position_ids"], positions)
        self.assertEqual(actual["sequence_lengths"], (8,))
        self.assertEqual(actual["max_seqlen"], 16)
        self.assertEqual(set(overrides), {"position_ids", "max_seqlen"})

    def test_invalid_metadata(self):
        config = Qwen2_5_VLConfig(vision_config={"spatial_merge_size": 2})
        grid = torch.tensor([[1, 2, 4]])
        cases = [
            ({"window_index": torch.arange(2)}, "Supply both"),
            ({"cu_seqlens": torch.tensor([0, 7])}, "cover all"),
            ({"sequence_lengths": (4, 4)}, "must match"),
            ({"max_seqlen": 1}, "longest attention"),
            ({"position_ids": torch.zeros(7, 2)}, "each patch"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                prepare_vision_inputs(grid, config.vision_config, overrides)
        with self.assertRaisesRegex(ValueError, "before moving"):
            prepare_vision_inputs(grid.to("meta"), config.vision_config)
        with self.assertRaisesRegex(ValueError, "divisible"):
            prepare_vision_inputs(torch.tensor([[1, 3, 4]]), config.vision_config)

    def test_multimodal_cpu_preparation_without_model(self):
        for config_class in (Qwen2VLConfig, Qwen2_5_VLConfig):
            with self.subTest(config_class=config_class):
                config = config_class(vision_config={"spatial_merge_size": 2})
                ids = torch.tensor([[8, 4, 4, 8, 5, 5, 5, 5, 8]])
                batch = {
                    "input_ids": ids,
                    "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 2, 2, 2, 2, 0]]),
                    "image_grid_thw": torch.tensor([[1, 2, 4]]),
                    "video_grid_thw": torch.tensor([[2, 2, 4]]),
                    "pixel_values": torch.empty(8, 1176),
                    "pixel_values_videos": torch.empty(16, 1176),
                    "second_per_grid_ts": torch.tensor([0.5]),
                }
                prepared = prepare_qwen_vl_inputs(config, batch)
                self.assertEqual(prepared["image_split_sizes"], (2,))
                self.assertEqual(prepared["video_split_sizes"], (4,))
                self.assertEqual(prepared["image_sequence_lengths"], (8,))
                self.assertEqual(prepared["video_sequence_lengths"], (8, 8))
                self.assertEqual(prepared["position_ids"].shape, (3, 1, 9))
                self.assertEqual(prepared["rope_deltas"].shape, (1, 1))
                self.assertNotIn("position_ids", batch)
                self.assertEqual(set(prepare_qwen_vl_inputs(config, {"input_ids": ids})), {"input_ids"})
                overridden = prepare_qwen_vl_inputs(
                    config, {**batch, "image_position_ids": prepared["image_position_ids"]}
                )
                self.assertIs(overridden["image_position_ids"], prepared["image_position_ids"])

    def test_different_layouts_equal_patch_totals(self):
        config = Qwen2VLConfig(vision_config={"spatial_merge_size": 2})
        grid_a = torch.tensor([[1, 2, 4]])
        grid_b = torch.tensor([[1, 4, 2]])
        prep_a = prepare_vision_inputs(grid_a, config.vision_config)
        prep_b = prepare_vision_inputs(grid_b, config.vision_config)
        self.assertEqual(prep_a["position_ids"].shape, prep_b["position_ids"].shape)
        self.assertEqual(prep_a["sequence_lengths"], prep_b["sequence_lengths"])
        self.assertFalse(torch.equal(prep_a["position_ids"], prep_b["position_ids"]))

    def test_text_only_passthrough_and_immutability(self):
        config = Qwen2VLConfig(vision_config={"spatial_merge_size": 2})
        raw_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        original_keys = set(raw_inputs.keys())
        prepared = prepare_qwen_vl_inputs(config, raw_inputs)
        self.assertEqual(set(prepared.keys()), original_keys)
        self.assertIsNot(prepared, raw_inputs)
        torch.testing.assert_close(prepared["input_ids"], raw_inputs["input_ids"])

    def test_temporal_scaling_qwen2_5_vs_qwen2(self):
        qwen2_config = Qwen2VLConfig(vision_config={"spatial_merge_size": 2})
        qwen2_5_config = Qwen2_5_VLConfig(vision_config={"spatial_merge_size": 2, "tokens_per_second": 20})
        batch = {
            "input_ids": torch.tensor([[8, 5, 5, 5, 5, 8]]),
            "mm_token_type_ids": torch.tensor([[0, 2, 2, 2, 2, 0]]),
            "video_grid_thw": torch.tensor([[2, 2, 4]]),
            "pixel_values_videos": torch.empty(16, 1176),
            "second_per_grid_ts": torch.tensor([0.5]),
        }
        prep_q2 = prepare_qwen_vl_inputs(qwen2_config, batch)
        prep_q25 = prepare_qwen_vl_inputs(qwen2_5_config, batch)
        # Qwen2.5 scales temporal positions by tokens_per_second * second_per_grid_ts = 20 * 0.5 = 10
        # Qwen2 leaves time_interval = 1
        pos_q2 = prep_q2["position_ids"]
        pos_q25 = prep_q25["position_ids"]
        self.assertEqual(pos_q2.shape, pos_q25.shape)
        # Temporal row is row 0
        self.assertFalse(torch.equal(pos_q2[0], pos_q25[0]))
        # Spatial rows (height, width) match exactly
        torch.testing.assert_close(pos_q2[1:], pos_q25[1:])

    def test_generation_with_precomputed_inputs(self):
        from tests.models.qwen2_vl.test_modeling_qwen2_vl import Qwen2VLVisionText2TextModelTester
        from transformers import Qwen2VLForConditionalGeneration

        tester = Qwen2VLVisionText2TextModelTester(parent=None)
        config, inputs = tester.prepare_config_and_inputs_for_common()
        inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        model = Qwen2VLForConditionalGeneration(config).eval()
        model.set_attn_implementation("eager")

        with torch.no_grad():
            out_eager = model.generate(**inputs, max_new_tokens=2, do_sample=False)

        prepared = prepare_qwen_vl_inputs(config, inputs)
        with torch.no_grad():
            out_prep = model.generate(**prepared, max_new_tokens=2, do_sample=False)

        torch.testing.assert_close(out_prep, out_eager)
