# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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
"""Unit tests for MiniCPM-V 4.6 mrope_canvas M-RoPE helpers."""

import unittest

from transformers import MiniCPMV4_6Config, is_torch_available
from transformers.models.minicpmv4_6.mrope_minicpmv4_6 import (
    compute_canvas_rope_index,
    expand_1d_position_ids_to_3d,
    uses_mrope_canvas,
)
from transformers.models.minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor


if is_torch_available():
    import torch

    from transformers import MiniCPMV4_6Model


class MiniCPMV4_6MropeUtilsTest(unittest.TestCase):
    def test_uses_mrope_canvas(self):
        self.assertFalse(uses_mrope_canvas(None))
        self.assertFalse(uses_mrope_canvas("disabled"))
        self.assertFalse(uses_mrope_canvas(""))
        self.assertTrue(uses_mrope_canvas("canvas"))
        self.assertTrue(uses_mrope_canvas("Canvas"))

    def test_config_property(self):
        disabled_cfg = MiniCPMV4_6Config()
        self.assertFalse(disabled_cfg.uses_mrope_canvas)

        canvas_cfg = MiniCPMV4_6Config(mrope_mode="canvas")
        self.assertTrue(canvas_cfg.uses_mrope_canvas)

    def test_sample_ids_per_visual(self):
        # Several visual inputs of one sample must all map back to that sample, otherwise their target sizes
        # end up scattered across the neighbouring samples of the batch.
        sample_ids = MiniCPMV4_6Processor._sample_ids_per_visual
        processor = MiniCPMV4_6Processor.__new__(MiniCPMV4_6Processor)

        self.assertEqual(sample_ids(processor, ["<image><image><image>"], "<image>"), [0, 0, 0])
        self.assertEqual(sample_ids(processor, ["<image><image>", "<image>"], "<image>"), [0, 0, 1])
        self.assertEqual(sample_ids(processor, ["no visual input"], "<image>"), [])
        self.assertEqual(sample_ids(processor, None, "<image>"), [])

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_expand_1d_position_ids_to_3d(self):
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

        pos = expand_1d_position_ids_to_3d(input_ids, attention_mask)
        self.assertEqual(tuple(pos.shape), (3, 2, 4))
        expected = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 3]])
        for dim in range(3):
            self.assertTrue(torch.equal(pos[dim], expected))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_mrope_canvas_text_only_matches_1d(self):
        input_ids = torch.tensor([[10, 11, 12, 13]])
        attention_mask = torch.ones_like(input_ids)

        canvas_pos, _ = compute_canvas_rope_index(
            input_ids,
            attention_mask,
            image_bounds=[torch.zeros(0, 2, dtype=torch.long)],
            target_sizes=[torch.zeros(0, 2, dtype=torch.int32)],
            special_token_ids={},
        )
        oned_pos = expand_1d_position_ids_to_3d(input_ids, attention_mask)
        self.assertTrue(torch.equal(canvas_pos, oned_pos))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_mrope_canvas_image_assigns_spatial_positions(self):
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        attention_mask = torch.ones_like(input_ids)
        special_token_ids = {
            "im_start_id": 10,
            "im_end_id": 11,
            "slice_start_id": 12,
            "slice_end_id": 13,
        }

        canvas_pos, _ = compute_canvas_rope_index(
            input_ids,
            attention_mask,
            image_bounds=[torch.tensor([[2, 6]], dtype=torch.long)],
            target_sizes=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=special_token_ids,
        )
        visual_h = canvas_pos[1, 0, 2:6]
        visual_w = canvas_pos[2, 0, 2:6]
        self.assertFalse(torch.equal(visual_h, visual_w))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_model_resolve_position_ids_respects_config(self):
        config = MiniCPMV4_6Config(
            mrope_mode="disabled",
            text_config={
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 64,
                "layer_types": ["full_attention"],
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "rope_parameters": {"rope_type": "default"},
            },
            vision_config={
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "image_size": 32,
                "patch_size": 8,
            },
        )
        model = MiniCPMV4_6Model(config)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        image_bounds = [torch.tensor([[1, 2]], dtype=torch.long)]
        target_sizes = [torch.tensor([[4, 4]], dtype=torch.int32)]

        disabled_pos = model._resolve_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            image_bounds=image_bounds,
            target_sizes_mrope=target_sizes,
            special_token_ids={},
        )
        expected = expand_1d_position_ids_to_3d(input_ids, attention_mask)
        self.assertTrue(torch.equal(disabled_pos, expected))
        self.assertIsNone(model.rope_deltas)

        model.config.mrope_mode = "canvas"
        canvas_pos, rope_deltas = model.compute_mrope_position_ids(
            input_ids,
            attention_mask=attention_mask,
            image_bounds=[torch.zeros(0, 2, dtype=torch.long)],
            target_sizes_mrope=[torch.zeros(0, 2, dtype=torch.int32)],
            special_token_ids={},
        )
        self.assertTrue(torch.equal(canvas_pos, expected))
        self.assertEqual(tuple(rope_deltas.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
