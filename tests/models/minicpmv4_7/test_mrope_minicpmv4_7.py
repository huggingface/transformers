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
"""Unit tests for MiniCPM-V 4.7 canvas M-RoPE."""

import unittest

from transformers import MiniCPMV4_7Config, is_torch_available
from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import (
    compute_canvas_rope_index,
    expand_1d_position_ids_to_3d,
)
from transformers.models.minicpmv4_7.processing_minicpmv4_7 import MiniCPMV4_7Processor


if is_torch_available():
    import torch

    from transformers import MiniCPMV4_7Model


SPECIAL_TOKEN_IDS = {
    "im_start_id": 10,
    "im_end_id": 11,
    "slice_start_id": 12,
    "slice_end_id": 13,
    "newline_id": 14,
}


def _tiny_config(**kwargs):
    return MiniCPMV4_7Config(
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
        image_start_id=10,
        image_end_id=11,
        slice_start_id=12,
        slice_end_id=13,
        newline_id=14,
        **kwargs,
    )


class MiniCPMV4_7MropeUtilsTest(unittest.TestCase):
    def test_sample_ids_per_visual(self):
        sample_ids = MiniCPMV4_7Processor._sample_ids_per_visual
        processor = MiniCPMV4_7Processor.__new__(MiniCPMV4_7Processor)

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

        canvas_pos, deltas = compute_canvas_rope_index(
            input_ids,
            attention_mask,
            target_sizes_mrope=[torch.zeros(0, 2, dtype=torch.int32)],
            special_token_ids={},
        )
        oned_pos = expand_1d_position_ids_to_3d(input_ids, attention_mask)
        self.assertTrue(torch.equal(canvas_pos, oned_pos))
        # amax(=3) + 1 - seq_len(=4) == 0
        self.assertTrue(torch.equal(deltas, torch.zeros(1, 1, dtype=torch.long)))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_mrope_canvas_image_assigns_spatial_positions(self):
        # [bos, im_start, 4 visual, im_end, eos]
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        attention_mask = torch.ones_like(input_ids)

        canvas_pos, _ = compute_canvas_rope_index(
            input_ids,
            attention_mask,
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        visual_h = canvas_pos[1, 0, 2:6]
        visual_w = canvas_pos[2, 0, 2:6]
        self.assertFalse(torch.equal(visual_h, visual_w))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_left_padding_matches_unpadded(self):
        # Unpadded baseline: [bos, im_start, 4 visual, im_end, eos]
        unpadded = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        unpadded_mask = torch.ones_like(unpadded)
        target_sizes = [torch.tensor([[2, 2]], dtype=torch.int32)]

        baseline, _ = compute_canvas_rope_index(
            unpadded, unpadded_mask, target_sizes_mrope=target_sizes, special_token_ids=SPECIAL_TOKEN_IDS
        )

        # Left-pad by 2 zeros
        left_padded = torch.tensor([[0, 0, 1, 10, 100, 100, 100, 100, 11, 2]])
        left_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
        left_pos, _ = compute_canvas_rope_index(
            left_padded, left_mask, target_sizes_mrope=target_sizes, special_token_ids=SPECIAL_TOKEN_IDS
        )

        self.assertTrue(torch.equal(left_pos[:, 0, 2:], baseline[:, 0, :]))
        self.assertTrue(torch.equal(left_pos[:, 0, :2], torch.zeros(3, 2, dtype=torch.long)))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_rope_deltas_plus_one(self):
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        attention_mask = torch.ones_like(input_ids)
        canvas_pos, deltas = compute_canvas_rope_index(
            input_ids,
            attention_mask,
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        expected = canvas_pos.amax(dim=(0, 2)).unsqueeze(1) + 1 - attention_mask.sum(-1).unsqueeze(1)
        self.assertTrue(torch.equal(deltas, expected.long()))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_compute_3d_position_ids_returns_4d(self):
        model = MiniCPMV4_7Model(_tiny_config())
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        attention_mask = torch.ones_like(input_ids)
        position_ids = model.compute_3d_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        self.assertEqual(tuple(position_ids.shape), (4, 1, 8))
        self.assertIsNotNone(model.rope_deltas)


if __name__ == "__main__":
    unittest.main()
