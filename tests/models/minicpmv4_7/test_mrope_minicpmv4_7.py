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

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_decode_step_continues_past_prefill_max(self):
        """The first decoded token must sit at `prefill_amax + 1`, never reuse the last prefill position."""
        model = MiniCPMV4_7Model(_tiny_config())
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        attention_mask = torch.ones_like(input_ids)

        prefill_pos = model.compute_3d_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        prefill_amax = int(prefill_pos[1:].amax())

        class _FakeCache:
            def __init__(self, length):
                self._length = length

            def get_seq_length(self):
                return self._length

        next_ids = torch.tensor([[3]])
        next_mask = torch.ones(1, input_ids.shape[1] + 1, dtype=attention_mask.dtype)
        decode_pos = model.compute_3d_position_ids(
            input_ids=next_ids,
            attention_mask=next_mask,
            past_key_values=_FakeCache(input_ids.shape[1]),
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        # Channels 1..3 are the spatial (T, H, W) axes shifted by `rope_deltas`.
        self.assertTrue(torch.equal(decode_pos[1:, :, -1], torch.full((3, 1), prefill_amax + 1, dtype=torch.long)))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_missing_special_token_ids_raises(self):
        """Silently falling back to 1-D positions on a visual input would be a quality regression."""
        config = _tiny_config()
        config.image_start_id = None
        config.image_end_id = None
        config.slice_start_id = None
        config.slice_end_id = None
        config.newline_id = None
        model = MiniCPMV4_7Model(config)

        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        with self.assertRaises(ValueError):
            model.get_rope_index(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            )

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_missing_special_token_ids_text_only_is_allowed(self):
        """Without visual grids there is nothing to place, so the text path must stay usable."""
        config = _tiny_config()
        config.image_start_id = None
        config.image_end_id = None
        config.slice_start_id = None
        config.slice_end_id = None
        config.newline_id = None
        model = MiniCPMV4_7Model(config)

        input_ids = torch.tensor([[1, 2, 3, 4]])
        pos, _ = model.get_rope_index(input_ids, attention_mask=torch.ones_like(input_ids))
        self.assertEqual(tuple(pos.shape), (3, 1, 4))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_span_grid_count_mismatch_raises(self):
        """One `<image>...</image>` span but two grids means processor/model disagree; do not guess."""
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        with self.assertRaises(ValueError):
            compute_canvas_rope_index(
                input_ids,
                torch.ones_like(input_ids),
                target_sizes_mrope=[torch.tensor([[2, 2], [2, 2]], dtype=torch.int32)],
                special_token_ids=SPECIAL_TOKEN_IDS,
            )

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_unbalanced_visual_markers_raise(self):
        """A dropped `</image>` used to silently truncate the remaining spans."""
        # Two `<image_start>` (id 10) but a single `<image_end>` (id 11).
        input_ids = torch.tensor([[1, 10, 100, 100, 10, 100, 100, 11, 2]])
        with self.assertRaises(ValueError):
            compute_canvas_rope_index(
                input_ids,
                torch.ones_like(input_ids),
                target_sizes_mrope=[torch.tensor([[2, 2], [2, 2]], dtype=torch.int32)],
                special_token_ids=SPECIAL_TOKEN_IDS,
            )

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_processor_kwargs_alone_drive_canvas(self):
        """End-to-end contract: the processor only emits `target_sizes_mrope` and `mm_token_type_ids`.

        `special_token_ids` and `image_bounds` are deliberately not returned (see
        `MiniCPMV4_7Processor.__call__`), so the model must resolve the structural ids from its own
        config and still produce genuine 4-channel canvas positions.
        """
        model = MiniCPMV4_7Model(_tiny_config())
        input_ids = torch.tensor([[1, 10, 100, 100, 100, 100, 11, 2]])
        attention_mask = torch.ones_like(input_ids)
        # Mirrors `create_mm_token_type_ids`: 1 marks the visual span, 0 the text tokens.
        mm_token_type_ids = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)

        position_ids = model.compute_3d_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            mm_token_type_ids=mm_token_type_ids,
        )

        self.assertEqual(tuple(position_ids.shape), (4, 1, 8))
        # Canvas actually engaged: the H and W axes disagree inside the visual span.
        visual_h, visual_w = position_ids[2, 0, 2:6], position_ids[3, 0, 2:6]
        self.assertFalse(torch.equal(visual_h, visual_w))
        # Identical to passing the ids explicitly, i.e. the config lookup is the sole source.
        explicit, _ = compute_canvas_rope_index(
            input_ids,
            attention_mask,
            target_sizes_mrope=[torch.tensor([[2, 2]], dtype=torch.int32)],
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        self.assertTrue(torch.equal(position_ids[1:], explicit))

    @unittest.skipUnless(is_torch_available(), "torch not available")
    def test_text_only_batch_stays_one_dimensional(self):
        """No visual grids means no canvas: every spatial axis collapses onto the 1-D ramp.

        `compute_3d_position_ids` returns 3 channels here (the 4th text channel is only prepended
        on the canvas path), and `mm_token_type_ids` must not accidentally trigger the canvas branch.
        """
        model = MiniCPMV4_7Model(_tiny_config())
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        position_ids = model.compute_3d_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=torch.zeros_like(input_ids, dtype=torch.int32),
        )
        self.assertEqual(tuple(position_ids.shape), (3, 1, 5))
        expected = torch.arange(5, dtype=torch.long)
        for channel in range(3):
            self.assertTrue(torch.equal(position_ids[channel, 0], expected))


if __name__ == "__main__":
    unittest.main()
