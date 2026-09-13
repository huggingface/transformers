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
"""Testing suite for the PyTorch HELIX model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import DynamicCache, HelixConfig, HelixForCausalLM, HelixModel


class HelixModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HelixModel

    def __init__(
        self,
        parent,
        num_hidden_layers=2,
        block_size=2,
        local_blocks=2,
        num_window_scales=2,
        index_layer_stride=2,
        landmark_dim=8,
        index_branching=2,
        index_beam_width=2,
        index_topk=2,
        index_max_levels=4,
        num_recurrent_heads=2,
        recurrent_head_dim=8,
        recurrent_value_head_dim=8,
        recurrent_chunk_size=2,
        conv_kernel_size=2,
        surprise_kernel_size=2,
    ):
        super().__init__(parent)
        self.num_hidden_layers = num_hidden_layers
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.num_window_scales = num_window_scales
        self.index_layer_stride = index_layer_stride
        self.landmark_dim = landmark_dim
        self.index_branching = index_branching
        self.index_beam_width = index_beam_width
        self.index_topk = index_topk
        self.index_max_levels = index_max_levels
        self.num_recurrent_heads = num_recurrent_heads
        self.recurrent_head_dim = recurrent_head_dim
        self.recurrent_value_head_dim = recurrent_value_head_dim
        self.recurrent_chunk_size = recurrent_chunk_size
        self.conv_kernel_size = conv_kernel_size
        self.surprise_kernel_size = surprise_kernel_size


@require_torch
class HelixModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HelixModelTester
    _torch_compile_train_cls = HelixForCausalLM if is_torch_available() else None
    # HELIX enforces causality with its block geometry rather than a 4D mask, and each of its three strands
    # normalizes separately, so there is no single attention matrix to report.
    has_attentions = False

    def test_left_padding_compatibility(self):
        self.skipTest(
            reason="HELIX anchors its memory blocks to absolute cache positions, so left padding shifts the "
            "block grid. Results stay causal and correct, but they are not bit-identical to an unpadded run "
            "-- the same caveat chunked-attention models carry. Right padding is exactly invariant; see "
            "`test_right_padding_is_exactly_invariant`."
        )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """
        HELIX's cache is the architecture's memory claim made concrete, so check it directly rather than
        through the generic helper: every layer carries a fixed-size recurrent state plus two fixed-size
        convolution states, `"helix_local"` layers cap their key/value history at the local window, and
        only the `"helix"` layers -- the ones whose index can reach the distant past -- keep it all.
        """
        self.assertEqual(config.num_hidden_layers, len(past_key_values))
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        key_dim = config.num_recurrent_heads * config.recurrent_head_dim
        conv_dim = 2 * key_dim + config.num_recurrent_heads * config.recurrent_value_head_dim

        for layer_idx, layer in enumerate(past_key_values.layers):
            layer_type = config.layer_types[layer_idx]
            self.assertEqual(
                layer.conv_states[0].shape, (batch_size, conv_dim, config.conv_kernel_size), msg=f"layer {layer_idx}"
            )
            self.assertEqual(
                layer.conv_states[1].shape,
                (batch_size, key_dim, config.surprise_kernel_size),
                msg=f"layer {layer_idx}",
            )
            self.assertEqual(
                layer.recurrent_states[0].shape,
                (batch_size, config.num_recurrent_heads, config.recurrent_head_dim, config.recurrent_value_head_dim),
                msg=f"layer {layer_idx}",
            )
            # A sliding layer retains `sliding_window - 1` tokens; the token being decoded makes up the
            # `sliding_window`-th one, which is exactly what the widest per-head window needs.
            expected_kv = seq_length if layer_type == "helix" else min(seq_length, config.sliding_window - 1)
            for tensor in (layer.keys, layer.values):
                self.assertEqual(
                    tensor.shape,
                    (batch_size, config.num_key_value_heads, expected_kv, head_dim),
                    msg=f"layer {layer_idx} ({layer_type})",
                )

    def _build_helix(self, dtype=torch.float64, **overrides):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for key, value in overrides.items():
            setattr(config, key, value)
        torch.manual_seed(0)
        return HelixForCausalLM(HelixConfig(**config.to_diff_dict())).to(torch_device, dtype).eval()

    def test_strictly_causal(self):
        """Editing token `t` must leave every logit before `t` untouched."""
        model = self._build_helix()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 24), device=torch_device)
        with torch.no_grad():
            reference = model(input_ids, use_cache=False).logits
        for position in (3, 11, 20):
            edited = input_ids.clone()
            edited[0, position] = (edited[0, position] + 5) % model.config.vocab_size
            with torch.no_grad():
                logits = model(edited, use_cache=False).logits
            torch.testing.assert_close(logits[:, :position], reference[:, :position], rtol=0, atol=0)

    def test_incremental_decoding_matches_prefill(self):
        """A fixed-size decode state must reproduce the one-shot forward, token for token."""
        model = self._build_helix()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 24), device=torch_device)
        with torch.no_grad():
            reference = model(input_ids, use_cache=False).logits
            cache = DynamicCache(config=model.config)
            steps = [
                model(input_ids[:, i : i + 1], past_key_values=cache, use_cache=True).logits
                for i in range(input_ids.shape[1])
            ]
        torch.testing.assert_close(torch.cat(steps, dim=1), reference, rtol=1e-5, atol=1e-5)

    def test_chunked_prefill_matches_prefill(self):
        """Prefilling in two chunks that do not land on a memory-block boundary must be equivalent."""
        model = self._build_helix()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 24), device=torch_device)
        with torch.no_grad():
            reference = model(input_ids, use_cache=False).logits
            cache = DynamicCache(config=model.config)
            first = model(input_ids[:, :11], past_key_values=cache, use_cache=True).logits
            second = model(input_ids[:, 11:], past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(torch.cat([first, second], dim=1), reference, rtol=1e-5, atol=1e-5)

    def test_right_padding_is_exactly_invariant(self):
        """Trailing padding cannot reach earlier positions, so it must not perturb them at all."""
        model = self._build_helix()
        length, padding = 24, 7
        input_ids = torch.randint(1, model.config.vocab_size, (1, length), device=torch_device)
        padded = torch.cat([input_ids, torch.zeros(1, padding, dtype=torch.long, device=torch_device)], dim=1)
        mask = torch.cat(
            [
                torch.ones(1, length, dtype=torch.long, device=torch_device),
                torch.zeros(1, padding, dtype=torch.long, device=torch_device),
            ],
            dim=1,
        )
        with torch.no_grad():
            reference = model(input_ids, use_cache=False).logits
            observed = model(padded, attention_mask=mask, use_cache=False).logits[:, :length]
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    def test_index_strand_reaches_beyond_the_local_window(self):
        """Every position past the widest local window must still have a live index selection."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        self.assertIn("helix", config.layer_types)
        model = self._build_helix()
        span = model.config.local_span
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8 * span), device=torch_device)

        selections = []
        braid = next(layer.mixer for layer in model.model.layers if layer.mixer.has_index)
        original = braid._select_memory_blocks

        def record(*args, **kwargs):
            blocks, scores = original(*args, **kwargs)
            selections.append((blocks, scores))
            return blocks, scores

        braid._select_memory_blocks = record
        with torch.no_grad():
            model(input_ids, use_cache=False)
        braid._select_memory_blocks = original

        blocks, scores = selections[0]
        live = scores > -1e29
        # Query blocks far enough in that a whole memory block precedes them must select something...
        reachable = live[..., 1:, :].any(-1)
        self.assertTrue(bool(reachable.all()), "an eligible query block ended up with no memory selected")
        # ...and it must be able to reach blocks the local window can no longer see.
        query_block = torch.arange(blocks.shape[2], device=blocks.device).view(1, 1, -1, 1)
        distant = live & (blocks < query_block - model.config.local_blocks)
        self.assertTrue(bool(distant.any()), "the index never selected a block outside the local window")

    def test_landmark_tree_depth_follows_the_memory(self):
        """
        The tree must always grow until its top level fits in the descent's seed beam. If it ever stopped
        short, the nodes above the cut would be unreachable and part of the memory would go silently dark.
        """
        model = self._build_helix(dtype=torch.float32)
        config = model.config
        braid = next(layer.mixer for layer in model.model.layers if layer.mixer.has_index)
        for num_blocks in (1, config.index_branching - 1, config.index_branching, 5 * config.index_branching**2):
            leaves = torch.zeros(1, config.num_key_value_heads, num_blocks, config.landmark_dim)
            with torch.no_grad():
                levels = braid._build_landmark_tree(leaves)
            self.assertEqual(len(levels) - 1, config.index_num_levels(num_blocks), msg=f"{num_blocks} blocks")
            self.assertLess(levels[-1].shape[2], config.index_branching, msg=f"{num_blocks} blocks")

    def test_tiling_does_not_change_the_result(self):
        """Query-block tiling is a memory knob, not a modelling one: it must be bitwise inert."""
        model = self._build_helix()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 29), device=torch_device)
        outputs = {}
        for tile in (0, 1, 3, 64):
            for layer in model.model.layers:
                layer.mixer.tile_blocks = tile
            with torch.no_grad():
                outputs[tile] = model(input_ids, use_cache=False).logits
        for tile, logits in outputs.items():
            torch.testing.assert_close(logits, outputs[0], rtol=0, atol=0, msg=f"tile={tile}")

    def test_attention_width_is_independent_of_context_length(self):
        """
        The direct evidence for `O(N)` compute: every query attends over the same number of keys no matter
        how long the context is, so total work grows linearly rather than quadratically.
        """
        model = self._build_helix(dtype=torch.float32)
        span = model.config.local_span
        for layer in model.model.layers:
            braid = layer.mixer
            original = braid._blocked_attention

            def record(*args, _braid=braid, _original=original, **kwargs):
                keys = args[1] if len(args) > 1 else kwargs["keys"]
                observed.append(keys.shape[3])
                return _original(*args, **kwargs)

            braid._blocked_attention = record

        seen = {}
        for length in (8 * span, 32 * span):
            observed = []
            input_ids = torch.randint(0, model.config.vocab_size, (1, length), device=torch_device)
            with torch.no_grad():
                model(input_ids, use_cache=False)
            seen[length] = sorted(set(observed))

        self.assertEqual(seen[8 * span], seen[32 * span])
        self.assertTrue(
            all(
                width <= (model.config.local_blocks + 1 + model.config.index_topk) * model.config.block_size
                for width in seen[8 * span]
            )
        )
