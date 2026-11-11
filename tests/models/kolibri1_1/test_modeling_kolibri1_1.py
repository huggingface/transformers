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
"""Testing suite for the PyTorch Kolibri1_1 model."""

import tempfile
import unittest
from collections import Counter
from unittest import mock

from huggingface_hub.errors import StrictDataclassClassValidationError
from safetensors.torch import load_file, save_file

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Kolibri1_1Config, Kolibri1_1ForCausalLM, Kolibri1_1Model
    from transformers.cache_utils import KVSharingDict
    from transformers.models.kolibri1_1 import modeling_kolibri1_1
    from transformers.models.kolibri1_1.modeling_kolibri1_1 import (
        Kolibri1_1Indexer,
        eager_attention_forward,
        nvfp4_fake_quantize,
    )


class Kolibri1_1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Kolibri1_1Model

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Covers all four sparse roles: an indexer and a reindexer, each producing its K/V or reading another layer's
        self.num_hidden_layers = 5
        self.layer_types = ["sliding_attention"] + ["full_attention"] * 4
        self.index_layers = [1, 3]
        self.kv_source_layers = [0, 1, 1, 1, 4]
        # Small budgets, so that keys are selected within the tester's sequence length
        self.index_topk = 2
        self.index_pool_size = 4
        self.index_compression = 2
        self.index_num_heads = 1
        self.index_num_local_tokens = 2
        self.index_num_sink_tokens = 1
        # A dense MLP layer in front, like the checkpoints of Kolibri 1.1
        self.mlp_only_layers = [0]
        self.mlp_only_intermediate_size = 40
        # Two query heads per key head, so that the indexers can score keys with more than one head
        self.num_key_value_heads = 1


def reference_selection(queries, keys, num_visible, pool, config, is_pool_source, threshold_ties):
    """Per-query reimplementation of the selection of the Kolibri 1.1 kernels, with ties sorted explicitly towards
    earlier keys. Returns the pool and the selected keys of one query, as sets of key indices. `threshold_ties` counts
    the selections whose last selected and first rejected candidates have the same score, by kind."""
    # Routing queries are rounded to NVFP4, like the block summaries and the routing keys below
    queries = nvfp4_fake_quantize(queries)

    def score(key):
        return sum(max(float(query @ key), 0.0) for query in queries) * config.head_dim**-0.5 / config.index_num_heads

    def top(candidates, scores, k, kind):
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))
        if len(ranked) > k and scores[ranked[k - 1]] == scores[ranked[k]]:
            threshold_ties[kind] += 1
        return set(ranked[:k])

    n, c = num_visible, config.index_compression
    if is_pool_source:
        if n <= config.index_pool_size:
            pool = set(range(n))
        else:
            summaries = {
                j: score(nvfp4_fake_quantize(keys[j * c : (j + 1) * c].sum(0) / c)) if (j + 1) * c <= n else 0.0
                for j in range(-(-n // c))
            }
            blocks = top(summaries, summaries, config.index_pool_size // c, "blocks")
            pool = {t for j in blocks for t in range(j * c, min((j + 1) * c, n))}
    budget = config.index_topk + config.index_num_local_tokens + config.index_num_sink_tokens
    if n <= budget:
        return pool, set(range(n))
    routing_keys = nvfp4_fake_quantize(keys)
    candidates = [t for t in pool if config.index_num_sink_tokens <= t < n - config.index_num_local_tokens]
    scores = {t: score(routing_keys[t]) for t in candidates}
    selected = top(candidates, scores, config.index_topk, "tokens")
    selected |= set(range(config.index_num_sink_tokens)) | set(range(n - config.index_num_local_tokens, n))
    return pool, selected


@require_torch
class Kolibri1_1ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Kolibri1_1ModelTester
    # The indexer groups queries by their first visible key (`Kolibri1_1Indexer.query_groups`): data-dependent control flow
    test_torch_exportable = False

    @unittest.skip(reason="Sparse layers always overlay their key selection as an attention mask, which flash rejects")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_sparse_layers_round_keys_and_values(self):
        # The cache and the K/V shared across layers stay unrounded: compare them with what each layer attends to
        config = self.model_tester.get_config()
        model = Kolibri1_1ForCausalLM(config).to(torch_device).eval()
        model.set_attn_implementation("eager")
        attended = {}

        def spy(module, query, key, value, *args, **kwargs):
            attended[module.layer_idx] = key, value
            return eager_attention_forward(module, query, key, value, *args, **kwargs)

        input_ids = torch.randint(3, config.vocab_size, (1, 12), device=torch_device)
        with mock.patch.object(modeling_kolibri1_1, "eager_attention_forward", spy), torch.no_grad():
            cache = model(input_ids, use_cache=True).past_key_values

        for layer_idx, source in enumerate(config.kv_source_layers):
            is_sparse = config.layer_types[layer_idx] == "full_attention"
            for name, states in zip(("keys", "values"), attended[layer_idx]):
                unrounded = getattr(cache.layers[source], name)
                expected = unrounded
                if is_sparse:
                    expected = nvfp4_fake_quantize(unrounded)
                    # Rounding moves the states by much more than the tolerance below
                    self.assertGreater((expected - unrounded).abs().max().item(), 1e-3)
                # The straight-through estimator adds the rounding error back, which is exact up to an ulp
                torch.testing.assert_close(states, expected, rtol=0, atol=1e-6, msg=f"{name} of layer {layer_idx}")

    def test_skipped_causal_mask_is_rebuilt(self):
        # Generic mask creation (e.g. `generate` with a static cache) may skip the causal mask without padding, but
        # sparse layers need it to overlay their selection
        config = self.model_tester.get_config()
        model = Kolibri1_1Model(config).to(torch_device).eval()
        input_ids = torch.randint(3, config.vocab_size, (1, 12), device=torch_device)
        self.assertGreater(config.sliding_window, input_ids.shape[1])  # so that skipping the sliding mask is exact
        with torch.no_grad():
            skipped = model(input_ids, attention_mask={"full_attention": None, "sliding_attention": None})
            torch.testing.assert_close(skipped.last_hidden_state, model(input_ids).last_hidden_state)

    def test_invalid_layouts(self):
        kwargs = {"num_hidden_layers": 4, "layer_types": ["sliding_attention", "full_attention"] * 2}
        Kolibri1_1Config(**kwargs)  # valid without the overrides below
        invalid = [
            ({"layer_types": ["sliding_attention", "full_attention", "linear_attention", "full_attention"]}, "Unsupported layer types"),
            ({"index_layers": [3]}, "first full-attention layer must be an indexer"),
            ({"index_layers": [3, 1]}, "`index_layers` must be increasing full-attention layers"),
            ({"index_layers": [1, 2]}, "`index_layers` must be increasing full-attention layers"),
            ({"kv_source_layers": [0, 1, 2]}, "`kv_source_layers` needs one entry per layer"),
            ({"kv_source_layers": [0, 1, 1, 1]}, "Sliding-attention layer 2 cannot read the K/V of another layer"),
            ({"kv_source_layers": [0, 3, 2, 3]}, "Layer 1 must read the K/V of an earlier layer"),
            ({"kv_source_layers": [0, 0, 2, 3]}, "Layer 1 must read the K/V of a layer with the same attention type"),
            ({"index_pool_size": 10, "index_compression": 4}, "`index_pool_size` must be a multiple"),
            ({"head_dim": 24}, "`head_dim` that is a multiple of 16"),
            ({"mlp_only_layers": [0]}, "`mlp_only_layers` need a positive `mlp_only_intermediate_size`"),
            ({"mlp_only_layers": [4], "mlp_only_intermediate_size": 8}, "`mlp_only_layers` must be layer indices"),
            ({"mlp_only_layers": [0], "mlp_only_intermediate_size": -8}, "`mlp_only_layers` need a positive"),
            # Indexers score keys with the query heads of key head 0
            ({"num_attention_heads": 8, "num_key_value_heads": 2, "index_num_heads": 5}, "`index_num_heads` must be"),
        ]  # fmt: skip
        for overrides, message in invalid:
            with self.subTest(**overrides), self.assertRaisesRegex(StrictDataclassClassValidationError, message):
                Kolibri1_1Config(**{**kwargs, **overrides})

    def test_kv_shared_layers_have_no_kv_weights(self):
        config = self.model_tester.get_config()
        model = Kolibri1_1Model(config)
        for layer_idx, layer in enumerate(model.layers):
            shared = config.kv_source_layers[layer_idx] != layer_idx
            for name in ("k_proj", "v_proj", "k_norm"):
                self.assertEqual(hasattr(layer.self_attn, name), not shared)
        self.assertEqual(config.kv_sharing_modes, [None, "store", "reuse", "reuse", None])

    def test_nvfp4_matches_reference_encoding(self):
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # the E2M1 magnitudes
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # half-way between consecutive magnitudes
        six = torch.tensor([6.0, -6.0])  # so that the scale is 1
        blocks_at_scale_1 = [
            torch.cat([levels, -levels]),  # all E2M1 values, with both zeros
            torch.cat([ties, -ties, six]),
            torch.cat([-torch.nextafter(ties, torch.tensor(0.0)), torch.nextafter(ties, torch.tensor(7.0)), six]),
        ]
        # Powers of two scale exactly: the smallest subnormal E4M3 scale, a subnormal one, the smallest normal one, 1,
        # and the largest power of two below 448
        blocks = [block * 2.0**exponent for block in blocks_at_scale_1 for exponent in (-9, -7, -6, 0, 3, 8)]

        def with_max(maximum):
            return maximum * torch.linspace(-1, 1, 16)

        blocks += [
            torch.zeros(16),  # scale at its minimum
            torch.tensor([2**-10, -(2**-11), 3 * 2**-12, 2**-20, 1e-30] + [0.0] * 11),  # scale at its minimum
            with_max(6 * 7 * 2**-9),  # largest subnormal scale
            with_max(6 * 1.5 * 2**-9),  # subnormal scale half-way between codes, rounded to even
            with_max(6 * 2.5 * 2**-9),  # subnormal scale half-way between codes, rounded to even
            with_max(6 * 1.0625),  # scale half-way between 1 and 1.125, rounded down: values saturate at 6
            with_max(6 * 1.1875),  # scale half-way between 1.125 and 1.25, rounded up
            with_max(6 * 1.07),  # scale rounded up to 1.125
            with_max(6 * 1.9375),  # scale rounded up to the next exponent
            with_max(6 * 448),  # largest scale
            with_max(6 * 464),  # scale above 448, saturated at 448
            torch.tensor([1e4, -1e4, 2688, -3000] + [100.0] * 12),  # scale and values saturated at 448 and 6 * 448
        ]
        # Three blocks of different magnitudes per row, so that sharing a scale across blocks shows
        values = torch.stack(blocks).view(-1, 3 * 16)

        def round_to_e4m3(scales):
            # 3 mantissa bits below the leading one, rounded to nearest-even. Subnormals share the exponent of the
            # smallest normal, 2**-6, so the smallest scale is 2**-9. The largest is 448
            exponent = (torch.frexp(scales).exponent - 1).clamp(min=-6)
            step = torch.ldexp(torch.ones_like(scales), exponent - 3)
            return (torch.round(scales / step) * step).clamp(max=448)

        def reference(values):
            # E2M1 encoding of the NVFP4 reference in torchtitan's Kolibri 1.1 tests, decoded again, block by block
            decoded = []
            for block in values.split(16, dim=-1):
                scales = round_to_e4m3((block.abs().amax(-1, keepdim=True) / 6).clamp(min=2**-9))
                normalized = block / scales
                levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
                # Find the closest level for each normalized value, breaking ties towards earlier levels
                distance = (normalized.abs()[..., None] - levels).abs()
                tied = distance == distance.amin(-1, keepdim=True)
                order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])  # even codes first
                code = torch.where(tied, order, 99).argmin(-1)
                decoded.append(levels[code].copysign(normalized) * scales)
            return torch.cat(decoded, dim=-1)

        torch.testing.assert_close(nvfp4_fake_quantize(values), reference(values), rtol=0, atol=0)
        values = values.bfloat16()
        torch.testing.assert_close(nvfp4_fake_quantize(values), reference(values.float()).bfloat16(), rtol=0, atol=0)

    def test_top_k_mask_breaks_ties_towards_earlier_keys(self):
        inf = float("inf")
        scores = torch.tensor(
            [
                [1.0, 3.0, 1.0, 2.0, 1.0, -inf],  # three keys tied for the last place
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # all keys tied, as when ReLU zeroes every score
                [5.0, 5.0, 5.0, 5.0, 1.0, -inf],  # more keys tied above the threshold than places
                [-inf, 2.0, -inf, -inf, 1.0, -inf],  # fewer candidates than places: masked keys are never selected
            ]
        )
        expected = torch.tensor(
            [
                [True, True, False, True, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [False, True, False, False, True, False],
            ]
        )
        self.assertEqual(Kolibri1_1Indexer.top_k_mask(scores, 3).tolist(), expected.tolist())

    def test_indexer_matches_reference(self):
        config = self.model_tester.get_config()
        config.index_topk, config.index_pool_size, config.index_compression = 5, 8, 2
        config.index_num_local_tokens, config.index_num_sink_tokens = 3, 2
        # Indexers score with the first 2 of the 3 query heads of key head 0, out of 2 key heads
        config.num_attention_heads, config.num_key_value_heads, config.index_num_heads = 6, 2, 2
        num_queries, num_keys, paddings = 25, 30, (4, 2)
        generator = torch.Generator().manual_seed(0)

        def random_values(*shape):
            # Not representable in NVFP4, so that rounding the routing queries and keys changes their scores
            return torch.randn(shape, generator=generator) * 3

        batch_size = len(paddings)
        queries = random_values(batch_size, config.num_attention_heads, num_queries, config.head_dim)
        keys = random_values(batch_size, config.num_key_value_heads, num_keys, config.head_dim)
        # The keys of the first row repeat 4 distinct keys, and repeated keys score the same: its selections have to
        # break ties, also between blocks. The keys of the second row are all distinct: they only tie at a zero score
        keys[0] = keys[0][:, torch.randint(4, (num_keys,), generator=generator)]
        positions = torch.arange(num_keys - num_queries, num_keys)
        # Rows of the batch are left-padded differently
        visible = torch.stack(
            [
                (torch.arange(num_keys) <= positions[:, None]) & (torch.arange(num_keys) >= padding)
                for padding in paddings
            ]
        )
        attention_mask = visible[:, None]

        shared_kv_states = KVSharingDict()
        source, reindexer = Kolibri1_1Indexer(config, 1), Kolibri1_1Indexer(config, 2)
        selected_by_source = source(queries, keys, attention_mask, shared_kv_states)[:, 0]
        (pool,) = shared_kv_states["index_pool_1"]
        rescoring_queries = random_values(batch_size, config.num_attention_heads, num_queries, config.head_dim)
        selected_by_reindexer = reindexer(rescoring_queries, keys, attention_mask, shared_kv_states)[:, 0]

        threshold_ties = [Counter() for _ in paddings]
        for batch_idx, padding in enumerate(paddings):
            document_keys = keys[batch_idx, 0, padding:]
            num_selected = []
            for row in range(num_queries):
                n = int(visible[batch_idx, row].sum())

                def keys_of(mask):
                    return set((mask[batch_idx, row].nonzero().squeeze(-1) - padding).tolist())

                expected_pool, expected = reference_selection(
                    queries[batch_idx, : config.index_num_heads, row],
                    document_keys,
                    n,
                    None,
                    config,
                    True,
                    threshold_ties[batch_idx],
                )
                self.assertEqual(keys_of(pool), expected_pool)
                self.assertEqual(keys_of(selected_by_source), expected)
                _, expected = reference_selection(
                    rescoring_queries[batch_idx, : config.index_num_heads, row],
                    document_keys,
                    n,
                    expected_pool,
                    config,
                    False,
                    threshold_ties[batch_idx],
                )
                self.assertEqual(keys_of(selected_by_reindexer), expected)
                num_selected.append(len(expected))
            # Each row covers dense queries, sparse queries from a full pool and sparse queries from a block pool
            self.assertEqual(min(num_selected), num_keys - num_queries + 1 - padding)
            self.assertEqual(max(num_selected), 10)
            self.assertGreater(int(visible[batch_idx].sum(-1).max()), config.index_pool_size)
        # The first row had to break ties at the threshold of both selections
        self.assertGreater(threshold_ties[0]["blocks"], 0)
        self.assertGreater(threshold_ties[0]["tokens"], 0)

    def test_sparse_attention_changes_long_sequences_only(self):
        config = self.model_tester.get_config()
        model = Kolibri1_1ForCausalLM(config).to(torch_device).eval()
        dense_config = self.model_tester.get_config()
        dense_config.index_topk = 64
        dense = Kolibri1_1ForCausalLM(dense_config).to(torch_device).eval()
        dense.load_state_dict(model.state_dict())
        input_ids = torch.randint(3, config.vocab_size, (1, 12), device=torch_device)
        with torch.no_grad():
            sparse_logits, dense_logits = model(input_ids).logits, dense(input_ids).logits
        budget = config.index_topk + config.index_num_local_tokens + config.index_num_sink_tokens
        # Queries that see no more keys than the budget attend to all of them, the others drop keys
        torch.testing.assert_close(sparse_logits[:, :budget], dense_logits[:, :budget])
        self.assertFalse(torch.allclose(sparse_logits[:, budget:], dense_logits[:, budget:]))

    def test_shared_kv_checkpoint_weights_are_ignored(self):
        # Checkpoints exported from training keep the unused key/value projections of K/V-sharing layers
        config = self.model_tester.get_config()
        for model_class, prefix in ((Kolibri1_1Model, ""), (Kolibri1_1ForCausalLM, "model.")):
            with self.subTest(model_class.__name__), tempfile.TemporaryDirectory() as tmp_dir:
                model_class(config).save_pretrained(tmp_dir)
                state_dict = load_file(f"{tmp_dir}/model.safetensors")
                for name in ("k_proj.weight", "v_proj.weight"):
                    state_dict[f"{prefix}layers.2.self_attn.{name}"] = state_dict[
                        f"{prefix}layers.1.self_attn.{name}"
                    ].clone()
                save_file(state_dict, f"{tmp_dir}/model.safetensors", metadata={"format": "pt"})
                _, loading_info = model_class.from_pretrained(tmp_dir, output_loading_info=True)
                self.assertEqual(loading_info["unexpected_keys"], set())
                self.assertEqual(loading_info["missing_keys"], set())

    def test_pools_are_freed_after_their_last_reader(self):
        # Layer 1 is an indexer without reindexers, like the 15th layer of Kolibri 1.1; layers 3 and 4 read the pool
        # of layer 2
        config = self.model_tester.get_config()
        config.index_layers = [1, 2]
        self.assertEqual(config.index_source_layers, [None, 1, 2, 2, 2])
        model = Kolibri1_1Model(config).to(torch_device).eval()
        events = []

        class RecordingKVSharingDict(KVSharingDict):
            def __delitem__(self, key):
                events.append(f"free {key}")
                super().__delitem__(key)

        for layer_idx, layer in enumerate(model.layers):
            layer.register_forward_hook(lambda *_, layer_idx=layer_idx: events.append(f"layer {layer_idx}"))
        layers = [f"layer {layer_idx}" for layer_idx in range(config.num_hidden_layers)]
        input_ids = torch.randint(3, config.vocab_size, (2, 12), device=torch_device)

        with mock.patch.object(modeling_kolibri1_1, "KVSharingDict", RecordingKVSharingDict):
            with torch.no_grad():
                freed = model(input_ids).last_hidden_state
            self.assertEqual(events, [*layers[:2], "free index_pool_1", *layers[2:], "free index_pool_2"])
            # With gradients, pools are kept until the end of the forward, in case layers are recomputed
            events.clear()
            kept = model(input_ids).last_hidden_state
            self.assertEqual(events, layers)
        torch.testing.assert_close(kept.detach(), freed)

    def test_left_padded_batch_matches_unpadded_rows(self):
        config = self.model_tester.get_config()
        model = Kolibri1_1ForCausalLM(config).to(torch_device).eval()
        long_ids = torch.randint(3, config.vocab_size, (1, 12), device=torch_device)
        short_ids = torch.randint(3, config.vocab_size, (1, 9), device=torch_device)
        pad = torch.full((1, 3), config.pad_token_id or 0, device=torch_device)
        input_ids = torch.cat([long_ids, torch.cat([pad, short_ids], dim=-1)])
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, :3] = 0
        with torch.no_grad():
            batched = model(input_ids, attention_mask=attention_mask).logits
            long_logits, short_logits = model(long_ids).logits, model(short_ids).logits
        # Both rows see more keys than the sparse budget, so key selection is active in them
        budget = config.index_topk + config.index_num_local_tokens + config.index_num_sink_tokens
        self.assertGreater(short_ids.shape[1], budget)
        torch.testing.assert_close(batched[0], long_logits[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(batched[1, 3:], short_logits[0], rtol=1e-4, atol=1e-4)
