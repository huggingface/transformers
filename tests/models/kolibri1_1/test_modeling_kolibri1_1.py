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

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized
from safetensors.torch import load_file, save_file

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Kolibri1_1Config, Kolibri1_1ForCausalLM, Kolibri1_1Model
    from transformers.models.kolibri1_1.modeling_kolibri1_1 import (
        Kolibri1_1Indexer,
        nvfp4_fake_quantize,
    )


class Kolibri1_1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Kolibri1_1Model

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Covers all four sparse roles: an indexer and a reindexer, each producing its K/V or reading another layer's
        self.num_hidden_layers = 5
        self.layer_types = ["sliding_attention"] + ["indexed_attention"] * 4
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


@require_torch
class Kolibri1_1ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Kolibri1_1ModelTester
    # The indexer groups queries by their first visible key (`Kolibri1_1Indexer.query_groups`): data-dependent control flow
    test_torch_exportable = False

    @unittest.skip(reason="Sparse layers always overlay their key selection as an attention mask, which flash rejects")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @parameterized.expand([("dynamic",), ("static",)])
    def test_cached_decode_matches_full_forward(self, cache_implementation):
        # A decoding step selects keys for one query from the cache, the full forward for all queries at once
        config = self.model_tester.get_config()
        model = Kolibri1_1ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(3, config.vocab_size, (2, 12), device=torch_device)
        outputs = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            min_new_tokens=6,
            max_new_tokens=6,
            do_sample=False,
            cache_implementation=cache_implementation,
            output_logits=True,
            return_dict_in_generate=True,
        )
        with torch.no_grad():
            full_logits = model(outputs.sequences).logits
        # Not bit-identical: both score the same keys, but in matrix products of different shapes
        decode_logits = torch.stack(outputs.logits, dim=1)
        torch.testing.assert_close(decode_logits, full_logits[:, input_ids.shape[1] - 1 : -1], rtol=1e-5, atol=1e-5)

    def test_invalid_layouts(self):
        # Checkpoints list the layers with an indexer as "full_attention"
        kwargs = {"num_hidden_layers": 4, "layer_types": ["sliding_attention", "full_attention"] * 2}
        # Valid without the overrides below
        self.assertEqual(Kolibri1_1Config(**kwargs).layer_types, ["sliding_attention", "indexed_attention"] * 2)
        invalid = [
            ({"layer_types": ["sliding_attention", "full_attention", "linear_attention", "full_attention"]}, "Unsupported layer types"),
            ({"index_layers": [3]}, "first indexed-attention layer must be an indexer"),
            ({"index_layers": [3, 1]}, "`index_layers` must be increasing indexed-attention layers"),
            ({"index_layers": [1, 2]}, "`index_layers` must be increasing indexed-attention layers"),
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
