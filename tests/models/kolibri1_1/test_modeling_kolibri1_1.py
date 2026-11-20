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

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator_memory, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


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

    @unittest.skip("Float8 quantization + TP numerical noise exceeds match threshold")
    def test_tp_generation_quantized(self):
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
        kwargs = {"num_hidden_layers": 4, "layer_types": ["sliding_attention", "indexed_attention"] * 2}
        # Valid without the overrides below
        Kolibri1_1Config(**kwargs)
        invalid = [
            ({"layer_types": ["sliding_attention", "full_attention"] * 2}, "Unsupported layer types"),
            ({"index_layers": [3]}, "first indexed-attention layer must be an indexer"),
            ({"index_layers": [3, 1]}, "`index_layers` must be increasing indexed-attention layers"),
            ({"index_layers": [1, 2]}, "`index_layers` must be increasing indexed-attention layers"),
            ({"kv_source_layers": [0, 1, 2]}, "`kv_source_layers` needs one entry per layer"),
            ({"kv_source_layers": [0, 1, 1, 1]}, "Sliding-attention layer 2 cannot read the K/V of another layer"),
            ({"kv_source_layers": [0, 3, 2, 3]}, "Layer 1 must read the K/V of an earlier layer"),
            ({"kv_source_layers": [0, 0, 2, 3]}, "Layer 1 must read the K/V of a layer with the same attention type"),
            ({"index_topk": 0}, "`index_topk` must be positive"),
            ({"index_pool_size": 0}, "`index_pool_size` must be positive"),
            ({"index_compression": 0}, "`index_compression` must be positive"),
            ({"index_pool_size": 10, "index_compression": 4}, "`index_pool_size` must be a multiple"),
            ({"head_dim": 24}, "`head_dim` that is a multiple of 16"),
            ({"mlp_only_layers": [4], "mlp_only_intermediate_size": 8}, "`mlp_only_layers` must be layer indices"),
            ({"mlp_only_layers": [0], "mlp_only_intermediate_size": -8}, "`mlp_only_layers` need a positive `mlp_only_intermediate_size`"),
            # Indexers score keys with the query heads of key head 0
            ({"num_attention_heads": 8, "num_key_value_heads": 2, "index_num_heads": 5}, "`index_num_heads` must be"),
        ]  # fmt: skip
        for overrides, message in invalid:
            with self.subTest(**overrides), self.assertRaisesRegex(StrictDataclassClassValidationError, message):
                Kolibri1_1Config(**{**kwargs, **overrides})

    def test_default_layout(self):
        # The layout of the released model
        config = Kolibri1_1Config()
        indexed_layers = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "indexed_attention"]
        self.assertEqual(indexed_layers, [4, 9, 14, 19, 24, 29, 34, 39, 44, 49])
        self.assertEqual(config.index_layers, [4, 14, 19, 29])
        self.assertEqual([config.kv_source_layers[i] for i in indexed_layers], [4, 4, 4, 19, 24, 29, 29, 29, 29, 4])
        self.assertEqual(config.mlp_only_layers, [0, 1])
        self.assertEqual(config.mlp_only_intermediate_size, 3584)

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

    def test_masked_tokens_stay_out_of_sparse_layers(self):
        # Masked tokens inside the prompt leave gaps in the keys a query sees, unlike padding at either end
        config = self.model_tester.get_config()
        # Dense MLPs only, so that the logits can be compared exactly: experts run on the tokens routed to them, and
        # a masked token that changes experts changes the size of these batches, which moves results by an ulp
        config.mlp_only_layers = list(range(config.num_hidden_layers))
        model = Kolibri1_1ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(3, config.vocab_size, (1, 20), device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, [5, 6, 13]] = 0
        visible = attention_mask[0].bool()
        other_ids = torch.where(visible, input_ids, (input_ids - 2) % (config.vocab_size - 3) + 3)
        self.assertTrue((other_ids != input_ids)[0, ~visible].all())

        # The content of a masked token reaches no other token, with float (eager) and boolean (sdpa) attention masks
        for attn_implementation in ("eager", "sdpa"):
            with self.subTest(attn_implementation=attn_implementation), torch.no_grad():
                model.set_attn_implementation(attn_implementation)
                logits = model(input_ids, attention_mask=attention_mask).logits
                other_logits = model(other_ids, attention_mask=attention_mask).logits
                torch.testing.assert_close(other_logits[0, visible], logits[0, visible], rtol=0, atol=0)

        # Every query still attends to itself: masked tokens don't shift its most recent tokens
        model.set_attn_implementation("eager")
        with torch.no_grad():
            attentions = model(input_ids, attention_mask=attention_mask, output_attentions=True).attentions
        for layer_idx, weights in enumerate(attentions):
            own_weights = weights[0, :, visible][:, :, visible].diagonal(dim1=-2, dim2=-1)
            self.assertTrue((own_weights > 0).all(), msg=f"layer {layer_idx}")


@slow
@require_torch_accelerator_memory(memory=90)
class Kolibri1_1IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    TEST_MODEL_ID = "Aleph-Alpha/Kolibri-1.1"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Dropped again by the mixin's `tearDownClass`.
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Kolibri1_1ForCausalLM.from_pretrained(cls.TEST_MODEL_ID, device_map="auto")
        return cls.model

    def generate_reply(self, content, max_new_tokens):
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = self.get_model()
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def test_model_logits(self):
        input_ids = [452, 22090, 493, 1678, 2459]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        # fmt: off
        EXPECTED_MEAN = torch.tensor([[-2.4225, -4.4516, -4.4104, -5.9657, -5.6187]])
        EXPECTED_SLICE = torch.tensor(
            [-6.1875, -8.1875, -9.8125, -9.4375, -8.2500, -7.5625, -9.0625, -8.1250, -6.3125, -1.2109]
        )
        # fmt: on
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    def test_model_generation(self):
        EXPECTED_TEXT = (
            "Ein Mixture-of-Experts-Modell ist ein neuronales Netzwerk, bei dem für jede Eingabe nur ein kleiner Teil "
            "der Parameter (die „Experten“) aktiviert wird, während ein Router entscheidet, welche Experten für die"
        )
        text = self.generate_reply("Erkläre in einem Satz, was ein Mixture-of-Experts-Modell ist.", 40)
        self.assertEqual(EXPECTED_TEXT, text)

    def test_model_recall_beyond_index_pool(self):
        # About 18k tokens, more than an indexer's pool of 16384 keys: the code is far beyond the sliding window and
        # the local tokens, so the sparse layers must keep its keys both in the pool and among the selected keys
        filler = "Der Himmel ist blau. Das Gras ist grün. Die Sonne scheint. " * 1300
        prompt = f"Merke dir: Der Geheimcode lautet 4721.\n\n{filler}\nWie lautet der Geheimcode? Antworte nur mit der Zahl."
        text = self.generate_reply(prompt, 10)
        self.assertEqual("4721", text)
