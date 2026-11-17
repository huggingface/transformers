# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from safetensors.torch import load_file, save_file

    from transformers import OlmoeConfig, OlmoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.integrations.experts_offload import (
        ExpertStore,
        SafetensorsReader,
        _split_expert,
        quantize_experts,
    )
    from transformers.integrations.metal_quantization import _affine_dequantize_tensor


HIDDEN, INTERMEDIATE, EXPERTS = 16, 8, 4
EXPERT_BYTES = 3 * HIDDEN * INTERMEDIATE * 4  # float32
IDS = [[1, 7, 3, 9, 2, 11, 5]]


def tiny_qwen3_moe(seed=0):
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=24,
        moe_intermediate_size=INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
    )
    config._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(config).eval()


def greedy(model, ids, tokens=6):
    with torch.no_grad():
        return model.generate(torch.tensor(ids), max_new_tokens=tokens, do_sample=False, pad_token_id=0)


class PaddedSource:
    """An external expert store over a model's experts: float32 gate_up and down, plus (layer 1) a padding tensor,
    so the two layers differ in size and get one pool each."""

    def __init__(self, model):
        self.experts = {}
        for index, layer in enumerate(model.model.layers):
            experts = layer.mlp.experts
            self.experts[f"source.{index}"] = (experts.gate_up_proj.detach(), experts.down_proj.detach(), index)
        self.prefixes = list(self.experts)
        self.read_bytes = self.read_calls = 0

    def prefix(self, module_name):
        return f"source.{module_name.split('.')[2]}"

    def layout(self, prefix):
        gate_up, down, index = self.experts[prefix]
        layout = [(tuple(gate_up.shape[1:]), torch.float32), (tuple(down.shape[1:]), torch.float32)]
        return layout + ([((16,), torch.uint8)] if index else [])

    def expert_bytes(self, prefix):
        return sum(math.prod(shape) * dtype.itemsize for shape, dtype in self.layout(prefix))

    def read(self, prefix, expert, out):
        gate_up, down, _ = self.experts[prefix]
        blob = torch.cat([gate_up[expert].reshape(-1).view(torch.uint8), down[expert].reshape(-1).view(torch.uint8)])
        out[: blob.numel()] = blob
        out[blob.numel() :] = 7
        self.read_bytes += out.numel()
        self.read_calls += 1


def reference_compute(module, store, hidden_states, top_k_index, top_k_weights, routing, experts, pinned):
    """`expert_compute` for :class:`PaddedSource`: every route through `store.expert_tensors`."""
    result = torch.zeros_like(hidden_states)
    prefix = module.offload_prefix
    for expert in experts:
        gate_up, down, *padding = store.expert_tensors(prefix, expert)
        if padding:
            assert bool((padding[0] == 7).all())
        token, slot = (t.to(hidden_states.device) for t in torch.where(routing == expert))
        activated = module._apply_gate(torch.nn.functional.linear(hidden_states[token], gate_up))
        output = torch.nn.functional.linear(activated, down) * top_k_weights[token, slot, None]
        result.index_add_(0, token, output.to(result.dtype))
    return result


# TODO(portability): run on Linux/macOS/Windows and parameterize available devices; current loads use CPU.
@require_torch
class ExpertsOffloadTest(unittest.TestCase):
    def load(self, path, **options):
        return Qwen3MoeForCausalLM.from_pretrained(
            path, attn_implementation="eager", experts_offload={"cache_bytes": 3 * EXPERT_BYTES, **options}
        )

    def test_loads_without_expert_banks_and_matches_eager(self):
        reference = tiny_qwen3_moe()
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            model, info = Qwen3MoeForCausalLM.from_pretrained(
                directory,
                attn_implementation="eager",
                experts_offload={"cache_bytes": 3 * EXPERT_BYTES},
                output_loading_info=True,
            )
            self.assertEqual((info["missing_keys"], info["unexpected_keys"]), (set(), set()))
            experts = model.model.layers[0].mlp.experts
            self.assertIsNone(experts.gate_up_proj)
            self.assertEqual(experts.config._experts_implementation, "experts_offload")
            self.assertFalse(any(name.endswith(("gate_up_proj", "down_proj")) for name in model.state_dict()))
            self.assertIsNone(model.experts_offload_store.pools[0].banks)  # device pools are taken at the first use
            torch.testing.assert_close(greedy(model, IDS), greedy(reference, IDS))
            self.assertGreater(model.experts_offload_store.misses, 0)

    def test_every_cache_size_and_kernel_matches_eager(self):
        reference = tiny_qwen3_moe(1)
        expected = greedy(reference, IDS)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            # Cover uncached, partial-cache and fully resident execution.
            for slots in (0, 1, 3, 8):
                for implementation in ("grouped_mm", "batched_mm"):
                    with self.subTest(slots=slots, implementation=implementation):
                        model = self.load(
                            directory, cache_bytes=slots * EXPERT_BYTES, experts_implementation=implementation
                        )
                        store = model.experts_offload_store
                        self.assertEqual(store.num_slots, slots)
                        torch.testing.assert_close(greedy(model, IDS), expected)
                        self.assertLessEqual(store.resident_bytes, slots * EXPERT_BYTES)
                        self.assertEqual(store.all_resident(), slots == 8)

    def test_reads_hits_evictions_and_zero_budget(self):
        reference = tiny_qwen3_moe(2)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            for budget in (0, EXPERT_BYTES):
                with self.subTest(budget=budget):
                    # No I/O threads: no reads ahead, so the counts below are exact.
                    model = self.load(directory, cache_bytes=budget, io_threads=0)
                    store = model.experts_offload_store
                    module, eager = model.model.layers[0].mlp.experts, reference.model.layers[0].mlp.experts
                    hidden, weights = torch.randn(2, HIDDEN), torch.ones(2, 1)
                    ids = torch.zeros(2, 1, dtype=torch.long)
                    with torch.inference_mode():
                        torch.testing.assert_close(module(hidden, ids, weights), eager(hidden, ids, weights))
                        self.assertEqual(store.stats()["read_bytes"], EXPERT_BYTES)
                        module(hidden, ids, weights)
                        self.assertEqual(store.hits, int(budget > 0))
                        module(hidden, ids + 1, weights)
                        torch.testing.assert_close(module(hidden, ids, weights), eager(hidden, ids, weights))
                    self.assertEqual(store.misses, 3 if budget else 4)
                    self.assertLessEqual(store.resident_bytes, budget)

    def test_prefetch_threads_and_depths_match_eager(self):
        reference = tiny_qwen3_moe(3)
        expected = greedy(reference, IDS, tokens=8)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            for io_threads, depth in ((0, 0), (1, 1), (4, 1), (4, 2)):
                with self.subTest(io_threads=io_threads, depth=depth):
                    model = self.load(directory, io_threads=io_threads, prefetch_depth=depth)
                    layers = model.model.layers
                    if depth:
                        self.assertIs(layers[0].mlp.experts.offload_next[1], layers[1].mlp.gate)
                    self.assertIsNone(layers[1].mlp.experts.offload_next)
                    torch.testing.assert_close(greedy(model, IDS, tokens=8), expected)
                    store = model.experts_offload_store
                    self.assertEqual(store.pending, {})
                    if io_threads:
                        self.assertGreater(store.prefetch_used, 0)
                    else:
                        self.assertEqual(store.prefetched, 0)

    def test_prefill_readahead_matches_eager(self):
        reference = tiny_qwen3_moe(3)
        expected = greedy(reference, IDS, tokens=8)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            # The 7-token prompt has 14 routes per layer, decode steps 2: a threshold of 8 reads ahead in prefill only.
            for routes in (0, 8):
                with self.subTest(prefill_readahead_routes=routes):
                    model = self.load(directory, io_threads=2, prefetch_depth=0, prefill_readahead_routes=routes)
                    layers = model.model.layers
                    next_prefix = layers[1].mlp.experts.offload_prefix
                    self.assertEqual(layers[0].mlp.experts.offload_next_prefix, next_prefix)
                    self.assertIsNone(layers[1].mlp.experts.offload_next_prefix)
                    store = model.experts_offload_store
                    calls = []
                    prefetch = store.prefetch

                    def spy(prefix, experts):
                        calls.append((prefix, experts))
                        return prefetch(prefix, experts)

                    with mock.patch.object(store, "prefetch", side_effect=spy):
                        torch.testing.assert_close(greedy(model, IDS, tokens=8), expected)
                    self.assertEqual(store.pending, {})
                    readahead = [call for call in calls if call == (next_prefix, range(EXPERTS))]
                    # Once per forward of layer 0 with enough routes: the prompt, not the decode steps.
                    self.assertEqual(len(readahead), 1 if routes else 0)

    def test_mps_commit_hooks(self):
        reference = tiny_qwen3_moe(3)
        expected = greedy(reference, IDS, tokens=8)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            for enabled in (True, False):
                with self.subTest(mps_commit=enabled):
                    with mock.patch("transformers.integrations.experts_offload._commit_mps") as commit:
                        model = self.load(directory, mps_commit=enabled)
                        torch.testing.assert_close(greedy(model, IDS, tokens=8), expected)
                    self.assertEqual(commit.called, enabled)

    def test_preload_and_fully_resident_layers_skip_the_host_copy(self):
        reference = tiny_qwen3_moe(4)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            with self.assertRaises(ValueError):
                self.load(directory, preload=True)  # 3 slots cannot hold 8 experts
            model = self.load(directory, cache_bytes=8 * EXPERT_BYTES, preload=True)
            store = model.experts_offload_store
            self.assertTrue(store.all_resident())
            read_bytes = store.stats()["read_bytes"]
            original = torch.Tensor.cpu
            with mock.patch.object(torch.Tensor, "cpu", autospec=True, side_effect=original) as cpu:
                torch.testing.assert_close(greedy(model, IDS), greedy(reference, IDS))
            self.assertFalse(any(call.args[0].dtype == torch.long for call in cpu.call_args_list))
            self.assertEqual((store.misses, store.stats()["read_bytes"]), (0, read_bytes))

    def test_per_expert_checkpoint_layout(self):
        torch.manual_seed(5)
        config = OlmoeConfig(
            vocab_size=64,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=EXPERTS,
            num_experts_per_tok=2,
        )
        config._attn_implementation = "eager"
        reference = OlmoeForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            reference.save_pretrained(path)
            # Match the Hub checkpoint: one tensor per expert and projection.
            tensors = load_file(path / "model.safetensors")
            for name in [name for name in tensors if name.endswith("experts.gate_up_proj")]:
                prefix = name.removesuffix(".gate_up_proj")
                gate_up, down = tensors.pop(name), tensors.pop(f"{prefix}.down_proj")
                for expert in range(EXPERTS):
                    gate, up = gate_up[expert].chunk(2)
                    tensors[f"{prefix}.{expert}.gate_proj.weight"] = gate.contiguous()
                    tensors[f"{prefix}.{expert}.up_proj.weight"] = up.contiguous()
                    tensors[f"{prefix}.{expert}.down_proj.weight"] = down[expert].contiguous()
            save_file(tensors, path / "model.safetensors", {"format": "pt"})
            model = OlmoeForCausalLM.from_pretrained(
                path, attn_implementation="eager", experts_offload={"cache_bytes": 3 * EXPERT_BYTES}
            )
            torch.testing.assert_close(greedy(model, IDS), greedy(reference, IDS))

    def test_quantized_experts_match_the_dequantized_model(self):
        reference = tiny_qwen3_moe(6)
        # 12 tokens x top-2 over 4 experts: prefill takes the sorted path, decode the pool path.
        ids = [[1, 7, 3, 9, 2, 11, 5, 13, 4, 8, 6, 10]]
        group = 8
        shapes = [
            ((2 * INTERMEDIATE, HIDDEN * 8 // 32), torch.uint32),
            ((2 * INTERMEDIATE, HIDDEN // group), torch.float32),
            ((2 * INTERMEDIATE, HIDDEN // group), torch.float32),
            ((HIDDEN, INTERMEDIATE * 8 // 32), torch.uint32),
            ((HIDDEN, INTERMEDIATE // group), torch.float32),
            ((HIDDEN, INTERMEDIATE // group), torch.float32),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path, quantized = Path(directory) / "model", Path(directory) / "experts-q8"
            reference.save_pretrained(path)
            quantize_experts(path, quantized, bits=8, group_size=group)
            # The reference runs the dequantized experts.
            with torch.no_grad():
                for index, layer in enumerate(reference.model.layers):
                    blobs = load_file(quantized / f"experts-{index:05d}.safetensors")
                    blobs = blobs[f"model.layers.{index}.mlp.experts.experts"]
                    parts = [_split_expert(blobs[expert], shapes) for expert in range(EXPERTS)]
                    for offset, projection in ((0, "gate_up_proj"), (3, "down_proj")):
                        dequantized = [_affine_dequantize_tensor(*p[offset : offset + 3], group, 8) for p in parts]
                        getattr(layer.mlp.experts, projection).copy_(torch.stack(dequantized))
            expected = greedy(reference, ids)
            for cache_bytes in (1000, 3000, 8000):
                with self.subTest(cache_bytes=cache_bytes):
                    model = self.load(path, cache_bytes=cache_bytes, quantized_experts=quantized)
                    torch.testing.assert_close(greedy(model, ids), expected)
            # The quantized experts can be the only copy: drop the BF16 experts from the checkpoint.
            tensors = {k: v for k, v in load_file(path / "model.safetensors").items() if ".mlp.experts." not in k}
            save_file(tensors, path / "model.safetensors", {"format": "pt"})
            model = self.load(path, cache_bytes=3000, quantized_experts=quantized)
            torch.testing.assert_close(greedy(model, ids), expected)

    def test_host_tier(self):
        reference = tiny_qwen3_moe(6)
        ids = [[1, 7, 3, 9, 2, 11, 5, 13, 4, 8, 6, 10]]
        with tempfile.TemporaryDirectory() as directory:
            path, quantized = Path(directory) / "model", Path(directory) / "experts-q8"
            reference.save_pretrained(path)
            quantize_experts(path, quantized, bits=8, group_size=8)
            expected = greedy(self.load(path, cache_bytes=3000, quantized_experts=quantized), ids)
            slot_bytes = self.load(
                path, cache_bytes=3000, quantized_experts=quantized
            ).experts_offload_store.slot_bytes
            total = 2 * EXPERTS  # two layers
            for host_experts, preload, io_threads in (
                (total, False, 2),
                (3, False, 2),
                (total, True, 0),
                (total, True, 4),
            ):
                with self.subTest(host_experts=host_experts, preload=preload, io_threads=io_threads):
                    model = self.load(
                        path, cache_bytes=3000, quantized_experts=quantized, io_threads=io_threads,
                        host_cache_bytes=host_experts * slot_bytes, host_preload=preload,
                    )  # fmt: skip
                    store = model.experts_offload_store
                    self.assertEqual(store.host_slots, host_experts)
                    reads = store.stats()["read_calls"]
                    self.assertEqual(reads, total if preload else 0)
                    torch.testing.assert_close(greedy(model, ids), expected)
                    torch.testing.assert_close(greedy(model, ids), expected)
                    if host_experts == total:
                        # Every expert was read at most once: device-cache misses are served from host memory.
                        self.assertEqual(store.stats()["read_calls"], total)
                        self.assertGreater(store.host_hits, 0)
                    store.synchronize()  # rows of the last copies are released at the next synchronization
                    self.assertEqual(store.host_busy, set())  # every handed-out row was released
            with self.assertRaises(ValueError):
                self.load(path, cache_bytes=3000, host_cache_bytes=1 << 20)  # needs quantized experts

    def test_device_map_sends_experts_to_disk(self):
        reference = tiny_qwen3_moe(10)
        expected = greedy(reference, IDS)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            for key in ("experts.*", "model.layers.*.mlp.experts"):
                with self.subTest(key=key):
                    model = Qwen3MoeForCausalLM.from_pretrained(
                        directory,
                        attn_implementation="eager",
                        device_map={"": "cpu", key: "disk"},
                        experts_offload={"cache_bytes": 3 * EXPERT_BYTES},
                    )
                    experts = model.model.layers[0].mlp.experts
                    self.assertIsNone(experts.gate_up_proj)
                    self.assertEqual(experts.config._experts_implementation, "experts_offload")
                    torch.testing.assert_close(greedy(model, IDS), expected)
                    self.assertGreater(model.experts_offload_store.misses, 0)
            for device_map, options in (
                ({"experts.*": "disk"}, {"cache_bytes": EXPERT_BYTES}),  # nothing on a device
                ({"": "cpu", "experts.*": "disk"}, None),  # no cache size
            ):
                with self.subTest(device_map=device_map), self.assertRaises(ValueError):
                    Qwen3MoeForCausalLM.from_pretrained(directory, device_map=device_map, experts_offload=options)

    def test_routing_profile_preload(self):
        reference = tiny_qwen3_moe(11)
        expected = greedy(reference, IDS)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            model = self.load(directory, cache_bytes=4 * EXPERT_BYTES, io_threads=0)
            greedy(model, IDS)
            profile = model.experts_offload_store.routing_profile()
            self.assertEqual(
                sum(map(sum, (p.values() for p in profile.values()))), model.experts_offload_store.lookups
            )
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile))
            for preload in (profile, str(path)):
                model = self.load(directory, cache_bytes=4 * EXPERT_BYTES, io_threads=2, preload_profile=preload)
                store = model.experts_offload_store
                self.assertIsNone(store.pools[0].banks)  # nothing read before the first forward
                torch.testing.assert_close(greedy(model, IDS), expected)
                for prefix, counts in profile.items():
                    top = sorted(counts, key=counts.get, reverse=True)[:2]  # 4 slots, 2 layers
                    self.assertTrue(all((prefix, expert) in store.cache for expert in top[:1]))
                self.assertLess(store.misses, model.experts_offload_store.lookups)

    def test_expert_source_with_mixed_layouts(self):
        reference = tiny_qwen3_moe(8)
        expected = greedy(reference, IDS, tokens=8)
        source = PaddedSource(reference)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            sizes = [source.expert_bytes(prefix) for prefix in source.prefixes]
            self.assertNotEqual(*sizes)  # one pool per layer
            for io_threads, depth, slots in ((0, 0, 0), (0, 0, 2), (2, 1, 2), (2, 2, EXPERTS)):
                with self.subTest(io_threads=io_threads, depth=depth, slots=slots):
                    model = self.load(
                        directory, cache_bytes=slots * sum(sizes), io_threads=io_threads, prefetch_depth=depth,
                        expert_source=source, expert_compute=reference_compute,
                    )  # fmt: skip
                    store = model.experts_offload_store
                    self.assertEqual(len(store.pools), 2)
                    self.assertEqual([pool.num_slots for pool in store.pools], [slots, slots])
                    self.assertEqual(model.model.layers[1].mlp.experts.offload_prefix, source.prefixes[1])
                    torch.testing.assert_close(greedy(model, IDS, tokens=8), expected)
                    self.assertEqual(store.pending, {})
                    self.assertGreater(store.stats()["read_calls"], 0)
                    for pool in store.pools:
                        self.assertLessEqual(len(pool.cache), pool.num_slots)
                    if slots == EXPERTS:
                        self.assertEqual(store.misses, 2 * EXPERTS - store.installed_ahead)
            with self.assertRaises(ValueError):
                self.load(directory, expert_source=source)  # needs expert_compute

    def test_cold_prefill_routes_matches_eager(self):
        reference = tiny_qwen3_moe(9)
        expected = greedy(reference, IDS, tokens=8)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            # 14 routes per layer in the 7-token prompt, 2 per decode step.
            model = self.load(
                directory, io_threads=2, prefetch_depth=0, cold_prefill_routes=8, prefill_readahead_routes=8,
                prefill_install_ahead=True,
            )  # fmt: skip
            torch.testing.assert_close(greedy(model, IDS, tokens=8), expected)
            store = model.experts_offload_store
            self.assertEqual(store.cold_installed, {})
            self.assertFalse(store.cold)

    def test_inference_only_and_option_validation(self):
        reference = tiny_qwen3_moe(7)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            with self.assertRaises(ValueError):
                self.load(directory, cache_size=1)
            with self.assertRaises(ValueError):
                Qwen3MoeForCausalLM.from_pretrained(directory, experts_offload={"io_threads": 2})
            with self.assertRaises(ValueError):
                self.load(directory, experts_implementation="eager")
            model = self.load(directory)
            with self.assertRaises(RuntimeError):
                model(torch.tensor(IDS))  # grad enabled
            model.train()
            with torch.no_grad(), self.assertRaises(RuntimeError):
                model(torch.tensor(IDS))


@require_torch
class ExpertStorePolicyTest(unittest.TestCase):
    def make_store(self, policy, cache_experts, experts=5, hidden=4, intermediate=2):
        banks = {
            "layer.gate_up_proj": torch.arange(experts * 2 * intermediate * hidden, dtype=torch.float32).view(
                experts, 2 * intermediate, hidden
            ),
            "layer.down_proj": torch.arange(experts * hidden * intermediate, dtype=torch.float32).view(
                experts, hidden, intermediate
            ),
        }
        reader = SimpleNamespace(
            tensors=banks,
            bypass_page_cache=False,
            shape=lambda name: tuple(banks[name].shape),
            read=lambda name, expert=None: banks[name][expert].clone(),
            read_bytes=0,
            read_calls=0,
        )
        reader.read_expert = lambda prefix, expert: SafetensorsReader.read_expert(reader, prefix, expert)
        expert_bytes = 3 * intermediate * hidden * 4
        store = ExpertStore(reader, cache_experts * expert_bytes, "cpu", torch.float32, policy=policy)
        store.register("layer", SimpleNamespace(num_experts=experts, intermediate_dim=intermediate, hidden_dim=hidden))
        store.allocate()
        return store, banks

    def test_lfu_keeps_a_hot_set_where_lru_thrashes(self):
        # Repeat a decoding cycle longer than the cache.
        pattern = [0, 1, 2, 3, 4] * 20
        hits = {}
        for policy in ("lru", "lfu"):
            store, banks = self.make_store(policy, cache_experts=3)
            for expert in pattern:
                gate_up, down = store.get("layer", expert)
                torch.testing.assert_close(gate_up, banks["layer.gate_up_proj"][expert])
                torch.testing.assert_close(down, banks["layer.down_proj"][expert])
            hits[policy] = store.hits
            self.assertLessEqual(store.resident_bytes, store.cache_bytes)
        self.assertEqual(hits["lru"], 0)
        # LFU keeps 3 of the 5 experts resident after warm-up: about 3 hits per cycle of 5.
        self.assertGreater(hits["lfu"], 0.5 * len(pattern))

    def test_lfu_follows_a_new_hot_set(self):
        store, _ = self.make_store("lfu", cache_experts=2, experts=6)
        store.decay_every = 8
        for expert in [0, 1] * 20 + [4, 5] * 20:
            store.get("layer", expert)
        self.assertEqual({key[1] for key in store.cache}, {4, 5})

    def test_cold_calls_keep_the_decode_set(self):
        store, _ = self.make_store("lru", cache_experts=3)
        for expert in (0, 1):
            store.acquire("layer", expert)
        store.cold = True  # a long prefill: installs 2, hits 0 without refreshing it
        store.acquire("layer", 2)
        store.acquire("layer", 0)
        store.demote("layer")
        store.cold = False
        store.acquire("layer", 3)  # evicts the cold install, not the decode set
        self.assertEqual({key[1] for key in store.cache}, {0, 1, 3})
