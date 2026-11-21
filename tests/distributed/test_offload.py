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
import tempfile
import unittest

from transformers import MetalConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.testing_utils import (
    require_kernels,
    require_torch,
    require_torch_gpu,
    require_torch_mps,
    require_triton,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from safetensors.torch import save_file


@require_torch
class ExpertsOffloadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=8,
            num_experts_per_tok=2,
        )
        self.model = Qwen3MoeForCausalLM(config).eval()
        self.ids = torch.randint(0, 64, (1, 12))
        self.expected = self.generate(self.model)
        self.expert_bytes = 3 * 16 * 32 * 4
        self.resident = sum(p.numel() * 4 for n, p in self.model.named_parameters() if "experts" not in n) + sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        )

    def generate(self, model):
        with torch.no_grad():
            return model.generate(self.ids, max_new_tokens=8, do_sample=False)

    def load(self, path, slots, **kwargs):
        model = Qwen3MoeForCausalLM.from_pretrained(
            path,
            device_map={"": "cpu", "experts": "disk"},
            max_memory={"cpu": self.resident + slots * self.expert_bytes},
            **kwargs,
        )
        return model, model._experts_offload

    def test_matches_eager_for_every_cache_size_and_lookahead(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path, save_original_format=False)  # stacked banks: one row per expert
            for slots in (1, 5, 24):
                for lookahead in (0, 1, 2):
                    with self.subTest(slots=slots, lookahead=lookahead):
                        model, store = self.load(path, slots)
                        store.lookahead = lookahead
                        torch.testing.assert_close(self.generate(model), self.expected)
                        self.assertEqual(sum(p.num_slots for p in store.pools.values()), slots)
                        self.assertGreater(store.reads, 0)

    def test_pinned_layers_stay_resident(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path, save_original_format=False)
            model, store = self.load(path, 8 + 3)  # layer 0 pinned (8 experts), 3 shared slots
            store.pin_layers = 1
            torch.testing.assert_close(self.generate(model), self.expected)
            pool = store.pool_of[store.order[0]]
            self.assertTrue(pool.complete(store.order[0]))
            self.assertFalse(store.pool_of[store.order[1]].complete(store.order[1]))
            self.assertEqual(store.pool_of[store.order[1]].num_slots, 3)

    def test_streamed_embedding_rows(self):
        """An embedding on "disk" reads only the rows it looks up."""
        self.assertFalse(self.model.config.tie_word_embeddings)
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path, save_original_format=False)
            model = Qwen3MoeForCausalLM.from_pretrained(
                path,
                device_map={"": "cpu", "experts": "disk", "embed_tokens": "disk"},
                max_memory={"cpu": self.resident + 5 * self.expert_bytes},
            )
            self.assertIn("model.embed_tokens", model._experts_offload.embeddings)
            torch.testing.assert_close(self.generate(model), self.expected)

    def test_lfu_matches_eager(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path, save_original_format=False)
            model, store = self.load(path, 5)
            store.policy = "lfu"  # pools are made at the first forward
            torch.testing.assert_close(self.generate(model), self.expected)
            self.assertTrue(any(pool.counts for pool in store.pools.values()))

    def test_original_format_is_refused(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)  # per-expert keys: every read would need a conversion
            with self.assertRaisesRegex(ValueError, "save_original_format=False"):
                self.load(path, 5)

    def test_stacked_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.config.save_pretrained(path)
            save_file({k: v.contiguous() for k, v in self.model.state_dict().items()}, f"{path}/model.safetensors")
            model, store = self.load(path, 5)
            torch.testing.assert_close(self.generate(model), self.expected)
            self.assertGreater(store.hits, 0)


@require_torch_mps
@require_kernels
class MetalExpertsOffloadTest(unittest.TestCase):
    def test_streamed_quantized_experts_match_in_memory(self):
        """Experts quantized on the fly (Metal `gather_qmm`) give the same tokens streamed or in memory."""
        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_tok=2,
        )
        ids = torch.randint(0, 64, (1, 12), device="mps")
        plan = {"*.mlp.experts": MetalConfig(bits=4, group_size=64)}
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as path:
            Qwen3MoeForCausalLM(config).save_pretrained(raw)
            kwargs = {"quantization_config": plan, "dtype": torch.float32}
            reference = Qwen3MoeForCausalLM.from_pretrained(raw, device_map="mps", **kwargs)
            # Quantized once, saved in the transformers format, streamed from there: no work per read
            reference.save_pretrained(path, save_original_format=False)
            streamed = Qwen3MoeForCausalLM.from_pretrained(
                path, device_map={"": "mps", "experts": "disk"}, max_memory={"mps": 10 << 20}, dtype=torch.float32
            )
            with torch.no_grad():
                expected = reference.generate(ids, max_new_tokens=8, do_sample=False)
                torch.testing.assert_close(streamed.generate(ids, max_new_tokens=8, do_sample=False), expected)
            store = streamed._experts_offload
            self.assertEqual(next(iter(store.pools.values())).layout["gate_up_proj"][1], torch.uint32)
            from transformers.integrations.mps_events import load

            if load() is not None:  # misses resolved by the host thread while the device waits
                self.assertIsNotNone(store.mps_router)
                self.assertGreater(store.misses, 0)
            self.assertGreater(store.reads, 0)

    def test_mixed_plan_streams_quantized_experts(self):
        """A saved plan mixing Metal experts and FP8 elsewhere: experts load through an identity converter with their
        scales, which still reads one row per bank."""
        from safetensors.torch import save_file

        from transformers.integrations.metal_quantization import _affine_quantize_tensor

        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_tok=2,
        )
        model = Qwen3MoeForCausalLM(config)
        state = {}
        for key, value in model.state_dict().items():
            if key.endswith(("experts.gate_up_proj", "experts.down_proj")):
                experts, rows, columns = value.shape
                packed, scales, biases = _affine_quantize_tensor(value.reshape(-1, columns).float(), 64, 4)
                state[key] = packed.reshape(experts, rows, -1)
                state[f"{key}_scales"] = scales.reshape(experts, rows, -1)
                state[f"{key}_qbiases"] = biases.reshape(experts, rows, -1)
            else:
                state[key] = value.contiguous()
        config.quantization_config = {
            "*.mlp.experts": {"quant_method": "metal", "bits": 4, "group_size": 64},
            "*": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head"],
            },
        }
        ids = torch.randint(0, 64, (1, 10), device="mps")
        with tempfile.TemporaryDirectory() as path:
            config.save_pretrained(path)
            save_file(state, f"{path}/model.safetensors", metadata={"format": "pt"})
            reference = Qwen3MoeForCausalLM.from_pretrained(path, device_map="mps", dtype=torch.float32)
            streamed = Qwen3MoeForCausalLM.from_pretrained(
                path, device_map={"": "mps", "experts": "disk"}, max_memory={"mps": 10 << 20}, dtype=torch.float32
            )
            with torch.no_grad():
                expected = reference.generate(ids, max_new_tokens=6, do_sample=False)
                torch.testing.assert_close(streamed.generate(ids, max_new_tokens=6, do_sample=False), expected)
            self.assertGreater(streamed._experts_offload.reads, 0)


@require_torch_gpu
@require_triton()
class TritonAffineExpertsTest(unittest.TestCase):
    def test_gather_qmv_matches_dequantized(self):
        from transformers.integrations.metal_quantization import _affine_dequantize_tensor, _affine_quantize_tensor
        from transformers.integrations.triton_affine_moe import gather_qmv

        torch.manual_seed(0)
        weight = torch.randn(4 * 96, 256, device="cuda")
        packed, scales, biases = _affine_quantize_tensor(weight, 64, 4)
        reference = _affine_dequantize_tensor(packed, scales, biases, 64, 4).view(4, 96, 256)
        x, ids = torch.randn(10, 256, device="cuda"), torch.randint(0, 4, (10,), device="cuda")
        out = gather_qmv(x, packed.view(4, 96, -1), scales.view(4, 96, -1), biases.view(4, 96, -1), ids, 64, 4)
        torch.testing.assert_close(out, torch.einsum("sk,snk->sn", x, reference[ids]), rtol=1e-4, atol=1e-4)

    def test_streamed_quantized_experts_match_in_memory(self):
        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=8,
            num_experts_per_tok=2,
        )
        ids = torch.randint(0, 64, (1, 12), device="cuda")
        plan = {"*.mlp.experts": MetalConfig(bits=4, group_size=64)}
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as path:
            Qwen3MoeForCausalLM(config).save_pretrained(raw)
            kwargs = {"quantization_config": plan, "dtype": torch.float32}
            reference = Qwen3MoeForCausalLM.from_pretrained(raw, device_map="cuda", **kwargs)
            # Quantized once, saved in the transformers format, streamed from there: no work per read
            reference.save_pretrained(path, save_original_format=False)
            streamed = Qwen3MoeForCausalLM.from_pretrained(
                path, device_map={"": "cuda", "experts": "disk"}, max_memory={"cuda": 10 << 20}, dtype=torch.float32
            )
            with torch.no_grad():
                expected = reference.generate(ids, max_new_tokens=8, do_sample=False)
                torch.testing.assert_close(streamed.generate(ids, max_new_tokens=8, do_sample=False), expected)
