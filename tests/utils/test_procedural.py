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

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3MoeConfig
    from transformers.utils.procedural import ProceduralTensor


def _tiny(config_cls, **kwargs):
    defaults = {
        "vocab_size": 300, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
    }  # fmt: skip
    return config_cls(**{**defaults, **kwargs})


@require_torch
class DummyModelTest(unittest.TestCase):
    def _pair(self, config, **dummy_kwargs):
        """A real model saved to disk, loaded normally and with `dummy=True`."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        AutoModelForCausalLM.from_config(config).save_pretrained(tmp.name)
        real = AutoModelForCausalLM.from_pretrained(tmp.name)
        dummy = AutoModelForCausalLM.from_pretrained(tmp.name, dummy=True, **dummy_kwargs)
        return real, dummy

    def test_same_structure_as_a_real_load(self):
        for config in (
            _tiny(LlamaConfig),
            _tiny(Qwen3MoeConfig, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32),
        ):
            real, dummy = self._pair(config)
            self.assertEqual(repr(real), repr(dummy))
            self.assertEqual(
                {n: (p.shape, p.dtype) for n, p in real.named_parameters()},
                {n: (p.shape, p.dtype) for n, p in dummy.named_parameters()},
            )
            self.assertEqual(real.num_parameters(), dummy.num_parameters())
            self.assertEqual({n for n, _ in real.named_buffers()}, {n for n, _ in dummy.named_buffers()})
            self.assertTrue(all(isinstance(p, ProceduralTensor) for p in dummy.parameters()))

    def test_deterministic_forward_and_generate(self):
        _, dummy = self._pair(_tiny(Qwen3MoeConfig, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32))
        ids = torch.randint(0, 300, (2, 7))
        with torch.no_grad():
            first, second = dummy(ids).logits, dummy(ids).logits
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(dummy.generate(ids, max_new_tokens=3, do_sample=False).shape, (2, 10))

    def test_seed(self):
        _, a = self._pair(_tiny(LlamaConfig))
        _, b = self._pair(_tiny(LlamaConfig), dummy_seed=69)
        wa, wb = a.model.layers[0].mlp.up_proj.weight, b.model.layers[0].mlp.up_proj.weight
        self.assertTrue(torch.equal(wa + 0, wa + 0))
        self.assertFalse(torch.equal(wa + 0, wb + 0))

    def test_slices_match_the_full_tensor(self):
        w = ProceduralTensor((1000, 96), torch.float32, seed=3)
        full = w + 0
        for index in (slice(500, 504), 500, slice(10, 40, 3), (slice(None), slice(5, 9)), (500, 7), slice(-3, None)):
            self.assertTrue(torch.equal(w[index] + 0, full[index]), index)
        self.assertTrue(torch.equal(w.t()[:, 7:9] + 0, full.t()[:, 7:9]))
        experts = ProceduralTensor((4, 32, 16), torch.float32, seed=5)
        self.assertTrue(torch.equal(experts[2] + 0, (experts + 0)[2]))

    def test_matmul_and_embedding_match_the_full_tensor(self):
        w = ProceduralTensor((1000, 96), torch.float32, seed=3)
        full = w + 0
        x = torch.randn(5, 96)
        torch.testing.assert_close(torch.nn.functional.linear(x, w), x @ full.T)
        ids = torch.tensor([[0, 999, 17, 500]])
        self.assertTrue(torch.equal(torch.nn.functional.embedding(ids, w), full[ids]))

    def test_read_only(self):
        _, dummy = self._pair(_tiny(LlamaConfig))
        self.assertFalse(any(p.requires_grad for p in dummy.parameters()))
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                dummy.model.layers[0].mlp.up_proj.weight.add_(1)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            dummy.save_pretrained(tmp)
