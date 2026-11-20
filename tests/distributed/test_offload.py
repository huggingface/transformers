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

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.testing_utils import require_torch
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
            self.model.save_pretrained(path)  # per-expert checkpoint keys: replays the conversion mapping
            for slots in (1, 5, 24):
                for lookahead in (0, 1, 2):
                    with self.subTest(slots=slots, lookahead=lookahead):
                        model, store = self.load(path, slots)
                        store.lookahead = lookahead
                        self.assertEqual(sum(p.num_slots for p in store.pools.values()), slots)
                        torch.testing.assert_close(self.generate(model), self.expected)
                        self.assertGreater(store.reads, 0)

    def test_lfu_matches_eager(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            model, store = self.load(path, 5)
            for pool in store.pools.values():
                pool.policy = "lfu"
            torch.testing.assert_close(self.generate(model), self.expected)
            self.assertTrue(any(pool.counts for pool in store.pools.values()))

    def test_stacked_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.config.save_pretrained(path)
            save_file({k: v.contiguous() for k, v in self.model.state_dict().items()}, f"{path}/model.safetensors")
            model, store = self.load(path, 5)
            torch.testing.assert_close(self.generate(model), self.expected)
            self.assertGreater(store.hits, 0)
