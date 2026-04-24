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
"""Testing suite for the PyTorch DeepseekV4 model."""

import unittest

from transformers import DeepseekV4Config, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import DeepseekV4ForCausalLM, DeepseekV4Model


_TINY_DEFAULTS = {
    "vocab_size": 99,
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "head_dim": 32,
    "qk_rope_head_dim": 8,
    "q_lora_rank": 32,
    "o_groups": 2,
    "o_lora_rank": 16,
    "moe_intermediate_size": 64,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "num_hash_layers": 1,
    "compress_ratios": [0, 4, 128, 0],
    "sliding_window": 8,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 3,
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 2,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 64,
}


def _tiny_config(**overrides):
    return DeepseekV4Config(**{**_TINY_DEFAULTS, **overrides})


@require_torch
class DeepseekV4ModelTest(unittest.TestCase):
    """Minimal smoke tests exercising the new architectural pieces.

    Not a ``CausalLMModelTest`` subclass — V4 attention has a per-head learnable sink,
    custom eager attention, and hyper-connections that diverge from the Llama shape
    assumptions that ``test_modeling_common`` hard-codes. Full coverage lands once the
    sink-aware attention backends (SDPA / flash) are wired up.
    """

    def test_forward_shapes(self):
        torch.manual_seed(0)
        cfg = _tiny_config()
        model = DeepseekV4ForCausalLM(cfg).eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 20))
        with torch.no_grad():
            out = model(input_ids)
        self.assertEqual(out.logits.shape, (2, 20, cfg.vocab_size))

    def test_layer_composition(self):
        cfg = _tiny_config()
        model = DeepseekV4Model(cfg)
        # Layer 0: pure SWA, no compressor, no indexer.
        self.assertIsNone(model.layers[0].self_attn.compressor)
        self.assertIsNone(model.layers[0].self_attn.indexer)
        # Layer 1: compress_ratio=4 → compressor + indexer present.
        self.assertIsNotNone(model.layers[1].self_attn.compressor)
        self.assertIsNotNone(model.layers[1].self_attn.indexer)
        # Layer 2: compress_ratio=128 → compressor only.
        self.assertIsNotNone(model.layers[2].self_attn.compressor)
        self.assertIsNone(model.layers[2].self_attn.indexer)

    def test_hash_routing_present_on_first_layers(self):
        cfg = _tiny_config(num_hash_layers=2)
        model = DeepseekV4Model(cfg)
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter, DeepseekV4TopKRouter

        self.assertIsInstance(model.layers[0].mlp.gate, DeepseekV4HashRouter)
        self.assertIsInstance(model.layers[1].mlp.gate, DeepseekV4HashRouter)
        self.assertIsInstance(model.layers[2].mlp.gate, DeepseekV4TopKRouter)

    def test_hyper_connections_shape(self):
        cfg = _tiny_config()
        model = DeepseekV4Model(cfg).eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 12))
        with torch.no_grad():
            out = model(input_ids)
        # The hc_head + final RMSNorm collapse the hc_mult streams inside the Model,
        # matching the Llama / Mixtral `Model(...) → [B, S, hidden] → lm_head` contract.
        self.assertEqual(out.last_hidden_state.shape, (1, 12, cfg.hidden_size))


if __name__ == "__main__":
    unittest.main()
