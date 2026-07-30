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

import torch

from transformers import DynamicCache
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
from transformers.models.hy_v4.modeling_hy_v4 import HYV4ForCausalLM, HYV4Model, HYV4MoE


class HYV4ModelTest(unittest.TestCase):
    def tiny_config(self):
        return HYV4Config(
            vocab_size=99,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            mlp_layer_types=["dense", "dense"],
            index_topk=8,
            index_head_dim=8,
            index_n_heads=4,
            indexer_types=["full", "full"],
            enable_lm_head_fp32=False,
            enable_ihc=False,
            gated_mla=False,
            learnable_sink=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

    def test_base_model_forward(self):
        model = HYV4Model(self.tiny_config()).eval()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=False)

        self.assertEqual(output.last_hidden_state.shape, (2, 5, model.config.hidden_size))

    def test_causal_lm_forward_and_loss(self):
        model = HYV4ForCausalLM(self.tiny_config()).eval()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

        with torch.no_grad():
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)

        self.assertEqual(output.logits.shape, (2, 5, model.config.vocab_size))
        self.assertEqual(output.loss.ndim, 0)

    def test_prefill_is_causal_with_default_attention_backend(self):
        config = self.tiny_config()
        config.indexer_types = ["full", "shared"]
        model = HYV4ForCausalLM(config).eval()
        prefix = torch.tensor([[1, 2, 3]])
        extended = torch.tensor([[1, 2, 3, 4]])

        with torch.no_grad():
            prefix_logits = model(input_ids=prefix, use_cache=False).logits
            extended_logits = model(input_ids=extended, use_cache=False).logits[:, : prefix.shape[1]]

        self.assertTrue(torch.allclose(prefix_logits, extended_logits, atol=1e-5, rtol=1e-5))

    def test_lm_head_fp32_contract(self):
        config = self.tiny_config()
        config.enable_lm_head_fp32 = True
        model = HYV4ForCausalLM(config).to(torch.bfloat16).eval()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
        self.assertEqual(logits.dtype, torch.float32)

    def test_cache_tracks_attention_and_indexer_state(self):
        config = self.tiny_config()
        config.indexer_types = ["full", "shared"]
        model = HYV4Model(config).eval()
        cache = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=True)
        # main K/V cache and the DSA indexer-key cache both advance to the prefill length
        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(cache.layers[0].indexer_keys.shape[1], 3)

    def test_greedy_generate_uses_hyv4_cache(self):
        model = HYV4ForCausalLM(self.tiny_config()).eval()
        with torch.no_grad():
            generated = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False)

        self.assertEqual(generated.shape, (1, 5))

    def test_save_and_reload(self):
        model = HYV4ForCausalLM(self.tiny_config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            expected_logits = model(input_ids=input_ids, use_cache=False).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = HYV4ForCausalLM.from_pretrained(directory).eval()
        with torch.no_grad():
            restored_logits = restored(input_ids=input_ids, use_cache=False).logits

        expected_config = model.config.to_dict()
        restored_config = restored.config.to_dict()
        expected_config.pop("_name_or_path", None)
        restored_config.pop("_name_or_path", None)
        self.assertEqual(restored_config, expected_config)
        self.assertEqual(restored.state_dict().keys(), model.state_dict().keys())
        self.assertTrue(torch.equal(restored_logits, expected_logits))

    def test_moe_uses_live_expert_parameters(self):
        config = self.tiny_config()
        config.mlp_layer_types = ["dense", "sparse"]
        config.n_routed_experts = 4
        config.num_experts_per_tok = 2
        moe = HYV4MoE(config)
        with torch.no_grad():
            for parameter in moe.parameters():
                parameter.normal_(mean=0.0, std=0.02)
        optimizer = torch.optim.SGD(moe.parameters(), lr=0.1)
        hidden_states = torch.randn(2, 3, config.hidden_size)

        before = moe.experts.gate_up_proj.detach().clone()
        loss = moe(hidden_states).square().mean()
        loss.backward()
        self.assertIsNotNone(moe.experts.gate_up_proj.grad)
        self.assertGreater(moe.experts.gate_up_proj.grad.abs().sum().item(), 0)
        optimizer.step()

        self.assertFalse(torch.equal(before, moe.experts.gate_up_proj))
        self.assertFalse(hasattr(moe.experts, "_unpacked_cache_ready"))

    def test_ihc_forward_and_parameter_layout(self):
        config = self.tiny_config()
        config.enable_ihc = True
        model = HYV4Model(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))

        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=False)

        self.assertEqual(output.last_hidden_state.shape, (2, 5, config.hidden_size))
        state_dict = model.state_dict()
        self.assertIn("layers.0.hc_attn_layer.hc_pre.hc_fn", state_dict)
        self.assertIn("layers.0.hc_mlp_layer.hc_pre.hc_fn", state_dict)
        self.assertIn("hc_head.hc_head_fn", state_dict)
        self.assertEqual(state_dict["layers.0.hc_attn_layer.hc_pre.hc_fn"].dtype, torch.float32)
        self.assertEqual(state_dict["hc_head.hc_head_fn"].dtype, torch.float32)

    def test_gated_mla_and_sink_parameter_layout(self):
        config = self.tiny_config()
        config.gated_mla = True
        config.learnable_sink = True
        model = HYV4Model(config).eval()
        attention = model.layers[0].self_attn

        self.assertEqual(
            attention.linear_gate.weight.shape, (config.num_attention_heads * config.v_head_dim, config.hidden_size)
        )
        self.assertEqual(attention.learnable_sink_param.shape, (config.num_attention_heads,))
        self.assertEqual(attention.learnable_sink_param.dtype, torch.float32)
        with torch.no_grad():
            output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_dsa_batch_and_shared_indexer(self):
        config = self.tiny_config()
        config.indexer_types = ["full", "shared"]
        model = HYV4Model(config).eval()

        with torch.no_grad():
            output = model(input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]), use_cache=False)

        self.assertEqual(output.last_hidden_state.shape, (2, 3, config.hidden_size))
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
        self.assertTrue(hasattr(model.layers[0].self_attn, "indexer"))
        self.assertFalse(hasattr(model.layers[1].self_attn, "indexer"))

    def test_dsa_sentinel_does_not_select_last_token(self):
        config = self.tiny_config()
        config.indexer_types = ["full", "shared"]
        attention = HYV4Model(config).layers[0].self_attn
        sparse_mask = attention._build_sparse_mask(
            torch.tensor([[[-1, 0]]]), attention_mask=None, key_length=3, dtype=torch.float32
        )

        self.assertEqual(sparse_mask[0, 0, 0, 0].item(), 0.0)
        self.assertLess(sparse_mask[0, 0, 0, 2].item(), -1e30)

    def test_dsa_cache_prefill_decode(self):
        config = self.tiny_config()
        config.indexer_types = ["full", "shared"]
        model = HYV4Model(config).eval()

        with torch.no_grad():
            prefill = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
            decoded = model(input_ids=torch.tensor([[4]]), past_key_values=prefill.past_key_values, use_cache=True)

        self.assertEqual(prefill.past_key_values.get_seq_length(), 4)
        self.assertEqual(prefill.past_key_values.layers[0].indexer_keys.shape[1], 4)
        self.assertEqual(decoded.last_hidden_state.shape, (1, 1, config.hidden_size))
        self.assertTrue(torch.isfinite(decoded.last_hidden_state).all())


if __name__ == "__main__":
    unittest.main()
