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
"""Testing suite for the PyTorch HYV4 model."""

import unittest

import torch
from parameterized import parameterized

from transformers import HYV4Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import HYV4ForCausalLM, HYV4Model


class HYV4ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HYV4Model
        causal_lm_class = HYV4ForCausalLM

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        moe_intermediate_size=16,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        num_hidden_layers=2,
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "shared"],
        index_topk=8,
        index_head_dim=8,
        index_n_heads=4,
        hc_mult=4,
    ):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers)
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mlp_layer_types = mlp_layer_types
        self.indexer_types = indexer_types
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.hc_mult = hc_mult


@require_torch
class HYV4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HYV4ModelTester
    # HYV4 routes each token to a subset of experts, so not every expert receives a gradient.
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.7, 0.8]

    def test_default_layer_types(self):
        config = HYV4Config(num_hidden_layers=8)
        self.assertEqual(config.mlp_layer_types, ["dense"] + ["sparse"] * 7)
        self.assertEqual(config.layer_types, ["deepseek_sparse_attention"] * 8)
        self.assertEqual(
            config.indexer_types,
            ["full", "full", "shared", "shared", "shared", "full", "shared", "shared"],
        )

    def test_lm_head_is_kept_in_fp32(self):
        self.assertIn("lm_head", HYV4ForCausalLM._keep_in_fp32_modules_strict)

    def test_mtp_weights_are_ignored_on_load(self):
        # Other runtimes (e.g. vLLM) keep the MTP tensors in the shared checkpoint; Transformers ignores them.
        self.assertEqual(HYV4ForCausalLM._keys_to_ignore_on_load_unexpected, [r"model\.mtp_layers\..*"])

    def test_ihc_hidden_state_and_parameters_are_fp32(self):
        config = self.model_tester.get_config()
        model = HYV4Model(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=False)

        self.assertEqual(output.last_hidden_state.shape, (2, 5, config.hidden_size))
        state_dict = model.state_dict()
        self.assertIn("layers.0.hc_attn_layer.hc_fn", state_dict)
        self.assertIn("hc_head.hc_head_fn", state_dict)
        self.assertEqual(state_dict["layers.0.hc_attn_layer.hc_fn"].dtype, torch.float32)
        self.assertEqual(state_dict["hc_head.hc_head_fn"].dtype, torch.float32)

    def test_sink_parameter_layout(self):
        config = self.model_tester.get_config()
        attention = HYV4Model(config).eval().layers[0].self_attn
        self.assertEqual(attention.learnable_sink_param.shape, (config.num_attention_heads,))
        self.assertEqual(attention.learnable_sink_param.dtype, torch.float32)

    def test_full_and_shared_indexer_layers(self):
        config = self.model_tester.get_config()
        model = HYV4Model(config).eval()
        self.assertIsNotNone(model.layers[0].self_attn.indexer)
        self.assertIsNone(model.layers[1].self_attn.indexer)

    def test_shared_indexer_requires_prior_full_indices(self):
        config = self.model_tester.get_config()
        config.indexer_types = ["shared", "full"]
        model = HYV4Model(config).eval()
        with self.assertRaisesRegex(ValueError, "Shared DSA layers require top-k indices"):
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)

    def test_dsa_sentinel_does_not_select_last_token(self):
        config = self.model_tester.get_config()
        attention = HYV4Model(config).eval().layers[0].self_attn
        sparse_mask = attention._build_sparse_mask(
            torch.tensor([[[-1, 0]]]), attention_mask=None, key_length=3, dtype=torch.float32
        )
        self.assertEqual(sparse_mask[0, 0, 0, 0].item(), 0.0)
        self.assertLess(sparse_mask[0, 0, 0, 2].item(), -1e30)

    def test_hidden_states_output(self):
        # HYV4 decoder layers carry a 4D `[batch, seq, hc_mult, hidden]` iHC stream; the
        # hc_mult streams are only collapsed at the top of the model via `hc_head`. The
        # common test assumes `(batch, seq, hidden)`, so accept the extra HC axis for the
        # per-layer states while still requiring the final state to be the collapsed 3D shape.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[-1]
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
            batch_size, seq_len = inputs_dict["input_ids"].shape[:2]
            for layer_h in hidden_states:
                if layer_h.ndim == 3:
                    self.assertEqual(layer_h.shape, (batch_size, seq_len, config.hidden_size))
                else:
                    self.assertEqual(layer_h.shape, (batch_size, seq_len, config.hc_mult, config.hidden_size))

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # HYV4's per-layer hidden states carry the extra `hc_mult` iHC stream axis, so the
        # base tester's exact `(batch, seq, hidden)` assertion does not hold; sanity-check
        # the batch and hidden dims instead.
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))
        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                self.assertEqual(layer_hidden.shape[0], batch_size)
                self.assertEqual(layer_hidden.shape[-1], config.hidden_size)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # HYV4 uses MLA, so the cached key/value head_dim differs from the parent tester's
        # assumption. Check the batch / head / head_dim invariants and skip the exact
        # seq-length axis (the DSA indexer cache advances separately).
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        for layer in past_key_values.layers:
            keys, values = layer.keys, layer.values
            self.assertIsInstance(keys, torch.Tensor)
            self.assertEqual(keys.shape[0], batch_size)
            self.assertEqual(keys.shape[1], num_kv_heads)
            self.assertEqual(keys.shape[3], config.qk_nope_head_dim + config.qk_rope_head_dim)
            self.assertEqual(values.shape[3], config.v_head_dim)

    # HYV4 has no default dynamic cache (`_supports_default_dynamic_cache` is False), and
    # assisted decoding requires a cache, so these do not apply.
    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("HYV4 does not provide a default dynamic cache required by assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("HYV4 does not provide a default dynamic cache required by assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass


if __name__ == "__main__":
    unittest.main()
