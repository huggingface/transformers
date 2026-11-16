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
"""Testing suite for the PyTorch ERNIE 5.5 MoE model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Ernie5_5_MoeConfig, Ernie5_5_MoeForCausalLM, Ernie5_5_MoeModel


class Ernie5_5_MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Ernie5_5_MoeModel

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.hidden_size = 64
        self.intermediate_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        # dense + HCA, then MoE + MLA/DSA
        self.num_hidden_layers = 2
        self.layer_types = ["heavily_compressed_attention", "indexed_attention"]
        self.first_k_dense_replace = 1
        # HCA
        self.head_dim = 32
        self.q_lora_rank = 16
        self.qk_rope_head_dim = 8
        self.o_groups = 2
        self.o_lora_rank = 16
        self.sliding_window = 4
        self.compress_rates = {"heavily_compressed_attention": 4}
        # The first (HCA) layer attends to the tokens plus one compressed entry per closed window
        self.key_length = self.seq_length + self.seq_length // 4
        # MLA + DSA: with `seq_length=7`, the last queries have more candidates older than the window than `index_topk`
        self.hybrid_mla_num_attention_heads = 4
        self.hybrid_mla_q_lora_rank = 16
        self.hybrid_mla_kv_lora_rank = 16
        self.hybrid_mla_qk_nope_head_dim = 8
        self.hybrid_mla_qk_rope_head_dim = 8
        self.hybrid_mla_v_head_dim = 16
        self.index_n_heads = 2
        self.index_head_dim = 16
        self.index_topk = 2
        self.vha_rank = 2
        # mHC
        self.hc_mult = 2
        self.hc_sinkhorn_iters = 3
        # Latent MoE
        self.moe_intermediate_size = 16
        self.moe_latent_size = 32
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.num_nextn_predict_layers = 0
        self.max_position_embeddings = 64


@require_torch
class Ernie5_5_MoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Ernie5_5_MoeModelTester

    # The lightning indexer only picks the top-k keys (non-differentiable); upstream trains it with a separate
    # objective, so its parameters get no gradient from the causal-LM loss
    test_all_params_have_gradient = False

    # No sequence classification / token classification / QA heads
    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    def test_hidden_states_output(self):
        # The decoder layers output the `hc_mult` residual streams, `[batch, seq, hc_mult, hidden]`; they are only
        # collapsed by `hc_head` at the top of the model
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        batch_size, seq_len = inputs_dict["input_ids"].shape
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            hidden_states = outputs.hidden_states
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
            for layer_hidden in hidden_states:
                if layer_hidden.ndim == 3:
                    self.assertEqual(layer_hidden.shape, (batch_size, seq_len, config.hidden_size))
                else:
                    self.assertEqual(layer_hidden.shape, (batch_size, seq_len, config.hc_mult, config.hidden_size))

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # Per-layer hidden states carry the extra `hc_mult` axis
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertEqual(layer_hidden.shape[0], batch_size)
                self.assertEqual(layer_hidden.shape[-1], config.hidden_size)

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_cache
    ):
        # The HCA layers attend to their sliding window plus the compressed entries, the MLA layers to every past
        # token: the key length differs per layer type
        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), output_length - prompt_length)
        for iter_attentions in attentions:
            self.assertIsInstance(iter_attentions, tuple)
            for layer_attention in iter_attentions:
                self.assertEqual(layer_attention.shape[0], batch_size)
                self.assertEqual(layer_attention.shape[1], config.num_attention_heads)

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        # The HCA layers keep a sliding window of shared K=V states; the MLA layers keep the compressed latent KV
        # (as keys) and the rotary keys (as values), which have different widths
        for layer, layer_type in zip(cache.layers, config.layer_types):
            self.assertEqual(layer.keys.shape[0], batch_size)
            self.assertEqual(layer.values.shape[0], batch_size)
            if layer_type == "heavily_compressed_attention":
                self.assertEqual(layer.keys.shape[-1], config.head_dim)
                self.assertLessEqual(layer.keys.shape[-2], config.sliding_window)
            else:
                self.assertEqual(layer.keys.shape[-1], config.hybrid_mla_kv_lora_rank)
                self.assertEqual(layer.values.shape[-1], config.hybrid_mla_qk_rope_head_dim)
                self.assertEqual(layer.keys.shape[-2], seq_length)

    @unittest.skip(
        reason="The HCA compressor pools fixed windows of `compress_rate` tokens before the attention mask is "
        "applied, so left padding shifts the window boundaries and folds pad tokens into the compressed entries "
        "(same as DeepSeek-V4)."
    )
    def test_left_padding_compatibility(self):
        pass

    def _make_model(self, **config_overrides):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config = Ernie5_5_MoeConfig.from_dict({**config.to_dict(), **config_overrides})
        torch.manual_seed(0)
        model = Ernie5_5_MoeForCausalLM(config).to(torch_device).eval()
        # The default init leaves the sinks, the head mix and MultiMax at their neutral values: perturb everything
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.05)
        return model

    def test_incremental_decoding_matches_full_forward(self):
        # Long enough to close several compression windows and to make the indexer pick among older keys; the
        # third layer covers HCA after a MoE block
        hca, dsa = "heavily_compressed_attention", "indexed_attention"
        model = self._make_model(num_hidden_layers=3, layer_types=[hca, dsa, hca])
        input_ids = torch.randint(0, model.config.vocab_size, (2, 30), device=torch_device)
        with torch.no_grad():
            full_logits = model(input_ids).logits
            prefix = 11  # two closed windows plus three buffered tokens
            outputs = model(input_ids[:, :prefix], use_cache=True)
            step_logits = [outputs.logits[:, -1]]
            for position in range(prefix, input_ids.shape[1] - 1):
                outputs = model(input_ids[:, position : position + 1], cache=outputs.cache, use_cache=True)
                step_logits.append(outputs.logits[:, -1])
        torch.testing.assert_close(torch.stack(step_logits, dim=1), full_logits[:, prefix - 1 : -1])

    def test_indexed_attention_is_dense_when_topk_covers_the_sequence(self):
        # When the indexer keeps every older key, MLA + DSA is plain causal attention and the sliding window
        # (which only decides what the indexer has to score) no longer matters
        overrides = {"layer_types": ["indexed_attention"] * 2, "index_topk": 64}
        input_ids = torch.randint(0, 99, (2, 20), device=torch_device)
        with torch.no_grad():
            narrow = self._make_model(sliding_window=2, **overrides)(input_ids).logits
            wide = self._make_model(sliding_window=64, **overrides)(input_ids).logits
        torch.testing.assert_close(narrow, wide)

    def test_config_from_released_format(self):
        # The released `config.json` describes the schedule with per-layer `compress_ratios` (plus one entry for the
        # ECHO layer) and the HCA YaRN with a legacy `rope_scaling` dict
        config = Ernie5_5_MoeConfig(
            num_hidden_layers=8,
            compress_ratios=[128] * 7 + [-2] + [128],
            hca_rope_type="yarn",
            rope_theta=8000000,
            compress_rope_theta=160000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )
        self.assertEqual(config.layer_types, ["heavily_compressed_attention"] * 7 + ["indexed_attention"])
        self.assertEqual(config.compress_rates, {"heavily_compressed_attention": 128})
        self.assertEqual(config.rope_parameters["main"]["rope_type"], "default")
        self.assertEqual(config.rope_parameters["main"]["rope_theta"], 8000000)
        compress = config.rope_parameters["compress"]
        self.assertEqual(compress["rope_type"], "yarn")
        self.assertEqual(compress["rope_theta"], 160000.0)
        self.assertEqual(compress["factor"], 16)
        self.assertEqual(compress["attention_factor"], 1.0)
        self.assertNotIn("type", compress)
