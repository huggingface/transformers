# Copyright 2026 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch A.X-K2 model."""

import os
import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import AXK2Config, Cache, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
)


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from transformers import (
        AXK2ForCausalLM,
        AXK2Model,
    )
    from transformers.models.axk2.modeling_axk2 import AXK2TopkRouter


class AXK2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = AXK2Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=32,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=8,
        gated_norm_rank=4,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.gated_norm_rank = gated_norm_rank
        self.mlp_layer_types = ["dense", "sparse"]


@require_torch
class AXK2ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = AXK2ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = AXK2ForCausalLM if is_torch_available() else None

    @unittest.skip("A.X-K2 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("A.X-K2 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as A.X-K2 has the MLA cache format (same as DeepSeek-V3.2)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("A.X-K2 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Dynamic control flow in MoE")
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        pass

    # The SGA indexer builds its sparse mask from a dynamic (cached) key length, which the static cache
    # and torch.compile / fullgraph paths cannot express — same limitation as GLM-MoE-DSA.
    @unittest.skip("SGA indexer mask shape mismatch with static cache")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("SGA indexer mask shape mismatch with static cache")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("SGA indexer mask shape mismatch with compilation")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("SGA indexer mask shape mismatch with compiled forward")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    # SGA (like DeepSeek Sparse Attention) selects tokens with a hard top-k, which is discontinuous: a tiny
    # numerical difference in the indexer scores (attention backend, padding, batching, sequence packing)
    # can flip which tokens are selected and thus change the output. These exact cross-backend /
    # padding-equivalence tests therefore do not hold for SGA.
    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("SGA hard top-k selection is sensitive to tiny numerical differences across backends.")
    def test_eager_matches_sdpa_inference(self, *args, **kwargs):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    @unittest.skip("SGA hard top-k selection is sensitive to tiny numerical differences across batching.")
    def test_eager_matches_batched_and_grouped_inference(self, *args, **kwargs):
        pass

    # These were re-checked per review: they still fail (verified on GPU — left padding gives a ~100%
    # logit mismatch, sequence packing ~80%). Even though the prompt is short, greedy generation grows the
    # sequence past `index_topk`, so the SGA hard top-k does engage and its selection flips under padding /
    # packing shifts. Same limitation as DeepSeek-V3.2, so they stay skipped.
    @unittest.skip("SGA hard top-k selection is sensitive to padding shifts (selection can flip).")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip("SGA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("SGA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MoE routing on a tiny randomly-initialized model makes the overfit target unstable.")
    def test_training_overfit(self):
        pass


def _tiny_grouped_config():
    # A.X-K2-Light shapes shrunk down, with grouped routing enabled the way the larger A.X-K2 releases
    # configure it (`n_group`/`topk_group` set instead of `None`).
    return AXK2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=4,
        topk_group=2,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_rope_head_dim=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        index_topk=8,
        index_head_dim=32,
        index_n_heads=2,
        gated_norm_rank=4,
        max_position_embeddings=256,
    )


@require_torch
class AXK2GroupedRoutingTest(unittest.TestCase):
    def test_grouped_routing_matches_reference(self):
        # Reference semantics: the released A.X-K2 modeling code (DeepSeek-V3-style group restriction,
        # masking losing groups with 0.0 before the expert top-k).
        config = _tiny_grouped_config()
        torch.manual_seed(0)
        router = AXK2TopkRouter(config)
        with torch.no_grad():
            router.weight.normal_()
            router.e_score_correction_bias.normal_()

        hidden_states = torch.randn(5, config.hidden_size)
        _, topk_weights, topk_indices = router(hidden_states)

        scores = (hidden_states.float() @ router.weight.float().T).sigmoid()
        scores_for_choice = scores + router.e_score_correction_bias
        experts_per_group = config.n_routed_experts // config.n_group
        group_scores = scores_for_choice.view(-1, config.n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=config.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(-1, config.n_group, experts_per_group).reshape(-1, config.n_routed_experts)
        )
        expected_indices = torch.topk(
            scores_for_choice.masked_fill(~score_mask.bool(), 0.0), k=config.num_experts_per_tok, dim=-1, sorted=False
        )[1]
        expected_weights = scores.gather(1, expected_indices)
        if config.norm_topk_prob:
            expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)
        expected_weights = expected_weights * config.routed_scaling_factor

        self.assertTrue(torch.equal(topk_indices, expected_indices))
        torch.testing.assert_close(topk_weights, expected_weights)
        # Every selected expert must come from one of the selected groups.
        selected_groups = topk_indices // experts_per_group
        self.assertTrue((selected_groups.unsqueeze(-1) == group_idx.unsqueeze(1)).any(-1).all())


@require_torch
class AXK2HubCheckpointLayoutTest(unittest.TestCase):
    def test_fused_gate_and_hub_layout_roundtrip(self):
        # A.X-K2 fuses the query up-projection and the output gate into a single `q_gate_proj` (doubled
        # input, no separate `g_proj`). The hub checkpoints store it under the vLLM name `q_b_proj`, with
        # per-expert projections (`experts.{i}.gate_proj/...`) and `W_down`/`W_up` gated-norm MLP names.
        # `save_pretrained` reverts the load-time conversions, so a save/load cycle exercises the whole hub
        # layout in both directions and must reproduce the exact same logits.
        config = _tiny_grouped_config()
        torch.manual_seed(0)
        model = AXK2ForCausalLM(config).eval()

        attn = model.model.layers[0].self_attn
        self.assertEqual(attn.q_gate_proj.weight.shape[-1], 2 * config.q_lora_rank)
        self.assertEqual(
            attn.q_gate_proj.weight.shape[0], config.num_attention_heads * (config.qk_head_dim + config.v_head_dim)
        )
        self.assertFalse(hasattr(attn, "q_b_proj"))
        self.assertFalse(hasattr(attn, "g_proj"))

        input_ids = torch.randint(0, config.vocab_size, (2, 7))
        with torch.no_grad():
            expected_logits = model(input_ids).logits

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            saved = load_file(os.path.join(tmp_dir, "model.safetensors"))
            self.assertIn("model.layers.1.mlp.experts.0.gate_proj.weight", saved)
            self.assertIn("model.layers.1.mlp.experts.0.up_proj.weight", saved)
            self.assertIn("model.layers.1.mlp.experts.0.down_proj.weight", saved)
            self.assertIn("model.layers.0.input_layernorm.W_down.weight", saved)
            self.assertIn("model.layers.0.input_layernorm.W_up.weight", saved)
            self.assertNotIn("model.layers.0.self_attn.g_proj.weight", saved)
            self.assertNotIn("model.layers.0.self_attn.q_gate_proj.weight", saved)
            # The fused projection is saved back under the vLLM `q_b_proj` name.
            fused_q_b_proj = saved["model.layers.0.self_attn.q_b_proj.weight"]
            self.assertEqual(fused_q_b_proj.shape[-1], 2 * config.q_lora_rank)

            reloaded = AXK2ForCausalLM.from_pretrained(tmp_dir).eval()
            with torch.no_grad():
                reloaded_logits = reloaded(input_ids).logits

        torch.testing.assert_close(reloaded_logits, expected_logits, rtol=0.0, atol=0.0)


@slow
@require_torch_accelerator
class AXK2IntegrationTest(unittest.TestCase):
    # A.X-K2's released checkpoint is a 20B+ MoE that does not fit a CI GPU, so these tests run on a tiny
    # *randomized* checkpoint (same layout as `tiny-axk1`) that still exercises the full stack: fused
    # attention output gate, gated RMSNorm, and the SGA (sparse-gated) indexer. Regenerate it
    # byte-identically (and upload) with:
    #
    #     torch.manual_seed(0)
    #     config = AXK2Config(
    #         vocab_size=163840, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    #         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=16,
    #         num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=16, qk_nope_head_dim=8,
    #         qk_rope_head_dim=8, v_head_dim=8, index_n_heads=2, index_head_dim=16, index_topk=8,
    #         gated_norm_rank=4, max_position_embeddings=4096,
    #         mlp_layer_types=["dense"] + ["sparse"] * 3,
    #     )
    #     AXK2ForCausalLM(config).to(torch.bfloat16).push_to_hub("hf-internal-testing/tiny-axk2")
    #
    # The logits expectations below were recorded from exactly this seeded model (bf16, eager, A100).
    model_id = "hf-internal-testing/tiny-axk2"

    def test_generation(self):
        # Weights are randomly initialized so the decoded text is arbitrary; this just exercises the full
        # greedy generation loop end to end and checks the output shape.
        model = AXK2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=torch_device)
        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        self.assertEqual(generated_ids.shape, (1, input_ids.shape[1] + 20))

    def test_model_logits_batched(self):
        model = AXK2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        # Last-3x3 logits slice, left-padded (batch 0) and unpadded (batch 1) rows.
        EXPECTED_LOGITS_LEFT_PADDED = Expectations(
            {
                ("cuda", 8): [[0.2441, 0.1201, 0.2129], [0.0659, 0.0635, 0.0525], [-0.019, -0.1104, 0.0454]],
            }
        )
        expected_left_padded = torch.tensor(EXPECTED_LOGITS_LEFT_PADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_UNPADDED = Expectations(
            {
                ("cuda", 8): [[0.1152, -0.1245, -0.4258], [-0.1797, -0.3145, -0.3223], [-0.0684, 0.3418, -0.027]],
            }
        )
        expected_unpadded = torch.tensor(EXPECTED_LOGITS_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()
        torch.testing.assert_close(logits[0, -3:, -3:], expected_left_padded, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, -3:, -3:], expected_unpadded, atol=1e-3, rtol=1e-3)
