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
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_triton,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester, torch_device


if is_torch_available():
    import torch

    from transformers import (
        MiMoV2FlashConfig,
        MiMoV2FlashForCausalLM,
        MiMoV2FlashModel,
    )
    from transformers.conversion_mapping import get_model_conversion_mapping


class MiMoV2FlashModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiMoV2FlashModel

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        hidden_act="silu",
        max_position_embeddings=64,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        super().__init__(
            parent=parent,
            batch_size=batch_size,
            seq_length=seq_length,
            is_training=is_training,
            use_input_mask=use_input_mask,
            use_labels=use_labels,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        # MiMo-V2-Flash specific test config
        self.head_dim = 8
        self.v_head_dim = 8
        self.layer_types = ["full_attention", "sliding_attention"]
        self.rope_parameters = {
            "full_attention": {"rope_type": "default", "rope_theta": 5_000_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5},
        }
        # 2 layers: [dense, moe]
        self.moe_layer_freq = [0, 1]
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 16
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.sliding_window = 64
        self.rms_norm_eps = 1e-5
        self.attention_bias = False
        self.attention_dropout = 0.0


# NOTE @casinca: some of these tests are re-used from GPT-OSS
# For MiMO, since we decouple sink and non sink layers for backends, these 2 tests are not skipped (unlike GPT-OSS):
# - test_sdpa_can_dispatch_non_composite_models
# - test_eager_matches_sdpa_inference
@require_torch
class MiMoV2FlashModelTest(CausalLMModelTest, unittest.TestCase):
    _is_stateful = True
    model_tester_class = MiMoV2FlashModelTester

    @unittest.skip(
        "Most probably because of the MoE, the MoE and router do not treat padded vs packed sequences like dense models"
    )
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "Most probably because of the MoE, the MoE and router do not treat padded vs packed sequences like dense models"
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @require_triton
    def test_flex_attention_with_grads(self):
        super().test_flex_attention_with_grads()

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        super().test_reverse_loading_mapping(check_keys_were_modified=check_keys_were_modified)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # SWA layers double the kv heads (see MiMoV2FlashAttention.__init__), so the per-layer
        # kv head count is layer-type dependent. Same override pattern as MiniMax.
        for layer_idx, layer in enumerate(past_key_values.layers):
            is_swa = config.layer_types[layer_idx] == "sliding_attention"
            num_kv_heads = config.num_key_value_heads * 2 if is_swa else config.num_key_value_heads
            expected_shape = (batch_size, num_kv_heads, seq_length, config.head_dim)
            self.assertEqual(layer.keys.shape, expected_shape)
            self.assertEqual(layer.values.shape, expected_shape)

    # NOTE: @casinca this is copy pasta tests from Gemma3, useful for MiMo RoPE
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("MiMo uses per-layer-type nested rope_parameters, not compatible with shared scaling config test")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Retrieves the RoPE layer class from the base model class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        partial_rotary_factor = 0.5  # from test config
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # MiMo uses per-layer-type nested rope_parameters and takes layer_type at __init__ (not forward)
        def _make_rope_params(**extra):
            params = {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": partial_rotary_factor}
            params.update(extra)
            return params

        # Sanity check original RoPE
        rope_params = _make_rope_params()
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="full_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        rope_params = _make_rope_params(rope_type="linear", factor=scaling_factor)
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="full_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        rope_params = _make_rope_params(rope_type="dynamic", factor=scaling_factor)
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="full_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        inv_freq_attr = "full_attention_inv_freq"
        self.assertTrue((getattr(ntk_scaling_rope, inv_freq_attr) <= getattr(original_rope, inv_freq_attr)).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        rope_params = _make_rope_params(rope_type="yarn", factor=scaling_factor)
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="full_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    def test_hub_config_backward_compatibility(self):
        """Hub config.json uses legacy field names and extra keys that must be normalized or stripped."""
        config = MiMoV2FlashConfig.from_dict(
            {
                "model_type": "mimo_v2_flash",
                "num_hidden_layers": 2,
                "vocab_size": 100,
                # Legacy fields that should be converted
                "hybrid_layer_pattern": [0, 1],
                "partial_rotary_factor": 0.5,
                "swa_rope_theta": 10000,
                # Hub-only fields that should be stripped
                "scoring_func": "sigmoid",
                "topk_method": "noaux_tc",
                "attention_value_scale": 0.707,
                "attention_chunk_size": 128,
                "sliding_window_size": 128,
                "n_shared_experts": None,
                # attribute_map alias (hub uses head_dim directly, same as native)
                "head_dim": 192,
                # Legacy SWA-prefixed fields that should be stripped (redundant with non-SWA counterparts)
                "swa_num_attention_heads": 64,
                "swa_num_key_value_heads": 8,
                "swa_head_dim": 192,
                "swa_v_head_dim": 128,
                # None -> default
                "routed_scaling_factor": None,
            }
        )
        config_dict = config.to_dict()

        # hybrid_layer_pattern -> layer_types
        self.assertEqual(config.layer_types, ["full_attention", "sliding_attention"])
        self.assertNotIn("hybrid_layer_pattern", config_dict)

        # partial_rotary_factor / swa_rope_theta -> rope_parameters
        self.assertEqual(config_dict["rope_parameters"]["full_attention"]["partial_rotary_factor"], 0.5)
        self.assertEqual(config_dict["rope_parameters"]["sliding_attention"]["rope_theta"], 10000)
        self.assertNotIn("partial_rotary_factor", config_dict)
        self.assertNotIn("swa_rope_theta", config_dict)

        # Hub-only fields stripped
        for key in (
            "scoring_func",
            "topk_method",
            "attention_value_scale",
            "attention_chunk_size",
            "sliding_window_size",
            "n_shared_experts",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_head_dim",
            "swa_v_head_dim",
        ):
            self.assertNotIn(key, config_dict)

        self.assertEqual(config.head_dim, 192)

        # None -> default
        self.assertEqual(config.routed_scaling_factor, 1.0)

    def test_layer_type_rope_parameters_keep_rotary_dims_in_sync(self):
        """Layer-specific rope parameters should produce position embeddings that match each attention rotary dim."""
        config = MiMoV2FlashConfig(
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            vocab_size=100,
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        config.rope_parameters["full_attention"]["partial_rotary_factor"] = 0.5
        config.rope_parameters["sliding_attention"]["partial_rotary_factor"] = 0.5

        model = MiMoV2FlashModel(config)
        input_ids = torch.tensor([[1, 2, 3]])
        hidden_states = model.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

        full_cos, _ = model.rotary_emb(hidden_states, position_ids, layer_type="full_attention")
        swa_cos, _ = model.rotary_emb(hidden_states, position_ids, layer_type="sliding_attention")

        self.assertEqual(full_cos.shape[-1], 4)
        self.assertEqual(swa_cos.shape[-1], 4)
        self.assertEqual(model(input_ids=input_ids).last_hidden_state.shape, (1, 3, 32))

    def test_moe_fused_expert_shapes(self):
        """Fused MixtralExperts layout: stacked gate_up and down per expert index."""
        config = MiMoV2FlashConfig(
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            vocab_size=100,
            hidden_size=32,
            moe_intermediate_size=16,
            n_routed_experts=4,
            moe_layer_freq=[0, 1],
        )
        model = MiMoV2FlashModel(config)
        experts = model.layers[1].mlp.experts
        self.assertEqual(experts.gate_up_proj.shape, (4, 32, 32))  # (E, 2*intermediate, hidden)
        self.assertEqual(experts.down_proj.shape, (4, 32, 16))  # (E, hidden, intermediate)

    def test_moe_legacy_conversion_mapping_registered(self):
        config = MiMoV2FlashConfig(
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            vocab_size=100,
            hidden_size=32,
            moe_intermediate_size=16,
            n_routed_experts=4,
            moe_layer_freq=[0, 1],
        )
        model = MiMoV2FlashModel(config)
        weight_mapping = get_model_conversion_mapping(model)

        found_gate_up_converter = any(
            "mlp.experts.*.gate_proj.weight" in mapping.source_patterns
            and "mlp.experts.gate_up_proj" in mapping.target_patterns
            for mapping in weight_mapping
        )
        found_down_converter = any(
            "mlp.experts.*.down_proj.weight" in mapping.source_patterns
            and "mlp.experts.down_proj" in mapping.target_patterns
            for mapping in weight_mapping
        )

        self.assertTrue(found_gate_up_converter)
        self.assertTrue(found_down_converter)

    # NOTE: @casinca can be dropped if HF fixes the DSV3 masking
    def test_router_group_mask_uses_negative_infinity(self):
        config = MiMoV2FlashConfig(
            num_hidden_layers=2,
            vocab_size=100,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            v_head_dim=4,
            layer_types=["full_attention", "sliding_attention"],
            moe_layer_freq=[0, 1],
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )
        model = MiMoV2FlashModel(config)
        router = model.layers[1].mlp.gate

        with torch.no_grad():
            router.weight.zero_()
            router.e_score_correction_bias.copy_(torch.tensor([-1.0, -1.0, -2.0, -2.0], dtype=torch.float32))

        hidden_states = torch.zeros((1, config.hidden_size), dtype=torch.float32)
        _, _, topk_idx = router(hidden_states)
        topk_idx = set(topk_idx[0].tolist())

        self.assertTrue(topk_idx.issubset({0, 1}))

    def test_generation_beyond_sliding_window(self):
        """Hybrid cache must handle full + sliding layers correctly when input exceeds the sliding window."""
        config = MiMoV2FlashConfig(
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            vocab_size=100,
            hidden_size=32,
            head_dim=8,
            v_head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            moe_layer_freq=[0, 1],
            moe_intermediate_size=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            sliding_window=4,
            max_position_embeddings=64,
            attn_implementation="eager",
        )
        model = MiMoV2FlashForCausalLM(config).eval()
        # Input longer than sliding_window (4)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
                disable_compile=True,
            )
        # Should generate without error and produce the expected number of tokens
        self.assertEqual(output.shape[1], input_ids.shape[1] + 3)


__all__ = ["MiMoV2FlashModelTest"]
