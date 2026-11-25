# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Omoikane model."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import OmoikaneConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_sdpa_inference,
)


if is_torch_available():
    import torch
    from safetensors.torch import load_file, save_file

    from transformers import OmoikaneForCausalLM, OmoikaneModel


class OmoikaneModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = OmoikaneModel

    small_config_overrides = {
        "hidden_act": "silu",
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "linear_num_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "moe_intermediate_size": 16,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
    }


@require_torch
class OmoikaneModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = OmoikaneModelTester
    model_tester: OmoikaneModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        conv_dim = config.linear_num_heads * (2 * config.linear_key_head_dim + config.linear_value_head_dim)
        return (batch_size, conv_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.linear_num_heads, config.linear_key_head_dim, config.linear_value_head_dim)

    def test_attention_outputs(self):
        """Only the MLA layers return attention weights."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = self.model_tester.seq_length
        expected = sum(layer == "full_attention" for layer in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), expected)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        # The PyTorch gated delta rule computes in float32 and casts back, which is not robust enough in float16
        if dtype == "fp16":
            self.skipTest("Not robust in fp16")
        _test_eager_matches_sdpa_inference(
            self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
        )

    @unittest.skip("MLA caches the compressed latents, not the standard key/value format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("MLA caches the compressed latents, not the standard key/value format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_nested_config_format(self):
        """The training checkpoints nest the token-mixer and MoE settings; the config converts them."""
        config = OmoikaneConfig.from_dict(
            {
                "model_type": "omoikane",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "vocab_size": 128,
                "gdn_config": {"num_heads": 2, "head_dim_k": 32, "head_dim_v": 16, "short_conv_kernel": 3},
                "mla_config": {
                    "num_heads": 4,
                    "q_lora_rank": 32,
                    "kv_lora_rank": 24,
                    "qk_nope_head_dim": 16,
                    "qk_rope_head_dim": 8,
                    "v_head_dim": 12,
                    "rope_theta": 2000000.0,
                },
                "moe_config": {
                    "n_routed_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 16,
                    "n_shared_experts": 1,
                    "group_limited_routing": {"n_group": 4, "topk_group": 2},
                },
                "mtp_config": {"depth": 1},
                "dense_layer_indices": [0, 3],
                "mla_layer_indices": [2],
            }
        )
        self.assertEqual(
            config.layer_types, ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
        )
        self.assertEqual(config.mlp_layer_types, ["dense", "sparse", "sparse", "dense"])
        self.assertEqual(
            (config.linear_num_heads, config.linear_key_head_dim, config.linear_value_head_dim), (2, 32, 16)
        )
        self.assertEqual(config.linear_conv_kernel_dim, 3)
        self.assertEqual((config.num_attention_heads, config.kv_lora_rank, config.v_head_dim), (4, 24, 12))
        self.assertEqual(config.rope_parameters["rope_theta"], 2000000.0)
        self.assertEqual((config.n_group, config.topk_group, config.num_local_experts), (4, 2, 8))
        self.assertFalse(hasattr(config, "gdn_config"))

    def test_training_checkpoint_layout(self):
        """Load the layout of the training checkpoints: one convolution per projection and the MTP module as the
        layer after the last. Saving writes the same convolution layout back."""
        config = self.model_tester.get_config()
        torch.manual_seed(0)
        model = OmoikaneForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 5, 9, 13, 2, 7]])
        with torch.no_grad():
            expected = model(input_ids).logits

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            state_dict = load_file(f"{tmpdirname}/model.safetensors")
            prefix = "model.layers.0.self_attn"
            self.assertNotIn(f"{prefix}.conv1d.weight", state_dict)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                self.assertEqual(state_dict[f"{prefix}.{name}.weight"].shape, (32, 1, 4))
            # the MTP module, which inference does not use
            mtp = f"model.layers.{config.num_hidden_layers}"
            state_dict[f"{mtp}.eh_proj.weight"] = torch.zeros(config.hidden_size, 2 * config.hidden_size)
            state_dict[f"{mtp}.self_attn.kv_b_proj.weight"] = torch.zeros(48, config.kv_lora_rank)
            save_file(state_dict, f"{tmpdirname}/model.safetensors", metadata={"format": "pt"})

            reloaded, info = OmoikaneForCausalLM.from_pretrained(tmpdirname, output_loading_info=True)

        self.assertEqual(info["missing_keys"], set())
        self.assertEqual(info["unexpected_keys"], set())
        with torch.no_grad():
            torch.testing.assert_close(reloaded.eval()(input_ids).logits, expected)

    def test_gated_delta_net_readout_scale(self):
        """Omoikane reads the recurrent state with L2-normalized queries and no `head_k_dim**-0.5` scale."""
        config = self.model_tester.get_config()
        torch.manual_seed(0)
        layer = OmoikaneModel(config).layers[0].self_attn.eval()
        x = torch.randn(1, 5, config.hidden_size)
        with torch.no_grad():
            output = layer(x)

            # Plain recurrence: S <- exp(g) S + beta k (v - S^T k)^T, o = S^T q
            mixed = torch.cat([layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)], -1).transpose(1, 2)
            mixed = torch.nn.functional.silu(layer.conv1d(mixed)[..., :5]).transpose(1, 2)
            q, k, v = mixed.split([layer.key_dim, layer.key_dim, layer.value_dim], -1)
            heads = config.linear_num_heads
            q = torch.nn.functional.normalize(q.view(1, 5, heads, -1), dim=-1)
            k = torch.nn.functional.normalize(k.view(1, 5, heads, -1), dim=-1)
            v = v.view(1, 5, heads, -1)
            g = -layer.A_log.exp() * torch.nn.functional.softplus(layer.a_proj(x) + layer.dt_bias)
            beta = layer.b_proj(x).sigmoid()
            state = torch.zeros(1, heads, config.linear_key_head_dim, config.linear_value_head_dim)
            outs = []
            for t in range(5):
                state = state * g[:, t].exp()[..., None, None]
                delta = (v[:, t] - torch.einsum("bhkv,bhk->bhv", state, k[:, t])) * beta[:, t, :, None]
                state = state + k[:, t, :, :, None] * delta[:, :, None, :]
                outs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
            o = torch.stack(outs, 1)
            gate = layer.g_proj(x).view(1, 5, heads, -1)
            expected = layer.o_proj(layer.o_norm(o, gate).reshape(1, 5, -1))

        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)
