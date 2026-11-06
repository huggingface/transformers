# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ling 3 / Bailing MoE V3 model."""

import re
import tempfile
import unittest
from unittest import mock

from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_kernels,
    require_torch,
    require_torch_accelerator_memory,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import load_file
    from torch import nn

    from transformers import BailingHybridConfig, BailingHybridForCausalLM, BailingHybridModel, FineGrainedFP8Config
    from transformers.generation.utils import ALL_CACHE_NAMES
    from transformers.integrations.finegrained_fp8 import FP8Linear
    from transformers.models.bailing_hybrid.modeling_bailing_hybrid import BailingHybridExperts, BailingHybridMLP
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts, DeepseekV3MLP
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer


class BailingHybridModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BailingHybridModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.hidden_act = "silu"
        self.num_hidden_layers = 2
        self.layer_types = ["linear_attention", "full_attention"]
        self.first_k_dense_replace = 1
        self.short_conv_kernel_size = 2
        # Keep the KDA head size distinct from the MLA RoPE size to cover config save/load semantics.
        self.head_dim = 24
        self.kv_lora_rank = 16
        self.q_lora_rank = None
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 16
        self.v_head_dim = 32
        self.moe_intermediate_size = 16
        self.num_local_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.num_mtp_layers = 0


@require_torch
class BailingHybridModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BailingHybridModelTester
    model_tester: BailingHybridModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, 3 * config.num_attention_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_attention_heads, config.linear_head_dim, config.linear_head_dim)

    def test_attention_outputs(self):
        """Only full-attention layers produce attention maps in this hybrid architecture."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        expected_num_attentions = sum(layer == "full_attention" for layer in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), expected_num_attentions)

            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            output_len = len(outputs)

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(output_len + 1, len(outputs))
            self.assertEqual(len(outputs.attentions), expected_num_attentions)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    def test_reverse_loading_mapping(self):
        """Exercise the two-stage Ling key mapping as an end-to-end exact round trip."""
        for q_lora_rank in (None, 12):
            with self.subTest(q_lora_rank=q_lora_rank):
                config = self.model_tester.get_config()
                config.q_lora_rank = q_lora_rank
                model = BailingHybridForCausalLM(config)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    saved_keys = set(load_file(f"{tmpdirname}/model.safetensors"))
                    reloaded = BailingHybridForCausalLM.from_pretrained(tmpdirname)

                expected_keys = {
                    "model.word_embeddings.weight",
                    "model.layers.0.attention.f_proj.weight",
                    "model.layers.0.attention.q_conv1d.weight",
                    "model.layers.0.attention.k_conv1d.weight",
                    "model.layers.0.attention.v_conv1d.weight",
                    "model.layers.1.attention.dense.weight",
                    "model.layers.1.mlp.gate.expert_bias",
                }
                query_projections = ("q_proj",) if q_lora_rank is None else ("q_a_layernorm", "q_a_proj", "q_b_proj")
                expected_keys.update(f"model.layers.1.attention.{name}.weight" for name in query_projections)
                self.assertEqual("model.layers.1.attention.q_proj.weight" in saved_keys, q_lora_rank is None)
                self.assertTrue(expected_keys.issubset(saved_keys))
                self.assertNotIn("model.layers.0.self_attn.conv1d.weight", saved_keys)
                state_dict = model.state_dict()
                reloaded_state_dict = reloaded.state_dict()
                self.assertNotEqual(config.linear_head_dim, config.qk_rope_head_dim)
                self.assertEqual(reloaded.config.linear_head_dim, config.linear_head_dim)
                self.assertEqual(reloaded.config.qk_rope_head_dim, config.qk_rope_head_dim)
                self.assertEqual(reloaded.config.q_lora_rank, config.q_lora_rank)
                self.assertEqual(state_dict.keys(), reloaded_state_dict.keys())
                for key, value in state_dict.items():
                    torch.testing.assert_close(value, reloaded_state_dict[key], rtol=0, atol=0)

    def test_mla_query_projection_variants(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        full_attention_idx = config.layer_types.index("full_attention")

        direct_model = BailingHybridForCausalLM(config)
        direct_attention = direct_model.model.layers[full_attention_idx].self_attn
        self.assertIsNotNone(direct_attention.q_proj)
        self.assertIsNone(direct_attention.q_a_proj)
        self.assertIsNone(direct_attention.q_a_layernorm)
        self.assertIsNone(direct_attention.q_b_proj)

        config = BailingHybridConfig.from_dict({**config.to_dict(), "q_lora_rank": 12})
        lora_model = BailingHybridForCausalLM(config)
        lora_attention = lora_model.model.layers[full_attention_idx].self_attn
        self.assertIsNone(lora_attention.q_proj)
        self.assertEqual(lora_attention.q_a_proj.weight.shape, (config.q_lora_rank, config.hidden_size))
        self.assertEqual(lora_attention.q_a_layernorm.weight.shape, (config.q_lora_rank,))
        self.assertEqual(
            lora_attention.q_b_proj.weight.shape,
            (config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim), config.q_lora_rank),
        )
        lora_model = lora_model.to(torch_device).eval()
        with torch.no_grad():
            outputs = lora_model(**self._prepare_for_class(inputs_dict, BailingHybridForCausalLM))
        self.assertEqual(outputs.logits.shape[:2], inputs_dict["input_ids"].shape)

    def test_fp8_mixed_precision_modules(self):
        config = self.model_tester.get_config()
        config.q_lora_rank = 12
        checkpoint_quantization_config = {
            "quant_method": "fp8",
            "modules_to_not_convert": [
                "lm_head",
                "model.word_embeddings",
                "model.layers.0.attention.b_proj",
                "model.layers.1.attention.g_proj",
                f"model.layers.{config.num_hidden_layers}.eh_proj",
            ],
            "weight_block_size": [128, 128],
        }
        fp8_config = BailingHybridConfig.from_dict(
            {**config.to_dict(), "quantization_config": checkpoint_quantization_config}
        )
        with torch.device("meta"):
            model = BailingHybridForCausalLM(fp8_config)
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config.from_dict(fp8_config.quantization_config))
        quantizer._process_model_before_weight_loading(model)

        linear_attention = model.model.layers[0].self_attn
        full_attention = model.model.layers[1].self_attn
        self.assertIs(type(linear_attention.b_proj), nn.Linear)
        self.assertIsInstance(linear_attention.q_proj, FP8Linear)
        self.assertIsInstance(linear_attention.forget_gate.f_proj, FP8Linear)
        self.assertIsInstance(linear_attention.g_proj, FP8Linear)
        self.assertIs(type(full_attention.g_proj), nn.Linear)
        self.assertIsInstance(full_attention.q_a_proj, FP8Linear)
        self.assertIsInstance(full_attention.q_b_proj, FP8Linear)
        self.assertIsInstance(full_attention.kv_b_proj, FP8Linear)
        self.assertIsInstance(model.model.layers[0].mlp.gate_proj, FP8Linear)
        self.assertIs(type(model.lm_head), nn.Linear)
        self.assertIsInstance(model.model.embed_tokens, nn.Embedding)
        self.assertEqual(fp8_config.quantization_config, checkpoint_quantization_config)

    def test_swiglu_limits_per_layer_and_round_trip(self):
        config = self.model_tester.get_config()
        config.first_k_dense_replace = 0
        config.mlp_layer_types = ["sparse"] * config.num_hidden_layers
        config.expert_swiglu_limit_list = [0, 2.5]
        config.share_expert_swiglu_limit_list = [1.5, 0]
        config = BailingHybridConfig.from_dict(config.to_dict())
        model = BailingHybridForCausalLM(config)
        for layer, expert_limit, shared_limit in zip(model.model.layers, [float("inf"), 2.5], [1.5, float("inf")]):
            self.assertEqual(layer.mlp.experts.limit, expert_limit)
            self.assertEqual(layer.mlp.shared_experts.limit, shared_limit)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = BailingHybridForCausalLM.from_pretrained(directory)
        self.assertEqual(reloaded.config.expert_swiglu_limit_list, [0, 2.5])
        self.assertEqual(reloaded.config.share_expert_swiglu_limit_list, [1.5, 0])
        self.assertEqual(reloaded.model.layers[1].mlp.experts.limit, 2.5)
        self.assertEqual(reloaded.model.layers[0].mlp.shared_experts.limit, 1.5)

        for name in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "one limit per decoder layer"):
                BailingHybridConfig.from_dict({**config.to_dict(), name: [1]})

    def test_swiglu_clipping_after_activation(self):
        config = self.model_tester.get_config()
        config.hidden_size = config.intermediate_size = config.moe_intermediate_size = 4
        config.num_local_experts = 2
        experts = BailingHybridExperts(config, swiglu_limit=1.5).to(torch_device)
        shared = BailingHybridMLP(config).to(torch_device)
        shared.limit = 2.5
        with torch.no_grad():
            for module in (experts, shared):
                for parameter in module.parameters():
                    parameter.fill_(0.5)
        inputs = torch.tensor(
            [[3.0, 3.0, 3.0, 3.0], [-3.0, -3.0, -3.0, -3.0]], device=torch_device, requires_grad=True
        )
        indices = torch.tensor([[0, 1], [1, 0]], device=torch_device)
        weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], device=torch_device)

        gate_up = torch.nn.functional.linear(inputs, experts.gate_up_proj[0])
        gate, up = gate_up.chunk(2, dim=-1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(gate).clamp(max=1.5) * up.clamp(-1.5, 1.5), experts.down_proj[0]
        )
        expected_gradient = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)[0]
        for implementation in ("eager", "batched_mm", "grouped_mm"):
            with self.subTest(implementation=implementation):
                config._experts_implementation = implementation
                output = experts(inputs, indices, weights)
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(torch.autograd.grad(output.sum(), inputs)[0], expected_gradient)

        expected_shared = shared.down_proj(
            torch.nn.functional.silu(shared.gate_proj(inputs)).clamp(max=2.5) * shared.up_proj(inputs).clamp(-2.5, 2.5)
        )
        torch.testing.assert_close(shared(inputs), expected_shared)
        self.assertFalse(
            torch.equal(torch.nn.functional.silu(gate).clamp(max=1.5), torch.nn.functional.silu(gate.clamp(max=1.5)))
        )

    def test_swiglu_disabled_matches_unclipped_experts(self):
        config = self.model_tester.get_config()
        config._experts_implementation = "eager"
        inputs = torch.randn(3, config.hidden_size, device=torch_device)
        indices = torch.tensor([[0, 1], [1, 2], [2, 3]], device=torch_device)
        weights = torch.rand(3, 2, device=torch_device)
        for new_class, reference_class in (
            (BailingHybridMLP, DeepseekV3MLP),
            (BailingHybridExperts, DeepseekV3Experts),
        ):
            with self.subTest(module=new_class.__name__):
                reference = reference_class(config).to(torch_device)
                candidate = new_class(config).to(torch_device)
                for parameter in reference.parameters():
                    nn.init.normal_(parameter, std=0.2)
                candidate.load_state_dict(reference.state_dict())
                args = (inputs, indices, weights) if new_class is BailingHybridExperts else (inputs,)
                torch.testing.assert_close(candidate(*args), reference(*args), rtol=0, atol=0)

    def test_fp8_preserves_layer_swiglu_clipping(self):
        config = self.model_tester.get_config()
        config.mlp_layer_types = ["sparse"] * config.num_hidden_layers
        config.expert_swiglu_limit_list = [0, 1.5]
        config.share_expert_swiglu_limit_list = [2.5, 0]
        with torch.device("meta"):
            model = BailingHybridForCausalLM(config)
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config())
        quantizer._process_model_before_weight_loading(model)
        gate_up = torch.tensor([[2.0, -2.0, 3.0, -3.0]])
        gate, up = gate_up.chunk(2, dim=-1)
        for layer, limit in zip(model.model.layers, [float("inf"), 1.5]):
            self.assertEqual(layer.mlp.experts.limit, limit)
            expected = torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
            torch.testing.assert_close(layer.mlp.experts._apply_gate(gate_up), expected, rtol=0, atol=0)
        self.assertEqual(model.model.layers[0].mlp.shared_experts.limit, 2.5)
        self.assertEqual(model.model.layers[1].mlp.shared_experts.limit, float("inf"))

    def test_sonicmoe_rejects_unsupported_clipping(self):
        config = self.model_tester.get_config()
        config.expert_swiglu_limit_list = [0, 10]
        model = BailingHybridForCausalLM(config)
        with self.assertRaisesRegex(ValueError, "post-activation SwiGLU clipping"):
            model.set_experts_implementation("sonicmoe")

    def test_mtp_ignored_keys_follow_configured_layer_count(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_mtp_layers = 2
        model = BailingHybridForCausalLM(config)
        patterns = model._keys_to_ignore_on_load_unexpected

        for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + config.num_mtp_layers):
            key = f"model.layers.{layer_idx}.attention.q_a_proj.weight"
            self.assertTrue(any(re.search(pattern, key) for pattern in patterns))

        last_main_layer = f"model.layers.{config.num_hidden_layers - 1}.attention.q_a_proj.weight"
        self.assertFalse(any(re.search(pattern, last_main_layer) for pattern in patterns))

        config.num_mtp_layers = 0
        self.assertFalse(BailingHybridForCausalLM(config)._keys_to_ignore_on_load_unexpected)

    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        # Ling's MoE and KDA chunk boundaries amplify harmless split-vs-full accumulation differences to roughly
        # 1e-5--1e-4. Keep the threshold well below the ~1e-3 padding-state contamination this regression targets.
        with mock.patch("transformers.utils.import_utils.is_torchdynamo_compiling", return_value=True):
            for model_class in self.all_generative_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                model = model_class(config).to(torch_device).eval()
                input_ids = inputs_dict["input_ids"][:2].to(torch_device)

                pad_token_id = 7
                seq_len = input_ids.shape[1]
                turn1_len = seq_len // 2 + 1
                pad_len = max(1, seq_len - turn1_len - 1)
                turn1, turn2 = input_ids[:, :turn1_len], input_ids[:, turn1_len:].clone()
                turn2[1, :pad_len] = pad_token_id
                attention_mask = torch.ones_like(input_ids)
                attention_mask[1, turn1_len : turn1_len + pad_len] = 0
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

                with torch.no_grad():
                    single = model(
                        input_ids=torch.cat([turn1, turn2], dim=-1),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    ).logits
                    out1 = model(
                        input_ids=turn1,
                        attention_mask=attention_mask[:, :turn1_len],
                        position_ids=position_ids[:, :turn1_len],
                        use_cache=True,
                    )
                    cache_kwarg, cache = next(
                        (
                            (name, getattr(out1, name))
                            for name in ALL_CACHE_NAMES
                            if getattr(out1, name, None) is not None
                        ),
                        ("past_key_values", None),
                    )
                    out2 = model(
                        input_ids=turn2,
                        attention_mask=attention_mask,
                        position_ids=position_ids[:, turn1_len:],
                        use_cache=True,
                        **{cache_kwarg: cache},
                    )

                torch.testing.assert_close(out2.logits[:, -1], single[:, -1], rtol=1e-3, atol=1e-4)

    @unittest.skip("MTP assisted generation cannot roll back the stateful KDA cache.")
    def test_generate_with_mtp(self):
        pass

    @unittest.skip("MLA uses different query/key and value head dimensions, so Flash SDPA cannot dispatch.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@slow
@require_kernels
@require_torch_accelerator_memory(memory=600)
class BailingHybridIntegrationTest(unittest.TestCase):
    model_id = "inclusionAI/Ling-3.1-flash"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_ling_3_1_flash_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", dtype="auto", attn_implementation="eager", trust_remote_code=False
        ).eval()
        linear_attention = model.model.layers[model.config.layer_types.index("linear_attention")].self_attn
        full_attention = model.model.layers[model.config.layer_types.index("full_attention")].self_attn
        for module in (full_attention.q_b_proj, full_attention.kv_b_proj, model.model.layers[0].mlp.gate_proj):
            self.assertIsInstance(module, FP8Linear)
            self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
        for module in (linear_attention.b_proj, full_attention.g_proj, model.lm_head):
            self.assertIs(type(module), nn.Linear)
            self.assertEqual(module.weight.dtype, torch.bfloat16)
        for i, layer in enumerate(model.model.layers):
            if model.config.mlp_layer_types[i] == "sparse":
                self.assertEqual(layer.mlp.experts.limit, model.config.expert_swiglu_limit_list[i] or float("inf"))
                self.assertEqual(
                    layer.mlp.shared_experts.limit, model.config.share_expert_swiglu_limit_list[i] or float("inf")
                )
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Please answer with one short sentence: What is the capital of France?"}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=2, do_sample=False)

        self.assertEqual(output[0, inputs.input_ids.shape[1] :].tolist(), [678, 2838])
