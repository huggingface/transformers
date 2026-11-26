# Copyright 2026 the Hugging Face team. All rights reserved.
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
from unittest.mock import patch

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_distributed_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AwqConfig,
        DistributedConfig,
        MiniCPM5MoeConfig,
        MiniCPM5MoeForCausalLM,
        MiniCPM5MoeModel,
    )


class MiniCPM5MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniCPM5MoeModel
        causal_lm_class = MiniCPM5MoeForCausalLM

    small_config_overrides = {"n_routed_experts": 4, "num_key_value_heads": 1, "attention_bias": True}


@require_torch
class MiniCPM5MoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPM5MoeModelTester

    def test_fp16_eager_attention_scales_before_narrowing(self):
        from transformers.models.minicpm5_moe.modeling_minicpm5_moe import minicpm5_moe_eager_attention_forward

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        attention = AutoModelForCausalLM.from_config(config).model.layers[0].self_attn
        query = torch.full((1, config.num_attention_heads, 1, 2), 256.0, dtype=torch.float16)
        key = torch.full((1, config.num_key_value_heads, 2, 2), 256.0, dtype=torch.float16)
        value = torch.ones_like(key)
        output, weights = minicpm5_moe_eager_attention_forward(attention, query, key, value, None, scaling=0.25)
        torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
        torch.testing.assert_close(output, torch.ones_like(output))

    def test_interleaved_query_and_gate_checkpoint(self):
        from transformers.core_model_loading import Interleave

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        heads = config.num_attention_heads
        query = torch.arange(heads * 6).view(heads, 3, 2)
        gate = -query - 1
        packed = torch.stack((query, gate), dim=1).flatten(0, 2)
        operation = Interleave(dim=0, num_groups_attribute="num_attention_heads")
        unpacked = operation.convert({"q": packed}, ["q"], ["split"], config=config)["split"]
        torch.testing.assert_close(unpacked, torch.cat((query.flatten(0, 1), gate.flatten(0, 1))))
        repacked = operation.reverse_op.convert({"split": unpacked}, ["split"], ["q"], config=config)["q"]
        torch.testing.assert_close(repacked, packed)

    @unittest.skipUnless(is_torch_distributed_available(), "test requires torch.distributed")
    def test_interleaved_checkpoint_loads_local_heads(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            torch.distributed.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
            try:
                sharded = AutoModelForCausalLM.from_pretrained(
                    directory, distributed_config=DistributedConfig(tp_size=2, tp_plan="auto")
                )
                for name in ("query_proj", "gate_proj"):
                    reference = getattr(model.model.layers[0].self_attn, name)
                    local = getattr(sharded.model.layers[0].self_attn, name)
                    for weight, expected in zip(local.parameters(), reference.parameters()):
                        torch.testing.assert_close(weight.to_local().cpu(), expected.chunk(2)[0], atol=0, rtol=0)
            finally:
                torch.distributed.destroy_process_group()

    def test_tp_size_must_divide_sharded_kv_heads(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(config)
        with self.assertRaisesRegex(ValueError, r"num_key_value_heads \(2\).*tp_size \(4\)"):
            model._resolve_parallel_plans(DistributedConfig(tp_size=4))
        tp_plan, _, _ = model._resolve_parallel_plans(DistributedConfig(tp_size=4, tp_plan={}))
        self.assertEqual(tp_plan, {})

    def test_awq_fused_projection_exclusion(self):
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        from transformers.quantizers.quantizers_utils import should_convert_module

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = AutoModelForCausalLM.from_config(config)
        quantizer = AwqQuantizer(AwqConfig(modules_to_not_convert=["model.layers.0.self_attn.q_proj"]))
        with (
            patch("transformers.integrations.replace_with_awq_linear", return_value=model),
            patch("transformers.integrations.replace_quantization_scales"),
        ):
            quantizer._process_model_before_weight_loading(model)
        skips = quantizer.modules_to_not_convert
        for name in ("query_proj", "gate_proj"):
            self.assertFalse(should_convert_module(f"model.layers.0.self_attn.{name}", skips))
            self.assertTrue(should_convert_module(f"model.layers.1.self_attn.{name}", skips))
        self.assertTrue(should_convert_module("model.layers.0.mlp.gate_proj", skips))

    def test_legacy_ungated_config(self):
        with self.assertRaisesRegex(ValueError, "use_gated_attention=False"):
            MiniCPM5MoeConfig.from_dict({"use_gated_attention": False})

    def test_router_preserves_fp32_under_autocast(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_experts_per_tok = 1
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)
        gate = model.model.layers[1].mlp.gate
        hidden_states = torch.ones(1, config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            gate.weight.zero_()
            gate.weight[:2, 0] = 1
            gate.weight[1, 1] = 2**-10
            expected = gate(hidden_states)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = gate(hidden_states)
        self.assertEqual(expected[2].item(), 1)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)

    def test_grouped_router_excludes_masked_experts_when_scores_are_negative(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.n_group = 2
        config.topk_group = 1
        config.num_experts_per_tok = 2
        gate = AutoModelForCausalLM.from_config(config).model.layers[1].mlp.gate
        with torch.no_grad():
            gate.e_score_correction_bias.copy_(torch.tensor([-1.0, -1.0, -2.0, -2.0]))
            _, _, indices = gate(torch.zeros(1, config.hidden_size))
        self.assertEqual(sorted(indices[0].tolist()), [0, 1])

    def test_cached_decode_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits[:, -1]
            prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
            decoded = model(input_ids=input_ids[:, -1:], cache=prefix.cache, use_cache=True)
        torch.testing.assert_close(decoded.logits[:, -1], expected, atol=1e-5, rtol=1e-4)
