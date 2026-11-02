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
import json
import os
import tempfile
import unittest

from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, TorchAoConfig
from transformers.testing_utils import require_accelerate, require_bitsandbytes, require_torch, require_torchao
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_torch
@require_accelerate
@require_torchao
@require_bitsandbytes
class QuantizationPlanTest(unittest.TestCase):
    def test_plan_quantizes_saves_and_reloads(self):
        import bitsandbytes as bnb
        from torchao.quantization import Int8WeightOnlyConfig
        from torchao.utils import TorchAOBaseTensor

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        plan = {"self_attn": TorchAoConfig(Int8WeightOnlyConfig()), "mlp": BitsAndBytesConfig(load_in_4bit=True)}
        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        def check_layers(model):
            for layer in model.model.layers:
                self.assertIsInstance(layer.self_attn.q_proj.weight, TorchAOBaseTensor)
                self.assertEqual(type(layer.self_attn.q_proj), torch.nn.Linear)
                self.assertIsInstance(layer.mlp.gate_proj, bnb.nn.Linear4bit)
            self.assertNotIsInstance(model.lm_head.weight, TorchAOBaseTensor)
            self.assertEqual(type(model.lm_head), torch.nn.Linear)

        with tempfile.TemporaryDirectory() as dense_dir, tempfile.TemporaryDirectory() as quantized_dir:
            LlamaForCausalLM(config).save_pretrained(dense_dir)
            model = LlamaForCausalLM.from_pretrained(dense_dir, quantization_config=plan, device_map="cpu")
            check_layers(model)
            model.save_pretrained(quantized_dir)

            with open(os.path.join(quantized_dir, "config.json")) as f:
                saved_plan = json.load(f)["quantization_config"]
            self.assertEqual(saved_plan["self_attn"]["quant_method"], "torchao")
            self.assertEqual(saved_plan["mlp"]["quant_method"], "bitsandbytes")

            reloaded = LlamaForCausalLM.from_pretrained(quantized_dir, device_map="cpu")
            check_layers(reloaded)
            with torch.no_grad():
                torch.testing.assert_close(reloaded(input_ids).logits, model(input_ids).logits)
