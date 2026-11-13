# Copyright 2026 Recursive and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Flint model."""

import tempfile
import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from transformers import FlintForCausalLM, FlintModel
    from transformers.models.flint.modeling_flint import eager_attention_forward


class FlintModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = FlintModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["full_attention", "sliding_attention"]
        self.mlp_only_layers = [0]
        self.sliding_window = 32


@require_torch
class FlintModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = FlintModelTester
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.8, 0.9]

    def test_attention_sinks(self):
        """Each sink adds a logit to the softmax denominator without contributing a value."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        attention = FlintModel(config).layers[0].self_attn
        attention.sinks.data = torch.linspace(-2.0, 2.0, config.num_attention_heads)
        query = torch.randn(1, config.num_attention_heads, 5, config.head_dim)
        key, value = torch.randn(2, 1, config.num_key_value_heads, 5, config.head_dim)
        mask = torch.full((5, 5), float("-inf")).triu(1)[None, None]
        output, _ = eager_attention_forward(attention, query, key, value, mask, scaling=attention.scaling)

        groups = config.num_attention_heads // config.num_key_value_heads
        scores = query.double() @ key.double().repeat_interleave(groups, 1).transpose(-1, -2) * attention.scaling
        scores = (scores + mask).exp()
        weights = scores / (scores.sum(-1, keepdim=True) + attention.sinks.double().exp()[None, :, None, None])
        expected = (weights @ value.double().repeat_interleave(groups, 1)).transpose(1, 2)
        torch.testing.assert_close(output.double(), expected, rtol=1e-5, atol=1e-5)

    def test_full_attention_layers_have_no_positional_encoding(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        model = FlintModel(config).to(torch_device).eval()
        input_ids = inputs["input_ids"][:1]
        position_ids = torch.arange(input_ids.shape[1], device=torch_device)[None]
        with torch.no_grad():
            output = model(input_ids, position_ids=position_ids).last_hidden_state
            # Doubling changes relative distances; rotary embeddings would not see a uniform shift
            doubled = model(input_ids, position_ids=2 * position_ids).last_hidden_state
        torch.testing.assert_close(output, doubled)

    def test_checkpoint_format_round_trip(self):
        """Checkpoints store one tensor per expert and the sinks as `softmax_offset`, which load and save unchanged."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = FlintForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as checkpoint, tempfile.TemporaryDirectory() as resaved:
            model.save_pretrained(checkpoint)
            state_dict = load_file(f"{checkpoint}/model.safetensors")
            self.assertIn("model.layers.1.mlp.experts.0.gate_proj.weight", state_dict)
            self.assertIn("model.layers.0.self_attn.softmax_offset", state_dict)
            self.assertFalse(any(name.endswith((".gate_up_proj", ".sinks")) for name in state_dict))

            reloaded = FlintForCausalLM.from_pretrained(checkpoint)
            for name, parameter in model.state_dict().items():
                torch.testing.assert_close(reloaded.state_dict()[name], parameter, rtol=0, atol=0)
            reloaded.save_pretrained(resaved)
            self.assertEqual(load_file(f"{resaved}/model.safetensors").keys(), state_dict.keys())


@slow
@require_torch_accelerator
class FlintIntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_greedy_chat(self):
        model_id = "RecursiveSI/flint"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FlintForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
        expected = {
            "Who are you?": "<think>The user asks who I am. Simple identity question. Answer concisely: Flint, made by "
            "Recursive. No need for extra disclaimers.</think>I'm Flint, an AI assistant "
            "developed by Recursive.",
            "What is 17 * 23? Answer briefly.": "<think>User asks simple multiplication, wants brief answer. "
            "17*23 = 391.</think>391",
        }
        for prompt, completion in expected.items():
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            generated = tokenizer.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            self.assertEqual(generated, completion)
