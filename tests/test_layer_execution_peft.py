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

"""Configuration through PEFT containers preserves adapters and execution-local caches."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_peft, require_torch


if is_torch_available():
    import torch

    from transformers import RepeatRange, get_layer_execution_plan, set_layer_execution_plan

    from . import test_layer_execution as execution_tests


@require_torch
@require_peft
class LayerExecutionPeftTest(unittest.TestCase):
    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_configuration_through_peft_and_parallel_containers(self, family):
        from peft import LoraConfig, get_peft_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        native = execution_tests.LayerExecutionModelTest().make_model(family).to(device)
        model = get_peft_model(native, LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["down_proj"]))
        model.eval()
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        state_keys = tuple(model.state_dict())
        inputs = torch.tensor([[1, 2, 3]], device=device)
        self.assertIsNone(get_layer_execution_plan(model))
        with torch.no_grad():
            original = model(inputs, use_cache=False).logits
        self.assertIs(set_layer_execution_plan(model, repeats=[RepeatRange(1, 2)]), model)
        self.assertEqual(get_layer_execution_plan(model).layer_order, (0, 1, 1, 2))
        self.assertEqual(model.get_layer_execution_plan(), get_layer_execution_plan(native))
        with torch.no_grad():
            repeated = model(inputs, use_cache=False).logits
            self.assertFalse(torch.equal(repeated, original))
            torch.testing.assert_close(
                model.generate(inputs, max_new_tokens=3),
                model.generate(inputs, max_new_tokens=3, use_cache=False),
            )
        wrapped = torch.nn.DataParallel(model)
        self.assertEqual(get_layer_execution_plan(wrapped), get_layer_execution_plan(model))
        self.assertIs(set_layer_execution_plan(wrapped, [0, 1, 2, 0, 1, 2]), wrapped)
        self.assertEqual(get_layer_execution_plan(model).layer_order, (0, 1, 2, 0, 1, 2))
        self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), parameter_ids)
        self.assertEqual(tuple(model.state_dict()), state_keys)
        self.assertIs(set_layer_execution_plan(model, None), model)
        self.assertIsNone(get_layer_execution_plan(wrapped))
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, use_cache=False).logits, original, atol=0, rtol=0)
