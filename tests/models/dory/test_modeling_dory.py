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
"""Testing suite for the PyTorch Dory model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import DoryForCausalLM, DoryModel


class DoryModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DoryModel

    def __init__(self, parent, **kwargs):
        # One loop keeps the common tests' one-cache-slot-per-layer assumption; loops are tested below
        super().__init__(parent, n_input_layers=1, n_output_layers=0, n_recurrent_loops=1, **kwargs)


@require_torch
class DoryModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DoryModelTester
    # The nGPT normalized updates start small (`ngpt_alpha_init`), so overfitting takes a few more steps
    training_overfit_steps = 400

    def test_reverse_loading_mapping(self):
        # Checkpoints are saved with the `backbone.` prefix, which only exists on the head model
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_chunked_prefill_and_decode_match_full_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_hidden_layers, config.layer_types, config.rope_parameters = 4, None, None
        config.n_input_layers, config.n_recurrent_layers, config.n_output_layers, config.n_recurrent_loops = 1, 2, 1, 3
        config.__post_init__()
        model = DoryForCausalLM(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        with torch.no_grad():
            full_logits = model(input_ids).logits
            outputs = model(input_ids[:, :4], use_cache=True)
            logits = [outputs.logits]
            for position in range(4, input_ids.shape[1]):
                outputs = model(input_ids[:, position : position + 1], past_key_values=outputs.past_key_values)
                logits.append(outputs.logits)
        torch.testing.assert_close(torch.cat(logits, dim=1), full_logits, rtol=1e-4, atol=1e-4)


@slow
@require_torch_accelerator
class DoryIntegrationTest(unittest.TestCase):
    checkpoint = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

    def test_bf16_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = DoryForCausalLM.from_pretrained(self.checkpoint, dtype=torch.bfloat16, device_map=torch_device)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(torch_device)
        output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        self.assertEqual(
            tokenizer.decode(output[0], skip_special_tokens=True),
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid. The",
        )
