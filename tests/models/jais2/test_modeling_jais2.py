# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Jais2 model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Jais2Config,
        Jais2ForCausalLM,
        Jais2Model,
    )


class Jais2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Jais2Config
        base_model_class = Jais2Model
        causal_lm_class = Jais2ForCausalLM

    config_overrides = {
        "hidden_act": "relu2",
    }


@require_torch
class Jais2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Jais2ModelTester
    all_model_classes = (
        (
            Jais2Model,
            Jais2ForCausalLM,
        )
        if is_torch_available()
        else ()
    )

    all_generative_model_classes = (Jais2ForCausalLM,) if is_torch_available() else ()

    pipeline_model_mapping = (
        {
            "feature-extraction": Jais2Model,
            "text-generation": Jais2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )


@slow
@require_torch
class Jais2IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_torch_accelerator
    @is_flaky(max_attempts=3)
    def test_model_logits(self):
        model_id = "inceptionai/Jais-2-8B-Chat"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = Jais2ForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()

        EXPECTED_LOGITS_BATCH0 = [-0.97509765625, -1.091796875, -0.9599609375]
        EXPECTED_LOGITS_BATCH1 = [-1.5361328125, -1.6328125, -1.5283203125]

        torch.testing.assert_close(
            logits[0, -1, :3],
            torch.tensor(EXPECTED_LOGITS_BATCH0, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -1, :3],
            torch.tensor(EXPECTED_LOGITS_BATCH1, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_model_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("inceptionai/Jais-2-8B-Chat")
        model = Jais2ForCausalLM.from_pretrained(
            "inceptionai/Jais-2-8B-Chat", torch_dtype=torch.float16, device_map="auto"
        )
        input_text = "The capital of France is"
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        model_inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        EXPECTED_TEXT = "The capital of France is Paris."
        self.assertEqual(generated_text, EXPECTED_TEXT)
