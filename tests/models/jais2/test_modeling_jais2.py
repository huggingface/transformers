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
    require_read_token,
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
@require_torch_accelerator
class Jais2IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_read_token
    def test_model_logits(self):
        model_id = "inceptionai/Jais-2-8B-Chat"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = Jais2ForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()

        EXPECTED_LOGITS_BATCH0 = [-0.9795, -1.0957, -0.9644, -0.9570, -0.9648, -0.9595, -0.9668, -0.9688, -0.9688, -0.9644, -0.9609, -0.9707, -0.9629, -0.9736, -0.9712]  # fmt: skip
        EXPECTED_LOGITS_BATCH1 = [-1.5332, -1.6289, -1.5264, -1.5195, -1.5264, -1.5215, -1.5303, -1.5303, -1.5312, -1.5264, -1.5234, -1.5322, -1.5254, -1.5352, -1.5332]  # fmt: skip

        torch.testing.assert_close(
            logits[0, -1, :15],
            torch.tensor(EXPECTED_LOGITS_BATCH0, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -1, :15],
            torch.tensor(EXPECTED_LOGITS_BATCH1, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )

    @require_read_token
    def test_model_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("inceptionai/Jais-2-8B-Chat")
        model = Jais2ForCausalLM.from_pretrained(
            "inceptionai/Jais-2-8B-Chat", torch_dtype=torch.float16, device_map="auto"
        )
        input_text = "Simply put, the theory of relativity states that"
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        model_inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        EXPECTED_TEXT = "Simply put, the theory of relativity states that the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is the same for all observers,"  # fmt: skip
        self.assertEqual(generated_text, EXPECTED_TEXT)
