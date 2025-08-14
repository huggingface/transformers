# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Testing suite for the PyTorch PhiMoE model."""

import unittest

from transformers import PhimoeConfig, StaticCache, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        PhimoeForCausalLM,
        PhimoeForSequenceClassification,
        PhimoeModel,
    )

    end_of_text_token = 32000

    class PhimoeMiniWithStaticCache(torch.nn.Module):
        def __init__(self, model: PhimoeForCausalLM, batch_size: int, max_seq_len: int):
            super().__init__()
            self.model = model
            self.cache = StaticCache(config=model.config, max_cache_len=max_seq_len)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
        ) -> torch.FloatTensor:
            return self.model.forward(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                past_key_values=self.cache,
            ).logits

        @staticmethod
        def generate(model: PhimoeForCausalLM, prompt_tokens: torch.LongTensor, max_seq_len: int) -> list[int]:
            model = PhimoeMiniWithStaticCache(model, 1, max_seq_len + prompt_tokens.shape[-1])

            response_tokens = []

            for input_pos in range(prompt_tokens.shape[-1]):
                result = model.forward(
                    input_ids=prompt_tokens[:, input_pos : input_pos + 1],
                )
                response_tokens.append(prompt_tokens[0][input_pos].item())

            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

            while current_token != end_of_text_token and len(response_tokens) < max_seq_len:
                result = model.forward(
                    input_ids=torch.tensor([[current_token]], dtype=torch.long),
                )
                current_token = torch.argmax(result[:, -1, :], dim=-1).item()
                response_tokens.append(current_token)

            return response_tokens


class PhimoeModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = PhimoeConfig
        base_model_class = PhimoeModel
        causal_lm_class = PhimoeForCausalLM
        sequence_class = PhimoeForSequenceClassification


@require_torch
class PhimoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (PhimoeModel, PhimoeForCausalLM, PhimoeForSequenceClassification) if is_torch_available() else ()
    )

    test_headmasking = False
    test_pruning = False
    test_all_params_have_gradient = False
    model_tester_class = PhimoeModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": PhimoeModel,
            "text-classification": PhimoeForSequenceClassification,
            "text-generation": PhimoeForCausalLM,
            "zero-shot": PhimoeForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79292/workflows/fa2ba644-8953-44a6-8f67-ccd69ca6a476/jobs/1012905
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True


@slow
@require_torch
class PhimoeIntegrationTest(unittest.TestCase):
    def test_model_phimoe_instruct_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhimoeForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[-3.5312, -2.5000, -1.2734,  0.3555, -0.7578, -0.4727,  0.5977, -0.4316,
          0.2256, -1.2188, -1.6797,  0.9961,  3.7656, 11.3125, -1.3828, -4.8438,
         -5.7500, -1.9375,  0.7227, -0.3438, -0.2100, -0.4277, -0.0444, -0.5352,
         -0.6406, -0.1016, -0.4258, -1.0234,  0.4297, -0.6250],
        [-0.9883,  0.1455, -0.4902,  2.3594,  0.7031,  3.1406,  0.4375,  0.2559,
          0.6172, -2.1094, -1.3359,  2.5938,  4.9062, 10.8125, -0.1094,  1.5781,
         -4.9375,  0.7148, -0.0972,  1.7656, -0.0801,  0.2217,  0.1875, -0.4629,
          1.5781,  0.3535,  0.0874,  0.6836, -0.0518, -1.2969]]).to(torch_device)  # fmt: skip

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

    def test_phimoe_instruct_generation(self):
        model = PhimoeForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = model.generate(inputs, max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits are both delicious and nutritious fruits that can be combined in various ways to create tast"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_phimoe_instruct_with_static_cache(self):
        model = PhimoeForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        response_tokens = PhimoeMiniWithStaticCache.generate(model, inputs, 64)

        output_text = tokenizer.batch_decode(torch.tensor([response_tokens], dtype=torch.long, device=torch_device))

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits are both delicious and nutritious fruits that can"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)
