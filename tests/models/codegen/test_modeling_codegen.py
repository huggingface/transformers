# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import unittest

from transformers import CodeGenConfig, is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import backend_manual_seed, require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, CodeGenForCausalLM, CodeGenModel


class CodeGenModelTester(CausalLMModelTester):
    config_class = CodeGenConfig
    if is_torch_available():
        base_model_class = CodeGenModel
        causal_lm_class = CodeGenForCausalLM


@require_torch
class CodeGenModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (CodeGenModel, CodeGenForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": CodeGenModel, "text-generation": CodeGenForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = CodeGenModelTester


@require_torch
class CodeGenModelLanguageGenerationTest(unittest.TestCase):
    @cached_property
    def cached_tokenizer(self):
        return AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    @cached_property
    def cached_model(self):
        return CodeGenForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

    @slow
    def test_lm_generate_codegen(self):
        tokenizer = self.cached_tokenizer
        for checkpointing in [True, False]:
            model = self.cached_model

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("def hello_world():", return_tensors="pt").to(torch_device)
            expected_output = 'def hello_world():\n    print("Hello World")\n\nhello_world()\n\n'

            output_ids = model.generate(**inputs, do_sample=False)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    @slow
    def test_codegen_sample(self):
        tokenizer = self.cached_tokenizer
        model = self.cached_model
        model.to(torch_device)

        torch.manual_seed(0)
        backend_manual_seed(torch_device, 0)

        tokenized = tokenizer("def hello_world():", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        token_type_ids = tokenized.token_type_ids.to(torch_device)
        output_seq = model.generate(input_ids=input_ids, do_sample=True, num_return_sequences=5)
        output_seq_tt = model.generate(
            input_ids=input_ids, token_type_ids=token_type_ids, do_sample=True, num_return_sequences=5
        )
        output_seq_strs = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        output_seq_tt_strs = tokenizer.batch_decode(output_seq_tt, skip_special_tokens=True)

        if torch_device == "cuda":
            EXPECTED_OUTPUT_STR = 'def hello_world():\n    print("Hello World")\n    return True\n\nresult ='
        else:
            EXPECTED_OUTPUT_STR = "def hello_world():\r\n    print('Hello, World.')\r\n\r\n\r"

        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
        self.assertTrue(
            all(output_seq_strs[idx] != output_seq_tt_strs[idx] for idx in range(len(output_seq_tt_strs)))
        )  # token_type_ids should change output
