# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# Run the test: CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/kernels/test_kernels.py

import os
import tempfile
import textwrap

from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    require_huggingface_hub_greater_or_equal,
    require_torch_multi_accelerator,
    torch_device,
    torchrun,
)


if is_torch_available():
    import torch


class TestHubKernels(TestCasePlus):
    def setUp(self):
        self.model_id = "unsloth/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, use_kernels=True)
        self.input = "Hello"
        self.EXPECTED_OUTPUT = "Hello, how are you?"

    def test_forward(self):
        tokenized_input = self.tokenizer(self.input, return_tensors="pt").input_ids.to(self.model.device)
        output = self.model.generate(tokenized_input, max_new_tokens=10)
        self.assertTrue(self.tokenizer.decode(output[0], skip_special_tokens=True) == self.EXPECTED_OUTPUT)

    def test_rmsnorm(self):
        layer = self.model.model.layers[0].input_layernorm
        print(layer)
