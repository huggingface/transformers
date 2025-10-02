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
import copy
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig
from transformers.testing_utils import (
    TestCasePlus,
    backend_empty_cache,
    require_kernels,
    require_torch_accelerator,
    torch_device,
)
from transformers.utils import is_kernels_available


if is_kernels_available():
    from kernels import Device, Mode, kernelize

@require_kernels
class TestHubKernels(TestCasePlus):
    def setUp(self):
        self.model_id = "unsloth/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model_kernelized = AutoModelForCausalLM.from_pretrained(
            self.model_id, use_kernels=True, device_map=torch_device
        )
        self.model_not_kernelized = AutoModelForCausalLM.from_pretrained(
            self.model_id, use_kernels=False, device_map=torch_device
        )
        self.input = "Hello"


    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @require_torch_accelerator
    def test_forward(self):
        tokenized_input = self.tokenizer(self.input, return_tensors="pt").input_ids.to(self.model_kernelized.device)
        output_ = self.model_kernelized.generate(tokenized_input, max_new_tokens=10, do_sample=False)
        output = self.tokenizer.decode(output_[0], skip_special_tokens=True)

        self.EXPECTED_OUTPUT = set()
        self.EXPECTED_OUTPUT.add("Hello, I'm looking for a reliable and trustworthy online")

        self.assertTrue(output in self.EXPECTED_OUTPUT)

    def test_getter_use_kernels(self):
        self.assertTrue(self.model_kernelized.use_kernels)
        self.assertFalse(self.model_not_kernelized.use_kernels)

    def test_kernelized_forward_is_different(self, kernelized_model, not_kernelized_model):
        """
        Iterate over modules and check if the forward method is different between
        the kernelized and not kernelized models. Break on first difference, else continue.
        Finally, assert that at least one forward is different.
        """
        found_difference = False
        for (name1, module1), (name2, module2) in zip(
            kernelized_model.named_modules(), not_kernelized_model.named_modules()
        ):
            # Only compare modules with the same name
            if name1 != name2:
                continue
            # Check if both modules have a 'forward' attribute
            if hasattr(module1, "forward") and hasattr(module2, "forward"):
                # Compare the code objects of the forward methods
                code1 = getattr(module1.forward, "__code__", None)
                code2 = getattr(module2.forward, "__code__", None)
                if code1 is not None and code2 is not None:
                    if code1 is not code2:
                        found_difference = True
                        break
        self.assertTrue(
            found_difference,
            "No module's forward method was different between kernelized and not kernelized models.",
        )

    def test_kernelized_forward_is_the_same(self, model_1, model_2):
        """
        Iterate over modules and check if the forward method is the same between
        the kernelized and not kernelized models. Break on first difference, else continue.
        Finally, assert that at least one forward is the same.
        """
        no_difference = True
        for (name1, module1), (name2, module2) in zip(model_1.named_modules(), model_2.named_modules()):
            # Only compare modules with the same name
            if name1 != name2:
                continue
            # Check if both modules have a 'forward' attribute
            if hasattr(module1, "forward") and hasattr(module2, "forward"):
                # Compare the code objects of the forward methods
                code1 = getattr(module1.forward, "__code__", None)
                code2 = getattr(module2.forward, "__code__", None)
                if code1 is not None and code2 is not None:
                    if code1 != code2:
                        no_difference = False
                        break
        self.assertTrue(
            no_difference,
            "All module's forward methods were the same between the two models",
        )

    def test_kernelize(self):
        model = copy.deepcopy(self.model_not_kernelized)
        kernelize(model, mode=Mode.INFERENCE, device=Device(type=model.device.type))
        self.test_kernelized_forward_is_different(model, self.model_not_kernelized)
        self.test_kernelized_forward_is_the_same(model, self.model_kernelized)
        del model

    def test_setter_use_kernels(self):
        model = copy.deepcopy(self.model_not_kernelized)
        model.use_kernels = True
        self.assertTrue(model.use_kernels)
        self.test_kernelized_forward_is_different(model, self.model_not_kernelized)
        self.test_kernelized_forward_is_the_same(model, self.model_kernelized)
        del model

    def test_unkernelize(self):
        model = copy.deepcopy(self.model_kernelized)

        with self.assertLogs("transformers.modeling_utils", level="WARNING") as cm:
            model.use_kernels = False

        self.assertTrue(
            any(
                "Disabling kernels at runtime is a no-op as there is no 'unkernelize' routine; keeping current kernels active."
                in msg
                for msg in cm.output
            )
        )

        self.assertFalse(model.use_kernels)

    def test_kernels_mapping(self):
        kernel_config = KernelConfig(kernel_mapping={"RMSNorm": "kernels-community/layer_norm:LlamaRMSNorm"})
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct", use_kernels=True, device_map=torch_device, kernel_config=kernel_config
        )

        EXPECTED_OUTPUT = set()
        EXPECTED_OUTPUT.add("Hello, I'm looking for a reliable and trustworthy online")

        tokenized_input = self.tokenizer(self.input, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(tokenized_input, max_new_tokens=10, do_sample=False)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertTrue(output in EXPECTED_OUTPUT)

        del model

    def test_faulty_kernel_mapping_layer_name(self):
        kernel_config = KernelConfig(kernel_mapping={"RMSNorm1": "kernels-community/layer_norm:LlamaRMSNorm"})
        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(
                "unsloth/Llama-3.2-1B-Instruct", use_kernels=True, device_map=torch_device, kernel_config=kernel_config
            )

    def test_faulty_kernel_mapping_type(self):
        kernel_config = KernelConfig(kernel_mapping={"RMSNorm": 1})
        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(
                "unsloth/Llama-3.2-1B-Instruct", use_kernels=True, device_map=torch_device, kernel_config=kernel_config
            )
