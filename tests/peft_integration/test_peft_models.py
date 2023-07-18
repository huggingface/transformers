# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import inspect
import unittest

from transformers import AutoModelForCausalLM, OPTForCausalLM
from transformers.testing_utils import require_peft, require_torch, require_torch_gpu, slow, torch_device
from transformers.utils import is_peft_available, is_torch_available


if is_peft_available():
    from peft import PeftModel

if is_torch_available():
    import torch


@require_peft
@require_torch
class PeftTesterMixin:
    # one safetensors adapters and one pickle adapter
    peft_test_model_ids = ("peft-internal-testing/opt-350m-lora-pickle", "peft-internal-testing/opt-350m-lora")
    transformers_test_model_classes = (AutoModelForCausalLM, OPTForCausalLM)


@slow
class PeftIntegrationTester(unittest.TestCase, PeftTesterMixin):
    r"""
    A testing suite that makes sure that the PeftModel class is correctly integrated into the transformers library.

    - test_peft_from_pretrained:
        Tests if the peft model is correctly loaded through `from_pretrained` method
    - test_peft_from_pretrained_kwargs:
        Tests if the kwargs are correctly passed to the peft model
    """

    def test_peft_from_pretrained(self):
        r"""
        Simple test that tests the basic usage of PEFT model through `from_pretrained`
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)
                self.assertTrue(isinstance(peft_model, PeftModel))

                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    @require_torch_gpu
    def test_peft_from_pretrained_kwargs(self):
        r"""
        Simple test that tests the basic usage of PEFT model through `from_pretrained` + additional kwargs
        and see if the integraiton behaves as expected.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
                self.assertTrue(isinstance(peft_model, PeftModel))

                module = inspect.getmodule(
                    peft_model.base_model.model.model.decoder.layers[0].self_attn.v_proj.__class__
                )

                # Check that the converted linear layers are from PEFT library - which is different from the
                # `Linear8bitLt` class from bnb.
                # here `module` should be `peft/src/peft/tuners/lora.py` thus contain the class `LoraModel`
                self.assertTrue("LoraModel" in dir(module))

                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))


class PeftFastIntegrationTester(unittest.TestCase, PeftTesterMixin):
    r"""
    A testing suite that makes sure that the PeftModel class is correctly integrated into the transformers library.
    "Fast" version with tiny random models

    - test_peft_from_pretrained:
        Tests if the peft model is correctly loaded through `from_pretrained` method
    - test_peft_from_pretrained_kwargs:
        Tests if the kwargs are correctly passed to the peft model
    """
    peft_test_model_ids = ("peft-internal-testing/tiny-OPTForCausalLM-lora",)

    def test_peft_from_pretrained(self):
        r"""
        Simple test that tests the basic usage of PEFT model through `from_pretrained`
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)
                self.assertTrue(isinstance(peft_model, PeftModel))

                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))
