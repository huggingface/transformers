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
import gc
import os
import re
import tempfile
import unittest

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
    logging,
)
from transformers.testing_utils import (
    CaptureLogger,
    require_bitsandbytes,
    require_peft,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import check_torch_load_is_safe, is_torch_available


if is_torch_available():
    import torch


@require_peft
@require_torch
class PeftTesterMixin:
    peft_test_model_ids = ("peft-internal-testing/tiny-OPTForCausalLM-lora",)
    transformers_test_model_ids = ("hf-internal-testing/tiny-random-OPTForCausalLM",)
    transformers_test_model_classes = (AutoModelForCausalLM, OPTForCausalLM)


# TODO: run it with CI after PEFT release.
@slow
class PeftIntegrationTester(unittest.TestCase, PeftTesterMixin):
    """
    A testing suite that makes sure that the PeftModel class is correctly integrated into the transformers library.
    """

    def _check_lora_correctly_converted(self, model):
        """
        Utility method to check if the model has correctly adapters injected on it.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        is_peft_loaded = False

        for _, m in model.named_modules():
            if isinstance(m, BaseTunerLayer):
                is_peft_loaded = True
                break

        return is_peft_loaded

    def _get_bnb_4bit_config(self):
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

    def _get_bnb_8bit_config(self):
        return BitsAndBytesConfig(load_in_8bit=True)

    def test_peft_from_pretrained(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`.
        This checks if we pass a remote folder that contains an adapter config and adapter weights, it
        should correctly load a model that has adapters injected on it.
        """
        logger = logging.get_logger("transformers.integrations.peft")

        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                with CaptureLogger(logger) as cl:
                    peft_model = transformers_class.from_pretrained(model_id, use_safetensors=False).to(torch_device)
                # ensure that under normal circumstances, there  are no warnings about keys
                self.assertNotIn("unexpected keys", cl.out)
                self.assertNotIn("missing keys", cl.out)

                self.assertTrue(self._check_lora_correctly_converted(peft_model))
                self.assertTrue(peft_model._hf_peft_config_loaded)
                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    def test_peft_state_dict(self):
        """
        Simple test that checks if the returned state dict of `get_adapter_state_dict()` method contains
        the expected keys.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, revision="refs/pr/2").to(torch_device)

                state_dict = peft_model.get_adapter_state_dict()

                for key in state_dict:
                    self.assertTrue("lora" in key)

    def test_peft_save_pretrained(self):
        """
        Test that checks various combinations of `save_pretrained` with a model that has adapters loaded
        on it. This checks if the saved model contains the expected files (adapter weights and adapter config).
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, revision="refs/pr/2").to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))

                    self.assertTrue("config.json" not in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

                    peft_model = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(peft_model))

    def test_peft_enable_disable_adapters(self):
        """
        A test that checks if `enable_adapters` and `disable_adapters` methods work as expected.
        """
        from peft import LoraConfig

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                peft_model.add_adapter(peft_config)

                peft_logits = peft_model(dummy_input).logits

                peft_model.disable_adapters()

                peft_logits_disabled = peft_model(dummy_input).logits

                peft_model.enable_adapters()

                peft_logits_enabled = peft_model(dummy_input).logits

                torch.testing.assert_close(peft_logits, peft_logits_enabled, rtol=1e-12, atol=1e-12)
                self.assertFalse(torch.allclose(peft_logits_enabled, peft_logits_disabled, atol=1e-12, rtol=1e-12))

    def test_peft_add_adapter(self):
        """
        Simple test that tests if `add_adapter` works as expected
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))
                # dummy generation
                _ = model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    def test_peft_add_adapter_from_pretrained(self):
        """
        Simple test that tests if `add_adapter` works as expected
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model_from_pretrained = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(model_from_pretrained))

    def test_peft_add_adapter_modules_to_save(self):
        """
        Simple test that tests if `add_adapter` works as expected when training with
        modules to save.
        """
        from peft import LoraConfig
        from peft.utils import ModulesToSaveWrapper

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

                model = transformers_class.from_pretrained(model_id).to(torch_device)
                peft_config = LoraConfig(init_lora_weights=False, modules_to_save=["lm_head"])
                model.add_adapter(peft_config)
                self._check_lora_correctly_converted(model)

                _has_modules_to_save_wrapper = False
                for name, module in model.named_modules():
                    if isinstance(module, ModulesToSaveWrapper):
                        _has_modules_to_save_wrapper = True
                        self.assertTrue(module.modules_to_save.default.weight.requires_grad)
                        self.assertTrue("lm_head" in name)
                        break

                self.assertTrue(_has_modules_to_save_wrapper)
                state_dict = model.get_adapter_state_dict()

                self.assertTrue("lm_head.weight" in state_dict)

                logits = model(dummy_input).logits
                loss = logits.mean()
                loss.backward()

                for _, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue(param.grad is not None)

    def test_peft_add_adapter_training_gradient_checkpointing(self):
        """
        Simple test that tests if `add_adapter` works as expected when training with
        gradient checkpointing.
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))

                # When attaching adapters the input embeddings will stay frozen, this will
                # lead to the output embedding having requires_grad=False.
                dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)
                frozen_output = model.get_input_embeddings()(dummy_input)
                self.assertTrue(frozen_output.requires_grad is False)

                model.gradient_checkpointing_enable()

                # Since here we attached the hook, the input should have requires_grad to set
                # properly
                non_frozen_output = model.get_input_embeddings()(dummy_input)
                self.assertTrue(non_frozen_output.requires_grad is True)

                # To repro the Trainer issue
                dummy_input.requires_grad = False

                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        self.assertTrue(param.requires_grad)

                logits = model(dummy_input).logits
                loss = logits.mean()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue("lora" in name.lower())
                        self.assertTrue(param.grad is not None)

    def test_peft_add_multi_adapter(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`. This test tests if
        add_adapter works as expected in multi-adapter setting.
        """
        from peft import LoraConfig
        from peft.tuners.tuners_utils import BaseTunerLayer

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                is_peft_loaded = False
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                logits_original_model = model(dummy_input).logits

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                logits_adapter_1 = model(dummy_input)

                model.add_adapter(peft_config, adapter_name="adapter-2")

                logits_adapter_2 = model(dummy_input)

                for _, m in model.named_modules():
                    if isinstance(m, BaseTunerLayer):
                        is_peft_loaded = True
                        break

                self.assertTrue(is_peft_loaded)

                # dummy generation
                _ = model.generate(input_ids=dummy_input)

                model.set_adapter("default")
                self.assertTrue(model.active_adapters() == ["default"])

                model.set_adapter("adapter-2")
                self.assertTrue(model.active_adapters() == ["adapter-2"])

                # Logits comparison
                self.assertFalse(
                    torch.allclose(logits_adapter_1.logits, logits_adapter_2.logits, atol=1e-6, rtol=1e-6)
                )
                self.assertFalse(torch.allclose(logits_original_model, logits_adapter_2.logits, atol=1e-6, rtol=1e-6))

                model.set_adapter(["adapter-2", "default"])
                self.assertTrue(model.active_adapters() == ["adapter-2", "default"])

                logits_adapter_mixed = model(dummy_input)
                self.assertFalse(
                    torch.allclose(logits_adapter_1.logits, logits_adapter_mixed.logits, atol=1e-6, rtol=1e-6)
                )
                self.assertFalse(
                    torch.allclose(logits_adapter_2.logits, logits_adapter_mixed.logits, atol=1e-6, rtol=1e-6)
                )

                # multi active adapter saving not supported
                with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)

    def test_delete_adapter(self):
        """
        Enhanced test for `delete_adapter` to handle multiple adapters,
        edge cases, and proper error handling.
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                # Add multiple adapters
                peft_config_1 = LoraConfig(init_lora_weights=False)
                peft_config_2 = LoraConfig(init_lora_weights=False)
                model.add_adapter(peft_config_1, adapter_name="adapter_1")
                model.add_adapter(peft_config_2, adapter_name="adapter_2")

                # Ensure adapters were added
                self.assertIn("adapter_1", model.peft_config)
                self.assertIn("adapter_2", model.peft_config)

                # Delete a single adapter
                model.delete_adapter("adapter_1")
                self.assertNotIn("adapter_1", model.peft_config)
                self.assertIn("adapter_2", model.peft_config)

                # Delete remaining adapter
                model.delete_adapter("adapter_2")
                self.assertFalse(hasattr(model, "peft_config"))
                self.assertFalse(model._hf_peft_config_loaded)

                # Re-add adapters for edge case tests
                model.add_adapter(peft_config_1, adapter_name="adapter_1")
                model.add_adapter(peft_config_2, adapter_name="adapter_2")

                # Attempt to delete multiple adapters at once
                model.delete_adapter(["adapter_1", "adapter_2"])
                self.assertFalse(hasattr(model, "peft_config"))
                self.assertFalse(model._hf_peft_config_loaded)

                # Test edge cases
                msg = re.escape("No adapter loaded. Please load an adapter first.")
                with self.assertRaisesRegex(ValueError, msg):
                    model.delete_adapter("nonexistent_adapter")

                model.add_adapter(peft_config_1, adapter_name="adapter_1")

                with self.assertRaisesRegex(ValueError, "The following adapter\\(s\\) are not present"):
                    model.delete_adapter("nonexistent_adapter")

                with self.assertRaisesRegex(ValueError, "The following adapter\\(s\\) are not present"):
                    model.delete_adapter(["adapter_1", "nonexistent_adapter"])

                # Deleting with an empty list or None should not raise errors
                model.add_adapter(peft_config_2, adapter_name="adapter_2")
                model.delete_adapter([])  # No-op
                self.assertIn("adapter_1", model.peft_config)
                self.assertIn("adapter_2", model.peft_config)

                # Deleting duplicate adapter names in the list
                model.delete_adapter(["adapter_1", "adapter_1"])
                self.assertNotIn("adapter_1", model.peft_config)
                self.assertIn("adapter_2", model.peft_config)

    def test_delete_adapter_with_modules_to_save(self):
        """
        Ensure that modules_to_save is accounted for when deleting an adapter.
        """
        from peft import LoraConfig

        # the test assumes a specific model architecture, so only test this one:
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        peft_config = LoraConfig(init_lora_weights=False, modules_to_save=["lm_head"])
        model.add_adapter(peft_config, adapter_name="adapter_1")

        # sanity checks
        self.assertIn("adapter_1", model.peft_config)
        self.assertNotIsInstance(model.lm_head, nn.Linear)  # a ModulesToSaveWrapper
        self.assertTrue(hasattr(model.lm_head, "modules_to_save"))
        self.assertTrue("adapter_1" in model.lm_head.modules_to_save)

        # now delete the adapter
        model.delete_adapter("adapter_1")
        self.assertFalse(hasattr(model, "peft_config"))
        self.assertFalse("adapter_1" in model.lm_head.modules_to_save)
        self.assertFalse(model.lm_head.modules_to_save)  # i.e. empty ModuleDict

    @require_torch_accelerator
    @require_bitsandbytes
    def test_peft_from_pretrained_kwargs(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained` + additional kwargs
        and see if the integraiton behaves as expected.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                bnb_config = self._get_bnb_8bit_config()
                peft_model = transformers_class.from_pretrained(
                    model_id, device_map="auto", quantization_config=bnb_config
                )

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    @require_torch_accelerator
    @require_bitsandbytes
    def test_peft_save_quantized(self):
        """
        Simple test that tests the basic usage of PEFT model save_pretrained with quantized base models
        """
        # 4bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                bnb_config = self._get_bnb_4bit_config()
                peft_model = transformers_class.from_pretrained(
                    model_id, device_map="auto", quantization_config=bnb_config
                )

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear4bit")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

        # 8-bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                bnb_config = self._get_bnb_8bit_config()
                peft_model = transformers_class.from_pretrained(
                    model_id, device_map="auto", quantization_config=bnb_config
                )

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

    @require_torch_accelerator
    @require_bitsandbytes
    def test_peft_save_quantized_regression(self):
        """
        Simple test that tests the basic usage of PEFT model save_pretrained with quantized base models
        """
        # 4bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                bnb_config = self._get_bnb_4bit_config()
                peft_model = transformers_class.from_pretrained(
                    model_id, device_map="auto", quantization_config=bnb_config
                )

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear4bit")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

        # 8-bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                bnb_config = self._get_bnb_8bit_config()
                peft_model = transformers_class.from_pretrained(
                    model_id, device_map="auto", quantization_config=bnb_config
                )

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

    def test_peft_pipeline(self):
        """
        Simple test that tests the basic usage of PEFT model + pipeline
        """
        from transformers import pipeline

        for adapter_id, base_model_id in zip(self.peft_test_model_ids, self.transformers_test_model_ids):
            peft_pipe = pipeline("text-generation", adapter_id)
            base_pipe = pipeline("text-generation", base_model_id)
            peft_params = list(peft_pipe.model.parameters())
            base_params = list(base_pipe.model.parameters())
            self.assertNotEqual(len(peft_params), len(base_params))  # Assert we actually loaded the adapter too
            _ = peft_pipe("Hello", max_new_tokens=20)

    def test_peft_add_adapter_with_state_dict(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`. This test tests if
        add_adapter works as expected with a state_dict being passed.
        """
        from peft import LoraConfig

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                with self.assertRaises(ValueError):
                    model.load_adapter(peft_model_id=None)

                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")

                check_torch_load_is_safe()
                dummy_state_dict = torch.load(state_dict_path, weights_only=True)

                model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=peft_config)
                with self.assertRaises(ValueError):
                    model.load_adapter(model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=None))
                self.assertTrue(self._check_lora_correctly_converted(model))

                # dummy generation
                _ = model.generate(input_ids=dummy_input)

    def test_peft_add_adapter_with_state_dict_low_cpu_mem_usage(self):
        """
        Check the usage of low_cpu_mem_usage, which is supported in PEFT >= 0.13.0
        """
        from peft import LoraConfig

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig()
                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")
                check_torch_load_is_safe()
                dummy_state_dict = torch.load(state_dict_path, weights_only=True)

                # this should always work
                model.load_adapter(
                    adapter_state_dict=dummy_state_dict, peft_config=peft_config, low_cpu_mem_usage=False
                )

                model.load_adapter(
                    adapter_state_dict=dummy_state_dict,
                    adapter_name="other",
                    peft_config=peft_config,
                    low_cpu_mem_usage=True,
                )
                # after loading, no meta device should be remaining
                self.assertFalse(any((p.device.type == "meta") for p in model.parameters()))

    def test_peft_from_pretrained_hub_kwargs(self):
        """
        Tests different combinations of PEFT model + from_pretrained + hub kwargs
        """
        peft_model_id = "peft-internal-testing/tiny-opt-lora-revision"

        # This should not work
        with self.assertRaises(OSError):
            _ = AutoModelForCausalLM.from_pretrained(peft_model_id)

        # This should work
        adapter_kwargs = {"revision": "test"}
        model = AutoModelForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        # note: always create new adapter_kwargs, avoid the test relying on the previous calls possibly mutating them
        adapter_kwargs = {"revision": "test"}
        model = OPTForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        adapter_kwargs = {"revision": "main", "subfolder": "test_subfolder"}
        model = AutoModelForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        adapter_kwargs = {"revision": "main", "subfolder": "test_subfolder"}
        model = OPTForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

    def test_peft_from_pretrained_unexpected_keys_warning(self):
        """
        Test for warning when loading a PEFT checkpoint with unexpected keys.
        """
        from peft import LoraConfig

        logger = logging.get_logger("transformers.integrations.peft")

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig()
                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")
                check_torch_load_is_safe()
                dummy_state_dict = torch.load(state_dict_path, weights_only=True)

                # add unexpected key
                dummy_state_dict["foobar"] = next(iter(dummy_state_dict.values()))

                with CaptureLogger(logger) as cl:
                    model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=peft_config)

                msg = "foobar | UNEXPECTED"
                self.assertIn(msg, cl.out)

    def test_peft_from_pretrained_missing_keys_warning(self):
        """
        Test for warning when loading a PEFT checkpoint with missing keys.
        """
        from peft import LoraConfig

        logger = logging.get_logger("transformers.integrations.peft")

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig()
                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")
                check_torch_load_is_safe()
                dummy_state_dict = torch.load(state_dict_path, weights_only=True)

                # remove a key so that we have missing keys
                key = next(iter(dummy_state_dict.keys()))
                del dummy_state_dict[key]

                with CaptureLogger(logger) as cl:
                    model.load_adapter(
                        adapter_state_dict=dummy_state_dict,
                        peft_config=peft_config,
                        low_cpu_mem_usage=False,
                        adapter_name="other",
                    )

                # Here we need to adjust the key name a bit to account for PEFT-specific naming.
                # 1. Remove PEFT-specific prefix
                peft_prefix = "base_model.model."
                key = key.removeprefix(peft_prefix)
                # 2. Insert adapter name
                prefix, _, suffix = key.rpartition(".")
                key = f".other.{suffix}"
                msg = f"{key} | MISSING |"
                self.assertIn(msg, cl.out)

    def test_peft_load_adapter_training_inference_mode_true(self):
        """
        By default, when loading an adapter, the whole model should be in eval mode and no parameter should have
        requires_grad=False.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    model = transformers_class.from_pretrained(peft_model.config._name_or_path)
                    model.load_adapter(tmpdirname)
                    assert not any(m.training for m in model.modules())
                    grads = [n for n,p in model.named_parameters() if p.requires_grad]
                    assert len(grads) == 0
                    del model

    def test_peft_load_adapter_training_inference_mode_false(self):
        """
        When passing is_trainable=True, the LoRA modules should be in training mode and their parameters should have
        requires_grad=True.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, use_safetensors=False).to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    model = transformers_class.from_pretrained(peft_model.config._name_or_path)
                    model.load_adapter(tmpdirname, is_trainable=True)

                    for name, module in model.named_modules():
                        if list(module.children()):
                            # only check leaf modules
                            continue

                        if "lora_" in name:
                            assert module.training
                            assert all(p.requires_grad for p in module.parameters())
                        else:
                            assert not module.training
                            assert all(not p.requires_grad for p in module.parameters())

    def test_prefix_tuning_trainer_load_best_model_at_end_error(self):
        # Original issue: https://github.com/huggingface/peft/issues/2256
        # There is a potential error when using load_best_model_at_end=True with a prompt learning PEFT method. This is
        # because Trainer uses load_adapter under the hood but with some prompt learning methods, there is an
        # optimization on the saved model to remove parameters that are not required for inference, which in turn
        # requires a change to the model architecture. This is why load_adapter will fail in such cases and users should
        # instead set load_best_model_at_end=False and use PeftModel.from_pretrained. As this is not obvious, we now
        # intercept the error and add a helpful error message.
        # This test checks this error message. It also tests the "happy path" (i.e. no error) when using LoRA.
        from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model

        # create a small sequence classification dataset (binary classification)
        dataset = []
        for i, row in enumerate(os.__doc__.splitlines()):
            dataset.append({"text": row, "label": i % 2})
        ds_train = Dataset.from_list(dataset)
        ds_valid = ds_train
        datasets = DatasetDict(
            {
                "train": ds_train,
                "val": ds_valid,
            }
        )

        # tokenizer for peft-internal-testing/tiny-OPTForCausalLM-lora cannot be loaded, thus using
        # hf-internal-testing/tiny-random-OPTForCausalLM
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", model_type="opt")

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")

        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        # lora works, prefix-tuning is expected to raise an error
        peft_configs = {
            "lora": LoraConfig(task_type=TaskType.SEQ_CLS),
            "prefix-tuning": PrefixTuningConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                prefix_projection=True,
                num_virtual_tokens=10,
            ),
        }

        for peft_type, peft_config in peft_configs.items():
            base_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
            base_model.config.pad_token_id = tokenizer.pad_token_id
            peft_model = get_peft_model(base_model, peft_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                training_args = TrainingArguments(
                    output_dir=tmpdirname,
                    num_train_epochs=3,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                )
                trainer = Trainer(
                    model=peft_model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["val"],
                )

                if peft_type == "lora":
                    # LoRA works with load_best_model_at_end
                    trainer.train()
                else:
                    # prefix tuning does not work, but at least users should get a helpful error message
                    msg = "When using prompt learning PEFT methods such as PREFIX_TUNING"
                    with self.assertRaisesRegex(RuntimeError, msg):
                        trainer.train()

    def test_peft_pipeline_no_warning(self):
        """
        Test to verify that the warning message "The model 'PeftModel' is not supported for text-generation"
        does not appear when using PeftModel with text-generation pipeline.
        """
        from peft import PeftModel

        from transformers import pipeline

        ADAPTER_PATH = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        BASE_PATH = "hf-internal-testing/tiny-random-OPTForCausalLM"

        # Input text for testing
        text = "Who is a Elon Musk?"

        model = AutoModelForCausalLM.from_pretrained(
            BASE_PATH,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

        lora_model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            device_map="auto",
        )

        # Create pipeline with PEFT model while capturing log output
        # Check that the warning message is not present in the logs
        pipeline_logger = logging.get_logger("transformers.pipelines.base")
        with self.assertNoLogs(pipeline_logger, logging.ERROR):
            lora_generator = pipeline(
                task="text-generation",
                model=lora_model,
                tokenizer=tokenizer,
                max_length=10,
            )

            # Generate text to verify pipeline works
            _ = lora_generator(text, max_new_tokens=20)

    def test_non_lora_load_adapter(self):
        """
        Check that loading a non-LoRA adapter works. Using LoKr as an example, not testing all possible PEFT methods.
        """
        from peft import LoKrConfig, get_peft_model

        inputs = torch.randint(0, 100, (1, 10)).to(torch_device)
        atol, rtol = 1e-4, 1e-4

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)
                with torch.inference_mode():
                    output_base = model(inputs).logits

                peft_config = LoKrConfig(init_weights=False)
                peft_model = get_peft_model(model, peft_config)
                with torch.inference_mode():
                    output_peft = peft_model(inputs).logits

                # sanity check: should be different
                assert not torch.allclose(output_base, output_peft, atol=atol, rtol=rtol)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    del model, peft_model

                    model = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    with torch.inference_mode():
                        output_transformers = model(inputs).logits
                    self.assertTrue(torch.allclose(output_peft, output_transformers, atol=atol, rtol=rtol))

    def test_non_lora_add_adapter(self):
        """
        Check that adding a non-LoRA adapter works. Using LoKr as an example, not testing all possible PEFT methods.
        """
        from peft import LoKrConfig

        inputs = torch.randint(0, 100, (1, 10)).to(torch_device)
        atol, rtol = 1e-4, 1e-4

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)
                with torch.inference_mode():
                    output_base = model(inputs).logits

                peft_config = LoKrConfig(init_weights=False)
                model.add_adapter(peft_config)
                with torch.inference_mode():
                    output_peft = model(inputs).logits
                # should be different
                assert not torch.allclose(output_base, output_peft, atol=atol, rtol=rtol)


@require_peft
@require_torch
@slow
class PeftHotswapIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # It is critical that the dynamo cache is reset for each test. Otherwise, if the test re-uses the same model,
        # there will be recompilation errors, as torch caches the model when run in the same process.
        torch.compiler.reset()
        gc.collect()

    def _check_model_hotswap(self, *, rank1, rank2, do_compile):
        # utility method that checks that we can successfully hotswap adapters, with the model outputs corresponding to
        # the respective adapters
        from peft import LoraConfig

        torch.manual_seed(0)
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        input = torch.randint(0, 100, (1, 10)).to(torch_device)
        with torch.inference_mode():
            base_output = model(input).logits

        # create 2 adapters
        model.add_adapter(LoraConfig(r=rank1, init_lora_weights=False), adapter_name="adapter_1")
        with torch.inference_mode():
            lora_1_output = model(input).logits

        # second adapter may have a different rank
        model.add_adapter(LoraConfig(r=rank2, init_lora_weights=False), adapter_name="adapter_2")
        model.set_adapter("adapter_2")
        with torch.inference_mode():
            lora_2_output = model(input).logits

        # sanity checks
        self.assertFalse(torch.allclose(base_output, lora_1_output, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(base_output, lora_2_output, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(lora_1_output, lora_2_output, atol=1e-6, rtol=1e-6))

        with tempfile.TemporaryDirectory() as tmpdirname:
            path_1 = os.path.join(tmpdirname, "adapter_1")
            path_2 = os.path.join(tmpdirname, "adapter_2")
            model.set_adapter("adapter_1")
            model.save_pretrained(path_1)
            model.set_adapter("adapter_2")
            model.save_pretrained(path_2)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
            enable_hotswap = do_compile or (rank1 != rank2)
            if enable_hotswap:
                # calling this is only needed if we want to compile the model or if the ranks are different
                model.enable_peft_hotswap(target_rank=max(rank1, rank2))

            # load the first adapter without hotswap (hotswap requires an existing adapter)
            model.load_adapter(path_1, adapter_name="adapter_1")
            if do_compile:
                # compile the model after loading the first adapter
                model = torch.compile(model, mode="reduce-overhead")

            with torch.inference_mode():
                lora_1_output_loaded = model(input).logits
            self.assertTrue(torch.allclose(lora_1_output, lora_1_output_loaded, atol=1e-6, rtol=1e-6))

            # hotswap in adapter_2 again, output should be same as lora_2_output
            if enable_hotswap:
                # after calling enable_peft_hotswap, hotswap will automatically be enabled
                model.load_adapter(path_2, adapter_name="adapter_1")
            else:
                # enable_peft_hotswap was not called, need to explicitly pass hotswap=True
                model.load_adapter(path_2, adapter_name="adapter_1", hotswap=True)

            with torch.inference_mode():
                lora_2_output_loaded = model(input).logits
            self.assertTrue(torch.allclose(lora_2_output, lora_2_output_loaded, atol=1e-6, rtol=1e-6))

    def test_hotswap_wrong_peft_type_raises(self):
        # only LoRA is supported for now
        from peft import IA3Config

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        peft_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        peft_config = IA3Config(feedforward_modules=[])
        model.add_adapter(peft_config, adapter_name="ia3")

        msg = "Hotswapping is currently only supported for LoRA"
        with self.assertRaisesRegex(ValueError, msg):
            model.load_adapter(peft_id, adapter_name="ia3", hotswap=True)

    def test_hotswap_without_existing_adapter_raises(self):
        # we can only hotswap if there is already an adapter with the same name
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        peft_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

        msg = "To hotswap an adapter, there must already be an existing adapter with the same adapter name"
        with self.assertRaisesRegex(ValueError, msg):
            model.load_adapter(peft_id, adapter_name="adapter_1", hotswap=True)

    def test_hotswap_different_adapter_name_raises(self):
        # we can only hotswap if there is already an adapter with the same name
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        peft_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        model.load_adapter(peft_id, adapter_name="adapter_1")

        other_name = "does_not_exist_yet"
        msg = "To hotswap an adapter, there must already be an existing adapter with the same adapter name"
        with self.assertRaisesRegex(ValueError, msg):
            model.load_adapter(peft_id, adapter_name=other_name, hotswap=True)

    def test_enable_peft_hotswap_called_after_adapter_added_raises(self):
        # ensure that when enable_peft_hotswap is called *after* loading the first adapter, an error is raised
        from peft import LoraConfig

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        lora_config = LoraConfig()
        model.add_adapter(lora_config)
        msg = re.escape("Call `enable_peft_hotswap` before loading the first adapter.")

        with self.assertRaisesRegex(RuntimeError, msg):
            model.enable_peft_hotswap(target_rank=32)

    def test_enable_peft_hotswap_called_after_adapter_added_warns(self):
        # ensure that when enable_peft_hotswap is called *after* loading the first adapter, there is a warning if
        # check_compiled="warn"
        from peft import LoraConfig

        logger = logging.get_logger("transformers.integrations.peft")
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        lora_config = LoraConfig()
        model.add_adapter(lora_config)
        msg = "It is recommended to call `enable_peft_hotswap` before loading the first adapter to avoid recompilation"

        with self.assertLogs(logger=logger, level="WARNING") as cm:
            model.enable_peft_hotswap(target_rank=32, check_compiled="warn")
            assert any(msg in log for log in cm.output)

    def test_enable_peft_hotswap_called_after_adapter_added_ignored(self):
        # Ensure that when enable_peft_hotswap is called *after* loading the first adapter, there is no error or
        # warning if check_compiled="ignore". Note that assertNoLogs only works with Python 3.10+.
        from peft import LoraConfig

        logger = logging.get_logger("transformers.integrations.peft")
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        lora_config = LoraConfig()
        model.add_adapter(lora_config)

        with self.assertNoLogs(logger, level="WARNING"):
            model.enable_peft_hotswap(target_rank=32, check_compiled="ignore")

    def test_hotswap_without_compile_and_same_ranks_works(self):
        self._check_model_hotswap(rank1=8, rank2=8, do_compile=False)

    def test_hotswap_without_compile_and_with_lower_rank_works(self):
        self._check_model_hotswap(rank1=13, rank2=7, do_compile=False)

    def test_hotswap_without_compile_and_with_higher_rank_works(self):
        self._check_model_hotswap(rank1=7, rank2=13, do_compile=False)

    def test_hotswap_with_compile_and_same_ranks_works(self):
        # It's important to add this context to raise an error on recompilation
        with (
            torch._dynamo.config.patch(error_on_recompile=True),
            torch._inductor.utils.fresh_inductor_cache(),
        ):
            self._check_model_hotswap(rank1=8, rank2=8, do_compile=True)

    def test_hotswap_with_compile_and_lower_rank_works(self):
        # It's important to add this context to raise an error on recompilation
        with (
            torch._dynamo.config.patch(error_on_recompile=True),
            torch._inductor.utils.fresh_inductor_cache(),
        ):
            self._check_model_hotswap(rank1=13, rank2=7, do_compile=True)

    def test_hotswap_with_compile_and_higher_rank_works(self):
        # It's important to add this context to raise an error on recompilation
        with (
            torch._dynamo.config.patch(error_on_recompile=True),
            torch._inductor.utils.fresh_inductor_cache(),
        ):
            self._check_model_hotswap(rank1=7, rank2=13, do_compile=True)
