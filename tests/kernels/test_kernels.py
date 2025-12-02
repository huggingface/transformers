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

# Run the test: CUDA_VISIBLE_DEVICES=0 RUN_SLOW=1 pytest -sv tests/kernels/test_kernels.py


import copy
import types
from unittest.mock import MagicMock, patch

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig
from transformers.integrations.hub_kernels import (
    _HUB_KERNEL_MAPPING,
    _KERNEL_MODULE_MAPPING,
    is_kernel,
    lazy_load_kernel,
    load_and_register_attn_kernel,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import (
    TestCasePlus,
    cleanup,
    require_kernels,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_kernels_available


if is_kernels_available():
    import kernels as kernels_pkg
    from kernels import Device, Mode, kernelize


@require_kernels
@slow
class TestHubKernels(TestCasePlus):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "unsloth/Llama-3.2-1B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.model_kernelized = AutoModelForCausalLM.from_pretrained(
            cls.model_id, use_kernels=True, device_map=torch_device
        )
        cls.model_not_kernelized = AutoModelForCausalLM.from_pretrained(
            cls.model_id, use_kernels=False, device_map=torch_device
        )
        cls.input = "Hello"

    @classmethod
    def tearDownClass(cls):
        for attr in [
            "model_kernelized",
            "model_not_kernelized",
            "tokenizer",
        ]:
            if hasattr(cls, attr):
                try:
                    delattr(cls, attr)
                except Exception:
                    pass

        # Clear any temporary kernel module cache entries populated by tests
        try:
            keys_to_remove = [
                k for k, v in list(_KERNEL_MODULE_MAPPING.items()) if v is None or isinstance(v, types.ModuleType)
            ]
            for k in keys_to_remove:
                _KERNEL_MODULE_MAPPING.pop(k, None)
        except Exception:
            pass

    def tearDown(self):
        # Free accelerator memory/cache and trigger GC
        cleanup(torch_device, gc_collect=True)

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

    def assert_kernelized_forward_is_different(self, kernelized_model, not_kernelized_model):
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

    def assert_kernelized_forward_is_the_same(self, model_1, model_2):
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
        kernelize(model, mode=Mode.INFERENCE, device=Device(type=model.device.type))  # type: ignore[arg-type]
        self.assert_kernelized_forward_is_different(model, self.model_not_kernelized)
        self.assert_kernelized_forward_is_the_same(model, self.model_kernelized)
        del model

    def test_setter_use_kernels(self):
        model = copy.deepcopy(self.model_not_kernelized)
        model.use_kernels = True
        self.assertTrue(model.use_kernels)
        self.assert_kernelized_forward_is_different(model, self.model_not_kernelized)
        self.assert_kernelized_forward_is_the_same(model, self.model_kernelized)
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
        del model

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


@require_kernels
class TestKernelUtilities(TestCasePlus):
    def test_is_kernel_regex(self):
        valid = [
            "org/model",
            "org/model@main",
            "org/model:my_func",
            "org/model@v1.2.3:my_func",
            "flash|org/model@rev:fn",
        ]
        invalid = [
            "org//model",
            "org/model:too:many",
            "org/model@rev:fn:extra",
            "/org/model",
            "org:model",
        ]
        for s in valid:
            self.assertTrue(is_kernel(s.split("|")[-1]))
        for s in invalid:
            self.assertFalse(is_kernel(s))

    def test_lazy_load_kernel_success_and_cache(self):
        sentinel = types.SimpleNamespace(name="sentinel")

        original_get_kernel = getattr(kernels_pkg, "get_kernel")
        try:

            def fake_get_kernel(repo_id, revision=None, version=None):
                self.assertIn(repo_id, {"kernels-community/causal-conv1d"})
                return sentinel

            setattr(kernels_pkg, "get_kernel", fake_get_kernel)
            _KERNEL_MODULE_MAPPING.pop("causal-conv1d", None)

            mod1 = lazy_load_kernel("causal-conv1d")
            self.assertIs(mod1, sentinel)
            mod2 = lazy_load_kernel("causal-conv1d")
            self.assertIs(mod2, sentinel)
        finally:
            setattr(kernels_pkg, "get_kernel", original_get_kernel)
            # Ensure cache is cleared to avoid holding onto module references across tests
            _KERNEL_MODULE_MAPPING.pop("causal-conv1d", None)

    def test_lazy_load_kernel_unknown(self):
        name = "unknown-kernel-name"
        _KERNEL_MODULE_MAPPING.pop(name, None)
        mod = lazy_load_kernel(name)
        self.assertIsNone(mod)
        self.assertIn(name, _KERNEL_MODULE_MAPPING)
        # Cleanup cache entry to avoid growth across tests
        _KERNEL_MODULE_MAPPING.pop(name, None)

    def test_lazy_load_kernel_version(self):
        HUB = _HUB_KERNEL_MAPPING
        name = "causal-conv1d"
        version_spec = ">=0.0.4,<0.1.0"
        original_get_kernel = getattr(kernels_pkg, "get_kernel")
        original_entry = HUB.get(name, None)

        # Use a real ModuleType so caching short-circuits on the second call
        sentinel_mod = types.ModuleType("sentinel_kernel_module")
        call_count = {"n": 0}

        try:
            # Inject dict-style mapping with repo_id and version
            HUB[name] = {"repo_id": "kernels-community/causal-conv1d", "version": version_spec}  # type: ignore[assignment]
            _KERNEL_MODULE_MAPPING.pop(name, None)

            def fake_get_kernel(repo_id, revision=None, version=None, user_agent=None):
                call_count["n"] += 1
                self.assertEqual(repo_id, "kernels-community/causal-conv1d")
                self.assertIsNone(revision, "revision must not be set when version is provided")
                self.assertEqual(version, version_spec)
                return sentinel_mod

            # Patch kernels.get_kernel so lazy_load_kernel picks it up on import
            setattr(kernels_pkg, "get_kernel", fake_get_kernel)

            # Act
            mod1 = lazy_load_kernel(name)
            mod2 = lazy_load_kernel(name)

            # Assert
            self.assertIs(mod1, sentinel_mod)
            self.assertIs(mod2, sentinel_mod)
            self.assertEqual(call_count["n"], 1, "second call should hit the cache")
        finally:
            # Restore patched function and mapping to avoid side effects
            setattr(kernels_pkg, "get_kernel", original_get_kernel)
            if original_entry is None:
                HUB.pop(name, None)
            else:
                HUB[name] = original_entry
            _KERNEL_MODULE_MAPPING.pop(name, None)


@require_kernels
class TestAttentionKernelRegistration(TestCasePlus):
    def test_load_and_register_flash_attn_like_kernel(self):
        kernel_obj = types.SimpleNamespace(flash_attn_varlen_func=lambda *a, **k: None)

        with (
            patch("transformers.integrations.hub_kernels.get_kernel", return_value=kernel_obj),
            patch("transformers.integrations.hub_kernels.lazy_import_flash_attention", return_value=None),
        ):
            attn_impl = "org/model"
            load_and_register_attn_kernel(attn_impl)
            self.assertIn(attn_impl, ALL_ATTENTION_FUNCTIONS.valid_keys())
            # Cleanup registration to avoid leaking functions across tests
            try:
                ALL_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception:
                pass
            try:
                ALL_MASK_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception:
                pass

    def test_load_and_register_named_function_kernel(self):
        def my_attention(*args, **kwargs):
            return None

        kernel_obj = types.SimpleNamespace(my_func=my_attention)
        with patch("transformers.integrations.hub_kernels.get_kernel", return_value=kernel_obj):
            attn_impl = "org/model:my_func"
            load_and_register_attn_kernel(attn_impl)
            self.assertIn(attn_impl, ALL_ATTENTION_FUNCTIONS.valid_keys())
            # Cleanup registration to avoid leaking functions across tests
            try:
                ALL_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception:
                pass
            try:
                ALL_MASK_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception:
                pass


@require_kernels
class TestUseKernelsLifecycle(TestCasePlus):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "unsloth/Llama-3.2-1B-Instruct"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, use_kernels=False, device_map=torch_device)

    @classmethod
    def tearDownClass(cls):
        # Delete large objects to drop references early
        if hasattr(cls, "model"):
            try:
                del cls.model
            except Exception:
                pass

    def tearDown(self):
        # Free accelerator memory/cache and trigger GC
        cleanup(torch_device, gc_collect=True)

    def test_setting_use_kernels_twice_does_not_rekernelize(self):
        call_count = {"n": 0}

        def spy_kernelize(*args, **kwargs):
            call_count["n"] += 1

        with patch.object(kernels_pkg, "kernelize", side_effect=spy_kernelize):
            self.model.use_kernels = True
            self.assertTrue(self.model.use_kernels)
            self.assertEqual(call_count["n"], 1)
            self.model.use_kernels = True
            self.assertEqual(call_count["n"], 1)

    def test_train_eval_calls_kernelize_with_correct_mode(self):
        last_modes = []

        def spy_kernelize(model, device=None, mode=None):
            last_modes.append(mode)

        with patch.object(kernels_pkg, "kernelize", side_effect=spy_kernelize):
            self.model.use_kernels = True
            self.model.train(True)
            self.assertTrue(any(m == Mode.TRAINING for m in last_modes))
            self.model.eval()
            self.assertTrue(any(m == Mode.INFERENCE for m in last_modes))


@require_kernels
class TestKernelMappingDeviceFiltering(TestCasePlus):
    """Test that kernel mappings correctly filter by current device."""

    def test_multi_device_mapping_filters_correctly(self):
        """
        Test that when a kernel_mapping contains multiple devices (cuda, rocm),
        only the current device's kernel is registered.
        Regression test for issue where ROCm overwrote CUDA mapping.
        """
        kernel_mapping = {
            "RMSNorm": {
                "cuda": "kernels-community/layer_norm:LlamaRMSNorm",
                "rocm": "kernels-community/layer_norm:LlamaRMSNorm",
            }
        }

        kernel_config = KernelConfig(kernel_mapping)

        # Create a mock model on CUDA device
        mock_model = MagicMock()
        mock_model.training = False

        # Mock parameter with CUDA device
        mock_param = MagicMock()
        mock_param.device.type = "cuda"
        mock_model.parameters.return_value = iter([mock_param])

        # Mock named_modules with RMSNorm layer
        mock_layer = MagicMock()
        mock_layer.kernel_layer_name = "RMSNorm"
        mock_model.named_modules.return_value = [("layers.0", mock_layer)]

        # Trigger the mapping creation
        kernel_config.create_compatible_mapping(mock_model)

        # Verify results
        result_mapping = kernel_config.kernel_mapping

        self.assertIn("RMSNorm", result_mapping, "RMSNorm should be in mapping")
        backends = list(result_mapping["RMSNorm"].keys())

        # Assert only CUDA is present, not ROCm
        self.assertIn("cuda", backends, "CUDA backend should be registered")
        self.assertNotIn("rocm", backends, "ROCm backend should NOT be registered on CUDA device")

    def test_single_device_mapping_still_works(self):
        """
        Test that single-device mappings continue to work as expected.
        """
        kernel_mapping = {"RMSNorm": "kernels-community/layer_norm:LlamaRMSNorm"}

        kernel_config = KernelConfig(kernel_mapping)

        # Create a mock model
        mock_model = MagicMock()
        mock_model.training = False

        mock_param = MagicMock()
        mock_param.device.type = "cuda"
        mock_model.parameters.return_value = iter([mock_param])

        mock_layer = MagicMock()
        mock_layer.kernel_layer_name = "RMSNorm"
        mock_model.named_modules.return_value = [("layers.0", mock_layer)]
        kernel_config.create_compatible_mapping(mock_model)

        result_mapping = kernel_config.kernel_mapping
        self.assertIn("RMSNorm", result_mapping, "RMSNorm should be in mapping")
