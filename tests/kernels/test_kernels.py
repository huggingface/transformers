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
import os
import tempfile
import types
from unittest.mock import MagicMock, patch

import torch
from huggingface_hub import snapshot_download

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig, ViTImageProcessor
from transformers.image_processing_backends import _resize_normalize_kernel, _resize_normalize_ragged_kernel
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.integrations.hub_kernels import (
    _HUB_KERNEL_MAPPING,
    _KERNEL_MODULE_MAPPING,
    _PROCESSING_KERNEL_ADAPTERS,
    is_kernel,
    lazy_load_kernel,
    load_and_register_attn_kernel,
    run_processing_kernel,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.monkey_patching import clear_patch_mapping, get_patch_mapping, register_patch_mapping
from transformers.testing_utils import (
    TestCasePlus,
    cleanup,
    require_kernels,
    require_rocm,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_kernels_available
from transformers.utils.kernel_config import add_to_mapping_local


if is_kernels_available():
    from kernels import Device, LocalLayerRepository, Mode, kernelize

    import transformers.integrations.hub_kernels as hub_kernels_pkg


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
                except Exception as e:
                    print(f"Could not delete attribute {attr}: {e}")

        # Clear any temporary kernel module cache entries populated by tests
        try:
            keys_to_remove = [
                k for k, v in list(_KERNEL_MODULE_MAPPING.items()) if v is None or isinstance(v, types.ModuleType)
            ]
            for k in keys_to_remove:
                _KERNEL_MODULE_MAPPING.pop(k, None)
        except Exception as e:
            print(f"Could not clear kernel module cache: {e}")

    def setUp(self):
        self._pre_test_patch_mapping = get_patch_mapping()

    def tearDown(self):
        # Restore monkey patch state to avoid leaking kernel patches across tests.
        clear_patch_mapping()
        if self._pre_test_patch_mapping:
            register_patch_mapping(self._pre_test_patch_mapping)
        # Free accelerator memory/cache and trigger GC
        cleanup(torch_device, gc_collect=True)

    @require_torch_accelerator
    def test_forward(self):
        tokenized_input = self.tokenizer(self.input, return_tensors="pt").input_ids.to(self.model_kernelized.device)
        output_ = self.model_kernelized.generate(tokenized_input, max_new_tokens=10, do_sample=False)
        output = self.tokenizer.decode(output_[0], skip_special_tokens=True)

        self.EXPECTED_OUTPUT = set()
        self.EXPECTED_OUTPUT.add("Hello, I'm looking for a reliable and trustworthy online")
        self.EXPECTED_OUTPUT.add("Hello! I'm excited to be a part of this")

        self.assertTrue(output in self.EXPECTED_OUTPUT)

    @require_rocm
    def test_rocm_rotary_kernel_forward_matches_baseline(self):
        """
        Regression test for the ROCm `rotary_pos_emb` function kernel (`kernels-community/aiter-rope`).

        On ROCm, `use_kernels=True` dispatches `apply_rotary_pos_emb` to the `aiter-rope` shim. A stale shim
        (e.g. the dropped `position_ids` signature mismatch fixed in
        https://github.com/huggingface/transformers/pull/46810) only blows up at runtime here, so comparing the
        kernelized forward against the non-kernelized baseline catches such breakages.
        """
        tokenized_input = self.tokenizer(self.input, return_tensors="pt").input_ids.to(torch_device)
        with torch.no_grad():
            kernelized_out = self.model_kernelized(tokenized_input).logits
            baseline_out = self.model_not_kernelized(tokenized_input).logits

        torch.testing.assert_close(baseline_out, kernelized_out, atol=1e-3, rtol=1e-3)

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
        kernel_config = KernelConfig(kernel_mapping={"RMSNorm": "kernels-community/layer-norm:LlamaRMSNorm"})
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

    def test_kernels_mapping_explicit_version(self):
        kernel_config = KernelConfig(
            kernel_mapping={"RMSNorm": ("kernels-community/layer-norm:LlamaRMSNorm", {"version": 1})}
        )
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

    @require_torch_accelerator
    def test_kernel_fusion(self):
        model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
        kernel_config = KernelConfig(
            {
                (
                    ("RMSNorm", "model.layers.*.post_attention_layernorm"),
                    ("MLP", "model.layers.*.mlp"),
                ): (
                    "AntonV/dummy-rmsnorm-mlp-with-transformations-and-init:RMSNormMLP",
                    {"revision": "d582b66bea8e567dd06e683eca611648cfe53a7b", "trust_remote_code": True},
                ),
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")

        baseline = AutoModelForCausalLM.from_pretrained(model_id, use_kernels=True, device_map=torch_device)
        baseline.eval()
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            baseline_out = baseline(**inputs).logits
        del baseline

        fused = AutoModelForCausalLM.from_pretrained(
            model_id, use_kernels=True, kernel_config=kernel_config, device_map=torch_device
        )
        fused.eval()
        with torch.no_grad():
            fused_out = fused(**inputs).logits

        torch.testing.assert_close(baseline_out, fused_out, atol=1e-4, rtol=1e-4)

        decoder_layers = [
            (name, m)
            for name, m in fused.named_modules()
            if hasattr(m, "post_attention_layernorm") and hasattr(m, "mlp")
        ]
        self.assertTrue(len(decoder_layers) > 0, "No decoder layers found")
        for name, layer in decoder_layers:
            self.assertIsInstance(
                layer.mlp,
                torch.nn.Identity,
                f"{name}.mlp should be nn.Identity after fusion",
            )
            self.assertTrue(
                hasattr(layer.post_attention_layernorm, "kernel_layer_name")
                or hasattr(type(layer.post_attention_layernorm), "kernel_layer_name"),
                f"{name}.post_attention_layernorm should carry kernel_layer_name after fusion",
            )

        del fused

    @require_torch_accelerator
    def test_kernel_replacement_with_layout(self):
        model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
        kernel_config = KernelConfig(
            {
                "RMSNorm": (
                    "AntonV/dummy-rmsnorm-kernel-with-init:CustomRMSNorm",
                    {"revision": "e5dcd4fe743b81fa2b065964ef9a108107496c4f", "trust_remote_code": True},
                )
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")

        baseline = AutoModelForCausalLM.from_pretrained(model_id, use_kernels=True, device_map=torch_device)
        baseline.eval()
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        original_rmsnorm_cls = type(next(m for m in baseline.modules() if "RMSNorm" in type(m).__name__))
        with torch.no_grad():
            baseline_out = baseline(**inputs).logits
        del baseline

        model = AutoModelForCausalLM.from_pretrained(
            model_id, use_kernels=True, kernel_config=kernel_config, device_map=torch_device
        )
        model.eval()
        with torch.no_grad():
            model_out = model(**inputs).logits

        torch.testing.assert_close(baseline_out, model_out, atol=1e-4, rtol=1e-4)

        replaced = [m for m in model.modules() if hasattr(type(m), "kernel_layer_name")]
        self.assertTrue(len(replaced) > 0, "No replaced kernel layout modules found")
        for m in replaced:
            self.assertNotIsInstance(m, original_rmsnorm_cls)

        del model

    def test_faulty_fusion_incomplete_pattern(self):
        model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
        # "layers.*.post_attention_layernorm" is missing the leading "model." segment.
        # re.fullmatch("layers.\w+", "model.layers.0") returns None, so no module
        # is ever matched and the function raises ValueError.
        kernel_config = KernelConfig(
            {
                (
                    ("RMSNorm", "layers.*.post_attention_layernorm"),
                    ("MLP", "layers.*.mlp"),
                ): (
                    "AntonV/dummy-rmsnorm-mlp-with-transformations-and-init:RMSNormMLP",
                    {"revision": "d582b66bea8e567dd06e683eca611648cfe53a7b", "trust_remote_code": True},
                ),
            }
        )
        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_kernels=True,
                kernel_config=kernel_config,
                device_map=torch_device,
            )

    def test_faulty_kernel_mapping_layer_name(self):
        kernel_config = KernelConfig(kernel_mapping={"RMSNorm1": "kernels-community/layer-norm:LlamaRMSNorm"})
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
class TestKernelsEnv(TestCasePlus):
    def test_disable_hub_kernels(self):
        import importlib

        original_state = hub_kernels_pkg.__dict__.copy()

        try:
            with patch.dict(os.environ, {"USE_HUB_KERNELS": "OFF"}):
                importlib.reload(hub_kernels_pkg)
                self.assertFalse(hub_kernels_pkg._kernels_enabled)
        finally:
            hub_kernels_pkg.__dict__.clear()
            hub_kernels_pkg.__dict__.update(original_state)

    def test_enable_hub_kernels(self):
        import importlib

        original_state = hub_kernels_pkg.__dict__.copy()

        try:
            with patch.dict(os.environ, {"USE_HUB_KERNELS": "ON"}):
                importlib.reload(hub_kernels_pkg)
                self.assertTrue(hub_kernels_pkg._kernels_enabled)
        finally:
            hub_kernels_pkg.__dict__.clear()
            hub_kernels_pkg.__dict__.update(original_state)


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
        sentinel = types.ModuleType("sentinel_kernel_module")

        def fake_get_kernel(repo_id, revision=None, version=None, allow_all_kernels=False):
            self.assertIn(repo_id, {"kernels-community/causal-conv1d"})
            self.assertFalse(allow_all_kernels)
            return sentinel

        patched_module_mapping = copy.copy(_KERNEL_MODULE_MAPPING)
        patched_module_mapping.pop("causal-conv1d", None)

        with patch.dict(
            lazy_load_kernel.__globals__,
            {
                "_KERNEL_MODULE_MAPPING": patched_module_mapping,
                "get_kernel": fake_get_kernel,
                "ALLOW_ALL_KERNELS": False,
            },
        ):
            mod1 = lazy_load_kernel("causal-conv1d", mapping=patched_module_mapping)
            self.assertIs(mod1, sentinel)

            mod2 = lazy_load_kernel("causal-conv1d", mapping=patched_module_mapping)
            self.assertIs(mod2, sentinel)

    def test_lazy_load_kernel_unknown(self):
        name = "unknown-kernel-name"
        _KERNEL_MODULE_MAPPING.pop(name, None)
        mod = lazy_load_kernel(name)
        self.assertIsNone(mod)
        self.assertIn(name, _KERNEL_MODULE_MAPPING)
        # Cleanup cache entry to avoid growth across tests
        _KERNEL_MODULE_MAPPING.pop(name, None)

    def test_lazy_load_kernel_version(self):
        name = "causal-conv1d"
        version_spec = ">=0.0.4,<0.1.0"

        sentinel_mod = types.ModuleType("sentinel_kernel_module")
        call_count = {"n": 0}

        def fake_get_kernel(repo_id, revision=None, version=None, allow_all_kernels=False):
            call_count["n"] += 1
            self.assertEqual(repo_id, "kernels-community/causal-conv1d")
            self.assertIsNone(revision)
            self.assertEqual(version, version_spec)
            self.assertFalse(allow_all_kernels)
            return sentinel_mod

        patched_hub_mapping = copy.deepcopy(_HUB_KERNEL_MAPPING)
        patched_hub_mapping[name] = {
            "repo_id": "kernels-community/causal-conv1d",
            "version": version_spec,
        }

        patched_module_mapping = copy.copy(_KERNEL_MODULE_MAPPING)
        patched_module_mapping.pop(name, None)

        with patch.dict(
            lazy_load_kernel.__globals__,
            {
                "_HUB_KERNEL_MAPPING": patched_hub_mapping,
                "_KERNEL_MODULE_MAPPING": patched_module_mapping,
                "get_kernel": fake_get_kernel,
                "ALLOW_ALL_KERNELS": False,
            },
        ):
            mod1 = lazy_load_kernel(name, mapping=patched_module_mapping)
            mod2 = lazy_load_kernel(name, mapping=patched_module_mapping)

            self.assertIs(mod1, sentinel_mod)
            self.assertIs(mod2, sentinel_mod)
            self.assertEqual(call_count["n"], 1)


@require_kernels
class TestAttentionKernelRegistration(TestCasePlus):
    def test_trust_remote_code_for_attention_kernels(self):
        """
        Test that using an untrusted kernel (any repo outside `kernels-community`) as attention requires
        passing an expplicit `allow_all_kernels=True`
        """
        from transformers import LlamaConfig, LlamaModel

        config = LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = LlamaModel(copy.deepcopy(config))
        untrusted_kernel = "untrusted/flash_attention_2"
        trusted_kernel = "kernels-community/flash-attn2"

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            # Test that an untrusted kernel will raise an error without the flag
            with self.assertRaisesRegex(
                ValueError,
                "Kernel repository 'untrusted/flash_attention_2' could not verify publisher trust status. Set trust_remote_code=True to allow loading kernels from untrusted sources.",
            ):
                _ = LlamaModel.from_pretrained(tmpdirname, attn_implementation=untrusted_kernel)

            def dummy_lazy_import(*args, **kwargs):
                pass

            # Test that it works with the flag - though the repo does not exist, so patch the dispatch
            with patch("transformers.modeling_utils.lazy_import_flash_attention", dummy_lazy_import):
                model = LlamaModel.from_pretrained(
                    tmpdirname, attn_implementation=untrusted_kernel, allow_all_kernels=True
                )
                self.assertEqual(model.config._attn_implementation, untrusted_kernel)

            # Test that a trusted kernel does not need trust_remote_code
            model = LlamaModel.from_pretrained(tmpdirname, attn_implementation=trusted_kernel)
            self.assertEqual(model.config._attn_implementation, trusted_kernel)

    def test_load_and_register_flash_attn_like_kernel(self):
        kernel_obj = types.SimpleNamespace(flash_attn_varlen_func=lambda *a, **k: None)

        with (
            patch("transformers.integrations.hub_kernels.get_kernel", return_value=kernel_obj),
            patch("transformers.modeling_flash_attention_utils.lazy_import_flash_attention", return_value=None),
        ):
            attn_impl = "org/model"
            load_and_register_attn_kernel(attn_impl)
            self.assertIn(attn_impl, ALL_ATTENTION_FUNCTIONS.valid_keys())
            # Cleanup registration to avoid leaking functions across tests
            try:
                ALL_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception as e:
                print(f"Could not clean up `ALL_ATTENTION_FUNCTIONS`: {e}")
            try:
                ALL_MASK_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception as e:
                print(f"Could not clean up `ALL_MASK_ATTENTION_FUNCTIONS`: {e}")

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
            except Exception as e:
                print(f"Could not clean up `ALL_ATTENTION_FUNCTIONS`: {e}")
            try:
                ALL_MASK_ATTENTION_FUNCTIONS.pop(attn_impl, None)
            except Exception as e:
                print(f"Could not clean up `ALL_MASK_ATTENTION_FUNCTIONS`: {e}")

    def test_add_to_mapping_local(self):
        repo_path = "/abs/path/kernel"
        compatible_mapping = {}
        add_to_mapping_local("RMSNorm", "cuda", f"{repo_path}:LlamaRMSNorm", Mode.INFERENCE, compatible_mapping)

        repo = compatible_mapping["RMSNorm"]["cuda"][Mode.INFERENCE]
        self.assertIsInstance(repo, LocalLayerRepository)
        self.assertEqual(repo.layer_name, "LlamaRMSNorm")

        with self.assertRaisesRegex(ValueError, "Only cuda, rocm, xpu, npu, neuron and tpu devices supported"):
            add_to_mapping_local("RMSNorm", "cpu", f"{repo_path}:LlamaRMSNorm", Mode.INFERENCE, compatible_mapping)

    @slow
    @require_torch_accelerator
    def test_add_to_mapping_local_then_load(self):
        repo_path = snapshot_download("kernels-community/layer-norm")
        compatible_mapping = {}
        add_to_mapping_local("RMSNorm", "cuda", f"{repo_path}:LlamaRMSNorm", Mode.INFERENCE, compatible_mapping)

        repo = compatible_mapping["RMSNorm"]["cuda"][Mode.INFERENCE]
        self.assertIsInstance(repo, LocalLayerRepository)
        self.assertEqual(repo.layer_name, "LlamaRMSNorm")

        layer_cls = repo.load()
        self.assertTrue(issubclass(layer_cls, torch.nn.Module))


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
            except Exception as e:
                print(f"Could not delete model: {e}")

    def tearDown(self):
        # Free accelerator memory/cache and trigger GC
        cleanup(torch_device, gc_collect=True)

    def test_setting_use_kernels_twice_does_not_rekernelize(self):
        with (
            patch.object(hub_kernels_pkg, "register_kernel_mapping_transformers") as mock_register,
            patch.object(hub_kernels_pkg, "_kernels_kernelize") as mock_kernelize,
        ):
            self.model.use_kernels = True

            self.assertTrue(self.model.use_kernels)
            # Check that both registraton and the underlying kernelize call happened
            mock_register.assert_called_once_with()
            self.assertEqual(mock_kernelize.call_count, 1)

            self.model.use_kernels = True

            mock_register.assert_called_once_with()
            self.assertEqual(mock_kernelize.call_count, 1)

    def test_train_eval_calls_kernelize_with_correct_mode(self):
        last_modes = []

        def spy_kernelize(model, device=None, mode=None):
            last_modes.append(mode)

        with patch.object(hub_kernels_pkg, "_kernels_kernelize", side_effect=spy_kernelize):
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
                "cuda": "kernels-community/layer-norm:LlamaRMSNorm",
                "rocm": "kernels-community/layer-norm:LlamaRMSNorm",
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
        kernel_mapping = {"RMSNorm": "kernels-community/layer-norm:LlamaRMSNorm"}

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


class RecordingKernel:
    """Stand-in for a loaded kernel module: records the call and returns a fixed output."""

    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def resize_normalize(self, images, size, image_mean, image_std, **kwargs):
        self.calls.append({"images": images, "size": size, "image_mean": image_mean, "image_std": image_std, **kwargs})
        return self.output


class TestProcessingKernels(TestCasePlus):
    """Dispatch and gating of processing kernels. Runs on CPU, `kernels` is not needed."""

    def setUp(self):
        super().setUp()
        self.images = [torch.randint(0, 255, (3, 40, 60), dtype=torch.uint8) for _ in range(2)]
        self.kwargs = {
            "do_resize": True,
            "size": SizeDict(height=8, width=8),
            "resample": PILImageResampling.BICUBIC,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }

    def run_adapter(self, output=None, images=None, **overrides):
        kernel = RecordingKernel(output=output)
        images = self.images if images is None else images
        # The kernel is CUDA only; pretend CPU is its device so the adapter can be tested without a GPU.
        with patch("transformers.image_processing_backends._KERNEL_DEVICE_TYPE", "cpu"):
            result = _resize_normalize_kernel(kernel, images, **{**self.kwargs, **overrides})
        return kernel, result

    def test_registration_records_repo_and_adapter(self):
        self.assertIs(_PROCESSING_KERNEL_ADAPTERS["resize_normalize"], _resize_normalize_kernel)
        self.assertIs(_PROCESSING_KERNEL_ADAPTERS["resize_normalize_ragged"], _resize_normalize_ragged_kernel)
        for name in ("resize_normalize", "resize_normalize_ragged"):
            self.assertEqual(_HUB_KERNEL_MAPPING[name]["repo_id"], "Molbap/kernel_image_resize")
            self.assertEqual(_HUB_KERNEL_MAPPING[name]["revision"], "v1.0.0")

    def test_kernel_outside_kernels_community_is_not_trusted_by_default(self):
        mapping = {}
        with patch.dict(_HUB_KERNEL_MAPPING, {"private_op": {"repo_id": "someone/private-kernel", "version": 1}}):
            self.assertIsNone(lazy_load_kernel("private_op", mapping))
            # Not cached: the answer changes inside `allow_all_hub_kernels()`, so it must be asked again.
            self.assertEqual(mapping, {})
            sentinel = types.ModuleType("trusted_kernel_module")
            with (
                patch("transformers.integrations.hub_kernels.ALLOW_ALL_KERNELS", True),
                patch("transformers.integrations.hub_kernels.is_kernels_available", return_value=True),
                patch("transformers.integrations.hub_kernels.get_kernel", return_value=sentinel),
            ):
                self.assertIs(lazy_load_kernel("private_op", mapping), sentinel)

    def test_ragged_adapter_translates_per_image_targets(self):
        kernel = MagicMock()
        kernel.resize_normalize_ragged_output.return_value = [torch.zeros(3, 8, 8), torch.zeros(3, 4, 4)]
        with patch("transformers.image_processing_backends._KERNEL_DEVICE_TYPE", "cpu"):
            result = _resize_normalize_ragged_kernel(
                kernel,
                self.images,
                [(8, 8), (4, 4)],
                **{key: self.kwargs[key] for key in self.kwargs if key != "size"},
            )
        self.assertEqual(len(result), 2)
        arguments = kernel.resize_normalize_ragged_output.call_args[0]
        self.assertEqual(arguments[1], [(8, 8), (4, 4)])
        self.assertEqual(arguments[2], [0.5, 0.5, 0.5])
        self.assertEqual(arguments[4], 1 / 255)
        self.assertEqual(arguments[5], "bicubic")

    def test_ragged_adapter_falls_back(self):
        kernel = MagicMock()
        with patch("transformers.image_processing_backends._KERNEL_DEVICE_TYPE", "cpu"):
            # One target per image is required, and the images must be uint8 CHW on one device.
            self.assertIsNone(_resize_normalize_ragged_kernel(kernel, self.images, [(8, 8)], **self.kwargs))
            self.assertIsNone(
                _resize_normalize_ragged_kernel(
                    kernel, [image.float() for image in self.images], [(8, 8), (4, 4)], **self.kwargs
                )
            )
        self.assertIsNone(_resize_normalize_ragged_kernel(kernel, self.images, [(8, 8), (4, 4)], **self.kwargs))
        kernel.resize_normalize_ragged_output.assert_not_called()

    def test_run_processing_kernel_calls_adapter(self):
        sentinel = types.ModuleType("sentinel_kernel_module")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.dict(
                run_processing_kernel.__globals__,
                {
                    "_kernels_enabled": True,
                    "_PROCESSING_KERNEL_ADAPTERS": {"dummy_op": lambda kernel, value: (kernel, value)},
                    "lazy_load_kernel": lambda name: sentinel,
                },
            ),
        ):
            self.assertEqual(run_processing_kernel("dummy_op", 3), (sentinel, 3))

    def test_run_processing_kernel_falls_back(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            # Unknown op, `USE_HUB_KERNELS=0`, and a kernel that cannot be loaded all keep the default path.
            self.assertIsNone(run_processing_kernel("not_a_registered_op"))
            with patch.dict(run_processing_kernel.__globals__, {"_kernels_enabled": False}):
                self.assertIsNone(run_processing_kernel("resize_normalize", self.images, **self.kwargs))
            with patch.dict(run_processing_kernel.__globals__, {"lazy_load_kernel": lambda name: None}):
                self.assertIsNone(run_processing_kernel("resize_normalize", self.images, **self.kwargs))

    def test_run_processing_kernel_needs_accelerator(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.dict(run_processing_kernel.__globals__, {"lazy_load_kernel": self.fail}):
                self.assertIsNone(run_processing_kernel("resize_normalize", self.images, **self.kwargs))

    def test_adapter_translates_square_resize(self):
        kernel, result = self.run_adapter(output=torch.zeros(2, 3, 8, 8))
        self.assertEqual(len(kernel.calls), 1)
        call = kernel.calls[0]
        self.assertEqual(call["size"], (8, 8))
        self.assertIsNone(call["crop_size"])
        self.assertEqual(call["resize_mode"], "square")
        self.assertEqual(call["resample"], "bicubic")
        self.assertEqual(call["rescale_factor"], 1 / 255)
        self.assertEqual(call["image_mean"], [0.5, 0.5, 0.5])
        # Same output shape as the default path: a list of images, stacked only when tensors are requested.
        self.assertEqual([image.shape for image in result["pixel_values"]], [(3, 8, 8)] * 2)
        _, stacked = self.run_adapter(output=torch.zeros(2, 3, 8, 8), return_tensors="pt")
        self.assertEqual(stacked["pixel_values"].shape, (2, 3, 8, 8))

    def test_adapter_translates_shortest_edge_resize_with_crop(self):
        kernel, _ = self.run_adapter(
            output=torch.zeros(2, 3, 6, 6),
            size=SizeDict(shortest_edge=8),
            do_center_crop=True,
            crop_size=SizeDict(height=6, width=6),
        )
        call = kernel.calls[0]
        self.assertEqual(call["size"], 8)
        self.assertEqual(call["crop_size"], (6, 6))
        self.assertEqual(call["resize_mode"], "shortest_edge")

    def test_adapter_scalar_normalization_stats(self):
        kernel, _ = self.run_adapter(output=torch.zeros(2, 3, 8, 8), image_mean=0.5, image_std=0.5)
        self.assertEqual(kernel.calls[0]["image_mean"], [0.5, 0.5, 0.5])

    def test_adapter_skips_rescale(self):
        kernel, _ = self.run_adapter(output=torch.zeros(2, 3, 8, 8), do_rescale=False)
        self.assertEqual(kernel.calls[0]["rescale_factor"], 1.0)

    def test_adapter_falls_back_on_unsupported_arguments(self):
        unsupported = {
            "padding": {"do_pad": True, "pad_size": SizeDict(height=8, width=8)},
            "no_resize": {"do_resize": False},
            "no_normalize": {"do_normalize": False},
            "lanczos": {"resample": PILImageResampling.LANCZOS},
            "bounded_resize": {"size": SizeDict(max_height=8, max_width=8)},
            "crop_without_size": {"do_center_crop": True, "crop_size": SizeDict()},
            "crop_larger_than_resize": {"do_center_crop": True, "crop_size": SizeDict(height=12, width=12)},
            "shortest_edge_below_crop": {
                "size": SizeDict(shortest_edge=4),
                "do_center_crop": True,
                "crop_size": SizeDict(height=6, width=6),
            },
            "shortest_edge_without_crop": {"size": SizeDict(shortest_edge=8)},
            "stats_per_channel_mismatch": {"image_mean": [0.5, 0.5], "image_std": [0.5, 0.5]},
        }
        for name, overrides in unsupported.items():
            with self.subTest(name):
                kernel, result = self.run_adapter(**overrides)
                self.assertIsNone(result)
                self.assertEqual(kernel.calls, [])

    def test_adapter_falls_back_on_unsupported_images(self):
        for name, images in {
            "empty": [],
            "float": [image.float() for image in self.images],
            "batched": [image.unsqueeze(0) for image in self.images],
            "mixed_channels": [self.images[0], self.images[1][:1]],
            "nested": [[image] for image in self.images],
        }.items():
            with self.subTest(name):
                kernel, result = self.run_adapter(images=images)
                self.assertIsNone(result)
                self.assertEqual(kernel.calls, [])

    def test_adapter_falls_back_on_images_the_kernel_cannot_reach(self):
        # Images stay on CPU unless the caller passes `device`, and the kernel runs where the images are.
        kernel = RecordingKernel()
        self.assertIsNone(_resize_normalize_kernel(kernel, self.images, **self.kwargs))
        self.assertEqual(kernel.calls, [])

    def test_resize_normalize_batch_matches_with_and_without_grouping(self):
        processor = ViTImageProcessor()
        images = [torch.randint(0, 255, (3, 40, 60), dtype=torch.uint8) for _ in range(2)] + [
            torch.randint(0, 255, (3, 30, 30), dtype=torch.uint8)
        ]
        targets = [(8, 12), (8, 12), (6, 6)]
        arguments = (targets, PILImageResampling.BICUBIC, True, 1 / 255, True, (0.5,) * 3, (0.5,) * 3)
        grouped = processor.resize_normalize_batch(images, *arguments, disable_grouping=False)
        ungrouped = processor.resize_normalize_batch(images, *arguments, disable_grouping=True)
        self.assertEqual([tuple(image.shape) for image in grouped], [(3, 8, 12), (3, 8, 12), (3, 6, 6)])
        for one, other in zip(grouped, ungrouped):
            torch.testing.assert_close(one, other)

    def test_use_kernels_is_a_runtime_flag(self):
        processor = ViTImageProcessor(use_kernels=True)
        self.assertNotIn("use_kernels", processor.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            self.assertFalse(ViTImageProcessor.from_pretrained(tmp_dir).use_kernels)
            self.assertTrue(ViTImageProcessor.from_pretrained(tmp_dir, use_kernels=True).use_kernels)

    def test_default_path_is_unchanged_when_the_kernel_cannot_run(self):
        images = [torch.randint(0, 255, (3, 40, 60), dtype=torch.uint8)]
        with patch.object(torch.cuda, "is_available", return_value=False):
            with_kernels = ViTImageProcessor(use_kernels=True)(images, return_tensors="pt")
            without_kernels = ViTImageProcessor(use_kernels=False)(images, return_tensors="pt")
        torch.testing.assert_close(with_kernels["pixel_values"], without_kernels["pixel_values"])


@require_kernels
@require_torch_gpu
@slow
class TestResizeNormalizeKernel(TestCasePlus):
    """Parity of the resize + normalize kernel against the torchvision path. Needs the kernel repo to be published."""

    def test_matches_default_path(self):
        if lazy_load_kernel("resize_normalize") is None:
            self.skipTest("kernel repo of `resize_normalize` is not available")
        images = [
            torch.randint(0, 255, (3, height, width), dtype=torch.uint8, device=torch_device)
            for height, width in [(480, 640), (300, 300), (1024, 768)]
        ]
        kernel_outputs = []

        def recording_adapter(*args, **kwargs):
            kernel_outputs.append(_resize_normalize_kernel(*args, **kwargs))
            return kernel_outputs[-1]

        with patch.dict(_PROCESSING_KERNEL_ADAPTERS, {"resize_normalize": recording_adapter}):
            with_kernels = ViTImageProcessor(use_kernels=True)(images, return_tensors="pt")["pixel_values"]
        # Without this the test would compare the default path against itself and pass on a silent fallback.
        self.assertTrue(
            kernel_outputs and kernel_outputs[0] is not None, "the adapter fell back, the kernel never ran"
        )
        without_kernels = ViTImageProcessor(use_kernels=False)(images, return_tensors="pt")["pixel_values"]
        self.assertEqual(with_kernels.shape, without_kernels.shape)
        # The kernel resizes in float where the default path resizes in uint8, so single pixels differ.
        self.assertLess((with_kernels - without_kernels).abs().mean().item(), 0.02)
