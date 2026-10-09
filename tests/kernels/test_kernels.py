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
import importlib
import inspect
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from huggingface_hub import snapshot_download
from parameterized import parameterized

import transformers
from tests.test_memory_cleanup_mixin import MemoryCleanupTestCase
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HunYuanVLImageProcessor,
    KernelConfig,
    LevitImageProcessor,
    PaddleOCRVLImageProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    Sam2ImageProcessor,
    ViTImageProcessor,
)
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_utils import PILImageResampling, SizeDict, load_image
from transformers.integrations.hub_kernels import (
    _HUB_KERNEL_MAPPING,
    _KERNEL_MODULE_MAPPING,
    is_kernel,
    lazy_load_kernel,
    load_and_register_attn_kernel,
    use_kernel_func_from_hub_with_fallback,
)
from transformers.integrations.hub_processing_kernels import (
    _PROCESSING_KERNEL_ADAPTERS,
    _connected_component_areas_kernel,
    _resize_normalize_kernel,
    _resize_normalize_patchify_kernel,
    run_processing_kernel,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.monkey_patching import clear_patch_mapping, get_patch_mapping, register_patch_mapping
from transformers.testing_utils import (
    TestCasePlus,
    require_kernels,
    require_rocm,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_kernels_available
from transformers.utils.kernel_config import add_to_mapping_local
from transformers.video_processing_utils import BaseVideoProcessor


if is_kernels_available():
    from kernels import Device, LocalLayerRepository, Mode, kernelize

    import transformers.integrations.hub_kernels as hub_kernels_pkg


@require_kernels
@slow
class TestHubKernels(MemoryCleanupTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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
        # Clear any temporary kernel module cache entries populated by tests; the mixin drops the two models.
        keys_to_remove = [
            k for k, v in list(_KERNEL_MODULE_MAPPING.items()) if v is None or isinstance(v, types.ModuleType)
        ]
        for k in keys_to_remove:
            _KERNEL_MODULE_MAPPING.pop(k, None)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self._pre_test_patch_mapping = get_patch_mapping()

    def tearDown(self):
        # Restore monkey patch state to avoid leaking kernel patches across tests.
        clear_patch_mapping()
        if self._pre_test_patch_mapping:
            register_patch_mapping(self._pre_test_patch_mapping)
        super().tearDown()

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

    def test_kernels_mapping_functions_registration(self):
        kernel_config = KernelConfig(
            kernel_mapping={"rotary_pos_emb": ("kernels-community/rotary:apply_rotary_transformers", {"version": 2})}
        )

        # Functions would previously raise if not properly handled
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct", use_kernels=True, device_map=torch_device, kernel_config=kernel_config
        )

        # Sanity checks making sure it is really been found / registered under the model
        self.assertIn("rotary_pos_emb", model.kernel_config.registered_layer_names.values())
        self.assertTrue(any(name.endswith(".rotary_pos_emb") for name in model.kernel_config.registered_layer_names))

        del model

    def test_kernels_mapping_no_inherit(self):
        kernel_config = KernelConfig(
            kernel_mapping={
                "RMSNorm": (
                    "kernels-community/layer-norm:LlamaRMSNorm",
                    {"version": 1},
                )
            },
            # Force to only inherit the mapping ^
            inherit_mapping=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct", use_kernels=True, device_map=torch_device, kernel_config=kernel_config
        )
        first_rms_norm = model.model.layers[0].input_layernorm
        first_self_attn = model.model.layers[0].self_attn

        # RoPE should still be registered under attn
        self.assertIn("rotary_pos_emb", getattr(first_self_attn, "_kernel_funcs", {}))
        first_rope = first_self_attn._kernel_funcs["rotary_pos_emb"]

        # Check kernelization by fwd matching
        self.assertIsNot(first_rms_norm.forward.__func__, type(first_rms_norm).forward)  # exchanged
        self.assertIs(first_rope.forward.__func__, type(first_rope).forward)  # not exchanged

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

        patched_hub_mapping = copy.deepcopy(_HUB_KERNEL_MAPPING)
        patched_hub_mapping["causal-conv1d"] = {
            "repo_id": "kernels-community/causal-conv1d",
            "version": 1,
        }

        patched_module_mapping = copy.copy(_KERNEL_MODULE_MAPPING)
        patched_module_mapping.pop("causal-conv1d", None)

        with patch.dict(
            lazy_load_kernel.__globals__,
            {
                "_HUB_KERNEL_MAPPING": patched_hub_mapping,
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

    def _make_fallback_func(self, torch_function):
        """Decorate `torch_function` with a fake original package installed, as if e.g. `fla` were available."""
        package_name = "fake_kernel_package"
        package = types.ModuleType(package_name)
        # Same name as the torch reference, but a different (recognisable) result and an extra kernel-only kwarg.
        package.fake_op = lambda hidden_states, weight, cu_seqlens=None: hidden_states * weight * 10

        with patch.dict(sys.modules, {package_name: package}):
            return use_kernel_func_from_hub_with_fallback("fake_op", package_name)(torch_function)

    def test_export_falls_back_to_torch_implementation(self):
        """Tests if a function decorated with use_kernel_func_from_hub_with_fallback is traced by `torch.export` as the
        original torch function and not the installed package's function, since the latter is generally not exportable.
        """

        def fake_op(hidden_states, weight, **kwargs):
            return hidden_states + weight

        decorated = self._make_fallback_func(fake_op)

        class Wrapper(torch.nn.Module):
            def forward(self, hidden_states, weight):
                return decorated(hidden_states, weight)

        inputs = (torch.ones(4), torch.full((4,), 3.0))
        package_result, torch_result = torch.full((4,), 30.0), torch.full((4,), 4.0)

        # Eager and `torch.compile` keep the package implementation ...
        self.assertTrue(torch.equal(decorated(*inputs), package_result))
        self.assertTrue(torch.equal(torch.compile(Wrapper(), fullgraph=True)(*inputs), package_result))
        # ... only `torch.export` swaps in the torch one.
        exported = torch.export.export(Wrapper(), inputs)
        self.assertTrue(torch.equal(exported.module()(*inputs), torch_result))

    def test_fallback_resolves_function_from_package_root(self):
        """The package-root implementation takes precedence over the nested-module fallback."""
        package = types.ModuleType("optional_backend")
        optimized_function = MagicMock(return_value="optimized")
        package.optimized_function = optimized_function
        torch_function = MagicMock(return_value="torch")

        with (
            patch(
                "transformers.integrations.hub_kernels.use_kernel_forward_from_hub",
                return_value=lambda function: function,
            ),
            patch(
                "transformers.integrations.hub_kernels.importlib.import_module",
                return_value=package,
            ) as import_module,
        ):
            wrapped = use_kernel_func_from_hub_with_fallback(
                func_name="optimized_function",
                package="optional_backend",
                internal_path="ops.kernel",
            )(torch_function)

        self.assertEqual(wrapped(), "optimized")
        import_module.assert_called_once_with("optional_backend")

    @parameterized.expand(
        [
            (
                "explicit_path",
                "optimized_function",
                "optional_backend",
                "ops.kernel",
                "optional_backend.ops.kernel",
            ),
            (
                "mapped_path",
                "chunk_gated_delta_rule",
                "fla",
                None,
                "fla.ops.gated_delta_rule",
            ),
        ]
    )
    def test_fallback_imports_nested_module(
        self,
        case_name,
        func_name,
        package_name,
        internal_path,
        internal_module_name,
    ):
        """A nested implementation can be resolved from an explicit or registered module path."""
        package = types.ModuleType(package_name)
        internal_module = types.ModuleType(internal_module_name)
        optimized_function = MagicMock(return_value="optimized")
        setattr(internal_module, func_name, optimized_function)
        torch_function = MagicMock(return_value="torch")

        with (
            patch(
                "transformers.integrations.hub_kernels.use_kernel_forward_from_hub",
                return_value=lambda function: function,
            ),
            patch(
                "transformers.integrations.hub_kernels.importlib.import_module",
                side_effect=[package, internal_module],
            ) as import_module,
        ):
            wrapped = use_kernel_func_from_hub_with_fallback(
                func_name=func_name,
                package=package_name,
                internal_path=internal_path,
            )(torch_function)

        self.assertEqual(wrapped(), "optimized")
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            [package_name, internal_module_name],
        )

    def test_fallback_uses_torch_when_nested_module_is_missing(self):
        """The reference implementation remains available when the nested module cannot be imported."""
        package = types.ModuleType("optional_backend")
        torch_function = MagicMock(return_value="torch")

        with (
            patch(
                "transformers.integrations.hub_kernels.use_kernel_forward_from_hub",
                return_value=lambda function: function,
            ),
            patch(
                "transformers.integrations.hub_kernels.importlib.import_module",
                side_effect=[package, ImportError],
            ),
        ):
            wrapped = use_kernel_func_from_hub_with_fallback(
                func_name="optimized_function",
                package="optional_backend",
                internal_path="ops.kernel",
            )(torch_function)

        self.assertEqual(wrapped(), "torch")


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
                "Kernel repository 'untrusted/flash_attention_2' could not verify publisher trust status. Set trust_remote_code=True or add the repository ID to the trust_remote_code allowlist to allow loading kernels from untrusted sources.",
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
class TestUseKernelsLifecycle(MemoryCleanupTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_id = "unsloth/Llama-3.2-1B-Instruct"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, use_kernels=False, device_map=torch_device)

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


def connected_component_areas_reference(regions):
    """Area of the 8-connected component of every pixel, by label propagation. Slow, for tests only."""
    flat_regions = regions.flatten(1)
    labels = torch.arange(1, flat_regions.shape[1] + 1, dtype=torch.float32, device=regions.device)
    labels = (labels * flat_regions).view(regions.shape)
    while True:
        propagated = torch.nn.functional.max_pool2d(labels, kernel_size=3, stride=1, padding=1) * regions
        if torch.equal(propagated, labels):
            break
        labels = propagated
    areas = torch.zeros_like(labels, dtype=torch.int32)
    for index in range(labels.shape[0]):
        _, inverse, counts = labels[index].unique(return_inverse=True, return_counts=True)
        areas[index] = counts[inverse].to(torch.int32) * regions[index]
    return areas


def patchify_kernel_layout_reference(
    frames,
    target_sizes,
    items,
    resample,
    rescale_factor,
    image_mean,
    image_std,
    patch_size,
    merge_size,
    temporal_patch_size,
):
    """What `resize_normalize_patchify` writes for frames that are already at their target size."""
    mean = torch.tensor(image_mean).view(-1, 1, 1)
    std = torch.tensor(image_std).view(-1, 1, 1)
    item_patches = []
    for item in items:
        padded_item = list(item) + [item[-1]] * (-len(item) % temporal_patch_size)
        video = (torch.stack([frames[index] for index in padded_item]).float() * rescale_factor - mean) / std
        frame_count, channels, height, width = video.shape
        patches = video.reshape(
            frame_count // temporal_patch_size,
            temporal_patch_size,
            channels,
            height // (patch_size * merge_size),
            merge_size,
            patch_size,
            width // (patch_size * merge_size),
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        item_patches.append(patches.reshape(-1, channels * temporal_patch_size * patch_size * patch_size))
    return torch.cat(item_patches)


def processors_calling_the_patchify_kernel():
    """Torchvision image and video processor classes whose module mentions the `resize_normalize_patchify` kernel."""
    classes = []
    for path in sorted(Path(transformers.__file__).parent.glob("models/*/*_processing_*.py")):
        if "resize_normalize_patchify" not in path.read_text() or path.stem.startswith("modular_"):
            continue
        module = importlib.import_module(f"transformers.models.{path.parent.name}.{path.stem}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and issubclass(cls, TorchvisionBackend):
                classes.append(cls)
    return classes


class ReferenceConnectedComponentsKernel:
    """Stand-in for `kernels-community/cv-utils`: records the input of `cc_2d` and answers with the reference."""

    def __init__(self):
        self.inputs = []

    def cc_2d(self, inputs, get_counts):
        self.inputs.append(inputs)
        return None, connected_component_areas_reference(inputs.bool())


class TestProcessingKernels(TestCasePlus):
    """Dispatch and gating of processing kernels. Runs on CPU, `kernels` is not needed."""

    def setUp(self):
        super().setUp()
        # One 8x8 object with a 2x2 hole, and a 1 pixel island away from it.
        self.mask_logits = torch.full((1, 1, 16, 16), -1.0)
        self.mask_logits[..., 2:10, 2:10] = 1.0
        self.mask_logits[..., 5:7, 5:7] = -1.0
        self.mask_logits[..., 13, 13] = 1.0

    def post_process(self, use_kernels, **kwargs):
        processor = Sam2ImageProcessor(use_kernels=use_kernels)
        return processor.post_process_masks([self.mask_logits.clone()], [(16, 16)], **kwargs)[0]

    def test_registration_records_repo_and_adapter(self):
        self.assertEqual(
            _PROCESSING_KERNEL_ADAPTERS["connected_component_areas"],
            ("cv-utils", _connected_component_areas_kernel),
        )
        self.assertEqual(_HUB_KERNEL_MAPPING["cv-utils"]["repo_id"], "kernels-community/cv-utils")

    def test_run_processing_kernel_calls_adapter(self):
        sentinel = types.ModuleType("sentinel_kernel_module")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.dict(
                _PROCESSING_KERNEL_ADAPTERS,
                {"dummy_op": ("dummy-kernel", lambda kernel, value: (kernel, value))},
                clear=True,
            ),
            patch.multiple(
                "transformers.integrations.hub_kernels",
                _kernels_enabled=True,
                lazy_load_kernel={"dummy-kernel": sentinel}.get,
            ),
        ):
            self.assertEqual(run_processing_kernel("dummy_op", 3), (sentinel, 3))

    def test_run_processing_kernel_falls_back(self):
        regions = self.mask_logits > 0
        with patch.object(torch.cuda, "is_available", return_value=True):
            # Unknown op, `USE_HUB_KERNELS=0`, and a kernel that cannot be loaded all keep the default path.
            self.assertIsNone(run_processing_kernel("not_a_registered_op"))
            with patch("transformers.integrations.hub_kernels._kernels_enabled", False):
                self.assertIsNone(run_processing_kernel("connected_component_areas", regions))
            with patch("transformers.integrations.hub_kernels.lazy_load_kernel", lambda name: None):
                self.assertIsNone(run_processing_kernel("connected_component_areas", regions))

    def test_run_processing_kernel_needs_accelerator(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch("transformers.integrations.hub_kernels.lazy_load_kernel", self.fail):
                self.assertIsNone(run_processing_kernel("connected_component_areas", self.mask_logits > 0))

    def test_connected_component_areas_adapter_pads_to_even_size(self):
        kernel = ReferenceConnectedComponentsKernel()
        regions = torch.zeros(2, 1, 5, 7, dtype=torch.bool)
        regions[:, :, 1:3, 1:4] = True
        # The kernel is CUDA only; pretend CPU is its device so the adapter can be tested without a GPU.
        with patch("transformers.integrations.hub_processing_kernels._KERNEL_DEVICE_TYPE", "cpu"):
            areas = _connected_component_areas_kernel(kernel, regions)
        self.assertEqual(tuple(kernel.inputs[0].shape), (2, 1, 6, 8))
        self.assertEqual(kernel.inputs[0].dtype, torch.uint8)
        self.assertEqual(tuple(areas.shape), (2, 1, 5, 7))
        self.assertTrue(torch.equal(areas, connected_component_areas_reference(regions)))

    def test_connected_component_areas_adapter_needs_cuda_tensors(self):
        kernel = ReferenceConnectedComponentsKernel()
        self.assertIsNone(_connected_component_areas_kernel(kernel, self.mask_logits > 0))
        self.assertEqual(kernel.inputs, [])

    def test_sam2_fills_holes_and_removes_sprinkles(self):
        with patch(
            "transformers.models.sam2.image_processing_sam2.run_processing_kernel",
            side_effect=lambda name, regions: connected_component_areas_reference(regions),
        ):
            masks = self.post_process(use_kernels=True, max_hole_area=4, max_sprinkle_area=1)
            too_small_thresholds = self.post_process(use_kernels=True, max_hole_area=3, max_sprinkle_area=0)
        expected = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        expected[..., 2:10, 2:10] = True
        self.assertTrue(torch.equal(masks, expected))
        self.assertTrue(torch.equal(too_small_thresholds, self.post_process(use_kernels=False)))

    def test_sam2_keeps_masks_without_the_kernel(self):
        default = self.post_process(use_kernels=False)
        self.assertTrue(default[0, 0, 5, 5].item() is False and default[0, 0, 13, 13].item() is True)
        self.assertTrue(torch.equal(self.post_process(False, max_hole_area=4, max_sprinkle_area=1), default))
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertTrue(torch.equal(self.post_process(True, max_hole_area=4, max_sprinkle_area=1), default))

    def run_resize_adapter(self, images=None, **overrides):
        kernel = MagicMock()
        arguments = {
            "size": SizeDict(height=8, width=8),
            "crop_size": None,
            "resample": PILImageResampling.BICUBIC,
            "rescale_factor": 1 / 255,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        images = (
            [torch.randint(0, 255, (3, 40, 60), dtype=torch.uint8) for _ in range(2)] if images is None else images
        )
        with patch("transformers.integrations.hub_processing_kernels._KERNEL_DEVICE_TYPE", "cpu"):
            result = _resize_normalize_kernel(kernel, images, **{**arguments, **overrides})
        return kernel, result

    def test_resize_adapter_translates_arguments(self):
        kernel, result = self.run_resize_adapter()
        self.assertIs(result, kernel.resize_normalize.return_value)
        arguments, keyword_arguments = kernel.resize_normalize.call_args
        self.assertEqual(arguments[1:], ((8, 8), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.assertEqual(keyword_arguments["resample"], "bicubic")
        self.assertEqual(keyword_arguments["resize_mode"], "square")
        self.assertTrue(keyword_arguments["round_to_uint8"])

        kernel, _ = self.run_resize_adapter(
            size=SizeDict(shortest_edge=8), crop_size=SizeDict(height=6, width=6), image_mean=0.5, image_std=0.5
        )
        arguments, keyword_arguments = kernel.resize_normalize.call_args
        self.assertEqual(arguments[1:], (8, [0.5] * 3, [0.5] * 3))
        self.assertEqual((keyword_arguments["resize_mode"], keyword_arguments["crop_size"]), ("shortest_edge", (6, 6)))

    def test_resize_adapter_falls_back(self):
        unsupported = {
            "lanczos": {"resample": PILImageResampling.LANCZOS},
            "crop_larger_than_resize": {"crop_size": SizeDict(height=12, width=12)},
            "empty_crop": {"crop_size": SizeDict()},
            "shortest_edge_without_crop": {"size": SizeDict(shortest_edge=8)},
            "bounded_resize": {"size": SizeDict(shortest_edge=8, longest_edge=16)},
            "stats_do_not_match_channels": {"image_mean": [0.5, 0.5], "image_std": [0.5, 0.5]},
            "float_images": {"images": [torch.rand(3, 40, 60)]},
            "mixed_channels": {
                "images": [torch.zeros(3, 4, 4, dtype=torch.uint8), torch.zeros(1, 4, 4, dtype=torch.uint8)]
            },
        }
        for name, overrides in unsupported.items():
            with self.subTest(name):
                kernel, result = self.run_resize_adapter(**overrides)
                self.assertIsNone(result)
                kernel.resize_normalize.assert_not_called()
        # Outside the patch the kernel device is CUDA, and these images are on CPU.
        kernel = MagicMock()
        self.assertIsNone(
            _resize_normalize_kernel(
                kernel,
                [torch.zeros(3, 4, 4, dtype=torch.uint8)],
                SizeDict(height=2, width=2),
                None,
                3,
                1 / 255,
                0.5,
                0.5,
            )
        )
        kernel.resize_normalize.assert_not_called()

    def test_patchify_adapter_translates_arguments(self):
        kernel = MagicMock()
        frames = [torch.zeros(3, 30, 40, dtype=torch.uint8), torch.zeros(3, 20, 20, dtype=torch.uint8)]
        with patch("transformers.integrations.hub_processing_kernels._KERNEL_DEVICE_TYPE", "cpu"):
            result = _resize_normalize_patchify_kernel(
                kernel, frames, [(28, 28), (28, 28)], [[0], [1]], 3, 1 / 255, [0.5] * 3, [0.5] * 3, 14, 2, 2
            )
        self.assertIs(result, kernel.resize_normalize_patchify.return_value)
        arguments, keyword_arguments = kernel.resize_normalize_patchify.call_args
        self.assertEqual(
            arguments[1:], ([(28, 28), (28, 28)], [[0], [1]], [0.5] * 3, [0.5] * 3, 1 / 255, "bicubic", True, 14, 2, 2)
        )
        self.assertTrue(keyword_arguments["round_to_uint8"])

    def test_qwen2_vl_uses_the_patchify_kernel_output(self):
        images = [torch.randint(0, 255, (3, 64, 96), dtype=torch.uint8)]
        pixel_values = torch.zeros(24, 3 * 2 * 14 * 14)
        with patch(
            "transformers.models.qwen2_vl.image_processing_qwen2_vl.run_processing_kernel",
            return_value=(pixel_values, [(1, 4, 6)]),
        ) as run_kernel:
            output = Qwen2VLImageProcessor(use_kernels=True)(images, return_tensors="pt")
            Qwen2VLImageProcessor(use_kernels=False)(images, return_tensors="pt")
        self.assertEqual(run_kernel.call_count, 1)
        self.assertEqual(run_kernel.call_args[0][0], "resize_normalize_patchify")
        self.assertIs(output["pixel_values"], pixel_values)
        self.assertEqual(output["image_grid_thw"].tolist(), [[1, 4, 6]])

    def test_qwen2_vl_video_groups_frames_per_video(self):
        videos = [
            torch.randint(0, 255, (3, 3, 64, 96), dtype=torch.uint8),
            torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8),
        ]
        with patch(
            "transformers.models.qwen2_vl.video_processing_qwen2_vl.run_processing_kernel",
            return_value=(torch.zeros(1, 1), [(2, 4, 6), (1, 2, 2)]),
        ) as run_kernel:
            output = Qwen2VLVideoProcessor(use_kernels=True)(
                videos, do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt"
            )
        frames, target_sizes, items = run_kernel.call_args[0][1:4]
        self.assertEqual(len(frames), 5)
        self.assertEqual(items, [[0, 1, 2], [3, 4]])
        self.assertEqual(len(set(target_sizes[:3])), 1)
        self.assertEqual(output["video_grid_thw"].tolist(), [[2, 4, 6], [1, 2, 2]])

    def test_processor_overriding_resize_does_not_use_the_resize_kernel(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.dict(_PROCESSING_KERNEL_ADAPTERS, {"resize_normalize": ("cv-utils", self.fail)}, clear=True),
            patch.multiple(
                "transformers.integrations.hub_kernels",
                _kernels_enabled=True,
                lazy_load_kernel=lambda name: types.ModuleType("sentinel_kernel_module"),
            ),
        ):
            output = LevitImageProcessor(use_kernels=True)([torch.randint(0, 255, (3, 64, 96), dtype=torch.uint8)])
        self.assertIn("pixel_values", output)

    def test_processors_reaching_the_patchify_kernel_use_its_layout(self):
        processor_classes = processors_calling_the_patchify_kernel()
        for processor_class in (Qwen2VLImageProcessor, PaddleOCRVLImageProcessor, HunYuanVLImageProcessor):
            self.assertIn(processor_class, processor_classes)
        qwen3_vl_image_settings = {"patch_size": 16, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
        cases = [(processor_class, {}) for processor_class in processor_classes]
        cases.append((Qwen2VLImageProcessor, qwen3_vl_image_settings))
        for processor_class, settings in cases:
            with self.subTest(processor_class=processor_class.__name__, **settings):
                processor = processor_class(use_kernels=True, **settings)
                is_video = issubclass(processor_class, BaseVideoProcessor)
                recorded_arguments = []

                def run(height, width):
                    recorded_arguments.clear()
                    if is_video:
                        videos = [torch.randint(0, 256, (3, 3, height, width), dtype=torch.uint8)]
                        return processor(videos, do_sample_frames=False, return_tensors="pt")["pixel_values_videos"]
                    images = [torch.randint(0, 256, (3, height, width), dtype=torch.uint8)]
                    return processor(images, return_tensors="pt")["pixel_values"]

                with (
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.dict(
                        _PROCESSING_KERNEL_ADAPTERS,
                        {
                            "resize_normalize_patchify": (
                                "cv-utils",
                                lambda kernel, *arguments: recorded_arguments.append(arguments),
                            )
                        },
                        clear=True,
                    ),
                    patch.multiple(
                        "transformers.integrations.hub_kernels",
                        _kernels_enabled=True,
                        lazy_load_kernel=lambda name: types.ModuleType("sentinel_kernel_module"),
                    ),
                ):
                    height, width = 112, 168
                    pixel_values = run(height, width)
                    if not recorded_arguments:
                        continue
                    for _ in range(3):
                        if recorded_arguments[0][1][0] == (height, width):
                            break
                        height, width = recorded_arguments[0][1][0]
                        pixel_values = run(height, width)
                frames, target_sizes = recorded_arguments[0][:2]
                self.assertEqual([tuple(frame.shape[-2:]) for frame in frames], [tuple(size) for size in target_sizes])
                torch.testing.assert_close(
                    pixel_values.reshape(-1), patchify_kernel_layout_reference(*recorded_arguments[0]).reshape(-1)
                )

    def test_use_kernels_is_a_runtime_flag(self):
        processor = Sam2ImageProcessor(use_kernels=True)
        self.assertNotIn("use_kernels", processor.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            self.assertFalse(Sam2ImageProcessor.from_pretrained(tmp_dir).use_kernels)
            self.assertTrue(Sam2ImageProcessor.from_pretrained(tmp_dir, use_kernels=True).use_kernels)


@require_kernels
@require_torch_gpu
@slow
class TestConnectedComponentsKernel(TestCasePlus):
    """The `kernels-community/cv-utils` connected components kernel against the reference, through the processor."""

    def test_matches_reference(self):
        generator = torch.Generator().manual_seed(0)
        masks = torch.randn(4, 3, 63, 65, generator=generator).to(torch_device)
        processor = Sam2ImageProcessor(use_kernels=True)
        regions = (masks.flatten(0, 1) > 0).unsqueeze(1)
        areas = run_processing_kernel("connected_component_areas", regions)
        self.assertIsNotNone(areas, "the kernel did not run")
        torch.testing.assert_close(areas.to(torch.int32).cpu(), connected_component_areas_reference(regions.cpu()))
        filled = processor.post_process_masks([masks], [(63, 65)], max_hole_area=8, max_sprinkle_area=8)[0]
        with patch(
            "transformers.models.sam2.image_processing_sam2.run_processing_kernel",
            side_effect=lambda name, regions: connected_component_areas_reference(regions),
        ):
            expected = processor.post_process_masks([masks], [(63, 65)], max_hole_area=8, max_sprinkle_area=8)[0]
        self.assertTrue(torch.equal(filled, expected))


@require_kernels
@require_torch_gpu
@slow
class TestResizeKernels(TestCasePlus):
    """The resize kernels of `kernels-community/cv-utils` against the default torchvision path, on photos."""

    def setUp(self):
        super().setUp()
        image = torch.from_numpy(np.array(load_image("http://images.cocodataset.org/val2017/000000039769.jpg")))
        image = image.permute(2, 0, 1).contiguous().to(torch_device)
        self.images = [image, image[:, :300, :500].contiguous(), image[:, 100:, 50:].contiguous()]

    def assert_kernel_matches_default(self, operation, processor_class, call):
        kernel_name, adapter = _PROCESSING_KERNEL_ADAPTERS[operation]
        if not hasattr(lazy_load_kernel(kernel_name), operation):
            self.skipTest(f"`{operation}` is not available in `{kernel_name}`")
        kernel_outputs = []

        def recording_adapter(*args, **kwargs):
            kernel_outputs.append(adapter(*args, **kwargs))
            return kernel_outputs[-1]

        with patch.dict(_PROCESSING_KERNEL_ADAPTERS, {operation: (kernel_name, recording_adapter)}):
            with_kernels = call(processor_class(use_kernels=True))
        self.assertTrue(
            kernel_outputs and kernel_outputs[0] is not None, "the adapter fell back, the kernel never ran"
        )
        without_kernels = call(processor_class(use_kernels=False))
        for key in without_kernels:
            self.assertEqual(with_kernels[key].shape, without_kernels[key].shape)
        # Both paths round to uint8 after each resize pass, in a different order of operations.
        one_level = 1 / 255 / min(processor_class().image_std)
        difference = (
            with_kernels.data[list(without_kernels)[0]] - without_kernels.data[list(without_kernels)[0]]
        ).abs()
        self.assertLess(difference.mean().item(), one_level / 4)
        self.assertLess((difference > one_level * 1.01).float().mean().item(), 1e-3)

    def test_resize_normalize(self):
        self.assert_kernel_matches_default(
            "resize_normalize", ViTImageProcessor, lambda processor: processor(self.images, return_tensors="pt")
        )

    def test_resize_normalize_patchify_images(self):
        self.assert_kernel_matches_default(
            "resize_normalize_patchify",
            Qwen2VLImageProcessor,
            lambda processor: processor(self.images, return_tensors="pt"),
        )

    def test_resize_normalize_patchify_video(self):
        video = torch.stack([self.images[0]] * 5)
        self.assert_kernel_matches_default(
            "resize_normalize_patchify",
            Qwen2VLVideoProcessor,
            lambda processor: processor(
                [video], do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt"
            ),
        )
