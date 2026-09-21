# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Unit tests for the ``transformers.exporters`` pieces the per-model export tests don't reach.

The per-model exporter mixins in ``tests/exporters/test_export.py`` end-to-end-exercise
``prepare_for_export``, ``apply_patches`` / ``apply_fx_*_fixes``, the leaf-tensor helpers,
and the bundled input preparers — so those get real coverage on every CI run. What they
DON'T touch:

- The **auto factory** (``AutoExportConfig`` / ``AutoHfExporter``) — models bypass it and
  instantiate concrete exporters directly.
- **Config dict round-trips** — configs are built via constructor calls, never serialised.
- **Registration edge cases** — collision warnings and type-check rejections in
  ``register_exporter`` / ``register_export_config``.
- **`patch_attributes` restore-on-exception** — the happy path is exercised but the exception
  branch never fires in real exports.
- The **`decompose_prefill_decode` guard** against generators that bypass the top-level
  forward — real generators call ``forward`` many times, so the guard is dead code without a
  targeted test.
- **`register_patch`** unresolvable-path fallback — real registrations point at real paths.

Everything below targets one of those gaps.
"""

import itertools
import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from transformers.exporters import utils as exporter_utils
from transformers.exporters.auto import (
    AUTO_EXPORT_CONFIG_MAPPING,
    AUTO_EXPORTER_MAPPING,
    AutoExportConfig,
    AutoHfExporter,
    register_export_config,
    register_exporter,
)
from transformers.exporters.base import HfExporter
from transformers.exporters.configs import DynamoConfig, ExecutorchConfig, ExportFormat, OnnxConfig
from transformers.testing_utils import require_executorch, require_onnx, require_onnxscript, require_torch
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DynamicCache, EncoderDecoderCache, GenerationConfig, LlamaConfig, StaticCache
    from transformers.exporters.utils import (
        cast_leaf_tensors,
        decompose_prefill_decode,
        duplicate_leaf_tensors,
        patch_attributes,
        register_patch,
    )


CONCRETE_CONFIGS = [
    (OnnxConfig, ExportFormat.ONNX),
    (DynamoConfig, ExportFormat.DYNAMO),
    (ExecutorchConfig, ExportFormat.EXECUTORCH),
]


# ─────────────────────────────────────────────────────────────────────────────
# Auto factory + config serialisation
# ─────────────────────────────────────────────────────────────────────────────


class ExportConfigMixinTest(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                original = config_cls(dynamic=True)
                restored = config_cls.from_dict(original.to_dict())
                self.assertEqual(restored, original)
                self.assertIs(restored.export_format, export_format)


class ExecutorchConfigTest(unittest.TestCase):
    def test_backend_options_roundtrip(self):
        for backend in ("xnnpack", "cuda", "mlx"):
            with self.subTest(backend=backend):
                config = ExecutorchConfig(
                    backend=backend,
                    dynamic=True,
                    strict=True,
                    dynamic_shapes={},
                    prefer_deferred_runtime_asserts_over_guards=True,
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                    alloc_mutable_buffers=False,
                )
                self.assertEqual(ExecutorchConfig.from_dict(config.to_dict()), config)
                self.assertEqual(AutoExportConfig.from_dict(config.to_dict()), config)

    def test_unknown_option_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "unknown_option"):
            ExecutorchConfig(unknown_option=True)
        for factory in (ExecutorchConfig.from_dict, AutoExportConfig.from_dict):
            with self.subTest(factory=factory), self.assertRaisesRegex(TypeError, "unknown_option"):
                factory({"export_format": "executorch", "unknown_option": True})


class AutoExportConfigTest(unittest.TestCase):
    def test_from_dict_dispatches_to_concrete_config(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format.value}), config_cls)
                # Enum inputs also work — serialised configs may hold either form.
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format}), config_cls)

    def test_from_dict_missing_export_format_raises(self):
        with self.assertRaisesRegex(ValueError, "export_format"):
            AutoExportConfig.from_dict({})

    def test_from_dict_unknown_format_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown exporter type"):
            AutoExportConfig.from_dict({"export_format": "not_a_real_backend"})


class AutoHfExporterTest(unittest.TestCase):
    def _check_dispatch(self, config):
        expected_cls = AUTO_EXPORTER_MAPPING[config.export_format.value]
        self.assertIsInstance(AutoHfExporter.from_config(config), expected_cls)
        # Same dispatch works when starting from a plain dict.
        self.assertIsInstance(AutoHfExporter.from_config(config.to_dict()), expected_cls)

    @require_torch
    def test_from_config_dispatches_dynamo(self):
        self._check_dispatch(DynamoConfig())

    @require_torch
    @require_onnx
    @require_onnxscript
    def test_from_config_dispatches_onnx(self):
        self._check_dispatch(OnnxConfig())

    @require_torch
    @require_executorch
    def test_from_config_dispatches_executorch(self):
        self._check_dispatch(ExecutorchConfig())

    def test_from_config_raises_on_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported export config"):
            AutoHfExporter.from_config({"export_format": "not_a_real_backend"})
        with self.assertRaisesRegex(ValueError, "Unsupported export config"):
            AutoHfExporter.from_config({})


class RegistrationTest(unittest.TestCase):
    """Cover the edge cases of `register_exporter` / `register_export_config` that normal
    registrations at module load don't hit — the type-check rejection paths. The mappings are
    temporarily patched so registrations never leak into other tests."""

    def test_register_exporter_rejects_non_subclass(self):
        with mock.patch.dict(AUTO_EXPORTER_MAPPING):
            with self.assertRaisesRegex(TypeError, "HfExporter"):

                @register_exporter("bad")
                class _NotAnExporter:
                    pass

    def test_register_export_config_rejects_non_subclass(self):
        with mock.patch.dict(AUTO_EXPORT_CONFIG_MAPPING):
            with self.assertRaisesRegex(TypeError, "ExportConfigMixin"):

                @register_export_config("bad_config")
                class _NotAConfig:
                    pass

    def test_register_exporter_installs_stub(self):
        # Sanity check that a legit registration is wired through — protects against a future
        # refactor that would break the decorator without breaking any real export test.
        with mock.patch.dict(AUTO_EXPORTER_MAPPING):

            @register_exporter("stub_exporter")
            class _StubExporter(HfExporter):
                required_packages = []

                def export(self, model, sample_inputs, config):
                    return None

            self.assertIs(AUTO_EXPORTER_MAPPING["stub_exporter"], _StubExporter)
        self.assertNotIn("stub_exporter", AUTO_EXPORTER_MAPPING)


@require_torch
class ExecutorchImportIsolationTest(unittest.TestCase):
    def test_exporter_import_without_optional_backends(self):
        code = textwrap.dedent(
            """
            import builtins
            import sys
            from unittest import mock

            forbidden = ("executorch.backends.cuda", "executorch.backends.mlx")
            original_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                names = (name, *(f"{name}.{item}" for item in (fromlist or ())))
                if any(item == prefix or item.startswith(prefix + ".") for item in names for prefix in forbidden):
                    raise AssertionError(f"Imported an optional backend: {name}")
                return original_import(name, globals, locals, fromlist, level)

            with (
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("builtins.__import__", side_effect=guarded_import),
            ):
                from transformers.exporters import exporter_executorch

            assert all(callable(prepare) for prepare in exporter_executorch._BACKEND_PREPARE.values())
            loaded = [name for name in sys.modules if any(
                name == prefix or name.startswith(prefix + ".") for prefix in forbidden
            )]
            assert not loaded, loaded
            """
        )
        self._run_in_subprocess(code)

    @require_executorch
    def test_mlx_prepares_unregistered_cropped_cache(self):
        code = textwrap.dedent(
            """
            import sys
            from types import SimpleNamespace
            from unittest import mock

            import torch
            from transformers import DynamicCache, EncoderDecoderCache
            from transformers.exporters.exporter_executorch import prepare_for_mlx

            cache = DynamicCache()
            cache.update(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8), 0)
            cache.crop(-1)
            original_keys = cache.layers[0].keys
            original_values = cache.layers[0].values
            assert not original_keys.is_contiguous()
            assert not original_values.is_contiguous()
            assert DynamicCache not in torch.utils._pytree.SUPPORTED_NODES
            assert EncoderDecoderCache not in torch.utils._pytree.SUPPORTED_NODES

            model = torch.nn.Linear(2, 2)
            inputs = {
                "input": torch.ones(1, 2),
                "past_key_values": EncoderDecoderCache(cache, cache),
            }
            modules = {"executorch.backends.mlx": SimpleNamespace(MLXPartitioner=mock.Mock())}
            with mock.patch.dict(sys.modules, modules):
                _, prepared, _ = prepare_for_mlx(model, inputs)
                _, repeated, _ = prepare_for_mlx(model, prepared)

            for name in ("self_attention_cache", "cross_attention_cache"):
                layer = getattr(prepared["past_key_values"], name).layers[0]
                repeated_layer = getattr(repeated["past_key_values"], name).layers[0]
                assert layer.keys.is_contiguous()
                assert layer.values.is_contiguous()
                torch.testing.assert_close(layer.keys, original_keys, rtol=0, atol=0)
                torch.testing.assert_close(layer.values, original_values, rtol=0, atol=0)
                assert repeated_layer.keys is layer.keys
                assert repeated_layer.values is layer.values
            assert prepared["input"] is inputs["input"]
            assert cache.layers[0].keys is original_keys
            assert cache.layers[0].values is original_values
            assert not original_keys.is_contiguous()
            assert not original_values.is_contiguous()
            """
        )
        self._run_in_subprocess(code)

    def _run_in_subprocess(self, code):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@require_torch
@require_executorch
class ExecutorchTensorReshapePatchTest(unittest.TestCase):
    @contextmanager
    def _patches_with_restoration(self):
        from transformers.exporters import exporter_executorch

        patches = [
            patch
            for patch in exporter_utils._PATCHES["executorch"]
            if patch[0] is torch.Tensor and patch[1] == "reshape"
        ]
        self.assertEqual(len(patches), 1)
        self.assertIs(patches[0][2], exporter_executorch._patch_tensor_reshape)
        original = torch.Tensor.reshape
        try:
            with patch_attributes(patches):
                self.assertIsNot(torch.Tensor.reshape, original)
                yield
        finally:
            self.assertIs(torch.Tensor.reshape, original)

    def test_shape_argument_forms(self):
        operations = {
            "varargs": lambda x: x.reshape(2, -1),
            "tuple": lambda x: x.reshape((2, -1)),
            "list": lambda x: x.reshape([2, -1]),
            "keyword_tuple": lambda x: x.reshape(shape=(2, -1)),
            "keyword_list": lambda x: x.reshape(shape=[2, -1]),
            "size": lambda x: x.reshape(x.shape),
            "single_dim": lambda x: x.reshape(-1),
        }
        for x in (torch.randn(3, 4), torch.randn(4, 3).t()):
            expected = {name: operation(x) for name, operation in operations.items()}
            with self._patches_with_restoration():
                for name, operation in operations.items():
                    with self.subTest(operation=name, contiguous=x.is_contiguous()):
                        actual = operation(x)
                        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
                        self.assertTrue(actual.is_contiguous())
                        self.assertEqual(actual.data_ptr() == x.data_ptr(), x.is_contiguous())

    def test_scalar_shape_forms(self):
        x = torch.tensor(1.0)
        operations = (
            lambda: x.reshape([]),
            lambda: x.reshape(()),
            lambda: x.reshape(shape=()),
            lambda: x.reshape(1),
        )
        expected = [operation() for operation in operations]
        with self._patches_with_restoration():
            for operation, reference in zip(operations, expected):
                torch.testing.assert_close(operation(), reference)

    def test_invalid_arguments(self):
        x = torch.randn(3, 4)
        invalid_arguments = (
            lambda: x.reshape(),
            lambda: x.reshape(2, shape=(2, 6)),
            lambda: x.reshape(size=(2, 6)),
            lambda: x.reshape(shape=(2, 6), unknown=True),
        )
        with self._patches_with_restoration():
            for operation in invalid_arguments:
                with self.assertRaises(TypeError):
                    operation()
            with self.assertRaisesRegex(RuntimeError, "invalid for input"):
                x.reshape(5)

    def _check_strict_export(self, operation, sample, expected_clones):
        class Model(nn.Module):
            def forward(self, x):
                return operation(x)

        with self._patches_with_restoration():
            program = torch.export.export(
                Model(),
                (sample[:3],),
                dynamic_shapes={"x": {0: torch.export.Dim("rows", min=2, max=6)}},
                strict=True,
            )
        self.assertTrue(program.range_constraints)
        clones = [node for node in program.graph.nodes if node.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clones), expected_clones)
        for node in clones:
            self.assertEqual(node.kwargs.get("memory_format"), torch.contiguous_format)
        exported = program.module()
        for rows in (2, 5):
            x = sample[:rows]
            actual = exported(x)
            torch.testing.assert_close(actual, operation(x), rtol=0, atol=0)
            self.assertTrue(actual.is_contiguous())
            self.assertEqual(actual.data_ptr() == x.data_ptr(), expected_clones == 0)

    def test_strict_symbolic_shape_forms(self):
        operations = {
            "varargs": lambda x: x.reshape(x.shape[0], -1),
            "tuple": lambda x: x.reshape((x.shape[0], -1)),
            "list": lambda x: x.reshape([x.shape[0], -1]),
            "keyword": lambda x: x.reshape(shape=(x.shape[0], -1)),
            "size": lambda x: x.reshape(x.shape),
        }
        sample = torch.randn(5, 2, 4)
        for name, operation in operations.items():
            for contiguous in (True, False):
                with self.subTest(operation=name, contiguous=contiguous):
                    self._check_strict_export(
                        operation, sample if contiguous else sample.transpose(1, 2), int(not contiguous)
                    )

    def test_strict_internal_transpose(self):
        self._check_strict_export(lambda x: x.transpose(1, 2).reshape(shape=(x.shape[0], -1)), torch.randn(5, 2, 4), 1)

    def test_restoration_after_strict_export_failure(self):
        class InvalidShape(nn.Module):
            def forward(self, x):
                return x.reshape(5)

        with self.assertRaisesRegex(RuntimeError, "invalid for input"):
            with self._patches_with_restoration():
                torch.export.export(InvalidShape(), (torch.randn(3, 4),), strict=True)


@require_torch
@require_executorch
class ExecutorchBackendRecipeTest(unittest.TestCase):
    def test_preparation_and_lowering(self):
        from transformers.exporters import exporter_dynamo
        from transformers.exporters import exporter_executorch as et

        static_cache = StaticCache(config=LlamaConfig(num_hidden_layers=1), max_cache_len=8)
        caches = {
            "none": (None, False),
            "dynamic": (DynamicCache(), False),
            "static": (static_cache, True),
            "encoder_decoder_dynamic": (EncoderDecoderCache(DynamicCache(), DynamicCache()), False),
            "static_self_attention": (EncoderDecoderCache(static_cache, DynamicCache()), True),
            "static_cross_attention": (EncoderDecoderCache(DynamicCache(), static_cache), True),
            "encoder_decoder_static": (EncoderDecoderCache(static_cache, static_cache), True),
        }
        for backend, cache_name in itertools.product(("xnnpack", "cuda", "mlx"), caches):
            cache, unsupported_on_mlx = caches[cache_name]
            with self.subTest(backend=backend, cache=cache_name):
                model = nn.Linear(2, 2)
                inputs = {"input": torch.randn(2, 3).t(), "nested": [torch.ones(1), "keep"], "past_key_values": cache}
                config = ExecutorchConfig(backend=backend, strict=True, dynamic=True, dynamic_shapes={})
                ep, events = SimpleNamespace(graph_module=object()), []
                mlx = mock.Mock()
                passes = mock.Mock(side_effect=lambda: events.append("passes") or [mock.sentinel.mlx_pass])
                modules = {
                    "executorch.backends.mlx": SimpleNamespace(MLXPartitioner=mlx),
                    "executorch.backends.mlx.passes": SimpleNamespace(get_default_passes=passes),
                }

                def capture(prepared_model, prepared_inputs, *, config):
                    events.append("capture")
                    self.assertIs(prepared_model, model)
                    self.assertFalse(any(p.requires_grad for p in model.parameters()))
                    self.assertEqual(model.weight.device.type, "cpu")
                    self.assertEqual(model.weight.dtype, torch.bfloat16 if backend == "cuda" else torch.float32)
                    self.assertTrue(prepared_inputs["input"].is_contiguous())
                    torch.testing.assert_close(prepared_inputs["input"], inputs["input"])
                    self.assertIs(prepared_inputs["nested"][0], inputs["nested"][0])
                    self.assertEqual(prepared_inputs["nested"][1], "keep")
                    return ep

                with (
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(torch.cuda, "is_available", return_value=True),
                    mock.patch.object(et, "XnnpackPartitioner") as xnn,
                    mock.patch.object(et, "CudaBackend", create=True),
                    mock.patch.object(et, "CudaPartitioner", create=True) as cuda,
                    mock.patch.object(exporter_dynamo.DynamoExporter, "export", side_effect=capture) as export,
                    mock.patch.object(
                        et, "apply_fx_program_fixes", side_effect=lambda *a: events.append("program")
                    ) as pf,
                    mock.patch.object(et, "apply_fx_node_fixes", side_effect=lambda *a: events.append("nodes")) as nf,
                    mock.patch.object(et, "EdgeCompileConfig") as compile_config,
                    mock.patch.object(et, "to_edge_transform_and_lower") as lower,
                ):
                    if backend == "mlx" and unsupported_on_mlx:
                        with self.assertRaisesRegex(
                            ValueError, "StaticCache is not supported by the ExecuTorch MLX backend.*Use DynamicCache"
                        ):
                            et.ExecutorchExporter().export(model, inputs, config)
                        export.assert_not_called()
                        lower.assert_not_called()
                        mlx.assert_not_called()
                        self.assertEqual(events, [])
                        self.assertTrue(all(p.requires_grad for p in model.parameters()))
                        continue
                    result = et.ExecutorchExporter().export(model, inputs, config)
                self.assertIs(export.call_args.kwargs["config"], config)
                self.assertEqual(events, ["capture", "program", "nodes"] + (["passes"] if backend == "mlx" else []))
                pf.assert_called_once_with("executorch", ep)
                nf.assert_called_once_with("executorch", ep.graph_module)
                partitioners = {"xnnpack": xnn, "cuda": cuda, "mlx": mlx}
                for name, partitioner in partitioners.items():
                    self.assertEqual(partitioner.call_count, int(name == backend))
                if backend == "mlx":
                    mlx.assert_called_once_with()
                    passes.assert_called_once_with()
                    compile_config.assert_called_once_with(_check_ir_validity=False, _skip_dim_order=True)
                else:
                    passes.assert_not_called()
                lower.assert_called_once_with(
                    ep,
                    partitioner=[partitioners[backend].return_value],
                    compile_config=compile_config.return_value,
                    transform_passes=[mock.sentinel.mlx_pass] if backend == "mlx" else None,
                )
                lower.return_value.to_executorch.assert_called_once_with(config=None)
                self.assertIs(result, lower.return_value.to_executorch.return_value)
                self.assertFalse(inputs["input"].is_contiguous())

    def test_allocation_options(self):
        from transformers.exporters import exporter_executorch as et

        for disabled in (None, "alloc_graph_input", "alloc_graph_output", "alloc_mutable_buffers"):
            with self.subTest(disabled=disabled):
                config = ExecutorchConfig(**({disabled: False} if disabled else {}))
                with (
                    mock.patch.object(et, "MemoryPlanningPass") as memory,
                    mock.patch.object(et, "ExecutorchBackendConfig") as backend_config,
                ):
                    result = et._get_backend_config(config)
                if disabled is None:
                    self.assertIsNone(result)
                    memory.assert_not_called()
                    backend_config.assert_not_called()
                else:
                    memory.assert_called_once_with(
                        alloc_graph_input=config.alloc_graph_input,
                        alloc_graph_output=config.alloc_graph_output,
                        alloc_mutable_buffers=config.alloc_mutable_buffers,
                    )
                    backend_config.assert_called_once_with(memory_planning_pass=memory.return_value)
                    self.assertIs(result, backend_config.return_value)


# ─────────────────────────────────────────────────────────────────────────────
# Registry edge cases the happy-path exports don't exercise
# ─────────────────────────────────────────────────────────────────────────────


class _Owner:
    def method(self):
        return "original"


@require_torch
class PatchRegistryEdgeCasesTest(unittest.TestCase):
    def test_patch_attributes_roll_back_on_exception(self):
        # Real exports never exit the trace via exception, so this rollback path is untested by
        # integration. If it ever regressed to leave already-installed patches in place when a
        # later factory raises, the *next* export would run against a leaked patch and fail in
        # a way that looks unrelated. Only this test would catch that.
        a, b = _Owner(), _Owner()

        def _bad_factory(original):
            raise RuntimeError("factory boom")

        with self.assertRaisesRegex(RuntimeError, "factory boom"):
            with patch_attributes(
                [
                    (a, "method", lambda original: (lambda: "a-patched")),
                    (b, "method", _bad_factory),
                ]
            ):
                pass
        self.assertEqual(a.method(), "original")
        self.assertEqual(b.method(), "original")

    def test_register_patch_skips_unresolvable_path(self):
        # Real backends only register paths that resolve; the silent-skip fallback is what lets
        # `exporter_onnx.py` and `exporter_executorch.py` co-exist when only one backend is
        # installed. If it ever started raising, one of the two backends would fail to import.
        backend = "_test_unresolvable"

        @register_patch(backend, "does.not.exist.at.all")
        def _patch(original):
            return original

        try:
            self.assertEqual(exporter_utils._PATCHES.get(backend, []), [])
        finally:
            exporter_utils._PATCHES.pop(backend, None)


# ─────────────────────────────────────────────────────────────────────────────
# Leaf-tensor invariants that integration tests wouldn't visibly catch
# ─────────────────────────────────────────────────────────────────────────────


@require_torch
class LeafTensorInvariantsTest(unittest.TestCase):
    def test_duplicate_leaf_tensors_only_clones_repeats(self):
        # If this ever regressed to ``.clone()``-everything, ONNX exports would still succeed
        # and just get a bit bigger — no integration test would notice. Similarly, if it
        # stopped cloning the second occurrence, ONNX's output-node dedup would rename ports
        # in a way that only manifests as a stale name mapping.
        shared = torch.zeros(2)
        distinct = torch.ones(3)
        result = duplicate_leaf_tensors({"a": shared, "b": shared, "c": distinct})
        self.assertIs(result["a"], shared)
        self.assertIsNot(result["b"], shared)
        self.assertTrue(torch.equal(result["b"], shared))
        self.assertIs(result["c"], distinct)

    def test_cast_leaf_tensors_preserves_integer_dtypes(self):
        # ``prepare_for_export`` casts input trees to the model's dtype. If this ever started
        # coercing integer tensors (``input_ids``, indices, positions) to float, most exports
        # would still trace but embedding-lookup / bincount / index-select paths would fail
        # far downstream with confusing errors. Only this test would attribute it to the cast.
        out = cast_leaf_tensors(
            {
                "input_ids": torch.zeros(2, dtype=torch.int64),
                "attention_mask": torch.ones(2, dtype=torch.int32),
                "hidden": torch.zeros(2, dtype=torch.float32),
            },
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertEqual(out["input_ids"].dtype, torch.int64)
        self.assertEqual(out["attention_mask"].dtype, torch.int32)
        self.assertEqual(out["hidden"].dtype, torch.float16)


# ─────────────────────────────────────────────────────────────────────────────
# decompose_prefill_decode guard (dead code without this test — no real generator
# calls forward < 2 times, so the branch would rot silently)
# ─────────────────────────────────────────────────────────────────────────────


@require_torch
class DecomposePrefillDecodeGuardTest(unittest.TestCase):
    def test_raises_when_generate_bypasses_forward(self):
        # Guards against generators that delegate to an inner model — the top-level ``forward``
        # captures at most one call, so the ``calls[0] / calls[1]`` indexing would raise a
        # confusing IndexError instead of the helpful RuntimeError below.
        class _FakeGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                # `decompose_prefill_decode` bases its capture config on the model's own (mimics a
                # real `PreTrainedModel`); the guard under test fires afterwards on the capture count.
                self.generation_config = GenerationConfig()

            def forward(self, input_ids=None, **kwargs):
                return input_ids

            def generate(self, input_ids=None, max_new_tokens=None, min_new_tokens=None, **kwargs):
                return self.forward(input_ids=input_ids)  # a single top-level forward call

        with self.assertRaisesRegex(RuntimeError, "captured 1"):
            decompose_prefill_decode(_FakeGenerator(), {"input_ids": torch.zeros(1, 1, dtype=torch.long)})
