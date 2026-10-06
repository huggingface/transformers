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
- **Registration edge cases** — type-check rejections in ``register_backend``.
- **`patch_attributes` restore-on-exception** — the happy path is exercised but the exception
  branch never fires in real exports.
- The **`decompose_prefill_decode` guard** against generators that bypass the top-level
  forward — real generators call ``forward`` many times, so the guard is dead code without a
  targeted test.
- **`register_patch`** unresolvable-path fallback — real registrations point at real paths.

Everything below targets one of those gaps.
"""

import unittest
from unittest import mock

from transformers.exporters import utils as exporter_utils
from transformers.exporters.auto import (
    EXPORT_BACKENDS,
    AutoExportConfig,
    AutoHfExporter,
    export_backend,
    register_backend,
)
from transformers.exporters.base import HfExporter, ModelRunner
from transformers.exporters.configs import (
    DynamoConfig,
    ExecutorchConfig,
    ExportConfigMixin,
    ExportFormat,
    OnnxConfig,
    OpenVINOConfig,
)
from transformers.testing_utils import require_executorch, require_onnx, require_onnxscript, require_torch
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn

    from transformers import GenerationConfig, PretrainedConfig
    from transformers.exporters.decompose import decompose_prefill_decode
    from transformers.exporters.utils import (
        cast_leaf_tensors,
        duplicate_leaf_tensors,
        patch_attributes,
        register_patch,
    )


CONCRETE_CONFIGS = [
    (OnnxConfig, ExportFormat.ONNX),
    (DynamoConfig, ExportFormat.DYNAMO),
    (ExecutorchConfig, ExportFormat.EXECUTORCH),
    (OpenVINOConfig, ExportFormat.OPENVINO),
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


class AutoExportConfigTest(unittest.TestCase):
    def test_from_dict_dispatches_to_concrete_config(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format.value}), config_cls)
                # Enum inputs also work — serialised configs may hold either form.
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format}), config_cls)


class AutoHfExporterTest(unittest.TestCase):
    def _check_dispatch(self, config):
        expected_cls = EXPORT_BACKENDS[config.export_format.value].exporter
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
        # Both name the formats that *are* registered, so the message says what to pass instead.
        with self.assertRaisesRegex(ValueError, "Unknown export format 'not_a_real_backend'"):
            AutoHfExporter.from_config({"export_format": "not_a_real_backend"})
        with self.assertRaisesRegex(ValueError, "No export format given"):
            AutoHfExporter.from_config({})


class RegistrationTest(unittest.TestCase):
    """`register_backend` wires a format through to `export_backend`, and refuses parts of the wrong type.
    `EXPORT_BACKENDS` is temporarily patched so registrations never leak into other tests."""

    def test_register_backend_rejects_non_subclass(self):
        with mock.patch.dict(EXPORT_BACKENDS):
            with self.assertRaisesRegex(TypeError, "HfExporter"):
                register_backend("bad", ExportConfigMixin, object, ModelRunner)

    def test_register_backend_installs_stub(self):
        class _StubExporter(HfExporter):
            required_packages = []

            def export_artifact(self, model, sample_inputs, config):
                return None, {}

            @classmethod
            def save_artifact(cls, artifact, path):
                return None

        with mock.patch.dict(EXPORT_BACKENDS):
            register_backend("stub", ExportConfigMixin, _StubExporter, ModelRunner)
            self.assertIs(export_backend("stub", "exporter"), _StubExporter)
        self.assertNotIn("stub", EXPORT_BACKENDS)

    def test_export_only_backend_refuses_to_run(self):
        with mock.patch.dict(EXPORT_BACKENDS):
            register_backend("export_only", ExportConfigMixin, HfExporter)
            self.assertIs(export_backend("export_only", "exporter"), HfExporter)
            with self.assertRaisesRegex(ValueError, "registers no runner"):
                export_backend("export_only", "runner")


# ─────────────────────────────────────────────────────────────────────────────
# Registry edge cases the happy-path exports don't exercise
# ─────────────────────────────────────────────────────────────────────────────


class _Owner:
    def method(self):
        return "original"


@require_torch
class PatchRegistryEdgeCasesTest(unittest.TestCase):
    def test_patch_attributes_roll_back_on_exception(self):
        # A factory raising mid-install must roll back the patches already installed.
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
        # An unresolvable path is skipped, so a backend imports when another's packages are missing.
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
        # Only repeated tensors are cloned, so ONNX's output dedup never renames ports.
        shared = torch.zeros(2)
        distinct = torch.ones(3)
        result = duplicate_leaf_tensors({"a": shared, "b": shared, "c": distinct})
        self.assertIs(result["a"], shared)
        self.assertIsNot(result["b"], shared)
        self.assertTrue(torch.equal(result["b"], shared))
        self.assertIs(result["c"], distinct)

    def test_cast_leaf_tensors_preserves_integer_dtypes(self):
        # Casting inputs to the model's dtype leaves integer tensors (ids, indices, positions) alone.
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
# decompose_prefill_decode guard
# ─────────────────────────────────────────────────────────────────────────────


@require_torch
class DecomposePrefillDecodeGuardTest(unittest.TestCase):
    def test_raises_when_generate_bypasses_forward(self):
        # A generator delegating to an inner model captures too few top-level forwards.
        class _FakeGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                # `decompose_prefill_decode` reads the model's own configs before capturing (mimics a
                # real `PreTrainedModel`); the guard under test fires afterwards on the capture count.
                self.config = PretrainedConfig()
                self.generation_config = GenerationConfig()

            def forward(self, input_ids=None, **kwargs):
                return input_ids

            def generate(self, input_ids=None, max_new_tokens=None, min_new_tokens=None, **kwargs):
                return self.forward(input_ids=input_ids)  # a single top-level forward call

        with self.assertRaisesRegex(RuntimeError, "captured 1"):
            decompose_prefill_decode(_FakeGenerator(), {"input_ids": torch.zeros(1, 1, dtype=torch.long)})
