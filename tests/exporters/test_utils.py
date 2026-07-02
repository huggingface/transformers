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

import unittest
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

            def forward(self, input_ids=None, **kwargs):
                return input_ids

            def generate(self, input_ids=None, max_new_tokens=None, min_new_tokens=None, **kwargs):
                return self.forward(input_ids=input_ids)  # a single top-level forward call

        with self.assertRaisesRegex(RuntimeError, "captured 1"):
            decompose_prefill_decode(_FakeGenerator(), {"input_ids": torch.zeros(1, 1, dtype=torch.long)})
