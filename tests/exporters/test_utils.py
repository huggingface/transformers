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
- **`patch_attribute` restore-on-exception** — the happy path is exercised but the exception
  branch never fires in real exports.
- The **`decompose_prefill_decode` guard** against generators that bypass the top-level
  forward — real generators call ``forward`` many times, so the guard is dead code without a
  targeted test (this branch was flagged in review).
- **`_resolve_dotted_path`** unresolvable-path fallback — real registrations point at real
  paths.

Everything below targets one of those gaps.
"""

import pytest
import torch
from torch import nn

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
from transformers.exporters.configs import (
    DynamoConfig,
    ExecutorchConfig,
    ExportFormat,
    OnnxConfig,
)
from transformers.exporters.exporter_dynamo import DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxExporter
from transformers.exporters.utils import (
    cast_leaf_tensors,
    decompose_prefill_decode,
    duplicate_leaf_tensors,
    patch_attribute,
    patch_attributes,
    register_patch,
)


CONCRETE_CONFIGS = [
    (DynamoConfig, ExportFormat.DYNAMO),
    (OnnxConfig, ExportFormat.ONNX),
    (ExecutorchConfig, ExportFormat.EXECUTORCH),
]


# ─────────────────────────────────────────────────────────────────────────────
# Auto factory + config serialisation
# ─────────────────────────────────────────────────────────────────────────────


class TestExportConfigMixin:
    @pytest.mark.parametrize("config_cls,fmt", CONCRETE_CONFIGS)
    def test_to_dict_from_dict_roundtrip(self, config_cls, fmt):
        original = config_cls(dynamic=True)
        restored = config_cls.from_dict(original.to_dict())
        assert restored == original
        assert restored.export_format is fmt

    def test_to_dict_is_a_copy(self):
        cfg = DynamoConfig(dynamic_shapes={"x": 0})
        d = cfg.to_dict()
        d["dynamic_shapes"]["y"] = 1
        assert cfg.dynamic_shapes == {"x": 0}


class TestAutoExportConfig:
    @pytest.mark.parametrize("config_cls,fmt", CONCRETE_CONFIGS)
    def test_from_dict_dispatches_to_concrete_config(self, config_cls, fmt):
        assert isinstance(AutoExportConfig.from_dict({"export_format": fmt.value}), config_cls)
        # Enum inputs also work — belt-and-suspenders since serialised configs may hold either.
        assert isinstance(AutoExportConfig.from_dict({"export_format": fmt}), config_cls)

    def test_from_dict_missing_export_format_raises(self):
        with pytest.raises(ValueError, match="export_format"):
            AutoExportConfig.from_dict({})

    def test_from_dict_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown exporter type"):
            AutoExportConfig.from_dict({"export_format": "openvino_ish"})


class TestAutoHfExporter:
    @pytest.mark.parametrize(
        "config,exporter_cls",
        [
            (DynamoConfig(), DynamoExporter),
            (OnnxConfig(), OnnxExporter),
            (ExecutorchConfig(), ExecutorchExporter),
        ],
    )
    def test_from_config_dispatches_by_export_format(self, config, exporter_cls):
        assert isinstance(AutoHfExporter.from_config(config), exporter_cls)
        # Same dispatch works when starting from a plain dict.
        assert isinstance(AutoHfExporter.from_config(config.to_dict()), exporter_cls)

    def test_from_config_raises_on_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported export config"):
            AutoHfExporter.from_config({"export_format": "not_a_real_backend"})
        with pytest.raises(ValueError, match="Unsupported export config"):
            AutoHfExporter.from_config({})

    def test_supports_export_format(self):
        assert AutoHfExporter.supports_export_format({"export_format": "dynamo"})
        assert AutoHfExporter.supports_export_format({"export_format": ExportFormat.DYNAMO})
        assert not AutoHfExporter.supports_export_format({"export_format": "not_a_real_backend"})
        assert not AutoHfExporter.supports_export_format({})


class TestRegistration:
    """Cover the edge cases of `register_exporter` / `register_export_config` that
    normal registrations at module load don't hit — the type-check rejection paths."""

    def _cleanup(self, name):
        AUTO_EXPORTER_MAPPING.pop(name, None)
        AUTO_EXPORT_CONFIG_MAPPING.pop(name, None)

    def test_register_exporter_rejects_non_subclass(self):
        try:
            with pytest.raises(TypeError, match="HfExporter"):

                @register_exporter("bad")
                class _NotAnExporter:
                    pass

        finally:
            self._cleanup("bad")

    def test_register_export_config_rejects_non_subclass(self):
        try:
            with pytest.raises(TypeError, match="ExportConfigMixin"):

                @register_export_config("bad_config")
                class _NotAConfig:
                    pass

        finally:
            self._cleanup("bad_config")

    def test_register_exporter_installs_stub(self):
        # Sanity check that a legit registration is wired through — protects against a future
        # refactor that would break the decorator without breaking any real export test.
        try:

            @register_exporter("stub_exporter")
            class _StubExporter(HfExporter):
                required_packages = []

                def export(self, model, sample_inputs, config):
                    return None

            assert AUTO_EXPORTER_MAPPING["stub_exporter"] is _StubExporter
        finally:
            self._cleanup("stub_exporter")


# ─────────────────────────────────────────────────────────────────────────────
# Registry edge cases the happy-path exports don't exercise
# ─────────────────────────────────────────────────────────────────────────────


class _Owner:
    def method(self):
        return "original"


class TestPatchRegistryEdgeCases:
    def test_patch_attribute_restores_on_exception(self):
        # Real exports never exit the trace via exception, so this rollback path is untested
        # by integration. If the ExitStack ever regresses, only this test would catch it.
        owner = _Owner()
        with pytest.raises(RuntimeError):
            with patch_attribute(owner, "method", lambda original: (lambda: "patched")):
                raise RuntimeError("boom")
        assert owner.method() == "original"

    def test_patch_attributes_rolls_back_all_when_middle_factory_raises(self):
        # If `patch_attributes`' ExitStack ever regressed to leave already-installed patches
        # in place when a later factory raises, the *next* export would run against a leaked
        # patch and fail in a way that looks unrelated. Only this test would catch that.
        a, b = _Owner(), _Owner()

        def _bad_factory(original):
            raise RuntimeError("factory boom")

        with pytest.raises(RuntimeError, match="factory boom"):
            with patch_attributes(
                [
                    (a, "method", lambda original: (lambda: "a-patched")),
                    (b, "method", _bad_factory),
                ]
            ):
                pass
        assert a.method() == "original"
        assert b.method() == "original"

    def test_register_patch_skips_unresolvable_path(self):
        # Real backends only register paths that resolve; the silent-skip fallback is what lets
        # ``exporter_onnx.py`` and ``exporter_openvino.py`` co-exist when only one backend is
        # installed. If it ever starts raising, one of the two backends would fail to import.
        backend = "_test_unresolvable"

        @register_patch(backend, "does.not.exist.at.all")
        def _patch(original):
            return original

        try:
            assert exporter_utils._PATCHES.get(backend, []) == []
        finally:
            exporter_utils._PATCHES.pop(backend, None)


# ─────────────────────────────────────────────────────────────────────────────
# Leaf-tensor invariants that integration tests wouldn't visibly catch
# ─────────────────────────────────────────────────────────────────────────────


class TestLeafTensorInvariants:
    def test_duplicate_leaf_tensors_only_clones_repeats(self):
        # If this ever regressed to ``.clone()``-everything, ONNX exports would still succeed
        # and just get a bit bigger — no integration test would notice. Similarly, if it
        # stopped cloning the second occurrence, ONNX's output-node dedup would rename ports
        # in a way that only manifests as a stale name mapping.
        shared = torch.zeros(2)
        distinct = torch.ones(3)
        result = duplicate_leaf_tensors({"a": shared, "b": shared, "c": distinct})
        assert result["a"] is shared
        assert result["b"] is not shared
        assert torch.equal(result["b"], shared)
        assert result["c"] is distinct

    def test_cast_leaf_tensors_preserves_integer_dtypes(self):
        # ``prepare_for_export`` casts input trees to the model's dtype. If this ever started
        # coercing integer tensors (``input_ids``, indices, positions) to float, most exports
        # would still trace but embedding-lookup / bincount / index-select paths would fail
        # far downstream with confusing errors. Only this test would attribute it to the cast.
        input_ids = torch.zeros(2, dtype=torch.int64)
        attention_mask = torch.ones(2, dtype=torch.int32)
        floats = torch.zeros(2, dtype=torch.float32)
        out = cast_leaf_tensors(
            {"input_ids": input_ids, "attention_mask": attention_mask, "hidden": floats},
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        assert out["input_ids"].dtype == torch.int64
        assert out["attention_mask"].dtype == torch.int32
        assert out["hidden"].dtype == torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# decompose_prefill_decode guard (dead code without this test — no real generator
# calls forward < 2 times, so the branch would rot silently)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeGenerator(nn.Module):
    """Mimics ``PreTrainedModel.generate`` but calls ``forward`` a configurable number of times."""

    def __init__(self, num_forward_calls: int):
        super().__init__()
        self._num_forward_calls = num_forward_calls
        self.linear = nn.Linear(1, 1)

    def forward(self, input_ids=None, **kwargs):
        return input_ids

    def generate(self, input_ids=None, max_new_tokens=None, min_new_tokens=None, **kwargs):
        for _ in range(self._num_forward_calls):
            self.forward(input_ids=input_ids)
        return input_ids


class TestDecomposePrefillDecodeGuard:
    def test_raises_when_generate_bypasses_forward(self):
        # Guards against generators that delegate to an inner model — the top-level ``forward``
        # captures nothing, so the ``calls[0] / calls[1]`` index would raise a confusing
        # IndexError instead of the helpful RuntimeError below.
        with pytest.raises(RuntimeError, match="captured 1"):
            decompose_prefill_decode(
                _FakeGenerator(num_forward_calls=1),
                {"input_ids": torch.zeros(1, 1, dtype=torch.long)},
            )
