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
- **``ExportArtifacts`` saving** — the manifest ``save_pretrained`` writes next to the components.
- **`patch_attributes` restore-on-exception** — the happy path is exercised but the exception
  branch never fires in real exports.
- The **`decompose_prefill_decode` guard** against generators that bypass the top-level
  forward — real generators call ``forward`` many times, so the guard is dead code without a
  targeted test.
- **`register_patch`** unresolvable-path fallback — real registrations point at real paths.

Everything below targets one of those gaps.
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transformers import BertConfig, GenerationConfig
from transformers.exporters import utils as exporter_utils
from transformers.exporters.auto import (
    EXPORT_BACKENDS,
    AutoExportConfig,
    AutoHfExporter,
    get_exporter_class,
    register_backend,
)
from transformers.exporters.base import ExportArtifacts, ExportedComponent, HfExporter
from transformers.exporters.configs import DynamoConfig, ExecutorchConfig, ExportFormat, OnnxConfig
from transformers.exporters.metadata import ExportMetadata
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
    def test_to_dict_from_dict_roundtrip_through_json(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                original = config_cls(dynamic=True)
                restored = AutoExportConfig.from_dict(json.loads(json.dumps(original.to_dict())))
                self.assertEqual(restored, original)
                self.assertEqual(restored.export_format, export_format)


class AutoExportConfigTest(unittest.TestCase):
    def test_from_dict_dispatches_to_concrete_config(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format.value}), config_cls)
                # Enum inputs also work — serialised configs may hold either form.
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format}), config_cls)

    def test_from_dict_missing_export_format_raises(self):
        with self.assertRaisesRegex(ValueError, "No export format"):
            AutoExportConfig.from_dict({})

    def test_from_dict_unknown_format_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown export format"):
            AutoExportConfig.from_dict({"export_format": "not_a_real_backend"})


class AutoHfExporterTest(unittest.TestCase):
    def _check_dispatch(self, config):
        expected_cls = get_exporter_class(config.export_format)
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
        with self.assertRaisesRegex(ValueError, "Unknown export format"):
            AutoHfExporter.from_config({"export_format": "not_a_real_backend"})
        with self.assertRaisesRegex(ValueError, "No export format"):
            AutoHfExporter.from_config({})


@dataclasses.dataclass
class _StubConfig(DynamoConfig):
    export_format: str = "stub"


class _StubExporter(HfExporter):
    """A custom backend writing each artifact (a string) as text, so saving is tested without tracing anything."""

    export_format = "stub"
    config_class = _StubConfig
    artifact_suffix = ".txt"

    def export_artifact(self, model, sample_inputs, config):
        return "program", {"input_names": list(sample_inputs)}

    @classmethod
    def save_artifact(cls, artifact, path):
        Path(path).write_text(artifact, encoding="utf-8")

    @classmethod
    def load_artifact(cls, path):
        return Path(path).read_text(encoding="utf-8")


class RegistrationTest(unittest.TestCase):
    """Cover the edge cases of `register_backend` that the built-in registrations don't hit. The registry is
    temporarily patched so registrations never leak into other tests."""

    def test_register_backend_rejects_non_subclass(self):
        class _NotAnExporter:
            pass

        class _NotAConfig:
            pass

        with mock.patch.dict(EXPORT_BACKENDS):
            with self.assertRaisesRegex(TypeError, "HfExporter"):
                register_backend("bad", DynamoConfig, _NotAnExporter)
            with self.assertRaisesRegex(TypeError, "ExportConfigMixin"):
                register_backend("bad", _NotAConfig, _StubExporter)

    def test_register_backend_installs_stub(self):
        with mock.patch.dict(EXPORT_BACKENDS):
            register_backend("stub", _StubConfig, _StubExporter)
            self.assertIs(get_exporter_class("stub"), _StubExporter)
            self.assertIsInstance(AutoHfExporter.from_config(_StubConfig()), _StubExporter)
            self.assertEqual(
                AutoExportConfig.from_dict(_StubConfig(dynamic=True).to_dict()), _StubConfig(dynamic=True)
            )
        self.assertNotIn("stub", EXPORT_BACKENDS)


class ExportArtifactsTest(unittest.TestCase):
    def test_export_returns_single_model_component(self):
        artifacts = _StubExporter().export(object(), {"input_ids": None}, _StubConfig())
        self.assertEqual(list(artifacts), ["model"])
        self.assertEqual(artifacts.artifact, "program")
        self.assertEqual(artifacts["model"].metadata.input_names, ("input_ids",))

    def test_artifact_refuses_several_components(self):
        component = ExportedComponent("program", ExportMetadata())
        artifacts = ExportArtifacts({"prefill": component, "decode": component}, "stub")
        with self.assertRaisesRegex(ValueError, "2 components"):
            artifacts.artifact

    def test_save_pretrained_from_pretrained_roundtrip(self):
        components = {
            "prefill": ExportedComponent("prefill program", ExportMetadata({"input_names": ["input_ids"]})),
            "decode": ExportedComponent("decode program", ExportMetadata()),
        }
        config, generation_config = BertConfig(hidden_size=32), GenerationConfig(max_new_tokens=3)
        with mock.patch.dict(EXPORT_BACKENDS):
            register_backend("stub", _StubConfig, _StubExporter)
            with tempfile.TemporaryDirectory() as directory:
                ExportArtifacts(components, "stub", config, generation_config).save_pretrained(directory)
                self.assertTrue((Path(directory) / "decode.txt").is_file())
                loaded = ExportArtifacts.from_pretrained(directory)
        self.assertEqual(loaded.export_format, "stub")
        self.assertEqual(
            {name: component.artifact for name, component in loaded.items()},
            {
                "prefill": "prefill program",
                "decode": "decode program",
            },
        )
        self.assertEqual(loaded["prefill"].metadata, components["prefill"].metadata)
        self.assertEqual(loaded.config.hidden_size, 32)
        self.assertEqual(loaded.generation_config.max_new_tokens, 3)

    def test_from_pretrained_downloads_only_the_export_from_the_hub(self):
        """From the Hub, the manifest is fetched first, then only the files it names (and the configs), so an
        export in a model repo's subfolder doesn't pull the model's own weights."""
        components = {"decode": ExportedComponent("decode program", ExportMetadata())}
        with mock.patch.dict(EXPORT_BACKENDS), tempfile.TemporaryDirectory() as repo:
            register_backend("stub", _StubConfig, _StubExporter)
            ExportArtifacts(components, "stub").save_pretrained(Path(repo) / "stub")

            def hf_hub_download(repo_id, filename, subfolder=None, **kwargs):
                return str(Path(repo) / (subfolder or "") / filename)

            def snapshot_download(repo_id, allow_patterns, **kwargs):
                self.assertEqual(
                    sorted(allow_patterns),
                    ["stub/config.json", "stub/decode.txt", "stub/export.json", "stub/generation_config.json"],
                )
                return repo

            with (
                mock.patch("transformers.exporters.base.hf_hub_download", hf_hub_download),
                mock.patch("transformers.exporters.base.snapshot_download", snapshot_download),
            ):
                loaded = ExportArtifacts.from_pretrained("org/model", subfolder="stub")
        self.assertEqual(loaded["decode"].artifact, "decode program")

    def test_save_pretrained_pushes_to_hub(self):
        components = {"decode": ExportedComponent("decode program", ExportMetadata())}
        artifacts = ExportArtifacts(components, "stub", generation_config=GenerationConfig(max_new_tokens=3))
        api = mock.Mock()
        api.create_repo.return_value.repo_id = "org/model"
        with (
            mock.patch.dict(EXPORT_BACKENDS),
            mock.patch("transformers.utils.hub.hf_api", return_value=api),
            mock.patch("transformers.exporters.base.hf_api", return_value=api),
            tempfile.TemporaryDirectory() as directory,
        ):
            register_backend("stub", _StubConfig, _StubExporter)
            artifacts.save_pretrained(directory, push_to_hub=True, repo_id="org/model")
        uploaded = {operation.path_in_repo for operation in api.create_commit.call_args.kwargs["operations"]}
        self.assertEqual(uploaded, {"decode.txt", "export.json", "generation_config.json"})


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
        # Silent-skip lets exporters co-exist when only one backend is installed.
        # Resolution moved from decoration time to apply_patches time (#50000): the string
        # path is now stored in _PATCHES, but apply_patches drops it (resolves to None) —
        # no patch installed, no exception, import-time safety unchanged.
        backend = "_test_unresolvable"

        @register_patch(backend, "does.not.exist.at.all")
        def _patch(original):
            return original

        try:
            # String is stored (lazy — not resolved yet)
            stored = exporter_utils._PATCHES.get(backend, [])
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0][:2], ("does.not.exist.at", "all"))
            # apply_patches silently skips it — no patch installed, no exception
            with exporter_utils.apply_patches(backend):
                pass
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
