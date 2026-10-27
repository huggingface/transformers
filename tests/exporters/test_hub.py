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
"""Pushing exports to the Hub and loading them back: against a mocked Hub, and through the staging Hub."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from huggingface_hub import HfApi

from tests.exporters.test_export import _run_onnx_program
from tests.exporters.test_utils import _StubConfig, _StubExporter
from transformers import BertConfig, GenerationConfig
from transformers.exporters import EXPORT_BACKENDS, ExportArtifacts, ExportedComponent, register_backend
from transformers.exporters.metadata import ExportMetadata
from transformers.testing_utils import (
    TOKEN,
    TemporaryHubRepo,
    is_staging_test,
    require_onnxruntime,
    require_onnxscript,
    require_torch,
)
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch


class ExportArtifactsMockedHubTest(unittest.TestCase):
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
        self.assertEqual(loaded["decode"].program, "decode program")

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


@is_staging_test
@require_torch
@require_onnxscript
@require_onnxruntime
class ExportArtifactsPushToHubTester(unittest.TestCase):
    """A real ONNX export (its `model.onnx.data` included) through the staging Hub."""

    def _export(self):
        from transformers import BertModel
        from transformers.exporters import OnnxConfig, OnnxExporter

        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37
        )
        inputs = {"input_ids": torch.randint(0, 99, (1, 7)), "attention_mask": torch.ones(1, 7, dtype=torch.long)}
        return OnnxExporter().export(BertModel(config).eval(), inputs, OnnxConfig()), inputs

    def _assert_loads_same(self, loaded, exported, inputs):
        self.assertEqual(loaded["model"].metadata, exported["model"].metadata)
        self.assertEqual(loaded.config.hidden_size, exported.config.hidden_size)
        torch.testing.assert_close(
            _run_onnx_program(loaded.program, inputs), _run_onnx_program(exported.program, inputs)
        )

    def test_push_to_hub(self):
        exported, inputs = self._export()
        with TemporaryHubRepo(token=TOKEN) as repo:
            exported.push_to_hub(repo.repo_id, token=TOKEN)
            loaded = ExportArtifacts.from_pretrained(repo.repo_id, token=TOKEN)
            self._assert_loads_same(loaded, exported, inputs)

    def test_push_to_hub_via_save_pretrained(self):
        exported, inputs = self._export()
        with TemporaryHubRepo(token=TOKEN) as repo, tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory, push_to_hub=True, repo_id=repo.repo_id, token=TOKEN)
            loaded = ExportArtifacts.from_pretrained(repo.repo_id, token=TOKEN)
            self._assert_loads_same(loaded, exported, inputs)

    def test_from_pretrained_subfolder_downloads_only_the_export(self):
        """An export in a model repo's `onnx/` folder loads without downloading the model's own weights."""
        exported, inputs = self._export()
        api = HfApi(token=TOKEN)
        with (
            TemporaryHubRepo(token=TOKEN) as repo,
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as cache_dir,
        ):
            exported.save_pretrained(directory)
            api.upload_folder(repo_id=repo.repo_id, folder_path=directory, path_in_repo="onnx")
            api.upload_file(repo_id=repo.repo_id, path_or_fileobj=b"weights", path_in_repo="model.safetensors")
            loaded = ExportArtifacts.from_pretrained(repo.repo_id, subfolder="onnx", token=TOKEN, cache_dir=cache_dir)
            downloaded = {path.name for path in Path(cache_dir).rglob("*") if path.is_file()}
            self._assert_loads_same(loaded, exported, inputs)
        self.assertIn("model.onnx.data", downloaded)
        self.assertNotIn("model.safetensors", downloaded)
