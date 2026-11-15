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
"""Saving and loading exports, one class per backend: each export saves, loads back as the type the export returned,
and names in its metadata the inputs and outputs of the saved file. ONNX adds its own cases: graphs past protobuf's
2GB limit, bfloat16, and exports loaded from the Hub cache.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pytest

from tests.exporters.test_runtime import _decompose_static_decode, _tiny_model
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.exporters import ExportedModel
from transformers.exporters.utils import get_leaf_tensors
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch,
    require_torch_gpu,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


class SaveLoadTesterMixin:
    def _save_and_load_decode(self, exporter, config, directory):
        """Export the static-cache decode, save it and load it back, checking that the backend's `program_files`
        (what a Hub download fetches) name every file it wrote. Returns the manifest entry and the loaded export."""
        decode_model, decode_inputs = _decompose_static_decode(_tiny_model(), torch.randint(0, 64, (1, 4)))
        exported = exporter.export(decode_model, copy.deepcopy(decode_inputs), config=config)
        exported.save_pretrained(directory)
        entry = json.loads((Path(directory) / "export.json").read_text(encoding="utf-8"))["components"]["model"]
        written = {path.name for path in Path(directory).iterdir()} - {"export.json", "config.json"}
        self.assertIn(entry["file"], written)
        self.assertLessEqual(written, set(exporter.program_files(entry["file"])))
        loaded = ExportedModel.from_pretrained(directory)
        self.assertEqual(loaded.components["model"].metadata, exported.components["model"].metadata)
        return entry, loaded


@slow
@require_torch
@pytest.mark.torch_export_test
class DynamoSaveLoadTest(SaveLoadTesterMixin, unittest.TestCase):
    def test_saved_program_loads_back(self):
        from transformers.exporters import DynamoConfig, DynamoExporter

        with tempfile.TemporaryDirectory() as directory:
            entry, loaded = self._save_and_load_decode(DynamoExporter(), DynamoConfig(dynamic=True), directory)
        self.assertEqual(entry["file"], "model.pt2")
        self.assertIsInstance(loaded.program, torch.export.ExportedProgram)


@slow
@require_torch
@require_onnxscript
@require_onnxruntime
@pytest.mark.onnx_export_test
class OnnxSaveLoadTest(SaveLoadTesterMixin, unittest.TestCase):
    def test_saved_graph_prefixes_mutated_inputs(self):
        """The saved `.onnx` file names a mutated cache's inputs with an `input.` prefix."""
        import onnxruntime as ort

        from transformers.exporters import OnnxConfig, OnnxExporter

        with tempfile.TemporaryDirectory() as directory:
            entry, _ = self._save_and_load_decode(OnnxExporter(), OnnxConfig(dynamic=True), directory)
            session = ort.InferenceSession(str(Path(directory) / entry["file"]), providers=["CPUExecutionProvider"])
        self.assertTrue(any(node.name.startswith("input.") for node in session.get_inputs()))

    def test_saved_program_runs(self):
        """A loaded `.onnx` runs like the exported program, ONNX Runtime reading its weights from
        `model.onnx.data`."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        exported = OnnxExporter().export(_tiny_model(), copy.deepcopy(inputs), OnnxConfig())
        expected = get_leaf_tensors(exported(**inputs))
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = ExportedModel.from_pretrained(directory)
            torch.testing.assert_close(get_leaf_tensors(loaded(**inputs)), expected)

    def test_loaded_through_symlinks_runs_and_saves(self):
        """Loaded as from the Hub cache, whose snapshot links each file to a blob outside it, an ONNX program runs and
        saves again."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        exported = OnnxExporter().export(_tiny_model(), copy.deepcopy(inputs), OnnxConfig())
        expected = get_leaf_tensors(exported(**inputs))
        with tempfile.TemporaryDirectory() as directory:
            blobs, snapshot = Path(directory, "blobs"), Path(directory, "snapshots", "main")
            exported.save_pretrained(blobs / "saved")
            snapshot.mkdir(parents=True)
            for file in (blobs / "saved").iterdir():
                blob = file.rename(blobs / f"blob-{file.name}")
                (snapshot / file.name).symlink_to(Path("..", "..", "blobs", blob.name))
            loaded = ExportedModel.from_pretrained(snapshot)
            torch.testing.assert_close(get_leaf_tensors(loaded(**inputs)), expected)
            loaded.save_pretrained(Path(directory, "resaved"))
            resaved = ExportedModel.from_pretrained(Path(directory, "resaved"))
            torch.testing.assert_close(get_leaf_tensors(resaved(**inputs)), expected)

    @require_torch_gpu
    def test_bfloat16_saves_and_loads(self):
        """A bfloat16 export runs in bfloat16, close to eager, and its saved graph runs like the exported one. On CUDA:
        ONNX Runtime's CPU provider has no bfloat16 kernels."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        model = _tiny_model().to(torch.bfloat16)
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        with torch.no_grad():
            expected = model(**inputs).logits
        exported = OnnxExporter().export(model, copy.deepcopy(inputs), OnnxConfig())
        outputs = get_leaf_tensors(exported(**inputs))
        self.assertEqual(outputs["logits"].dtype, torch.bfloat16)
        torch.testing.assert_close(outputs["logits"].cpu(), expected, atol=1e-2, rtol=1e-2)
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = ExportedModel.from_pretrained(directory)
            torch.testing.assert_close(get_leaf_tensors(loaded(**inputs)), outputs)

    def test_quantized_saves_and_loads(self):
        """A graph ONNX Runtime's quantizer rewrote saves and loads: its large `Constant` values (a static mask's
        indices, 8 x 32 int64) come back in memory, not pointing at the quantizer's deleted temporary file."""
        from transformers.exporters import OnnxConfig, OnnxExporter, OnnxRuntimeDynamicConfig, OnnxRuntimeQuantizer

        inputs = {"input_ids": torch.randint(0, 64, (8, 32)), "attention_mask": torch.ones(8, 32, dtype=torch.long)}
        quantizer = OnnxRuntimeQuantizer(OnnxRuntimeDynamicConfig(op_types_to_quantize=["MatMul", "Gemm"]))
        exported = OnnxExporter().export(_tiny_model(), copy.deepcopy(inputs), OnnxConfig(quantizer=quantizer))
        expected = get_leaf_tensors(exported(**inputs))
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            torch.testing.assert_close(get_leaf_tensors(ExportedModel.from_pretrained(directory)(**inputs)), expected)

    def test_past_2gb_saves_and_loads(self):
        """A model past protobuf's 2GB limit saves with its weights in `model.onnx.data`, loads, and runs like the
        exported program. A one-layer Llama with a 150k vocabulary holds 2.3GB of untied embeddings."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        config = LlamaConfig(
            hidden_size=2048,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=150_000,
        )
        model = LlamaForCausalLM(config).eval()
        self.assertGreater(sum(p.numel() * p.element_size() for p in model.parameters()), 2**31)
        inputs = {"input_ids": torch.randint(0, 64, (1, 8)), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        exported = OnnxExporter().export(model, copy.deepcopy(inputs), OnnxConfig())
        expected = get_leaf_tensors(exported(**inputs))

        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            self.assertLess((Path(directory) / "model.onnx").stat().st_size, 2**31)
            self.assertGreater((Path(directory) / "model.onnx.data").stat().st_size, 2**31)
            outputs = get_leaf_tensors(ExportedModel.from_pretrained(directory)(**inputs))
        torch.testing.assert_close(outputs, expected, atol=0, rtol=0)


@slow
@require_torch
@require_executorch
@pytest.mark.executorch_export_test
class ExecutorchSaveLoadTest(SaveLoadTesterMixin, unittest.TestCase):
    def test_saved_program_loads_back(self):
        from executorch.runtime import Program

        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        with tempfile.TemporaryDirectory() as directory:
            entry, loaded = self._save_and_load_decode(ExecutorchExporter(), ExecutorchConfig(), directory)
            self.assertEqual(entry["file"], "model.pte")
            self.assertIsNotNone(entry["metadata"]["in_spec"])
            self.assertIsNotNone(entry["metadata"]["out_spec"])
            self.assertIsInstance(loaded.program, Program)
            self.assertIn("forward", loaded.program.method_names)
            saved = Path(directory, "model.pte").read_bytes()
            with tempfile.TemporaryDirectory() as elsewhere:
                loaded.save_pretrained(elsewhere)
                self.assertEqual(Path(elsewhere, "model.pte").read_bytes(), saved)
            loaded.save_pretrained(directory)
            self.assertEqual(Path(directory, "model.pte").read_bytes(), saved)
            self.assertIn("forward", ExportedModel.from_pretrained(directory).program.method_names)


@slow
@require_torch
@require_openvino
@pytest.mark.openvino_export_test
class OpenVINOSaveLoadTest(SaveLoadTesterMixin, unittest.TestCase):
    def test_saved_model_is_stateful(self):
        """The saved `.xml` model, `stateful`, has no cache ports but a `beam_idx` input."""
        import openvino

        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        with tempfile.TemporaryDirectory() as directory:
            entry, _ = self._save_and_load_decode(OpenVINOExporter(), OpenVINOConfig(dynamic=True), directory)
            self.assertEqual(entry["file"], "openvino_model.xml")
            ov_model = openvino.Core().read_model(str(Path(directory) / entry["file"]))
        self.assertIn("beam_idx", [port.get_any_name() for port in ov_model.inputs])

    def test_loaded_model_saves_in_place(self):
        """A loaded model saves back into the directory it was read from, whose weights OpenVINO still maps."""
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        with tempfile.TemporaryDirectory() as directory:
            _, loaded = self._save_and_load_decode(OpenVINOExporter(), OpenVINOConfig(dynamic=True), directory)
            loaded.save_pretrained(directory)
            reloaded = ExportedModel.from_pretrained(directory)
            self.assertEqual(reloaded.components["model"].metadata, loaded.components["model"].metadata)
            self.assertEqual(
                [port.get_any_name() for port in reloaded.program.inputs],
                [port.get_any_name() for port in loaded.program.inputs],
            )
