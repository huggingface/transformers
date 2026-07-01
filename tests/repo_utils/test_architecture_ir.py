# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers.testing_utils import require_torch


REPO_PATH = Path(__file__).resolve().parents[2]
ARCHITECTURE_IR_PATH = REPO_PATH / "utils" / "architecture_ir"
if str(ARCHITECTURE_IR_PATH) not in sys.path:
    sys.path.append(str(ARCHITECTURE_IR_PATH))

import generate_architecture_ir  # noqa: E402


@require_torch
class ArchitectureIrGeneratorTest(unittest.TestCase):
    def test_generates_manifest_and_artifacts_without_checkpoint_loading(self):
        from transformers.models.auto import configuration_auto, modeling_auto

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
                patch.object(
                    configuration_auto.AutoConfig,
                    "from_pretrained",
                    side_effect=AssertionError("Architecture IR generation must not load checkpoint configs."),
                ),
                patch.object(
                    modeling_auto.AutoModel,
                    "from_pretrained",
                    side_effect=AssertionError("Architecture IR generation must not load checkpoint weights."),
                ),
            ):
                exit_code = generate_architecture_ir.main(
                    ["--architectures", "llama", "bert", "t5", "--output-dir", tmp_dir]
                )

            self.assertEqual(exit_code, 0)
            manifest_path = Path(tmp_dir) / "manifest.json"
            self.assertTrue(manifest_path.exists())

            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["schema_version"], "architecture-ir-manifest-v0")
            self.assertEqual(manifest["failures"], [])
            self.assertEqual({entry["model_type"] for entry in manifest["architectures"]}, {"llama", "bert", "t5"})

            artifacts = {}
            for model_type in ("llama", "bert", "t5"):
                artifact_path = Path(tmp_dir) / "artifacts" / f"{model_type}.json"
                self.assertTrue(artifact_path.exists())
                with artifact_path.open(encoding="utf-8") as handle:
                    artifacts[model_type] = json.load(handle)

                for field in (
                    "schema_version",
                    "model_type",
                    "metadata",
                    "entrypoints",
                    "config",
                    "components",
                    "templates",
                    "repeats",
                    "edges",
                    "provenance",
                ):
                    self.assertIn(field, artifacts[model_type])

                artifact_text = json.dumps(artifacts[model_type], indent=2)
                self.assertLessEqual(artifact_text.count("\n"), 700)

            llama_repeats = _repeats_by_id(artifacts["llama"])
            self.assertEqual(llama_repeats["decoder_layers"]["body"], "decoder_layer")
            self.assertEqual(llama_repeats["decoder_layers"]["count_expr"], "config.num_hidden_layers")
            self.assertEqual(llama_repeats["decoder_layers"]["item_path_pattern"], "model.layers.{i}")
            self.assertIn("model.layers.{i}.self_attn", _path_patterns(artifacts["llama"]))

            bert_repeats = _repeats_by_id(artifacts["bert"])
            self.assertEqual(bert_repeats["encoder_layers"]["body"], "encoder_layer")
            self.assertEqual(bert_repeats["encoder_layers"]["count_expr"], "config.num_hidden_layers")
            self.assertEqual(bert_repeats["encoder_layers"]["item_path_pattern"], "model.encoder.layer.{i}")
            self.assertIn("model.encoder.layer.{i}.attention", _path_patterns(artifacts["bert"]))

            t5_repeats = _repeats_by_id(artifacts["t5"])
            self.assertEqual(t5_repeats["encoder_blocks"]["body"], "encoder_block")
            self.assertEqual(t5_repeats["encoder_blocks"]["count_expr"], "config.num_layers")
            self.assertEqual(t5_repeats["encoder_blocks"]["item_path_pattern"], "model.encoder.block.{i}")
            self.assertEqual(t5_repeats["decoder_blocks"]["body"], "decoder_block")
            self.assertEqual(t5_repeats["decoder_blocks"]["count_expr"], "config.num_decoder_layers")
            self.assertEqual(t5_repeats["decoder_blocks"]["item_path_pattern"], "model.decoder.block.{i}")

            edge_kinds = set()
            for artifact in artifacts.values():
                edge_kinds.update(edge["kind"] for edge in artifact["edges"])
            self.assertTrue({"data", "residual", "mask", "position", "cross_attention"}.issubset(edge_kinds))

            canonical_text = json.dumps(artifacts, indent=2)
            for concrete_path in (
                "model.layers.0",
                "model.layers.1",
                "model.encoder.layer.0",
                "model.encoder.layer.1",
                "model.encoder.block.0",
                "model.decoder.block.0",
            ):
                self.assertNotIn(concrete_path, canonical_text)


def _repeats_by_id(artifact):
    return {repeat["id"]: repeat for repeat in artifact["repeats"]}


def _path_patterns(artifact):
    return {component["path_pattern"] for component in [*artifact["components"], *artifact["templates"]]}


if __name__ == "__main__":
    unittest.main()
