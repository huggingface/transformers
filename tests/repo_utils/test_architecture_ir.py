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
SCHEMA_PATH = ARCHITECTURE_IR_PATH / "schema"
if str(ARCHITECTURE_IR_PATH) not in sys.path:
    sys.path.append(str(ARCHITECTURE_IR_PATH))

import generate_architecture_ir  # noqa: E402
from architecture_ir import resolve_template_to_graph  # noqa: E402


try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def _load_schema(name):
    with (SCHEMA_PATH / name).open(encoding="utf-8") as handle:
        return json.load(handle)


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
            self.assertEqual(manifest["artifact_schema_version"], "architecture-template-v0")
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

                # The canonical artifact is an ArchitectureTemplate.
                self.assertEqual(artifacts[model_type]["schema_version"], "architecture-template-v0")
                self.assertEqual(artifacts[model_type]["metadata"]["level"], "architecture_template")

                # The full config must NOT be serialized: only referenced + salient scalar fields.
                config = artifacts[model_type]["config"]
                self.assertIn("referenced_fields", config)
                self.assertIn("salient_fields", config)
                self.assertNotIn("fields", config)
                config_text = json.dumps(config)
                for junk in ("id2label", "transformers_version", "return_dict", "output_attentions", "bos_token_id"):
                    self.assertNotIn(junk, config_text)

                artifact_text = json.dumps(artifacts[model_type], indent=2)
                self.assertLessEqual(artifact_text.count("\n"), 700)

            # referenced_fields must expose exactly the config knobs the repeats reference.
            self.assertEqual(artifacts["llama"]["config"]["referenced_fields"], {"num_hidden_layers": 32})
            self.assertEqual(artifacts["llama"]["config"]["salient_fields"]["hidden_size"], 4096)
            self.assertEqual(set(artifacts["t5"]["config"]["referenced_fields"]), {"num_layers", "num_decoder_layers"})

            # Semantic-depth facts: view / family / attention variant / positional scheme.
            self.assertEqual(artifacts["llama"]["architecture"]["view"], "decoder")
            self.assertEqual(artifacts["llama"]["architecture"]["family"], "causal_lm")
            self.assertEqual(artifacts["llama"]["architecture"]["attention_variant"], "MHA")
            self.assertEqual(artifacts["llama"]["architecture"]["positional"], "rope")
            self.assertEqual(artifacts["bert"]["architecture"]["view"], "encoder")
            self.assertEqual(artifacts["bert"]["architecture"]["family"], "masked_lm")
            self.assertEqual(artifacts["bert"]["architecture"]["positional"], "learned")
            self.assertEqual(artifacts["t5"]["architecture"]["view"], "enc_dec")
            self.assertEqual(artifacts["t5"]["architecture"]["positional"], "relative")

            # Attention components carry normalized attributes.
            llama_attn = next(c for c in artifacts["llama"]["templates"] if c["kind"] == "attention")
            self.assertEqual(llama_attn["attributes"]["variant"], "MHA")
            self.assertTrue(llama_attn["attributes"]["rope"])

            # Reserved edge kinds are now emitted: decoder self-attention has a KV cache;
            # an encoder (bert) has none.
            self.assertTrue([e for e in artifacts["llama"]["edges"] if "cache" in e["kind"]])
            self.assertEqual([e for e in artifacts["bert"]["edges"] if "cache" in e["kind"]], [])

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

            if HAS_JSONSCHEMA:
                template_schema = _load_schema("architecture-template-v0.schema.json")
                for model_type, artifact in artifacts.items():
                    with self.subTest(model_type=model_type):
                        jsonschema.validate(instance=artifact, schema=template_schema)

    def test_resolved_graph_evaluates_counts_and_stays_symbolic(self):
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
                generate_architecture_ir.main(["--architectures", "llama", "--output-dir", tmp_dir])

            with (Path(tmp_dir) / "artifacts" / "llama.json").open(encoding="utf-8") as handle:
                template = json.load(handle)

            # A checkpoint config with a layer count that differs from the model default.
            checkpoint_config = {"model_type": "llama", "num_hidden_layers": 80}
            resolved = resolve_template_to_graph(template, checkpoint_config, config_source="fake/llama-checkpoint")

            self.assertEqual(resolved["schema_version"], "resolved-graph-v0")
            self.assertEqual(resolved["metadata"]["level"], "resolved_graph")
            self.assertEqual(resolved["template_ref"]["schema_version"], "architecture-template-v0")
            self.assertEqual(resolved["config"]["referenced_fields"]["num_hidden_layers"], 80)

            resolved_repeats = _repeats_by_id(resolved)
            self.assertEqual(resolved_repeats["decoder_layers"]["count"], 80)
            self.assertTrue(resolved_repeats["decoder_layers"]["count_resolved"])
            # count_expr is preserved for provenance.
            self.assertEqual(resolved_repeats["decoder_layers"]["count_expr"], "config.num_hidden_layers")

            # The resolved graph stays symbolic: exactly one repeated body, not 80 instances.
            decoder_bodies = [c for c in resolved["templates"] if c["id"] == "decoder_layer"]
            self.assertEqual(len(decoder_bodies), 1)
            resolved_text = json.dumps(resolved)
            self.assertNotIn("model.layers.0", resolved_text)
            self.assertNotIn("model.layers.79", resolved_text)

            if HAS_JSONSCHEMA:
                jsonschema.validate(instance=resolved, schema=_load_schema("resolved-graph-v0.schema.json"))

    def test_moe_semantic_depth_and_route_edges(self):
        from transformers.models.auto import configuration_auto, modeling_auto

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
                patch.object(
                    configuration_auto.AutoConfig,
                    "from_pretrained",
                    side_effect=AssertionError("must not load checkpoint configs."),
                ),
                patch.object(
                    modeling_auto.AutoModel,
                    "from_pretrained",
                    side_effect=AssertionError("must not load checkpoint weights."),
                ),
            ):
                generate_architecture_ir.main(["--architectures", "mixtral", "--output-dir", tmp_dir])

            with (Path(tmp_dir) / "artifacts" / "mixtral.json").open(encoding="utf-8") as handle:
                artifact = json.load(handle)

            arch = artifact["architecture"]
            self.assertTrue(arch["is_moe"])
            self.assertEqual(arch["moe"]["num_experts"], 8)
            self.assertEqual(arch["moe"]["experts_per_token"], 2)
            # Mixtral uses grouped-query attention.
            self.assertEqual(arch["attention_variant"], "GQA")

            # A MoE block routes to its experts (reserved 'route' edge kind is now emitted).
            route_edges = [e for e in artifact["edges"] if e["kind"] == "route"]
            self.assertTrue(route_edges)
            self.assertTrue(any("expert" in e["target"].lower() for e in route_edges))

            if HAS_JSONSCHEMA:
                jsonschema.validate(instance=artifact, schema=_load_schema("architecture-template-v0.schema.json"))

    def test_observed_dataflow_has_symbolic_shapes(self):
        from transformers.models.auto import configuration_auto, modeling_auto

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
                patch.object(
                    configuration_auto.AutoConfig,
                    "from_pretrained",
                    side_effect=AssertionError("must not load checkpoint configs."),
                ),
                patch.object(
                    modeling_auto.AutoModel,
                    "from_pretrained",
                    side_effect=AssertionError("must not load checkpoint weights."),
                ),
            ):
                generate_architecture_ir.main(["--architectures", "llama", "--output-dir", tmp_dir])

            with (Path(tmp_dir) / "artifacts" / "llama.json").open(encoding="utf-8") as handle:
                artifact = json.load(handle)

            self.assertIn("dataflow", artifact)
            dataflow = artifact["dataflow"]
            self.assertEqual(dataflow["source"], "observed_forward_meta")

            # Shapes are symbolized (config-parametric), not baked concrete integers.
            self.assertEqual(dataflow["input"]["shape"], ["B", "S"])
            self.assertEqual(dataflow["output"]["shape"], ["B", "S", "config.hidden_size"])

            stages = {stage["id"]: stage for stage in dataflow["stages"] if stage["id"]}
            self.assertEqual(stages["embed_tokens"]["out_shape"], ["B", "S", "config.hidden_size"])
            self.assertEqual(stages["decoder_layers"]["out_shape"], ["B", "S", "config.hidden_size"])
            # The concrete default hidden size must not leak into the observed shapes.
            self.assertNotIn("4096", json.dumps(dataflow))

            # Observed data edges skip position modules (rotary is not a data-path carrier).
            observed_data = {
                (e["source"], e["target"]) for e in artifact["edges"] if e.get("provenance") == "observed_forward"
            }
            self.assertNotIn("rotary_emb", {node for edge in observed_data for node in edge})

    def test_modular_diff_extends_and_patches(self):
        from transformers.models.auto import configuration_auto, modeling_auto

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
                patch.object(
                    configuration_auto.AutoConfig,
                    "from_pretrained",
                    side_effect=AssertionError("must not load checkpoint configs."),
                ),
                patch.object(
                    modeling_auto.AutoModel,
                    "from_pretrained",
                    side_effect=AssertionError("must not load checkpoint weights."),
                ),
            ):
                generate_architecture_ir.main(["--architectures", "gemma", "qwen2", "bert", "--output-dir", tmp_dir])

            artifacts = {}
            for model_type in ("gemma", "qwen2", "bert"):
                with (Path(tmp_dir) / "artifacts" / f"{model_type}.json").open(encoding="utf-8") as handle:
                    artifacts[model_type] = json.load(handle)

            # bert is standalone: no modular file.
            bert = artifacts["bert"]
            self.assertIsNone(bert["extends"])
            self.assertFalse(bert["modularity"]["is_modular"])
            self.assertEqual(bert["patches"], [])

            # gemma and qwen2 are modular and inherit from llama.
            for model_type in ("gemma", "qwen2"):
                art = artifacts[model_type]
                self.assertEqual(art["extends"], "llama")
                self.assertTrue(art["modularity"]["is_modular"])
                self.assertGreater(len(art["patches"]), 0)
                self.assertGreater(art["modularity"]["diff_size"], 0)

            # qwen2 is a canonical clean-modular exemplar: its attention is overridden, not rebuilt.
            qwen2 = artifacts["qwen2"]
            attn_patches = [p for p in qwen2["patches"] if p["component_kind"] == "attention"]
            self.assertTrue(attn_patches)
            self.assertTrue(all(p["relation"] == "inherits" for p in attn_patches))
            # A clean modular model is much smaller than a heavier one.
            self.assertLess(qwen2["modularity"]["diff_size"], artifacts["gemma"]["modularity"]["diff_size"])

            if HAS_JSONSCHEMA:
                template_schema = _load_schema("architecture-template-v0.schema.json")
                for art in artifacts.values():
                    jsonschema.validate(instance=art, schema=template_schema)

    def test_missing_config_field_is_a_warning_not_a_failure(self):
        template = {
            "schema_version": "architecture-template-v0",
            "model_type": "llama",
            "repeats": [
                {
                    "id": "decoder_layers",
                    "kind": "symbolic_repeat",
                    "body": "decoder_layer",
                    "count_expr": "config.num_hidden_layers",
                    "count_source": "config",
                    "index_symbol": "i",
                    "item_path_pattern": "model.layers.{i}",
                }
            ],
        }
        resolved = resolve_template_to_graph(template, {"model_type": "llama"})
        resolved_repeats = _repeats_by_id(resolved)
        self.assertFalse(resolved_repeats["decoder_layers"]["count_resolved"])
        self.assertTrue(resolved["warnings"])


def _repeats_by_id(artifact):
    return {repeat["id"]: repeat for repeat in artifact["repeats"]}


def _path_patterns(artifact):
    return {component["path_pattern"] for component in [*artifact["components"], *artifact["templates"]]}


if __name__ == "__main__":
    unittest.main()
