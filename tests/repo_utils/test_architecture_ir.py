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
from architecture_ir import build_modular_graph, resolve_template_to_graph  # noqa: E402


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
                # Boilerplate blocks were removed for leanness — must NOT reappear.
                for gone in ("metadata", "entrypoints", "modularity"):
                    self.assertNotIn(gone, artifacts[model_type])
                # Provenance is slimmed to the source classes; no per-node `module`.
                self.assertIn("config_class", artifacts[model_type]["provenance"])
                self.assertNotIn("module", artifacts[model_type]["templates"][0])

                # The full config must NOT be serialized: only referenced + salient scalar fields.
                config = artifacts[model_type]["config"]
                self.assertIn("referenced_fields", config)
                self.assertIn("salient_fields", config)
                self.assertNotIn("fields", config)
                config_text = json.dumps(config)
                for junk in ("id2label", "transformers_version", "return_dict", "output_attentions", "bos_token_id"):
                    self.assertNotIn(junk, config_text)

                # Soft compactness ceiling. Richer per-template detail (projections with dims) grows
                # this, but symbolic repeats + the no-concrete-path checks below are the real guards
                # against full-layer expansion (which would be many thousands of lines).
                artifact_text = json.dumps(artifacts[model_type], indent=2)
                self.assertLessEqual(artifact_text.count("\n"), 1200)

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

            # The IR descends into attention/MLP projections (once per template, still symbolic),
            # with config-parametric dims — so a viewer can render q/k/v/o and gate/up/down.
            templates_by_id = {c["id"]: c for c in artifacts["llama"]["templates"]}
            self.assertIn("decoder_layer.self_attn.q_proj", templates_by_id)
            self.assertEqual(templates_by_id["decoder_layer.self_attn.q_proj"]["kind"], "projection")
            gate = templates_by_id["decoder_layer.mlp.gate_proj"]
            self.assertEqual(gate["attributes"]["in_features"], "config.hidden_size")
            self.assertEqual(gate["attributes"]["out_features"], "config.intermediate_size")
            down = templates_by_id["decoder_layer.mlp.down_proj"]
            self.assertEqual(down["attributes"]["in_features"], "config.intermediate_size")
            # Tensor-parallel style is annotated per projection from the model's base_model_tp_plan.
            self.assertEqual(gate["attributes"]["tp"], "colwise")
            self.assertEqual(down["attributes"]["tp"], "rowwise")

            # Capabilities block: attention backends, patterns, and available task heads.
            caps = artifacts["llama"]["capabilities"]
            self.assertIn("sdpa", caps["attention_backends"])
            self.assertIn("eager", caps["attention_backends"])
            self.assertEqual(caps["attention_patterns"], ["causal"])
            self.assertIn("causal_lm", caps["task_heads"])
            self.assertTrue(caps["tensor_parallel"])
            self.assertEqual(artifacts["bert"]["capabilities"]["attention_patterns"], ["bidirectional"])
            self.assertIn("masked_lm", artifacts["bert"]["capabilities"]["task_heads"])

            # Kernelizable layers: llama's RMSNorm is @use_kernel_forward_from_hub, with Hub repos.
            self.assertIn("RMSNorm", caps["kernels"])
            self.assertTrue(caps["kernels"]["RMSNorm"])  # non-empty repo list (read from source, offline)
            norm = next(c for c in artifacts["llama"]["templates"] if c["kind"] == "normalization")
            self.assertEqual(norm["attributes"]["kernel"], "RMSNorm")
            self.assertEqual(norm["attributes"]["norm_type"], "rms")  # kernel merges with existing attrs
            # The MLP container carries dims + activation for its caption.
            mlp = templates_by_id["decoder_layer.mlp"]
            self.assertEqual(mlp["attributes"]["activation"], "silu")
            self.assertEqual(mlp["attributes"]["intermediate_size"], 11008)
            # Projections are attached as children of their host so the tree is navigable.
            self.assertIn("decoder_layer.mlp.gate_proj", mlp["children"])

            # Embedding and position nodes carry config-parametric detail (no bare structural nodes
            # left except pure containers).
            embed = next(c for c in artifacts["llama"]["components"] if c["kind"] == "embedding")
            self.assertEqual(embed["attributes"]["num_embeddings"], "config.vocab_size")
            self.assertEqual(embed["attributes"]["embedding_dim"], "config.hidden_size")
            rotary = next(c for c in artifacts["llama"]["components"] if c["kind"] == "position")
            self.assertEqual(rotary["attributes"]["scheme"], "rope")
            self.assertEqual(rotary["attributes"]["head_dim"], "config.head_dim")
            # t5's relative position bias is recognized as its own scheme.
            t5_pos = next(c for c in artifacts["t5"]["templates"] if c["kind"] == "position")
            self.assertEqual(t5_pos["attributes"]["scheme"], "relative")

            # Intra-module dataflow: q/k/v fan out from attention and fan into o_proj (parallel,
            # not a spurious chain); gate/up fan into down_proj. Drives the poster-style layout.
            data = {(e["source"], e["target"]) for e in artifacts["llama"]["edges"] if e["kind"] == "data"}
            for proj in ("q_proj", "k_proj", "v_proj"):
                self.assertIn(("decoder_layer.self_attn", f"decoder_layer.self_attn.{proj}"), data)
                self.assertIn((f"decoder_layer.self_attn.{proj}", "decoder_layer.self_attn.o_proj"), data)
            self.assertIn(("decoder_layer.mlp", "decoder_layer.mlp.gate_proj"), data)
            self.assertIn(("decoder_layer.mlp.gate_proj", "decoder_layer.mlp.down_proj"), data)
            # q/k/v are siblings, not chained to each other.
            self.assertNotIn(("decoder_layer.self_attn.q_proj", "decoder_layer.self_attn.k_proj"), data)
            # The re-grounded block-level chain is still intact alongside the finer edges.
            self.assertIn(("decoder_layer.input_layernorm", "decoder_layer.self_attn"), data)

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

    def test_nested_submodels_stay_connected(self):
        # Multimodal models nest full sub-models (vision_tower, language_model, ...) as attributes.
        # Those wrappers must survive as component nodes so the tree stays reachable from the root.
        from transformers.models.auto import configuration_auto, modeling_auto

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
                patch.object(configuration_auto.AutoConfig, "from_pretrained", side_effect=AssertionError("no")),
                patch.object(modeling_auto.AutoModel, "from_pretrained", side_effect=AssertionError("no")),
            ):
                generate_architecture_ir.main(["--architectures", "gemma3", "--output-dir", tmp_dir])

            with (Path(tmp_dir) / "artifacts" / "gemma3.json").open(encoding="utf-8") as handle:
                artifact = json.load(handle)

            root = next(c for c in artifact["components"] if c["id"] == "model")
            self.assertTrue(root["children"], "root model must list its sub-model children")
            for wrapper in ("vision_tower", "language_model", "multi_modal_projector"):
                self.assertIn(wrapper, root["children"])
                self.assertTrue(any(c["id"] == wrapper for c in artifact["components"]))

            # No component (other than the root) may be orphaned from the children tree.
            reachable = set()
            for c in artifact["components"]:
                reachable.update(c.get("children", []))
            orphans = [c["id"] for c in artifact["components"] if c["id"] != "model" and c["id"] not in reachable]
            self.assertEqual(orphans, [])

            # Multimodal facts come from the text backbone, not None.
            self.assertEqual(artifact["architecture"]["view"], "multimodal")
            self.assertIsNotNone(artifact["architecture"].get("attention_variant"))

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

            # dataflow is a flat shape lookup keyed by semantic id (ordering lives in edges).
            shapes = dataflow["shapes"]
            self.assertEqual(shapes["embed_tokens"]["out"], ["B", "S", "config.hidden_size"])
            self.assertEqual(shapes["decoder_layers"]["out"], ["B", "S", "config.hidden_size"])
            self.assertNotIn("stages", dataflow)
            self.assertNotIn("blocks", dataflow)
            # The concrete default hidden size must not leak into the observed shapes.
            self.assertNotIn("4096", json.dumps(dataflow))

            # Observed data edges skip position modules (rotary is not a data-path carrier).
            observed_data = {
                (e["source"], e["target"]) for e in artifact["edges"] if e.get("provenance") == "observed_forward"
            }
            self.assertNotIn("rotary_emb", {node for edge in observed_data for node in edge})

            # Intra-block order is re-grounded from the observed forward (pre-norm), not module
            # registration order — the input norm precedes attention rather than dangling at the end.
            data_edges = {(e["source"], e["target"]) for e in artifact["edges"] if e["kind"] == "data"}
            self.assertIn(("decoder_layer.input_layernorm", "decoder_layer.self_attn"), data_edges)
            self.assertIn(("decoder_layer.self_attn", "decoder_layer.post_attention_layernorm"), data_edges)
            # The old registration-order edge (attention straight to mlp) is gone.
            self.assertNotIn(("decoder_layer.self_attn", "decoder_layer.mlp"), data_edges)

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

            # A rough per-model diff metric derived from the (lean) patches — the modularity summary
            # block was removed from the artifact; the diff_size metric now lives in modular_graph.json.
            def diff_metric(art):
                size = 0
                for p in art.get("patches", []):
                    for bucket in ("overridden", "added", "deleted"):
                        m = p.get(bucket, {})
                        size += len(m.get("methods", [])) + len(m.get("attrs", []))
                    size += 3 if p["relation"] == "new" else 0
                return size

            # bert is standalone: no modular file → extends null, no patches.
            bert = artifacts["bert"]
            self.assertIsNone(bert["extends"])
            self.assertNotIn("patches", bert)

            # gemma and qwen2 are modular and inherit from llama.
            for model_type in ("gemma", "qwen2"):
                art = artifacts[model_type]
                self.assertEqual(art["extends"], "llama")
                self.assertGreater(len(art["patches"]), 0)

            # qwen2 is a canonical clean-modular exemplar: its attention is overridden, not rebuilt.
            qwen2 = artifacts["qwen2"]
            attn_patches = [p for p in qwen2["patches"] if p["component_kind"] == "attention"]
            self.assertTrue(attn_patches)
            self.assertTrue(all(p["relation"] == "inherits" for p in attn_patches))
            # A clean modular model is much smaller than a heavier one.
            self.assertLess(diff_metric(qwen2), diff_metric(artifacts["gemma"]))

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


class ModularGraphTest(unittest.TestCase):
    """The library-wide modular inheritance forest (pure ast, no torch/model build)."""

    def test_lineage_chain_and_roots(self):
        # Seed a known chain; ancestors are resolved transitively for correct roots.
        graph = build_modular_graph(only=["qwen3", "gemma"])
        nodes = graph["nodes"]

        self.assertEqual(nodes["qwen3"]["extends"], "qwen2")
        self.assertEqual(nodes["qwen2"]["extends"], "llama")
        self.assertEqual(nodes["gemma"]["extends"], "llama")

        # Transitive root + depth make lineage explicit: qwen3 -> qwen2 -> llama.
        self.assertEqual(nodes["qwen3"]["root"], "llama")
        self.assertEqual(nodes["qwen3"]["depth"], 2)
        self.assertEqual(nodes["llama"]["root"], "llama")
        self.assertEqual(nodes["llama"]["depth"], 0)

        # Reverse links: children point back down the tree.
        self.assertIn("qwen3", nodes["qwen2"]["children"])
        self.assertIn("qwen2", nodes["llama"]["children"])
        self.assertIn("gemma", nodes["llama"]["children"])

        # A spine base with descendants is a root; a leaf is not.
        self.assertIn("llama", graph["roots"])
        self.assertNotIn("qwen3", graph["roots"])

        if HAS_JSONSCHEMA:
            jsonschema.validate(instance=graph, schema=_load_schema("modular-graph-v0.schema.json"))

    def test_isolated_standalone_is_pruned(self):
        # A standalone model with no descendants in the seed set is not a node.
        graph = build_modular_graph(only=["bert"])
        self.assertNotIn("bert", graph["nodes"])
        self.assertEqual(graph["roots"], [])


class KernelizationTest(unittest.TestCase):
    """Kernel detection is pure source `ast` — no torch, no `kernels` package, offline."""

    def test_decorated_layers_and_repos_from_source(self):
        from architecture_ir.kernelization import detect_kernel_layers, kernel_repositories
        from architecture_ir.modular_graph import models_root

        root = models_root()
        # gpt_oss decorates both a norm and its MoE MLP.
        gpt_oss = detect_kernel_layers(os.path.join(root, "gpt_oss"))
        self.assertEqual(gpt_oss.get("GptOssRMSNorm"), "RMSNorm")
        self.assertEqual(gpt_oss.get("GptOssMLP"), "MegaBlocksMoeMLP")

        repos = kernel_repositories()
        self.assertTrue(repos.get("RMSNorm"))  # e.g. kernels-community/rmsnorm
        self.assertTrue(repos.get("MegaBlocksMoeMLP"))  # e.g. kernels-community/megablocks
        self.assertTrue(all(isinstance(r, str) for r in repos["RMSNorm"]))


def _repeats_by_id(artifact):
    return {repeat["id"]: repeat for repeat in artifact["repeats"]}


def _path_patterns(artifact):
    return {component["path_pattern"] for component in [*artifact["components"], *artifact["templates"]]}


if __name__ == "__main__":
    unittest.main()
