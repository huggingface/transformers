# Architecture IR — Specification v0 (Experimental)

This document specifies the **Architecture IR v0** contract: the on-disk shape of the artifacts produced by
`utils/architecture_ir` and consumed by downstream tooling (viewers, diffs, docs, Hub integrations).

Its purpose is to be a **stable contract** so that generator work and consumer/viewer work can proceed in parallel.
The prose here is normative for v0; the machine-checkable rules live in:

- `schema/architecture-template-v0.schema.json`
- `schema/resolved-graph-v0.schema.json`

See [`IR.md`](../../IR.md) at the repository root for the broader motivation and long-term vision. This file is the
concrete v0 data contract.

> **Status:** experimental. This is a local-only prototype. Nothing here is a stable public API under
> `src/transformers`.

---

## 1. Goals and non-goals

### Goals

- Describe the **semantic structure** of a model architecture independently of any checkpoint weights.
- Define **one reusable artifact per `model_type`** that is parameterized by configuration values and reused by every
  checkpoint of that architecture.
- Keep the canonical artifact **compact and human-readable**: repeated structures stay symbolic.
- Provide a second, **resolved** level obtained by combining the template with a checkpoint `config.json`, which
  consumers primarily read.
- Use **stable semantic IDs** as identity, so diffs, documentation, and inheritance are meaningful across versions.
- Make artifacts **schema-validatable** so producers and consumers agree on shape.

### Non-goals

- Checkpoint weights or tensor values.
- Full runtime/computation graphs, `torch.fx`, `torch.export`, or ONNX graphs.
- Full tensor-op tracing (the IR is coarse and semantic, not an executable graph).
- Kernel/backend implementations, quantization layouts.
- **Visualization layout** (positions, sizes, colors) — the IR is renderer-agnostic; layout is a consumer concern.
- Checkpoint-specific adapters (LoRA/PEFT, etc.).

---

## 2. Two levels of representation

The IR defines two artifact levels. They share the same semantic vocabulary (components, repeats, edges) but differ in
what is evaluated.

### 2.1 `ArchitectureTemplate` (canonical artifact)

- **One artifact per `model_type`** (e.g. `artifacts/llama.json`).
- **Config-parametric**: numeric structural values that depend on configuration are stored as **config expression
  strings** (e.g. `"config.num_hidden_layers"`), not as baked-in numbers.
- **Compact / template-based**: repeated blocks are represented **once** as a template body plus a symbolic repeat.
  The template MUST NOT serialize every concrete layer instance.
- This is the artifact intended to be reused across all checkpoints of an architecture and eventually stored on the Hub.
- `schema_version`: `"architecture-template-v0"`.

> The template MAY additionally carry values observed under the model's **default** config at generation time (for
> example a repeat's optional `count`). These are **provenance/defaults**, not the authoritative parametric form. The
> authoritative parametric form is always the config expression string.
>
> The template MUST NOT serialize the full checkpoint/default config. Instead `config` carries two compact, curated
> maps: **`referenced_fields`** (exactly the config knobs referenced by config expressions in the artifact, with their
> default values — the authoritative parametric surface) and **`salient_fields`** (a whitelist of architecture-defining
> scalar defaults such as `hidden_size`, `num_attention_heads`, `hidden_act`). Label maps, tokenizer ids, runtime flags,
> version strings, and nested config objects are deliberately excluded.

### 2.2 `ResolvedGraph`

- Produced by resolving an `ArchitectureTemplate` against a specific checkpoint `config.json`.
- Config-parametric values are **evaluated** to concrete numbers (e.g. `count_expr: "config.num_hidden_layers"` →
  `count: 32`).
- **Still symbolic and compact**: repeats keep a single body and are **not** expanded into one component per instance.
  Expanding to per-instance nodes is an optional, explicit consumer step and is out of scope for v0.
- Consumers (viewers, diffs, docs) should primarily read the resolved graph.
- `schema_version`: `"resolved-graph-v0"`.

```
ArchitectureTemplate  +  checkpoint config.json   ->   ResolvedGraph
       (per model_type)      (per checkpoint)              (per checkpoint)
```

---

## 3. Semantic components

A **component** is a semantic unit of the architecture. Components represent concepts such as embeddings, attention,
feed-forward/MLP, normalization, transformer blocks, encoders/decoders, poolers, position modules, etc. — rather than
arbitrary Python modules whenever possible.

Each component has:

| Field          | Required | Meaning                                                                        |
| -------------- | -------- | ------------------------------------------------------------------------------ |
| `id`           | yes      | **Stable semantic ID** — the component's primary identity (see §6).            |
| `kind`         | yes      | Semantic role (`embedding`, `attention`, `feed_forward`, `normalization`, …).  |
| `children`     | yes      | Semantic IDs of child components.                                              |
| `class_name`   | no       | **Provenance**: originating Python class name.                                 |
| `module`       | no       | **Provenance**: originating Python module path.                                |
| `path_pattern` | no       | **Provenance**: symbolic module path with `{i}` placeholders.                  |

Components are split across two arrays:

- `components`: semantic units reachable **outside** of any symbolic repeat body.
- `templates`: reusable **repeat bodies** and their descendants — serialized once, referenced by repeats.

`kind` is an open vocabulary (validated as a string). v0 recognizers commonly emit: `model`, `embedding`, `position`,
`attention`, `cross_attention`, `feed_forward`, `moe`, `normalization`, `pooler`, `transformer_block`, `encoder`,
`decoder`, `stack`, `repeated_container`. Consumers MUST tolerate unknown `kind` values.

### Semantic facts (`architecture` + component `attributes`)

Beyond `kind`, the generator derives normalized **semantic facts** (generically, from config + module tree — no
per-model special-casing):

- A top-level **`architecture`** block with model-level facts: `view` (`decoder`/`encoder`/`enc_dec`/`multimodal`),
  `family` (task family via the Auto\* mappings, e.g. `causal_lm`/`masked_lm`/`seq2seq`), `attention_variant`
  (`MHA`/`GQA`/`MQA`/`MLA`), `positional` (`rope`/`relative`/`alibi`/`learned`), `is_moe` (+ `moe` params),
  `sliding_window`, `tie_word_embeddings`. Fields that don't apply are omitted; a value being absent means
  "not present / undeterminable".
- Per-component **`attributes`**: attention components carry `{variant, n_heads, n_kv_heads, head_dim, rope,
  sliding_window}`; feed-forward components carry `{hidden_size, intermediate_size, activation}`; normalization
  components carry `{norm_type}` (`rms`/`layer`/…); MoE components carry `{num_experts, experts_per_token, …}`.

The IR also **descends into the projections** inside attention / MLP / MoE bodies — the `q/k/v/o_proj` and
`gate/up/down_proj` (kind `projection`) appear as children of their host with `{in_features, out_features}`, each
**symbolized to a config expression** where it matches a salient value (e.g. `config.hidden_size`,
`config.intermediate_size`; dims like `num_kv_heads · head_dim` that match no single field stay integers). This stays
compact — projections are serialized once in the template body, never per repeated layer.

These facts also drive the reserved edge kinds: decoder-side self-attention emits `cache_read`/`cache_write` edges to a
`state:kv_cache` pseudo-node, and a MoE block emits a `route` edge to its experts container.

---

## 4. Symbolic repeats

Homogeneous repeated blocks (e.g. a stack of identical decoder layers) are collapsed into a single **symbolic repeat**
plus one template body. The IR MUST NOT emit `layers.0`, `layers.1`, … as separate components.

A repeat has:

| Field                    | Required | Meaning                                                                       |
| ------------------------ | -------- | ----------------------------------------------------------------------------- |
| `id`                     | yes      | Stable semantic ID for the repeat (e.g. `decoder_layers`).                    |
| `kind`                   | yes      | Always `"symbolic_repeat"`.                                                   |
| `body`                   | yes      | Semantic ID of the repeated template body (e.g. `decoder_layer`).            |
| `count_expr`             | yes      | **Config expression string** for the count (see §7).                          |
| `count_source`           | yes      | `"config"` or `"module_tree"`.                                                |
| `index_symbol`           | yes      | Symbol used for the index in path patterns (e.g. `i`).                        |
| `item_path_pattern`      | yes      | **Provenance**: symbolic path of one item (e.g. `model.layers.{i}`).          |
| `count`                  | no       | Count observed under the **default** config — provenance/default only.        |
| `container_path_pattern` | no       | **Provenance**: symbolic path of the repeat container.                        |
| `repeated_class_name`    | no       | **Provenance**: originating Python class of the repeated block.               |
| `provenance`             | no       | Additional provenance.                                                        |

In a `ResolvedGraph`, a repeat additionally carries an evaluated `count` (concrete integer) and `count_resolved`
(boolean); `count_expr` is preserved for provenance.

---

## 5. Coarse dataflow edges

Edges capture the **high-level** flow of information between semantic IDs. They are intentionally coarse: the goal is
semantic understanding, not a full tensor-op graph.

An edge has `source`, `target`, and `kind`. `source`/`target` are semantic component/repeat IDs, or an
`input:<name>` pseudo-node (e.g. `input:attention_mask`, `input:encoder_hidden_states`) for model-level inputs that are
not themselves components.

### Edge kinds

| Kind              | Meaning                                                                              | Emitted in v0 for llama/bert/t5 |
| ----------------- | ------------------------------------------------------------------------------------ | ------------------------------- |
| `data`            | Coarse forward dataflow between components.                                          | yes                             |
| `residual`        | Residual/skip connection around a sub-block.                                         | yes                             |
| `mask`            | An attention mask consumed by a component.                                           | yes                             |
| `position`        | Positional information (RoPE, learned/relative position) consumed by a component.    | yes                             |
| `cross_attention` | Encoder hidden states consumed by decoder cross-attention.                           | yes (t5)                        |
| `route`           | MoE router dispatch to experts.                                                      | reserved                        |
| `cache_read`      | Read from the KV cache (cached attention).                                           | reserved                        |
| `cache_write`     | Write to the KV cache (cached attention).                                            | reserved                        |

All eight kinds are part of the v0 contract and are valid in both schemas. `route`, `cache_read`, and `cache_write`
are **defined but not yet emitted** by the v0 recognizers for `llama`/`bert`/`t5`; they are reserved for MoE and
cached-attention recognizers. Consumers MUST accept all eight kinds and SHOULD tolerate additional kinds gracefully.

### Observed dataflow (`dataflow`)

Structural edges are inferred from the module tree. To **ground** the top-level flow in what actually executes, the
generator additionally runs one forward pass on the meta-device model (meta tensors carry shapes but allocate ~no
memory) and records the top-level stages in real call order, with their tensor shapes. This is emitted as an optional
top-level `dataflow` object: `input`, `output`, and an ordered `stages` list, each stage carrying its semantic `id`
(mapped from the executing module), `module_name`/`class_name` (provenance), and `in_shape`/`out_shape`.

Because the template is config-parametric, observed **integer** shapes are **symbolized**: a dim is either an integer
or a token — `"B"` (batch), `"S"` (sequence), or a config expression such as `"config.hidden_size"` when the dim
matches a salient config value. So `[1, 8, 4096]` is serialized as `["B", "S", "config.hidden_size"]`, and a
`ResolvedGraph` consumer can evaluate it per checkpoint.

The forward is also hooked **inside** the representative element of each repeated block, so the `dataflow` object
carries a `blocks` map (repeat id → the body's direct children in observed call order, with shapes). This **re-grounds
the intra-block `data` edges**: module registration order is not forward order (it would leave a pre-norm block's
norms dangling at the end), so the block's data edges are rewritten to the observed order — e.g. for a Llama layer,
`input_layernorm → self_attn → post_attention_layernorm → mlp`. Only `data` edges internal to the block are rewritten;
`residual`/`mask`/`position`/`cache` edges are left intact.

Observed consecutive-stage `data` edges that the structural pass missed are added to `edges` (tagged with a
`"provenance": "observed_forward"` field); `position`/`mask` stages are excluded from that chain since they are not
main-path carriers. The `dataflow` block is **best-effort and optional**: models whose forward can't run on meta
(data-dependent control flow, unusual inputs) simply omit it and keep their structural edges. Blocks whose body wraps a
further nested `ModuleList` of heterogeneous sub-layers (e.g. T5) are not re-grounded and keep their structural edges.

---

## 6. Stable semantic IDs

Every component and repeat has a **stable semantic ID** that is its **primary identity**. IDs are hierarchical and
semantic, e.g.:

```
decoder_layer
decoder_layer.self_attn
decoder_layer.mlp
decoder_layer.input_layernorm
```

rather than instance paths such as `layers.0.self_attn`, `layers.1.self_attn`, …

**Python class names and module paths are provenance, not identity.** They are carried in `class_name`, `module`, and
`path_pattern` (and in `provenance` blocks) so that tooling can trace an ID back to source, but two artifacts that
differ only in class/module naming should ideally share the same semantic IDs.

Stable IDs are the anchor for diffs, modular inheritance, documentation, and cross-version comparison.

---

## 7. Config expressions

Config-parametric values (currently repeat counts) are stored as **expression strings**, never as Python code to be
`exec`'d.

- A config expression is a small, safe arithmetic DSL over `config.<field>` references and numeric literals.
- Supported forms in v0: `config.<field>` references, integer/float literals, and the operators
  `+ - * / // % **` with parentheses. Example: `"config.num_hidden_layers"`, `"config.hidden_size // config.num_attention_heads"`.
- No function calls, arbitrary names, subscripts, or attribute chains beyond `config.<field>` are permitted.
- Resolution is performed by `resolve_template_to_graph`, which parses the expression with a restricted evaluator
  (`evaluate_config_expression`) — it never calls `eval`/`exec` on the string.

If an expression cannot be evaluated (e.g. the config lacks the referenced field), the resolver records a warning and
leaves the repeat's `count` unresolved (`count_resolved: false`) rather than failing the whole graph.

---

## 8. Source provenance

Provenance answers "where did this come from?" without being part of the semantic identity. It appears at three levels:

- **Artifact `provenance`**: generator name, resolution/instantiation strategy, config/model class + module,
  schema version, model type.
- **`entrypoints`**: the config/model classes and auto-class entrypoints used to introspect the architecture.
- **Per-component / per-repeat**: `class_name`, `module`, `path_pattern`, and optional `provenance` blocks.

The v0 generator instantiates models from configuration only, on the meta device, with weight init disabled; it does
**not** load checkpoint weights or call `from_pretrained`.

---

## 9. Compatibility and versioning policy

- Each artifact carries an explicit `schema_version`: `"architecture-template-v0"` or `"resolved-graph-v0"`.
- **v0 is experimental** and may change without deprecation guarantees.
- Both schemas set `additionalProperties: true`. Producers MAY add fields within a version; **consumers MUST ignore
  unknown fields** rather than rejecting the artifact.
- Enumerations (edge `kind`s, `count_source`) MAY gain new members within v0; consumers SHOULD degrade gracefully on
  unrecognized members.
- **Breaking changes** (removing/renaming required fields, changing field meaning, tightening types) require a new
  `schema_version` (e.g. `architecture-template-v1`). The version string is the compatibility signal.
- Semantic IDs are intended to be stable across regenerations of the same architecture; changing an ID for an existing
  component is a semantically breaking change even if the schema is unchanged.

---

## 10. Modular inheritance

Many Transformers models are defined as a `modular_<name>.py` whose classes inherit from another model
(`class GemmaModel(LlamaModel)`). The IR captures this inheritance — mirroring the philosophy of Modular Transformers
while staying independent from Python implementation details — via three top-level fields on `ArchitectureTemplate`:

- **`extends`**: the dominant parent `model_type` this architecture inherits from, or `null` for standalone models.
- **`modularity`**: a diff summary — `is_modular`, `parent_model(s)`, per-bucket `totals`
  (`overridden`/`added`/`deleted`/`new_classes`/`trivial`), and a single-number **`diff_size`**
  (`overridden + added + deleted + 3·new_classes`). `diff_size` is an automatic, per-model measure of how modular a
  model is: a clean modular model (e.g. `qwen2` extending `llama`) has a tiny diff; a model that declares a parent but
  overrides nothing while adding many classes has a large one. It doubles as a **modularity linter**.
- **`patches`**: the per-class change sets, each projected onto a semantic `component_kind` (e.g. `GemmaAttention` →
  `attention`) alongside the provenance `target_class`/`parent_class`/`parent_model` and the `overridden`/`added`/
  `deleted` method & attr lists. Empty for standalone models.

### How it is computed

The generator parses the model's `modular_<name>.py` with the stdlib `ast` (no `libcst` dependency, no torch, no import
of the model) and mirrors the semantics of `utils/modular_model_converter.py`:

- a method/attr present in **both** the modular class and its named parent class → **overridden**;
- present **only** in the modular class → **added**;
- a deletion sentinel (`attr = AttributeError(...)` / `def f(): raise AttributeError`) → **deleted**.

The parent model of each class is read from the modular file's `from ..<model>.modeling_<model> import <Class>` imports.
Computation is best-effort: on any failure the artifact still carries a `modularity` summary marking the model
non-modular with a `note`, keeping the artifact shape stable.

### Library-wide modular graph

The per-artifact `extends` records only a model's own parent. To answer *which architectures are linked, and which share
a lineage*, the generator can also emit a standalone **`modular_graph.json`** (schema
`schema/modular-graph-v0.schema.json`, `schema_version: modular-graph-v0`): a forest over the whole library with, per
node, its `extends`/`parents`, `children`, the transitive `root` ancestor, `depth`, `is_modular`, and `diff_size`, plus a
`roots` list of the spine base models. Models sharing a `root` are in the same lineage — the pairs whose semantic IDs
align cleanly and are the natural defaults for comparison. Like the per-model diff it is built purely by `ast`-parsing
`modular_*.py` files (no torch, no model build), so the full forest computes in a few seconds independently of which
artifacts are generated (`generate_architecture_ir.py --modular-graph`).

### Not yet implemented

`patches` are keyed to Python classes projected onto semantic kinds — not yet to individual semantic component **IDs**
(e.g. `decoder_layer.self_attn`), and there is no patch grammar for edge/config-default overrides or for *applying* a
patch to reconstruct a child template from its parent. Those remain future work; the current payload is a faithful,
schema-stable representation of the modular diff and the modularity metric.

---

## 11. Artifact layout on disk

The generator writes:

```
<output-dir>/
  manifest.json                # schema_version: architecture-ir-manifest-v0
  modular_graph.json           # schema_version: modular-graph-v0 (only with --modular-graph)
  artifacts/
    <model_type>.json          # ArchitectureTemplate
```

The `manifest.json` is an index over generated templates (with per-architecture status and any failures). It is not
part of the two-level IR contract and has its own `schema_version` (`architecture-ir-manifest-v0`).

The `modular_graph.json` (written when `--modular-graph` is passed) is the library-wide modular inheritance forest
described in §10; it is independent of the two-level IR contract and covers the whole library, not just the generated
architectures.
