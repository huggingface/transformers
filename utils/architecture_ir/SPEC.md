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
> example `config.fields` and a repeat's optional `count`). These are **provenance/defaults**, not the authoritative
> parametric form. The authoritative parametric form is always the config expression string.

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
`attention`, `cross_attention`, `feed_forward`, `normalization`, `pooler`, `transformer_block`, `encoder`, `decoder`,
`stack`, `repeated_container`. Consumers MUST tolerate unknown `kind` values.

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

## 10. Future: modular inheritance (reserved, not implemented)

The IR is expected to eventually support **architecture inheritance**, mirroring the philosophy of Modular Transformers
while staying independent from Python implementation details. Instead of duplicating an entire architecture, an
artifact would extend another and apply semantic patches (e.g. replace attention, add a projector, modify an edge,
override config defaults).

For this, two fields are **reserved** at the top level of `ArchitectureTemplate`:

- `extends`: the `model_type` (or template reference) this template inherits from.
- `patches`: an ordered list of semantic patch operations applied to the base.

**v0 does not implement modular diffing.** These fields are documented and accepted by the schema (as optional) so that
consumers can be forward-compatible, but the v0 generator does not emit them, and no patch grammar is standardized yet.
A full modular-diff grammar will be specified in a later revision.

---

## 11. Artifact layout on disk

The generator writes:

```
<output-dir>/
  manifest.json                # schema_version: architecture-ir-manifest-v0
  artifacts/
    <model_type>.json          # ArchitectureTemplate
```

The `manifest.json` is an index over generated templates (with per-architecture status and any failures). It is not
part of the two-level IR contract and has its own `schema_version` (`architecture-ir-manifest-v0`).
