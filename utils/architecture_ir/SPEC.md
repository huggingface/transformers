# Architecture IR — Specification v0 (Experimental)

This document specifies the **Architecture IR v0** contract: the on-disk shape of the artifacts produced by
`utils/architecture_ir` and consumed by downstream tooling (viewers, diffs, docs, Hub integrations).

Its purpose is to be a **stable contract** so that generator work and consumer/viewer work can proceed in parallel.
The prose here is normative for v0; the machine-checkable rules live in:

- `schema/architecture-template-v0.schema.json`
- `schema/resolved-graph-v0.schema.json`
- `schema/modular-graph-v0.schema.json`

**Looking for the key-by-key contract?** Jump to [§12 Field reference](#12-field-reference) — a complete table of every
key, its type/allowed values, and a one-line description for each artifact. The sections before it explain the *why*.
For filled-in, annotated samples, see [`examples/`](./examples/) — four tiered JSONC files (a trimmed, commented
`mistral.json`) that build up by depth: `01-structure` → `02-capabilities` → `03-modularity` → `04-full`.

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
| `path_pattern` | yes      | **Provenance**: symbolic module path with `{i}` placeholders.                  |
| `class_name`   | no       | **Provenance**: originating Python class name.                                 |
| `children`     | no       | Semantic IDs of child components; **omitted for leaf nodes**.                  |

Components are split across two arrays:

- `components`: semantic units reachable **outside** of any symbolic repeat body.
- `templates`: reusable **repeat bodies** and their descendants — serialized once, referenced by repeats.

The component tree is always **connected from the root**: a structural container is retained even if its own `kind`
isn't independently interesting, whenever it is an ancestor of a component that is. This matters for models that nest
whole sub-models as attributes (a VLM's `vision_tower` / `language_model` / `multi_modal_projector`) — those wrappers
are emitted as nodes and listed in the root's `children`, so a consumer can always walk from the root to every leaf.

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
- Per-component **`attributes`**: attention → `{variant, n_heads, n_kv_heads, head_dim, rope, sliding_window}`;
  feed-forward → `{hidden_size, intermediate_size, activation}`; normalization → `{norm_type}` (`rms`/`layer`/…);
  MoE → `{num_experts, experts_per_token, …}`; embedding → `{num_embeddings, embedding_dim}`; position →
  `{scheme, …}` (`rope` + `rope_theta`/`head_dim`, `relative` + `num_buckets`, `learned` + `max_position_embeddings`).
  Only pure structural containers (the model root, a transformer-block wrapper) carry no `attributes` — their meaning
  is their composition.

The IR also **descends into the projections** inside attention / MLP / MoE bodies — the `q/k/v/o_proj` and
`gate/up/down_proj` (kind `projection`) appear as children of their host with `{in_features, out_features}`, each
**symbolized to a config expression** where it matches a salient value (e.g. `config.hidden_size`,
`config.intermediate_size`; dims like `num_kv_heads · head_dim` that match no single field stay integers). This stays
compact — projections are serialized once in the template body, never per repeated layer.

The **intra-module dataflow** among those projections is emitted as `data` edges (tagged `"provenance":
"intra_module"`): the block fans out to its input projections (`self_attn → q/k/v`, `mlp → gate/up`) and they fan in to
the output projection (`q/k/v → o_proj`, `gate/up → down_proj`). Roles are assigned semantically (by projection name),
not from execution order — q/k/v run in parallel, so a call-order trace would wrongly chain them. When no output
projection is recognized, only the fan-out is emitted, leaving the projections as unordered siblings.

These facts also drive the reserved edge kinds: decoder-side self-attention emits `cache_read`/`cache_write` edges to a
`state:kv_cache` pseudo-node, and a MoE block emits a `route` edge to its experts container.

For multimodal models the `architecture` facts (attention variant, positional, MoE) are read from the **text
backbone** (`config.text_config`) rather than the composite config, so a VLM reports its LLM's attention variant
instead of `None`; `view`/`family` still come from the composite (they depend on the vision/audio sub-configs).

### Capabilities (`capabilities`)

Distinct from `architecture` (what the model *is*), the `capabilities` block records — implementation-agnostically —
what the architecture *can do / run with*:

- **`attention_backends`**: implementations the model supports, read from its class support flags — `eager` (always,
  the reference impl) plus `sdpa` / `flash_attention` / `flex_attention` when opted in. This is "can run with", not
  "installed here" (the actual backend is gated at runtime by the installed library).
- **`attention_patterns`** / **`attention_schedule`**: the distinct per-layer attention pattern kinds
  (`causal`/`bidirectional`/`sliding`/`chunked`/`compressed`) and, for a *non-uniform* schedule (e.g. gpt-oss's
  sliding/full alternation), the raw `config.layer_types` list. This is what a viewer needs to draw per-layer
  attention-mask patterns; a uniform schedule collapses to `null`.
- **`task_heads`**: every task family available for the `model_type` across the Auto\* mappings (e.g. llama →
  `causal_lm, question_answering, sequence_classification, token_classification`) — what the architecture can be used
  for, beyond the single base-model `family`.
- **`tensor_parallel`**: whether the model ships a base tensor-parallel plan; when it does, each projection component
  also carries a `tp` attribute (`colwise`/`rowwise`) resolved from `base_model_tp_plan`.
- **`kernels`**: layers augmented by `@use_kernel_forward_from_hub` (kernel-swappable) present in the model, mapped to
  their compatible Hub kernel repos (e.g. `RMSNorm → [kernels-community/liger-kernels, …]`). Each such node also
  carries `attributes.kernel` (the layer name) pointing back here. Both the decorated classes and the repo mapping are
  read from source with `ast` (from each model's `modeling_*.py` and the `_KERNEL_MAPPING` in `hub_kernels.py`) — no
  torch, no network, and no dependency on the optional `kernels` package.

A single unambiguous attention pattern is also stamped on each attention component as `attributes.pattern`.

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

Structural edges are inferred from the module tree. To **ground** the flow in what actually executes, the generator
runs one forward pass on the meta-device model (meta tensors carry shapes but allocate ~no memory), hooking the
top-level stages *and* the direct children inside the representative element of each repeated block. Two things come
out of it:

1. **Grounded edges.** The observed call order rewrites the `data` edges so they reflect real forward order rather than
   module-registration order — e.g. a Llama layer becomes `input_layernorm → self_attn → post_attention_layernorm → mlp`
   (registration order would dangle the norms at the end). Only `data` edges are rewritten; `residual`/`mask`/`position`/
   `cache` edges are left intact. Edges the structural pass missed are added tagged `"provenance": "observed_forward"`;
   `position`/`mask` nodes are excluded from the main-path chain.
2. **Observed shapes.** The `dataflow` object carries only the non-redundant part — a `shapes` map from semantic id to
   `{in, out}` (the *ordering* is already in `edges`). Because the template is config-parametric, observed **integer**
   shapes are **symbolized**: a dim is an integer or a token — `"B"` (batch), `"S"` (sequence), or a config expression
   like `"config.hidden_size"` when it matches a salient value. So `[1, 8, 4096]` becomes `["B", "S", "config.hidden_size"]`,
   which a `ResolvedGraph` consumer evaluates per checkpoint.

The `dataflow` block is **best-effort and optional**: models whose forward can't run on meta
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

**Python class names and module paths are provenance, not identity.** They are carried in `class_name` and
`path_pattern` per node (and the source module in the artifact-level `provenance`) so that tooling can trace an ID back
to source, but two artifacts that differ only in class/module naming should ideally share the same semantic IDs.

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

Provenance answers "where did this come from?" without being part of the semantic identity. It is kept deliberately
minimal:

- **Artifact `provenance`**: just the source classes — `{config_class, config_module, model_class, model_module}`. The
  resolution strategy (meta device, config-only, no weights) is invariant and documented here, not repeated per artifact.
- **Per-component**: `class_name` and `path_pattern`. (Per-node `module` was dropped — it repeated the same
  modeling-file string on every node; the source module is in the artifact-level `provenance`.)

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
while staying independent from Python implementation details — via two top-level fields on `ArchitectureTemplate`:

- **`extends`**: the dominant parent `model_type` this architecture inherits from, or `null` for standalone models.
- **`patches`**: the per-class change sets, each projected onto a semantic `component_kind` (e.g. `GemmaAttention` →
  `attention`) alongside the provenance `target_class`/`parent_class` and the non-empty `overridden`/`added`/`deleted`
  method & attr lists. **Trivial** classes (inherit everything, change nothing) are omitted, and `patches` itself is
  omitted for standalone models.

There is deliberately **no per-artifact `modularity` summary block**: `is_modular` is just `extends != null`, and the
per-model **`diff_size`** metric (`overridden + added + deleted + 3·new_classes`; a clean model like `qwen2` extending
`llama` scores tiny, a declare-a-parent-but-rebuild-everything model scores large — a modularity linter) is a
*cross-model* number, so it lives once in `modular_graph.json` (§ below) rather than in every artifact.

### How it is computed

The generator parses the model's `modular_<name>.py` with the stdlib `ast` (no `libcst` dependency, no torch, no import
of the model) and mirrors the semantics of `utils/modular_model_converter.py`:

- a method/attr present in **both** the modular class and its named parent class → **overridden**;
- present **only** in the modular class → **added**;
- a deletion sentinel (`attr = AttributeError(...)` / `def f(): raise AttributeError`) → **deleted**.

The parent model of each class is read from the modular file's `from ..<model>.modeling_<model> import <Class>` imports.
Computation is best-effort: on any failure the artifact simply carries `extends: null` and no `patches`.

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

---

## 12. Field reference

The complete key-by-key contract. **Req** legend: **●** required by schema · **▲** always emitted by the generator
(not schema-required, so consumers should still tolerate absence) · **○** optional / conditional. Objects set
`additionalProperties: true`, so producers may add keys within v0 and **consumers must ignore unknown keys**.

### 12.1 `ArchitectureTemplate` — top level (`artifacts/<model_type>.json`)

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `schema_version` | ● | `"architecture-template-v0"` | Contract version. |
| `model_type` | ● | string | The `model_type` this template describes (one per). |
| `config` | ● | object | Config identity + parametric knobs — §12.6. |
| `components` | ● | `component[]` | Semantic nodes outside any repeat body — §12.2. |
| `templates` | ● | `component[]` | Repeat-body nodes, serialized once — §12.2. |
| `repeats` | ● | `repeat[]` | Symbolic repeats — §12.3. |
| `edges` | ● | `edge[]` | Coarse dataflow edges — §12.4. |
| `provenance` | ● | object | `{config_class, config_module, model_class, model_module}` — the source classes only. |
| `architecture` | ▲ | object | Model-level semantic facts — §12.5. |
| `capabilities` | ▲ | object | What the model can do / run with — §12.7. |
| `extends` | ▲ | string \| null | Dominant parent `model_type` (modular), else null. |
| `patches` | ○ | `patch[]` | Per-class modular changes — §12.9. Omitted for standalone models. |
| `dataflow` | ○ | object | Observed tensor shapes — §12.10. Absent when the meta forward can't run. |
| `warnings` | ○ | string[] | Non-fatal generation warnings; omitted when empty. |

> **Removed in the lean revision** (all derivable / boilerplate — don't reintroduce): `metadata` (level implied by
> `schema_version`), `entrypoints` (folded into `provenance`), and the `modularity` summary block (`is_modular` =
> `extends != null`; `diff_size`/`totals` derive from `patches`; the metric lives in `modular_graph.json`).

### 12.2 `component` — entries of `components` and `templates`

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `id` | ● | string | Stable semantic ID; **primary identity** (e.g. `decoder_layer.self_attn`). |
| `kind` | ● | string (§12.11) | Semantic role. |
| `path_pattern` | ● | string | Provenance: symbolic module path with `{i}` (e.g. `model.layers.{i}.self_attn`). |
| `class_name` | ▲ | string | Provenance: Python class name. |
| `children` | ○ | string[] | Semantic IDs of child components / repeats. **Omitted for leaf nodes** (absence ⇒ none). |
| `attributes` | ○ | object (§12.12) | Normalized per-node facts; keys depend on `kind`. |

(Per-node `module` was removed — it was the same modeling-file string on every node; the source module is in top-level `provenance`.)

### 12.3 `repeat` — entries of `repeats`

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `id` | ● | string | Semantic ID (e.g. `decoder_layers`). |
| `kind` | ● | `"symbolic_repeat"` | Constant. |
| `body` | ● | string | Semantic ID of the repeated template body. |
| `count_expr` | ● | string | Config expression for the count (e.g. `config.num_hidden_layers`). Not `exec`'d — §7. |
| `count_source` | ● | `config` \| `module_tree` | Whether `count_expr` resolves against config or was a literal. |
| `index_symbol` | ● | string | Index symbol used in path patterns (`i`). |
| `item_path_pattern` | ● | string | Provenance: symbolic path of one item (`model.layers.{i}`). |
| `count` | ○ | int \| null | Count under the **default** config (provenance; evaluated count lives in the ResolvedGraph). |
| `container_path_pattern` | ○ | string | Provenance: symbolic path of the container. |
| `repeated_class_name` | ○ | string | Provenance: Python class of the repeated block. |
| `provenance` | ○ | object | Extra provenance. |

### 12.4 `edge` — entries of `edges`

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `source` | ● | string | Semantic ID, or a pseudo-node `input:<name>` / `state:<name>` (e.g. `input:attention_mask`, `state:kv_cache`). |
| `target` | ● | string | Same as `source`. |
| `kind` | ● | `data` \| `residual` \| `mask` \| `position` \| `cross_attention` \| `route` \| `cache_read` \| `cache_write` | Edge semantics (§5). |
| `provenance` | ○ | string | Origin tag when present, e.g. `observed_forward`, `intra_module`. |

### 12.5 `architecture` — model-level facts (multimodal reads the text backbone)

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `view` | ▲ | `decoder` \| `encoder` \| `enc_dec` \| `multimodal` | High-level layout. |
| `family` | ▲ | string \| null | Task family: `causal_lm`, `masked_lm`, `seq2seq`, `image_text_to_text`, `image_classification`, … |
| `attention_variant` | ○ | `MHA` \| `GQA` \| `MQA` \| `MLA` \| null | |
| `positional` | ○ | `rope` \| `relative` \| `alibi` \| `learned` \| `sinusoidal` \| null | |
| `is_moe` | ▲ | bool | |
| `moe` | ○ | object | `{num_experts, experts_per_token, num_shared_experts?}` when MoE. |
| `sliding_window` | ○ | int \| null | |
| `tie_word_embeddings` | ○ | bool \| null | |

Fields that don't apply are omitted (absent ⇒ "not present / undeterminable").

### 12.6 `config`

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `class_name` | ● | string | Config class. |
| `module` | ● | string | Config module path. |
| `model_type` | ● | string \| null | |
| `referenced_fields` | ● | object `{field: value}` | Config knobs referenced by config expressions, with default values — the parametric surface. |
| `salient_fields` | ○ | object `{field: value}` | Curated scalar architecture defaults (`hidden_size`, heads, `hidden_act`, …). The full config is **not** serialized. |

### 12.7 `capabilities`

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `attention_backends` | ▲ | string[] ⊆ {`eager`, `sdpa`, `flash_attention`, `flex_attention`} | Impls the model supports. "Can run with", not "installed here". |
| `attention_patterns` | ▲ | string[] ⊆ {`causal`, `bidirectional`, `sliding`, `chunked`, `compressed`} | Distinct per-layer mask kinds. |
| `attention_schedule` | ▲ | string[] \| null | Raw `config.layer_types` when non-uniform, else null. |
| `task_heads` | ▲ | string[] | Task families available for the `model_type` across Auto\* mappings. |
| `tensor_parallel` | ▲ | bool | Ships a base TP plan (adds `tp` to projection nodes — §12.11). |
| `kernels` | ○ | object `{layer: [repo_id, …]}` | Kernelizable layers present → compatible Hub kernel repos. Nodes point in via `attributes.kernel`. Present only when the model has kernelizable layers. |

### 12.8 Modular fields — `extends` + `patches`

`extends` is at the top level (§12.1). The `modularity` summary block was **removed**: `is_modular` = `extends != null`,
and `diff_size`/`totals` are a one-line sum over `patches`. The cross-model `diff_size` metric lives in
`modular_graph.json` (§12.14).

**`patch`** — entries of `patches` (trivial inherit-everything classes are omitted; empty member buckets are omitted):

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `relation` | ● | `inherits` \| `new` | Overrides an inherited class vs a brand-new class. |
| `target_class` | ● | string | Provenance: the modular Python class. |
| `component_kind` | ○ | string \| null | Semantic kind the class maps to (`attention`, `feed_forward`, …). |
| `parent_class` | ○ | string \| null | Provenance. |
| `parent_model` | ○ | string \| null | Only when not the dominant `extends` parent (multi-parent case). |
| `overridden` / `added` / `deleted` | ○ | `{methods?: string[], attrs?: string[]}` | Non-empty member change sets only. |

### 12.9 `dataflow`

Node **order** is not repeated here — it lives in `edges`. `dataflow` carries only the non-redundant part: observed
tensor shapes keyed by semantic id.

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `source` | ● | string | How the flow was obtained (`observed_forward_meta`). |
| `input` | ○ | `{name: string, shape: sym_shape}` | Model input tensor. |
| `output` | ○ | `{shape: sym_shape}` | Model output tensor. |
| `shapes` | ● | object `{semantic_id: {in: sym_shape, out: sym_shape}}` | Per-node observed shapes; join to nodes by id. |

**`sym_shape`**: `array | null`; each dim is an integer or a token string — `"B"` (batch), `"S"` (sequence), or a
config expression like `"config.hidden_size"`.

### 12.10 Component `kind` vocabulary

`model`, `embedding`, `position`, `attention`, `cross_attention`, `feed_forward`, `moe`, `normalization`,
`projection`, `pooler`, `transformer_block`, `encoder`, `decoder`, `stack`, `repeated_container`, `module` (a retained
structural connector, e.g. a VLM sub-model wrapper). **Open set** — consumers must tolerate unknown kinds.

### 12.11 `attributes` by component `kind`

| kind | attribute keys | notes |
|------|----------------|-------|
| `attention`, `cross_attention` | `variant, n_heads, n_kv_heads, head_dim, rope, sliding_window, pattern` | `rope` bool; `pattern` only when the model has a single attention pattern. |
| `feed_forward` | `hidden_size, intermediate_size, activation` | |
| `moe` | `num_experts, experts_per_token, num_shared_experts` | |
| `normalization` | `norm_type` | `rms` \| `layer` \| `group` \| `batch`. |
| `embedding` | `num_embeddings, embedding_dim` | values are a config expression (`config.vocab_size`) or an int. |
| `position` | `scheme` + one of `rope_theta`/`head_dim` \| `num_buckets` \| `max_position_embeddings` | keyed by scheme. |
| `projection` | `in_features, out_features, tp` | features are a config expression or int; `tp` = `colwise` \| `rowwise` (from the TP plan). |
| *(any kernelizable node)* | `kernel` | Hub kernel layer name (e.g. `RMSNorm`); joins to `capabilities.kernels[name]` for the compatible repos. |

### 12.12 `ResolvedGraph` (`resolve_template_to_graph` output) — delta vs the template

Same component/edge/repeat vocabulary as the template. Differences:

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `schema_version` | ● | `"resolved-graph-v0"` | |
| `template_ref` | ● | `{schema_version, model_type}` | Back-pointer to the template it was resolved from. |
| `config` | ● | `{source, model_type, referenced_fields}` | Checkpoint config values actually used. |

Each `repeats[]` entry additionally carries `count` (evaluated int), `count_resolved` (bool); `count_expr` is retained.
Repeats stay **symbolic** (one body per repeat, not expanded per instance). No `entrypoints` / modular fields.

### 12.13 `ModularGraph` (`modular_graph.json`)

| Key | Req | Type / values | Description |
|-----|-----|---------------|-------------|
| `schema_version` | ● | `"modular-graph-v0"` | |
| `roots` | ● | string[] | Spine base models (no parent, ≥1 descendant), most-descendants-first. |
| `nodes` | ● | object `{model: node}` | The forest. |

**`node`**: `{extends: string|null, parents: string[], children: string[], root: string|null, depth: int,
is_modular: bool, diff_size: int|null}`. Two models sharing a `root` are in the same lineage (align cleanly for
comparison).
