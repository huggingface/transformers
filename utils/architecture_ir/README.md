# Experimental Architecture IR Generator

This is a local-only prototype for generating a coarse, human-readable architecture IR from a Transformers
`model_type`.

It is intentionally isolated under `utils/architecture_ir` and does not add runtime dependencies to Transformers. The
generator resolves configs with `AutoConfig.for_model`, instantiates `AutoModel.from_config` on the meta device under
`no_init_weights`, and inspects the module tree. It does not load checkpoint weights, call `from_pretrained`, upload to
the Hub, or require network access.

## Usage

```bash
python utils/architecture_ir/generate_architecture_ir.py \
  --architectures llama bert t5 \
  --output-dir /tmp/architecture-ir
```

Output layout:

```text
/tmp/architecture-ir/
  manifest.json
  artifacts/
    llama.json
    bert.json
    t5.json
```

The data contract is specified in [`SPEC.md`](./SPEC.md), which defines two levels: the compact, config-parametric
`ArchitectureTemplate` (one per `model_type`, the canonical artifact produced here) and the `ResolvedGraph` obtained by
resolving a template against a checkpoint `config.json`.

Each generated artifact is an `ArchitectureTemplate`, validated by `schema/architecture-template-v0.schema.json`, and
contains:

- config class plus a compact `referenced_fields` (config knobs referenced by config expressions) and `salient_fields`
  (curated architecture-defining scalar defaults) — the full config is intentionally not serialized
- resolved config/model entrypoints
- compact semantic components and reusable templates
- repeated `ModuleList` blocks collapsed into symbolic repeats when detectable
- coarse semantic edges (`data`, `residual`, `mask`, `position`, `cross_attention`, `route`, `cache_read`,
  `cache_write`)
- an `architecture` block of normalized semantic facts (view, task family, attention variant, positional scheme, MoE
  params) plus per-component `attributes`
- a `modularity` block + `extends`/`patches`: the modular-diff of the model's `modular_<name>.py` vs its parent,
  including a `diff_size` modularity metric
- an observed `dataflow` block: top-level flow from a real meta forward, with config-parametric (symbolized) tensor
  shapes like `["B", "S", "config.hidden_size"]`
- path-pattern provenance such as `model.layers.{i}.self_attn`

The canonical `artifacts/<model_type>.json` files do not serialize every module instance. For example, repeated Llama
decoder layers are represented once as a `decoder_layer` template and a `decoder_layers` repeat:

```json
{
  "id": "decoder_layers",
  "kind": "symbolic_repeat",
  "body": "decoder_layer",
  "count_expr": "config.num_hidden_layers",
  "item_path_pattern": "model.layers.{i}"
}
```

To inspect the full expanded module dump while debugging, pass `--debug-expanded`:

```bash
python utils/architecture_ir/generate_architecture_ir.py \
  --architectures llama bert t5 \
  --output-dir /tmp/architecture-ir \
  --debug-expanded
```

Expanded debug artifacts are written under `debug/expanded/` and are not the canonical artifact.

## Current Scope

The first milestone is tuned for useful output on `llama`, `bert`, and `t5`, but the implementation is a single generic
generator. Architecture-specific behavior should be added as reusable semantic recognizers in
`architecture_ir/recognizers.py`, not as one exporter per architecture.

Known limitations:

- dataflow is semantic and coarse, not an executable graph; the observed `dataflow` block is top-level only (a
  forward that can't run on meta simply omits it)
- full forward methods are not traced, and `torch.fx` / `torch.export` are not primary sources
- `patches` are keyed to Python classes projected onto semantic kinds, not yet to individual semantic component IDs;
  there is no grammar for *applying* a patch to reconstruct a child from its parent
- `cache_read`/`cache_write` edges cover decoder self-attention; cross-attention KV caching is not yet modeled
- repeat detection currently focuses on homogeneous `ModuleList` containers
- edge recognition is based on module names/classes and common Transformer conventions
