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

Each artifact uses `schema/architecture-artifact-v0.schema.json` and contains:

- config class and config fields from local defaults
- resolved config/model entrypoints
- compact semantic components and reusable templates
- repeated `ModuleList` blocks collapsed into symbolic repeats when detectable
- coarse semantic edges with `data`, `residual`, `mask`, `position`, and `cross_attention` kinds
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

- dataflow is semantic and coarse, not an executable graph
- forward methods are not traced, and `torch.fx` / `torch.export` are not primary sources
- repeat detection currently focuses on homogeneous `ModuleList` containers
- edge recognition is based on module names/classes and common Transformer conventions
