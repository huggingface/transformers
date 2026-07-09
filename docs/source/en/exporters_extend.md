<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Extending the exporters

The exporters keep their workarounds as reversible patches and FX-level fixes applied at
well-defined points in the export flow, out of the modeling code wherever possible. Adding
export support for a new architecture or backend means registering one of these, and each fix
belongs at the lowest stage that can express it cleanly.

Each exporter's source file labels its stages as `# ── Stage N: … ──` comment blocks. The sections
below follow that same layout 1:1, so the file you read and the doc you read are the same map.

## Patches and fixes

The exporters use two lifecycles consistently.

Patches are registered with `@register_patch(backend, *dotted_paths)` and installed with
`apply_patches(backend)`. A patch reversibly swaps an attribute (a `torch` op, an ExecuTorch
internal, or a model class method) for the duration of the export. Pass multiple paths to a single
decorator to share one factory across targets, which is useful when the same method shape needs
patching on several classes (for example `_update_mamba_mask` on Jamba, Bamba, and others).
Originals are restored on exit, even if the export raises.

Fixes are registered with `@register_fx_node_fix(backend)` or `@register_fx_program_fix(backend)`
and applied with `apply_fx_node_fixes(backend, gm)` or `apply_fx_program_fixes(backend, ep)`.
ONNX-IR fixes are listed in `_IR_FIXES` and applied with `apply_onnx_ir_fixes`. A fix mutates the
in-progress graph or program in place. There's no revert, since fixes permanently repair the
artifact before the next pipeline step.

Every patch and fix sits in a backend-keyed registry (`_PATCHES`, `_FX_NODE_FIXES`,
`_FX_PROGRAM_FIXES` in [exporters/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/utils.py)).
Adding a new one is *write a function and decorate it*, nothing else.

## DynamoExporter

The base exporter has one patch stage and four structural helpers. They run in this order inside
`DynamoExporter.export`, against the original `nn.Module`. Source:
[exporter_dynamo.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_dynamo.py).

### Stage 1: Forward-signature patch

`patch_forward_signature`, under the `# ── Stage 1: Model signature patch ──` marker.

Replaces `model.forward` with an explicit flat-arg signature derived from the inputs dict, so
`torch.export` doesn't bundle `**kwargs` into a single tuple. This is the entry contract
`torch.export` reads before tracing.

This stage is internal and has no extension knob.

### Stage 2: Model patches

`_PATCHES["dynamo"]` via `apply_patches("dynamo")`, under the `# ── Stage 2: Model patches ──` marker.

Reversible class-attribute swaps applied during tracing. Each `_patch_*(original) → replacement`
factory targets one or more `Class.method` paths and replaces a non-exportable model pattern
(data-dependent loops, in-place ops, mask checks, chunked-attention `split → zip → cat`) with an
export-safe equivalent.

To add one, define `_patch_*(original)` and decorate it with `@register_patch("dynamo", *dotted_paths)`.
Pass multiple paths to share the factory across classes. Existing examples cover the
mamba/linear-attn mask, the NLLB classifier cast, and chunked-vision attention.

### Stage 3: Pytree registration

`register_cache_pytrees_for_model`, under the `# ── Stage 3: Pytree registration ──` marker.

Registers flatten/unflatten via `torch.utils._pytree.register_pytree_node` for every captured
`Cache` and `ModelOutput`. It's reflection-driven and tuned for tensor containers, not a general
serializer.

This is usually automatic. If a type isn't reflectable, add a branch to `_flatten_to_context` or
`_unflatten_from_context`.

### Stage 4: Dynamic shapes

`get_auto_dynamic_shapes`, under the `# ── Stage 4: Dynamic shapes ──` marker.

Auto-assigns `Dim.AUTO` to every tensor and cache leaf when `DynamoConfig.dynamic=True` and the
user did not pass `dynamic_shapes` explicitly.

Override per-export via `DynamoConfig.dynamic_shapes`.

### Stage 5: State cleanup

`reset_model_state` and `_STATEFUL_CACHE_ATTRS`, under the `# ── Stage 5: Model state cleanup ──` marker.

Resets non-`Cache` tensor attributes set inside `forward` (for example glm_moe_dsa `_cached_keys`
and wav2vec2_bert `cached_rotary_positional_embedding`) that `torch.export` leaves as FakeTensors,
so a follow-up eager forward is safe.

To extend, append the attribute name to `_STATEFUL_CACHE_ATTRS`.

## OnnxExporter

`OnnxExporter` extends `DynamoExporter` with five numbered stages applied around
`torch.onnx.export`. Source:
[exporter_onnx.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_onnx.py).

A complete inventory of patches in the file is one grep away:

```bash
grep -nE "^def (_patch_|_fix_|_aten_)" src/transformers/exporters/exporter_onnx.py
```

### Stage 1: Torch patches

`_PATCHES["onnx"]`, reversible via `apply_patches("onnx")`, under the `# ── Stage 1: Torch patches ──`
marker. Runs during `torch.export` and `torch.onnx.export`.

Reversible swaps of `torch` ops (`where`, `unsqueeze`, `scaled_dot_product_attention`,
`searchsorted`, and others) that the ONNX decomposer can't lower as-is. Each `_patch_*(original)`
closes over the original.

To add one, define `_patch_*(original)` and decorate it with `@register_patch("onnx", "dotted.path")`.

### Stage 2: ONNX patches

`_PATCHES["onnx"]`, reversible via `apply_patches("onnx")`, under the `# ── Stage 2: ONNX patches ──`
marker. Runs during `torch.onnx.export`.

Hooks the private `_prepare_exported_program_for_export` step so the FX node fixes (stage 3) run
again right after `run_decompositions`. Any new symbolic-guard nodes the ONNX decomposition
introduces get repaired before the FX to ONNX lowering picks them up.

Uses the same registry as stage 1: define `_patch_*(original)` and decorate it with
`@register_patch("onnx", "dotted.path")`.

### Stage 3: FX node fixes

`_FX_NODE_FIXES["onnx"]`, in-place via `apply_fx_node_fixes("onnx", gm)`, under the
`# ── Stage 3: FX node fixes ──` marker. Runs after `torch.export`, then again after
`run_decompositions`.

Per-node rewrites on the `GraphModule` to drop or replace nodes the ONNX decomposer can't lower
(alias ops, in-place views, `_assert_*`, dead comparisons, in-place `triu_`, `fill_diagonal_`,
`sort(stable=True)`). DCE runs automatically at the end of the walk.

To add one, define `_fix_*(gm, node) → bool` (return `True` to consume the node) and decorate it
with `@register_fx_node_fix("onnx")`.

### Stage 4: ONNX translations

`_ONNX_TRANSLATION_TABLE`, under the `# ── Stage 4: ONNX translations ──` marker. Runs during the
FX to ONNX lowering.

Overrides `torchlib`'s default lowering for specific aten ops where the default is buggy or missing.
Currently `aten.index_put` (bool-mask path), `aten.bincount` (`OneHot + ReduceSum`), and
`aten._grouped_mm` / `transformers.grouped_mm_fallback` (MoE grouped-matmul lowered to an unrolled
`Slice + MatMul + Concat`).

To add one, implement an `_aten_*` onnxscript function and add it to `_ONNX_TRANSLATION_TABLE`.

### Stage 5: ONNX IR fixes

`_IR_FIXES` applied via `apply_onnx_ir_fixes`, under the `# ── Stage 5: ONNX IR fixes ──` marker.
Runs after `torch.onnx.export` returns.

Post-export rewrites on the `ONNXProgram` IR to work around ORT validation and runtime bugs (for
example forcing `TopK(sorted=True)`). Applied to both the top-level graph and every function.

To add one, implement `_fix_ir_*(graph_like)` and append it to `_IR_FIXES`.

## ExecutorchExporter

`ExecutorchExporter` extends `DynamoExporter` with four stages applied around
`to_edge_transform_and_lower` and `to_executorch`, plus a backend-preparation step that runs first.
Source:
[exporter_executorch.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_executorch.py).

### Stage 1: Backend preparation

`_BACKEND_PREPARE`, under the `# ── Stage 1: Backend preparation ──` marker. Runs before
`torch.export` and is a one-shot step, not a reversible patch.

`prepare_for_xnnpack` moves the model to CPU/fp32 and selects `XnnpackPartitioner`;
`prepare_for_cuda` moves it to CUDA/bf16 and selects `CudaPartitioner`. Each returns
`(model, sample_inputs, partitioner)`.

To add a backend, implement `prepare_for_<name>` and register it in `_BACKEND_PREPARE`.

### Stage 2: Torch patches

`_PATCHES["executorch"]`, reversible via `apply_patches("executorch")`, under the
`# ── Stage 2: Torch patches ──` marker. Runs during `torch.export` tracing.

Replaces `torch` ops the ExecuTorch backends can't accept (`split_copy`, `chunk`, `topk(k>dim)`,
non-divisible `avg_pool2d`, `dropout`, in-place `view`, GQA-shaped SDPA) with decomposed equivalents.

To add one, define `_patch_*(original)` and decorate it with `@register_patch("executorch", "dotted.path")`.

### Stage 3: ExecuTorch patches

`_PATCHES["executorch"]`, reversible via `apply_patches("executorch")`, under the
`# ── Stage 3: ExecuTorch patches ──` marker. Runs during `to_edge_transform_and_lower` and
`to_executorch`.

Reversibly swaps ExecuTorch internals that crash on legitimate dynamic-shape patterns:
`SpecPropPass.update_placeholder_tensor_specs`, `PruneEmptyTensorsPass.remove_empty_tensors_from_cat`,
`eval_upper_bound`, `dim_order_from_stride` (rebound on every importer), the XNNPACK
squeeze/unsqueeze define-node, the complex-dtype validator, and the edge-dialect sym-op allowlist.

Uses the same registry as stage 2: define `_patch_*(original)` and decorate it with
`@register_patch("executorch", "dotted.path")`.

### Stage 4: FX program fixes

`_FX_PROGRAM_FIXES["executorch"]`, in-place via `apply_fx_program_fixes("executorch", ep)`, under
the `# ── Stage 4: FX program fixes ──` marker. Runs after `torch.export`, before
`to_edge_transform_and_lower`.

Repairs the `ExportedProgram` where the fix needs program-level context: widen `int_oo` upper
bounds in `range_constraints`, and fill missing placeholder `meta["val"]` from `state_dict`.

To add one, define `_fix_*(exported_program) → None` and decorate it with
`@register_fx_program_fix("executorch")`.

### Stage 5: FX node fixes

`_FX_NODE_FIXES["executorch"]`, in-place via `apply_fx_node_fixes("executorch", gm)`, under the
`# ── Stage 5: FX node fixes ──` marker. Runs after stage 4, before `to_edge_transform_and_lower`.

Per-node rewrites: swap Python sym ops for `executorch_prim.*` equivalents, rewrite `pow` as a
`mul` chain, normalize amax/max negative dim, and force a contiguous clone. DCE runs automatically
at the end of the walk.

To add one, define `_fix_*(gm, node) → bool` (return `True` to consume the node) and decorate it
with `@register_fx_node_fix("executorch")`.

## When to patch the exporter vs. fix the model

The split is intentional.

Make a modeling change when the pattern blocks export across multiple backends: data-dependent
loops, stateful caches outside `Cache`, or hand-written split-loop attention. Fix it once in the
model, and every exporter benefits.

Write an exporter patch when the issue is a single backend's lowering bug: a missing ONNX
translation, an ORT validation quirk, or an FX decomposition that emits a dead op. Keep the
workaround in the exporter, and the modeling code stays clean.

## Known upstream workarounds

A small number of model classes hit confirmed bugs in `onnxscript`'s graph optimizer (constant
folding crashing on `SplitToSequence`, FPN initializers being dropped). For those, ONNX
optimization is selectively disabled via
[`ONNX_DISABLE_OPTIMIZE_MODEL_CLASSES`](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_utils.py)
in the test suite, and each entry is annotated with the upstream issue it works around. This list
is expected to shrink as upstream bugs land. It is not an extension point for arbitrary skipping,
and new entries must reference a specific upstream bug.

A second list,
[`EXPORT_SKIP_MODEL_CLASSES`](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_utils.py),
opts a handful of model classes out of the entire export sweep when the model itself is
fundamentally non-exportable as-is (data-dependent control flow that can't be vectorized, or
modules treated as forward arguments). Same expectations: every entry carries a TODO naming the
underlying model change needed, and the list is expected to shrink, not grow.
