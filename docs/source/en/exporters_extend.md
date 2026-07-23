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

`torch.export` traces the model into a graph, later stages transform
that graph, and a final stage lowers or emits it for the target runtime. Most models pass through
untouched. When there's a PyTorch pattern the backend can't handle, the exporter applies a
small workaround at that stage rather than editing the model.

Add a workaround by writing one function and registering it with a decorator. Each workaround belongs at the lowest stage that can express it cleanly.

## Patches and fixes

A workaround is either a patch or a fix. The two differ in whether they can be reverted.

|              | Patch                                                                          | Fix                                                                                   |
| ------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| What it does | Swaps out an attribute (a `torch` op, an ExecuTorch internal, or a model method) for the duration of the export | Rewrites the traced graph or program                                                  |
| Reverted     | Yes, the original is restored afterward                                        | No, it repairs the artifact before the next stage runs                                |
| Register with | `@register_patch(backend, *paths)`                                            | `@register_fx_node_fix(backend)` or `@register_fx_program_fix(backend)`               |

Both live in a registry in
[exporters/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/utils.py),
and the exporter installs everything registered for its backend at the right stage.

Reach for a patch when the issue is a single backend's lowering bug: a missing ONNX
translation, an ORT validation quirk, or an FX decomposition that emits a dead op. The workaround
stays in the exporter, and the modeling code stays clean.

When the pattern blocks export across multiple backends, such as data-dependent loops, stateful
caches outside `Cache`, or hand-written split-loop attention, fix the model instead. Fixing it
once in the model helps every exporter.

## Add a patch

Suppose a model method does something `torch.export` can't trace. NLLB-MoE's
`NllbMoeTop2Router._cast_classifier` casts the classifier weights to another dtype,
which isn't traceable. Replace it with a no-op for the duration of the export.

Write a factory that takes the original method and returns its replacement, then register the
factory against the method's dotted path:

```python
from transformers.exporters.utils import register_patch

@register_patch("dynamo", "transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier")
def _patch_classifier_cast(_original):
    # Replace the untraceable dtype cast with a no-op during export.
    return lambda self, *args, **kwargs: None
```

The exporter swaps the method in before tracing and restores it afterward,
so the patch only affects export. A few variations:

- Pass extra paths to share one factory across call sites, for example
  `@register_patch("dynamo", path_a, path_b)`.
- Patch a `torch` op by pointing the path at it, for example `@register_patch("onnx", "torch.where")`.
  The factory receives the real op as its argument, so the replacement can call through to it.
- Write a fix instead of a patch when you need to rewrite the graph after tracing. The mechanism is
  the same, a decorated function in the matching registry.

## Stage reference

Each exporter's source labels its stages as `# ── Stage N: … ──` comment blocks, so the file and
this reference line up. Look there for the exact ops and classes each stage handles.

### DynamoExporter

The base exporter runs one patch stage and four helpers, in order, inside `DynamoExporter.export`
(see [exporter_dynamo.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_dynamo.py)).

1. Forward-signature patch: gives `model.forward` a flat argument signature so `torch.export`
   doesn't bundle inputs into one `**kwargs` tuple. This is internal and not an extension point.
2. Model patches: swap untraceable model methods for export-safe equivalents during tracing. Extend
   with `@register_patch("dynamo", ...)`.
3. Pytree registration: register each `Cache` and `ModelOutput` so `torch.export` can flatten and
   rebuild it (usually happens automatically). Add a branch to `_flatten_to_context` / `_unflatten_from_context`
   for a type the attribute walk can't reach.
4. Dynamic shapes: assign `Dim.AUTO` to every tensor and cache leaf when `dynamic=True`. Override
   with `DynamoConfig.dynamic_shapes`.
5. State cleanup: reset tensor attributes a model sets inside `forward` that `torch.export` leaves
   as fake tensors. Extend by adding the attribute name to `_STATEFUL_CACHE_ATTRS`.

### OnnxExporter

`OnnxExporter` adds five stages around `torch.onnx.export` (see
[exporter_onnx.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_onnx.py)).
Grep the file for the full list of patches:

```bash
grep -nE "^def (_patch_|_fix_|_aten_)" src/transformers/exporters/exporter_onnx.py
```

1. Torch patches: swap `torch` ops the ONNX exporter can't translate as-is. Extend with
   `@register_patch("onnx", ...)`.
2. ONNX patches: re-run the node fixes after `run_decompositions` so newly introduced shape-guard
   nodes get repaired before lowering. Uses the same `@register_patch("onnx", ...)` registry.
3. FX node fixes: rewrite graph nodes the ONNX exporter can't lower, such as alias ops, in-place
   views, and dead asserts. Extend with `@register_fx_node_fix("onnx")`.
4. ONNX translations: supply a custom lowering for an aten op where the default is missing or buggy
   (for example `aten.index_put` or `aten._grouped_mm`). Add an `_aten_*` function to
   `_ONNX_TRANSLATION_TABLE`.
5. ONNX IR fixes: rewrite the finished ONNX program to work around ONNX Runtime bugs (for example
   forcing `TopK(sorted=True)`). Add a `_fix_ir_*` function to `_IR_FIXES`.

### ExecutorchExporter

`ExecutorchExporter` adds five stages around `to_edge_transform_and_lower` and `to_executorch`,
starting with backend preparation (see
[exporter_executorch.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_executorch.py)).

1. Backend preparation: move the model to the target device and dtype and pick its partitioner
   (`prepare_for_xnnpack`, `prepare_for_cuda`). Add a backend by registering `prepare_for_<name>` in
   `_BACKEND_PREPARE`.
2. Torch patches: replace `torch` ops the ExecuTorch backends can't accept, such as `split_copy`,
   `chunk`, and `topk(k>dim)`. Extend with `@register_patch("executorch", ...)`.
3. ExecuTorch patches: swap ExecuTorch internals that crash on valid dynamic-shape graphs. Uses the
   same `@register_patch("executorch", ...)` registry.
4. FX program fixes: repair the exported program where the fix needs whole-program context, such as
   widening range constraints or filling missing placeholder metadata. Extend with
   `@register_fx_program_fix("executorch")`.
5. FX node fixes: rewrite individual nodes, such as mapping Python sym ops to `executorch_prim.*` or
   rewriting `pow` as a `mul` chain. Extend with `@register_fx_node_fix("executorch")`.

## Known upstream workarounds

A small number of model classes hit confirmed bugs in `onnxscript`'s graph optimizer (constant
folding crashing on `SplitToSequence`, FPN initializers being dropped). For those, ONNX optimization
is selectively disabled via
[ONNX_DISABLE_OPTIMIZE_MODEL_CLASSES](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_utils.py)
in the test suite, and each entry is annotated with the upstream issue it works around. This list is
expected to shrink as upstream bugs land. It is not an extension point for arbitrary skipping, and
new entries must reference a specific upstream bug.

A second list,
[EXPORT_SKIP_MODEL_CLASSES](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_utils.py),
opts a handful of model classes out of the entire export sweep when the model itself is fundamentally
non-exportable as-is (data-dependent control flow that can't be vectorized, or modules treated as
forward arguments). Every entry carries a `TODO` naming the underlying model
change needed, and the list is expected to shrink, not grow.
