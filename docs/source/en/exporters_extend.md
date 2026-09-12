<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
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

1. Backend preparation: the selected `ExecutorchBackendRecipe` prepares the model, inputs, shape
   bounds, and backend state. Built-in and external backends use the same recipe contract; external
   backends register a factory with `register_executorch_backend`.
2. Torch patches: replace `torch` ops the ExecuTorch backends can't accept, such as `split_copy`,
   `chunk`, and `topk(k>dim)`. Extend with `@register_patch("executorch", ...)`.
3. ExecuTorch patches: swap ExecuTorch internals that crash on valid dynamic-shape graphs. Uses the
   same `@register_patch("executorch", ...)` registry.
4. FX program fixes: repair the exported program where the fix needs whole-program context, such as
   widening range constraints or filling missing placeholder metadata. Extend with
   `@register_fx_program_fix("executorch")`.
5. FX node fixes: rewrite individual nodes, such as mapping Python sym ops to `executorch_prim.*` or
   rewriting `pow` as a `mul` chain. Extend with `@register_fx_node_fix("executorch")`.

## Register an ExecuTorch backend

Register a factory that parses backend-owned options and returns an `ExecutorchBackendRecipe`.
The exporter invokes the factory once per export. A recipe can wrap or transform the source model,
choose its input ABI, carry metadata from preparation to lowering, and supply its own lowering
implementation. Adding an ExecuTorch backend does not require registering another export format.

```python
from transformers.exporters import (
    ExecutorchBackendPreparation,
    ExecutorchBackendRecipe,
    ExecutorchCompatibilityPolicy,
    register_executorch_backend,
)


def make_backend(options):
    if options:
        raise ValueError(f"Unsupported backend options: {sorted(options)}")

    def prepare(model, inputs, config):
        return ExecutorchBackendPreparation(
            model=model,
            sample_inputs=inputs,
            state={"constant_methods": {}},
        )

    def lower(program, preparation, config):
        # Delegate to your backend's edge lowering and ExecuTorch serialization.
        return lower_to_my_backend(program, preparation.state, config)

    return ExecutorchBackendRecipe(
        prepare=prepare,
        lower=lower,
        compatibility=ExecutorchCompatibilityPolicy(),
    )


register_executorch_backend("my_backend", make_backend)
```

Registering the same factory again is a no-op. Replacing another external factory requires
`overwrite=True`; built-in names cannot be replaced. Backend-specific options belong in
`ExecutorchConfig.backend_options` and should be validated by the factory. Built-in backends reject
options they do not support. Configuration fields, including backend options, must support
`copy.deepcopy`; mutable configuration is snapshotted and object identity is not preserved.
These values need not be JSON-serializable, but they should not contain live resources such as locks,
open handles, or generators. Construct resources during preparation and retain them in
`preparation.state`, which is not copied. Snapshot failures report which configuration copy failed.

### Preparation and compatibility

By default, Transformers normalizes the recipe's returned inputs: it strips output-control flags,
precomputes supported model-specific inputs, and aligns input dtype/device with the prepared model.
A recipe that already defines its complete input ABI sets `normalize_inputs=False` on
`ExecutorchBackendPreparation` to bypass that normalization. It then owns input keys, dtype/device,
and their agreement with `dynamic_shapes`. Configuration-only output flags can be supplied separately
in `output_flags`; forward arguments with those names remain intact in prepared-input mode.

Preparation's explicit `dynamic_shapes` take precedence over configuration shapes. `None` falls back
to the configuration and, if requested, automatic dynamic shapes. `capture_contexts` must be fresh
context managers created for each preparation; they cover tracing only, not transforms or lowering.

`ExecutorchCompatibilityPolicy` preserves the existing common behavior by default:
`common_patches=True` and `common_graph_fixes=True`. These bundles contain compatibility workarounds,
including shape-bound heuristics and runtime-assert removal; they are not merely cosmetic cleanup.
A backend can disable either bundle. Explicit recipe patches and backend-namespaced patches still
apply. These options do not disable Dynamo's tracing support, including signature adaptation,
model patches, input copying, and pytree handling.

Recipe patches and attention registries cover preparation through immediate lowering. Attention
selection on the declared preparation target starts after preparation, before normalization.
Common and backend patch bundles start after normalization. After tracing, capture contexts exit,
then the recipe's optional `transform_exported_program` runs, followed by enabled common program
and node fixes, then lowering. A transform must return an `ExportedProgram`.

Recipe patches exclusively own their target attributes while active. A conflicting installation
through the shared patch helpers raises an error rather than silently shadowing a recipe patch.
Ordinary registry patches still compose in registration order when no exclusive owner is present.
To replace one shared workaround without disabling its entire bundle, exclude its target explicitly:

```python
from transformers.exporters import ExecutorchCompatibilityPolicy, ExecutorchExportPatch

reshape_target = "torch.Tensor.reshape"
recipe = ExecutorchBackendRecipe(
    prepare=prepare,
    lower=lower,
    patches=(ExecutorchExportPatch((reshape_target,), make_backend_reshape),),
    compatibility=ExecutorchCompatibilityPolicy(
        excluded_patch_targets=(reshape_target,),
    ),
)
```

Exclusions filter the common ExecuTorch, selected-backend, and Dynamo patch registries for this
export. They do not filter recipe patches, graph fixes, signature/configuration helpers, or arbitrary
capture contexts, and are not implicitly inherited by nested exports. Exclusions match the resolved
owner and attribute, so two paths naming the same slot agree; separate bindings to the same function
are different slots. Duplicate recipe targets are rejected. Exclusions cannot undo patches already
installed by an outer scope. Raw third-party attribute assignments outside the shared patch helpers
are not covered by collision detection.

### Scoped attention

An `ExecutorchAttention` requires an explicit masking choice. For example, a Llama-compatible
eager attention implementation must be paired with an eager-compatible mask builder:

```python
from transformers.exporters import ExecutorchAttention, scoped_executorch_attention
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

attention = ExecutorchAttention(
    implementation="my_eager_attention",
    attention_function=eager_attention_forward,
    mask_function=eager_mask,
)


def prepare(model, inputs, config):
    return ExecutorchBackendPreparation(
        model=model,
        sample_inputs=inputs,
        attention_target=model,
    )
```

Assign this descriptor to the recipe's `attention`. The exporter installs its registries before
calling `prepare`, then calls `set_attn_implementation()` on `preparation.attention_target`.
A wrapper can return its underlying HF model as the target; a replacement model can name itself.
The original input model need not provide a setter. `attention_target=None` means registry-only:
the recipe owns configuration selection, and the exporter does not guess a target.

`mask_function=None` explicitly means the backend owns masking: the scope temporarily removes
any existing mask registration for that name, so shared mask generation is skipped. This does not
supply causal, padding, sliding-window, or other masks on the backend's behalf. Already-prepared
4-D masks can still pass through the mask utilities. Use this choice only when the backend preserves
all masking semantics required by its supported inputs.

For custom capture of a plain module that already selects its implementation, use registry-only scope:

```python
with scoped_executorch_attention(attention):
    program = torch.export.export(custom_module, args, strict=True)
```

Passing `model=` selects attention on that model and restores its tracked attention configuration
fields afterward. This also supports preparation-time work that requires selection: use a nested
scope inside `prepare`, then return the desired target for capture and lowering. The setter may run
once in that preparation scope and again for capture, but is not replayed during deferred lowering.
Existing attention and mask registrations are restored on success or failure. Managed selection
restores tracked configuration fields, not arbitrary side effects of a custom setter.

### Capture now, lower later

```python
from transformers.exporters import ExecutorchConfig, ExecutorchExporter

exporter = ExecutorchExporter()
captured = exporter.capture(model, inputs, ExecutorchConfig(backend="my_backend"))
program = captured.exported_program
metadata = captured.preparation.state
artifact = exporter.lower(captured)
```

The capture retains the resolved recipe, prepared model/inputs/state, and a configuration snapshot.
Later registry replacement or changes to the caller's configuration do not redirect lowering.
Deferred lowering reenters fresh recipe, attention, and selected compatibility patch scopes. It
restores the captured attention-field values temporarily without calling the model setter again.
It does not rerun preparation, capture contexts, tracing, transforms, or graph fixes. Patch factories
must therefore support fresh invocations rather than depending on one-shot side effects.

A capture permits one lowering attempt, even if that attempt fails: backend lowering may mutate
the graph or backend state. Captures retain live Python objects and are not serializable bundles or
model snapshots. Do not independently mutate their model/state between capture and lowering.
`export()` keeps capture and lowering inside one uninterrupted scope. `capture()` is the intermediate
API: use `captured.exported_program` for the transformed, compatibility-fixed graph and
`captured.preparation.state` for backend data.

### Mutation and concurrency

Temporary attribute patches, attention registrations, and tracked config changes are restored on
scope exit. Backend preparation is not transactional: device/dtype conversion, quantization, cache
installation, and other source-model changes remain the backend/caller's responsibility.

Cooperating exporters and patch helpers use one reentrant mutation lock, held from before snapshots
until restoration completes. Same-thread nested scopes are supported; concurrent cooperating
exports are serialized. Unrelated eager inference and third-party mutations do not acquire this
lock and are not isolated. Use separate processes for independent parallel export. Do not start a
cooperating export on another thread and wait for it while holding an export scope.

Backend-only patch dependencies can use `register_patch(..., lazy=True)` so target owners are
resolved when their bundle is applied, not when the exporter is imported. This preserves patch
registration order while avoiding unnecessary backend imports.

## Known upstream workarounds

A few model classes hit confirmed bugs in the `onnxscript` graph optimizer (constant folding crashing
on `SplitToSequence`, FPN initializers being dropped). [ONNX_DISABLE_OPTIMIZE](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_export.py) 
disables `onnxscript` optimization for those models. Each entry records the
upstream issue next to the model name. The list is expected to shrink as upstream bugs land, so a
new entry must reference a specific upstream bug rather than disable optimization arbitrarily.

[EXPORT_SKIPS](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_export.py),
opts a handful of model classes out of the export sweep entirely when the model is
fundamentally non-exportable as-is (data-dependent control flow that can't be vectorized, or modules
treated as forward arguments). Each entry carries a reason naming the model-side change needed. This
list is also expected to shrink, not grow.
