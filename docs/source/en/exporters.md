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

# Exporters

Export any [`PreTrainedModel`] to ONNX, ExecuTorch, or a standalone PyTorch program — same model,
same two lines of code, any runtime.

```python
exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))  # or OnnxExporter, ExecutorchExporter
exported = exporter.export(model, inputs)
```

Because the exporters live inside Transformers, they evolve with the models. Every architecture
change, new attention pattern, or custom cache type is supported at export time from day one —
no waiting for a downstream library to catch up.

| Exporter               | Output                     | Runtime                                       |
| ---------------------- | -------------------------- | --------------------------------------------- |
| [`DynamoExporter`]     | `ExportedProgram`          | Any PyTorch runtime, AOT compilation          |
| [`OnnxExporter`]       | `ONNXProgram`              | Any ONNX runtime (ORT, TensorRT, OpenVINO, …) |
| [`ExecutorchExporter`] | `ExecutorchProgramManager` | Mobile and edge devices (ExecuTorch)          |

## Installation

<hfoptions id="exporters-install">
<hfoption id="Dynamo">

```bash
pip install transformers torch
```

</hfoption>
<hfoption id="ONNX">

```bash
pip install transformers torch onnxscript onnxruntime
```

</hfoption>
<hfoption id="ExecuTorch">

```bash
pip install transformers torch executorch
```

</hfoption>
</hfoptions>

## Quick start

All exporters share the same interface: create an exporter with a config, call `.export(model, inputs)`.
Switch between runtimes by swapping the exporter class — nothing else changes.

<hfoptions id="exporters-quickstart">
<hfoption id="Dynamo">

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> inputs = dict(tokenizer("Hello, world!", return_tensors="pt"))

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> exported = exporter.export(model, inputs)

>>> # run the exported graph directly
>>> outputs = exported.module()(**inputs)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer("Hello, world!", return_tensors="pt"))

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
onnx_program = exporter.export(model, inputs)

# save and load with ONNX Runtime
onnx_program.save("model.onnx")

import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
ort_inputs = {k: v.numpy() for k, v in inputs.items()}
outputs = session.run(None, ort_inputs)
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer("Hello, world!", return_tensors="pt"))

exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend="xnnpack", dynamic=True))
et_program = exporter.export(model, inputs)

# save for on-device deployment
et_program.save("model.pte")
```

</hfoption>
</hfoptions>

## Dynamic shapes

Set `dynamic=True` on any config to export with symbolic input shapes. All tensor dimensions
are automatically marked as dynamic, so the exported graph accepts inputs of any size at
runtime without retracing.

<hfoptions id="dynamic-shapes">
<hfoption id="Dynamo">

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> inputs = dict(tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt"))

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> exported = exporter.export(model, inputs)

>>> # works with any batch size or sequence length
>>> outputs = exported.module()(**dict(tokenizer("A single input", return_tensors="pt")))
>>> outputs = exported.module()(**dict(tokenizer(["One", "Two", "Three"], padding=True, return_tensors="pt")))
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt"))

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
onnx_program = exporter.export(model, inputs)

# works with any batch size or sequence length
onnx_program(**dict(tokenizer("A single input", return_tensors="pt")))
onnx_program(**dict(tokenizer(["One", "Two", "Three"], padding=True, return_tensors="pt")))
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt"))

exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend="xnnpack", dynamic=True))
et_program = exporter.export(model, inputs)
```

</hfoption>
</hfoptions>

For fine-grained control over which dimensions are dynamic, pass explicit `dynamic_shapes` instead.
This is passed directly to `torch.export.export` — see the
[torch.export documentation](https://pytorch.org/docs/stable/export.html) for the expected format.

<hfoptions id="explicit-dynamic-shapes">
<hfoption id="Dynamo">

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> inputs = dict(tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt"))

>>> batch = torch.export.Dim("batch", min=1, max=32)
>>> seq = torch.export.Dim("seq", min=1, max=2048)

>>> exporter = DynamoExporter(export_config=DynamoConfig(
...     dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
...     prefer_deferred_runtime_asserts_over_guards=True,
... ))
>>> exported = exporter.export(model, inputs)
```

</hfoption>
<hfoption id="ONNX">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt"))

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = OnnxExporter(export_config=OnnxConfig(
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    prefer_deferred_runtime_asserts_over_guards=True,
))
onnx_program = exporter.export(model, inputs)
```

</hfoption>
<hfoption id="ExecuTorch">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt"))

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = ExecutorchExporter(export_config=ExecutorchConfig(
    backend="xnnpack",
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    prefer_deferred_runtime_asserts_over_guards=True,
))
et_program = exporter.export(model, inputs)
```

</hfoption>
</hfoptions>

## Generative models

For autoregressive generation, the model's `forward` has different shapes at the prefill step
(full prompt, no KV cache) versus the decode step (single token, populated KV cache). Exporters
expose [`~HfExporter.export_for_generation`] which splits both stages and exports each.
For multi-modal generative models it additionally splits the prefill into vision/audio encoder,
projector, language model, and `lm_head`. Encoder and language-model discovery uses the canonical
[`~PreTrainedModel.get_encoder`] (`modality="image"` / `"audio"`) and
[`~PreTrainedModel.get_decoder`] accessors, so any new architecture that wires those up
correctly works out of the box. Projector lookup (`multi_modal_projector`, `connector`,
`embed_vision`, `embed_audio`) is heuristic — see
[`~transformers.exporters.utils._MULTIMODAL_PROJECTOR_NAMES`] in
[exporters/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/utils.py)
and append a new attribute name there if needed.

<hfoptions id="generate">
<hfoption id="Dynamo">

```python
>>> from transformers import AutoModelForImageTextToText, AutoProcessor
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
>>> messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
>>> text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
>>> inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> components = exporter.export_for_generation(model, inputs)
>>> # components = {"image_encoder": ExportedProgram, "language_model": ExportedProgram, "lm_head": ExportedProgram, "decode": ExportedProgram}
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
components = exporter.export_for_generation(model, inputs)
# components = {"image_encoder": ONNXProgram, "language_model": ONNXProgram, "lm_head": ONNXProgram, "decode": ONNXProgram}
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend="xnnpack", dynamic=True))
components = exporter.export_for_generation(model, inputs)
# components = {"image_encoder": ExecutorchProgramManager, "language_model": ..., "lm_head": ..., "decode": ...}
```

</hfoption>
</hfoptions>

<Tip warning={true}>

The exported components are **independent graphs**, not a turnkey inference pipeline.
The caller is responsible for running each encoder, projecting embeddings, and orchestrating
the generation loop. We are actively working to reduce the glue required between components.

</Tip>

<details>
<summary>What <code>export_for_generation</code> does under the hood</summary>

[`~exporters.utils.decompose_for_generation`] runs `model.generate(**inputs, max_new_tokens=2)`
once and hooks `model.forward` to capture the real prefill and decode kwargs (and the
per-submodule kwargs via hooks on each encoder / projector / language model if the model is
multi-modal). That's why it works for any architecture — decoder-only, SSM, encoder-decoder,
multi-modal — without per-model glue. `export_for_generation` is a one-liner over it.

Call `decompose_for_generation` directly when you want to do something between decomposing
and exporting — run an eager forward for verification, swap a submodule's inputs, skip a stage:

```python
from transformers.exporters.utils import decompose_for_generation

components = decompose_for_generation(model, inputs)
# {"image_encoder": (submodel, fwd_kwargs), "language_model": (...), ..., "decode": (...)}

exported = {}
for name, (submodel, subinputs) in components.items():
    eager_outputs = submodel(**subinputs)
    exported[name] = exporter.export(submodel, subinputs)
```

</details>

## Export pipeline

`torch.export`, `torch.onnx.export`, and ExecuTorch each have rough edges around specific
PyTorch patterns. The exporters work around these with a small set of patches and fixes
applied at well-defined points in the export flow. Each exporter's source file labels
these as `# ── Stage N: … ─────` blocks; the sections below mirror that layout 1:1, so
the file you read and the doc you read are the same map.

Two lifecycles are used consistently:

- **Patches** (registered via `@register_patch(backend, "dotted.path")`, installed via
  `apply_patches(backend)`) reversibly swap an attribute (a `torch` op, an ExecuTorch
  internal, an instance method) for the duration of the export. Originals are restored
  on exit, even if the body raises.
- **Fixes** (registered via `@register_fx_node_fix(backend)` /
  `@register_fx_program_fix(backend)`, applied via `apply_fx_node_fixes(backend, gm)` /
  `apply_fx_program_fixes(backend, ep)`; ONNX-IR fixes still listed in `_IR_FIXES` and
  applied via `apply_onnx_ir_fixes`) mutate the in-progress graph or program in place.
  There's no revert — they're meant to permanently repair the artifact before the next
  pipeline step.

Every patch / fix sits in a backend-keyed registry (`_PATCHES`, `_FX_NODE_FIXES`,
`_FX_PROGRAM_FIXES` in [exporters/utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/utils.py)).
Adding a new one is *write a function and decorate it* — nothing else.

### `DynamoExporter`

The base exporter has one patch stage and three structural helpers. They run in this order
inside `DynamoExporter.export`, against the original `nn.Module`:

| #     | Stage                                                                | Section in [exporter_dynamo.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_dynamo.py) | What it does                                                                                                                                                                                                                                | How to extend                                                                                                                                                  |
| ----- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Model patches** (`patch_untraceable_patterns` / `_MODEL_PATCHERS`) | `# ── Stage 1: Model patches ──`                                                                                                     | Walks every submodule. Each `_patch_*(module) → (attr_name, replacement) \| None` factory matches on attribute / source-string and, on a hit, swaps the attribute for a tracing-safe replacement. Originals are saved and restored on exit. | Define `_patch_*` and append to `_MODEL_PATCHERS`. Examples: mamba mask, linear-attn mask, NLLB classifier cast, chunked-vision-attention `split → zip → cat`. |
| **2** | **Pytree registration** (`register_cache_pytrees_for_model`)         | `# ── Stage 2: Pytree registration ──`                                                                                               | Registers flatten/unflatten via `torch.utils._pytree.register_pytree_node` for every captured `Cache` / `ModelOutput`. Reflection-driven, tuned for tensor containers (not a general serialiser).                                           | Usually automatic. If a type isn't reflectable, add a branch to `_flatten_to_context` / `_unflatten_from_context`.                                             |
| **3** | **Forward-signature patch** (`patch_forward_signature`)              | `# ── Stage 3: Model signature patch ──`                                                                                             | Replaces `model.forward` with an explicit flat-arg signature derived from the inputs dict, so `torch.export` doesn't bundle `**kwargs` into a single tuple.                                                                                 | Internal — no extension knob.                                                                                                                                  |
| **4** | **Dynamic shapes** (`get_auto_dynamic_shapes`)                       | `# ── Stage 4: Dynamic shapes ──`                                                                                                    | Auto-assigns `Dim.AUTO` to every tensor and cache leaf when `DynamoConfig.dynamic=True` and the user did not pass `dynamic_shapes` explicitly.                                                                                              | Override per-export via `DynamoConfig.dynamic_shapes`.                                                                                                         |
| **5** | **State cleanup** (`cleanup_state` / `_STATEFUL_CACHE_ATTRS`)        | `# ── Stage 5: State cleanup ──`                                                                                                     | Resets non-`Cache` tensor attributes set inside `forward` (e.g. glm_moe_dsa `_cached_keys`, wav2vec2_bert `cached_rotary_positional_embedding`) that `torch.export` leaves as FakeTensors, so a follow-up eager forward is safe.            | Append the attribute name to `_STATEFUL_CACHE_ATTRS`.                                                                                                          |

### `OnnxExporter`

`OnnxExporter` extends `DynamoExporter` with five numbered stages applied around
`torch.onnx.export`. The labels match the `# ── Stage N: … ──` headers in the source:

| #     | Stage                                                       | Section in [exporter_onnx.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_onnx.py) | When it runs                                     | Lifecycle                            | What it does                                                                                                                                                                                                                                                                | How to extend                                                                                |
| ----- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **1** | **Torch patches** (`_PATCHES["onnx"]`)                      | `# ── Stage 1: Torch patches ──`                                                                                                 | During `torch.export` / `torch.onnx.export`      | Reversible (`apply_patches("onnx")`) | Reversible swaps of `torch` ops (`where`, `unsqueeze`, `scaled_dot_product_attention`, `searchsorted`, …) that the ONNX decomposer can't lower as-is. Each `_patch_*(original)` closes over the original.                                                                   | Define `_patch_*(original)` and decorate with `@register_patch("onnx", "dotted.path")`.       |
| **2** | **ONNX patches** (`_PATCHES["onnx"]`)                       | `# ── Stage 2: ONNX patches ──`                                                                                                  | During `torch.onnx.export`                       | Reversible (`apply_patches("onnx")`) | Hooks the private `_prepare_exported_program_for_export` step so the FX node fixes (stage 3) run again right after `run_decompositions` — any new symbolic-guard nodes the ONNX decomposition introduces get repaired before the FX → ONNX lowering picks them up.          | Same registry as stage 1 — define `_patch_*(original)` and decorate with `@register_patch("onnx", "dotted.path")`. |
| **3** | **FX node fixes** (`_FX_NODE_FIXES["onnx"]`)                | `# ── Stage 3: FX node fixes ──`                                                                                                 | After `torch.export`, again after `run_decompositions` | In-place (`apply_fx_node_fixes("onnx", gm)`) | Per-node rewrites on the `GraphModule` to drop or replace nodes the ONNX decomposer can't lower (alias ops, in-place views, `_assert_*`, dead comparisons, in-place `triu_`, `fill_diagonal_`, `sort(stable=True)`). DCE runs automatically at the end of the walk.   | Define `_fix_*(gm, node) → bool` (return `True` to consume) and decorate with `@register_fx_node_fix("onnx")`. |
| **4** | **ONNX translations** (`_ONNX_TRANSLATION_TABLE`)           | `# ── Stage 4: ONNX translations ──`                                                                                             | During FX → ONNX lowering                        | n/a (translation table)              | Overrides `torchlib`'s default lowering for specific aten ops where the default is buggy or missing. Currently `aten.index_put` (bool-mask path) and `aten.bincount` (`OneHot + ReduceSum`).                                                                                | Implement an `_aten_*` onnxscript function and add it to `_ONNX_TRANSLATION_TABLE`.          |
| **5** | **ONNX IR fixes** (`_IR_FIXES` / `apply_onnx_ir_fixes`)     | `# ── Stage 5: ONNX IR fixes ──`                                                                                                 | After `torch.onnx.export` returns                | In-place (`apply_onnx_ir_fixes`)     | Post-export rewrites on the `ONNXProgram` IR to work around ORT validation/runtime bugs (e.g. forcing `TopK(sorted=True)`). Applied to both the top-level graph and every function.                                                                                         | Implement `_fix_ir_*(graph_like)` and append to `_IR_FIXES`.                                 |

A complete inventory of patches in the file is one grep away:

```bash
grep -nE "^def (_patch_|_fix_|_aten_)" src/transformers/exporters/exporter_onnx.py
```

### `ExecutorchExporter`

`ExecutorchExporter` extends `DynamoExporter` with four numbered stages applied around
`to_edge_transform_and_lower` and `to_executorch`:

| #     | Stage                                                              | Section in [exporter_executorch.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_executorch.py) | When it runs                                            | Lifecycle                            | What it does                                                                                                                                                                                                                                                                                              | How to extend                                                                                                                |
| ----- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Backend preparation** (`_BACKEND_PREPARE`)                       | `# ── Stage 1: Backend preparation ──`                                                                                                       | Before `torch.export`                                   | n/a (one-shot)                       | `prepare_for_xnnpack` moves the model to CPU/fp32 and selects `XnnpackPartitioner`; `prepare_for_cuda` moves to CUDA/bf16 and selects `CudaPartitioner`. Returns `(model, sample_inputs, partitioner)`.                                                                                                   | Implement `prepare_for_<name>` and register it in `_BACKEND_PREPARE`.                                                        |
| **2** | **Torch patches** (`_PATCHES["executorch"]`)                       | `# ── Stage 2: Torch patches ──`                                                                                                             | During `torch.export` tracing                           | Reversible (`apply_patches("executorch")`) | Replaces `torch` ops the ExecuTorch backends can't accept (`split_copy`, `chunk`, `topk(k>dim)`, non-divisible `avg_pool2d`, `dropout`, in-place `view`, GQA-shaped SDPA) with decomposed equivalents.                                                                                              | Define `_patch_*(original)` and decorate with `@register_patch("executorch", "dotted.path")`.                                |
| **3** | **ExecuTorch patches** (`_PATCHES["executorch"]`)                  | `# ── Stage 3: ExecuTorch patches ──`                                                                                                        | During `to_edge_transform_and_lower` / `to_executorch`  | Reversible (`apply_patches("executorch")`) | Reversibly swaps ExecuTorch internals that crash on legitimate dynamic-shape patterns: `SpecPropPass.update_placeholder_tensor_specs`, `PruneEmptyTensorsPass.remove_empty_tensors_from_cat`, `eval_upper_bound`, `dim_order_from_stride` (rebound on every importer), XNNPACK squeeze/unsqueeze define-node, complex-dtype validator, edge-dialect sym-op allowlist. | Same registry as stage 2 — define `_patch_*(original)` and decorate with `@register_patch("executorch", "dotted.path")`.     |
| **4** | **FX program fixes** (`_FX_PROGRAM_FIXES["executorch"]`)           | `# ── Stage 4: FX program fixes ──`                                                                                                          | After `torch.export`, before `to_edge_transform_and_lower` | In-place (`apply_fx_program_fixes("executorch", ep)`) | Repair the `ExportedProgram` where the fix needs program-level context: widen `int_oo` upper bounds in `range_constraints`, fill missing placeholder `meta["val"]` from `state_dict`.                                                                                                          | Define `_fix_*(exported_program) → None` and decorate with `@register_fx_program_fix("executorch")`. |
| **5** | **FX node fixes** (`_FX_NODE_FIXES["executorch"]`)                 | `# ── Stage 5: FX node fixes ──`                                                                                                             | After stage 4, before `to_edge_transform_and_lower`     | In-place (`apply_fx_node_fixes("executorch", gm)`) | Per-node rewrites: swap Python sym ops for `executorch_prim.*` equivalents, rewrite `pow` as `mul` chain, normalize amax/max negative dim, force contiguous clone. DCE runs automatically at the end of the walk.                                                                                | Define `_fix_*(gm, node) → bool` (return `True` to consume) and decorate with `@register_fx_node_fix("executorch")`. |

### When to patch the exporter vs. fix the model

The split is intentional:

- **Modeling change** if the pattern blocks export across multiple backends — data-dependent
  loops, stateful caches outside `Cache`, hand-written split-loop attention. Fix it once in
  the model and every exporter benefits.
- **Exporter patch** if the issue is a single backend's lowering bug — a missing ONNX
  translation, an ORT validation quirk, an FX decomposition that emits a dead op. Keep the
  workaround in the exporter and the modeling code stays clean.

### Known upstream workarounds

A small number of model classes hit confirmed bugs in `onnxscript`'s graph optimizer
(constant folding crashing on `SplitToSequence`, FPN initialisers being dropped). For those,
ONNX optimisation is selectively disabled via
[`ONNX_DISABLE_OPTIMIZE_MODEL_CLASSES`](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_utils.py)
in the test suite — each entry is annotated with the upstream issue it works around. This
list is **expected to shrink** as upstream bugs land; it is not an extension point for
arbitrary skipping, and new entries should reference a specific upstream bug.

## API reference

### Exporter classes

[[autodoc]] transformers.exporters.exporter_dynamo.DynamoExporter
    - export

[[autodoc]] transformers.exporters.exporter_onnx.OnnxExporter
    - export

[[autodoc]] transformers.exporters.exporter_executorch.ExecutorchExporter
    - export

### Configuration

[[autodoc]] transformers.utils.export_config.DynamoConfig

[[autodoc]] transformers.utils.export_config.OnnxConfig

[[autodoc]] transformers.utils.export_config.ExecutorchConfig

### Utilities

[[autodoc]] transformers.exporters.utils.get_leaf_tensors

[[autodoc]] transformers.exporters.utils.prepare_for_export

[[autodoc]] transformers.exporters.utils.decompose_prefill_decode

[[autodoc]] transformers.exporters.utils.decompose_multimodal

[[autodoc]] transformers.exporters.utils.decompose_for_generation

[[autodoc]] transformers.exporters.utils.is_multimodal
