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
exporter = DynamoExporter()
config = DynamoConfig(dynamic=True)  # or OnnxExporter, ExecutorchExporter
exported = exporter.export(model, inputs, config=config)
```

Because the exporters live inside Transformers, they evolve with the models. Every architecture
change, new attention pattern, or custom cache type is supported at export time from day one —
no waiting for a downstream library to catch up.

<Tip warning={true}>

The exporters are **experimental**. Many of the patches in this module work around specific
upstream bugs (torch, onnxscript, onnxruntime, executorch) and will be removed as soon as the
fix lands upstream. Until the API stabilises, treat the patches as tied to the versions used in
the test suite — pin those versions in production tooling, and expect both new patches and
removals as we follow upstream.

</Tip>

| Exporter               | Output                     | Runtime                                       |
| ---------------------- | -------------------------- | --------------------------------------------- |
| [`DynamoExporter`]     | `ExportedProgram`          | Any PyTorch runtime, AOT compilation          |
| [`OnnxExporter`]       | `ONNXProgram`              | Any ONNX runtime (ORT, TensorRT, OpenVINO, …) |
| [`OpenVINOExporter`]   | `openvino.Model`           | OpenVINO runtime (Intel CPU/GPU/NPU)          |
| [`ExecutorchExporter`] | `ExecutorchProgramManager` | Mobile and edge devices (ExecuTorch)          |

[`AutoHfExporter`] picks the right exporter from a config and [`AutoExportConfig`] picks the right
config class from a dict — the same auto-class idiom the rest of `transformers` uses, useful when
the backend is selected at runtime rather than hard-coded in the call site.

## Installation

<hfoptions id="exporters-install">
<hfoption id="Dynamo">

```bash
pip install transformers "torch==2.12.0"
```

</hfoption>
<hfoption id="ONNX">

```bash
pip install transformers "torch==2.12.0" "onnx==1.21.0" "onnxscript==0.7.0" onnxruntime
```

</hfoption>
<hfoption id="ExecuTorch">

```bash
pip install transformers "torch==2.12.0" "executorch==1.3.1"
```

</hfoption>
<hfoption id="OpenVINO">

```bash
pip install transformers "torch==2.12.0" "openvino==2025.0.0"
```

</hfoption>
</hfoptions>

<Tip>
The versions above are the ones the exporter test suite is pinned against — newer / older releases
often work but the exporter patches target a specific API surface, so for production tooling pin
these and expect [`HfExporter`] to log a warning when it detects drift.
</Tip>

## Quick start

All exporters share the same interface: create an exporter with a config, call `.export(model, inputs)`.
Switch between runtimes by swapping the exporter class — nothing else changes.

<hfoptions id="exporters-quickstart">
<hfoption id="Dynamo">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import DynamoExporter, DynamoConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = DynamoExporter()
config = DynamoConfig(dynamic=True)
exported = exporter.export(model, inputs, config=config)

# run the exported graph directly
outputs = exported.module()(**inputs)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = OnnxExporter()
config = OnnxConfig(dynamic=True)
onnx_program = exporter.export(model, inputs, config=config)

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
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = ExecutorchExporter()
config = ExecutorchConfig(backend="xnnpack", dynamic=True)
et_program = exporter.export(model, inputs, config=config)

# save for on-device deployment
et_program.save("model.pte")

# load and run via the ExecuTorch Python runtime
from executorch.runtime import Runtime

program = Runtime.get().load_program("model.pte")
method = program.load_method("forward")
outputs = method.execute(list(inputs.values()))
```

</hfoption>
<hfoption id="OpenVINO">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OpenVINOExporter, OpenVINOConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = OpenVINOExporter()
config = OpenVINOConfig(dynamic=True)
ov_model = exporter.export(model, inputs, config=config)

ov_model.save("model.xml")

# compile and run on CPU (or "GPU" / "NPU" if available)
import openvino as ov
compiled = ov.Core().compile_model(ov_model, "CPU")
ov_inputs = {k: v.numpy() for k, v in inputs.items()}
outputs = compiled(ov_inputs)
```

</hfoption>
</hfoptions>

## Dynamic shapes

The quick-start examples above already pass `dynamic=True`, which marks every tensor
dimension as dynamic so the exported graph accepts inputs of any size at runtime without
retracing.

For fine-grained control over which dimensions are dynamic, pass explicit `dynamic_shapes`
instead. This is forwarded directly to `torch.export.export` — see the
[torch.export documentation](https://pytorch.org/docs/stable/export.html) for the expected format.

<hfoptions id="explicit-dynamic-shapes">
<hfoption id="Dynamo">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import DynamoExporter, DynamoConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt")

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = DynamoExporter()
config = DynamoConfig(
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    # Emit data-dependent shape guards as runtime asserts instead of failing the export when a
    # guard wouldn't hold across the explicit symbolic range — most LLMs need this under fine-grained
    # ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``, where torch.export
    # infers shape relations instead of verifying them against user-stated bounds.
    prefer_deferred_runtime_asserts_over_guards=True,
)
exported = exporter.export(model, inputs, config=config)
```

</hfoption>
<hfoption id="ONNX">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt")

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = OnnxExporter()
config = OnnxConfig(
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    # Emit data-dependent shape guards as runtime asserts instead of failing the export when a
    # guard wouldn't hold across the explicit symbolic range — most LLMs need this under fine-grained
    # ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``, where torch.export
    # infers shape relations instead of verifying them against user-stated bounds.
    prefer_deferred_runtime_asserts_over_guards=True,
)
onnx_program = exporter.export(model, inputs, config=config)
```

</hfoption>
<hfoption id="ExecuTorch">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt")

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = ExecutorchExporter()
config = ExecutorchConfig(
    backend="xnnpack",
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    # Emit data-dependent shape guards as runtime asserts instead of failing the export when a
    # guard wouldn't hold across the explicit symbolic range — most LLMs need this under fine-grained
    # ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``, where torch.export
    # infers shape relations instead of verifying them against user-stated bounds.
    prefer_deferred_runtime_asserts_over_guards=True,
)
et_program = exporter.export(model, inputs, config=config)
```

</hfoption>
<hfoption id="OpenVINO">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OpenVINOExporter, OpenVINOConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt")

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = OpenVINOExporter()
config = OpenVINOConfig(
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    # Emit data-dependent shape guards as runtime asserts instead of failing the export when a
    # guard wouldn't hold across the explicit symbolic range — most LLMs need this under fine-grained
    # ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``, where torch.export
    # infers shape relations instead of verifying them against user-stated bounds.
    prefer_deferred_runtime_asserts_over_guards=True,
)
ov_model = exporter.export(model, inputs, config=config)
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
correctly works out of the box. Projector lookup falls back to a heuristic name list
(`multi_modal_projector`, `connector`, `embed_vision`, `embed_audio`); new architectures
should align their projector attribute to one of these canonical names rather than growing
the list.

<hfoptions id="generate">
<hfoption id="Dynamo">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters import DynamoExporter, DynamoConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exporter = DynamoExporter()
config = DynamoConfig(dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config)
# components = {"image_encoder": ExportedProgram, "language_model": ExportedProgram, "lm_head": ExportedProgram, "decode": ExportedProgram}
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exporter = OnnxExporter()
config = OnnxConfig(dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config)
# components = {"image_encoder": ONNXProgram, "language_model": ONNXProgram, "lm_head": ONNXProgram, "decode": ONNXProgram}
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exporter = ExecutorchExporter()
config = ExecutorchConfig(backend="xnnpack", dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config)
# components = {"image_encoder": ExecutorchProgramManager, "language_model": ..., "lm_head": ..., "decode": ...}
```

</hfoption>
<hfoption id="OpenVINO">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters import OpenVINOExporter, OpenVINOConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exporter = OpenVINOExporter()
config = OpenVINOConfig(dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config)
# components = {"image_encoder": openvino.Model, "language_model": openvino.Model, "lm_head": openvino.Model, "decode": openvino.Model}
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

The capture runs the model eagerly on `inputs`, so pass **small but representative** values —
one short prompt, a single small image, a few audio frames. The exported program isn't tied
to those sizes (dynamic shapes still flow through), but smaller capture inputs make
`decompose_for_generation` cheaper and keep symbolic-shape inference tractable.

Call `decompose_for_generation` directly when you want to do something between decomposing
and exporting — run an eager forward for verification, swap a submodule's inputs, skip a stage:

```python
from transformers.exporters.utils import decompose_for_generation

components = decompose_for_generation(model, inputs)
# {"image_encoder": (submodel, fwd_kwargs), "language_model": (...), ..., "decode": (...)}

exported = {}
for name, (submodel, subinputs) in components.items():
    eager_outputs = submodel(**subinputs)
    exported[name] = exporter.export(submodel,subinputs, config=config)
```

</details>

## Limitations and workarounds

`torch.export`, `torch.onnx.export`, and ExecuTorch each have rough edges around specific
PyTorch patterns. The exporters work around these with a small set of reversible patches
and FX-level fixes applied at well-defined points in the export flow. None of this is
visible from the public `export()` API, but the most common things to know:

- Flash-attention and flex-attention are not exportable on any backend; `sdpa` is the preferred
  setting and `eager` also works (slower). Set one of them on the model before calling `export()`
  if it's using something else.
- `grouped_mm` traces fine through `DynamoExporter` and is auto-translated for `OnnxExporter`;
  for `ExecutorchExporter` with the XNNPACK backend, the exporter swaps MoE experts to
  `batched_mm` because XNNPACK has no `_grouped_mm.out` kernel.
- A short list of models (`EXPORT_SKIP_MODEL_CLASSES`) is skipped from the export sweep when
  the model itself is fundamentally non-exportable; each entry carries a TODO with the
  model-side change needed.

<details>
<summary>Export pipeline — internals (per-backend stages and how to extend)</summary>

Each exporter's source file labels its stages as `# ── Stage N: … ─────` blocks; the
tables below mirror that layout 1:1, so the file you read and the doc you read are the
same map.

Two lifecycles are used consistently:

- **Patches** (registered via `@register_patch(backend, *dotted_paths)`, installed via
  `apply_patches(backend)`) reversibly swap an attribute (a `torch` op, an ExecuTorch
  internal, a model class method) for the duration of the export. Pass multiple paths
  to a single decorator to share the same factory across targets — useful when the
  same method shape needs to be patched on several classes (e.g. `_update_mamba_mask`
  on Jamba/Bamba/…). Originals are restored on exit, even if the body raises.
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

The base exporter has one patch stage and four structural helpers. They run in this order
inside `DynamoExporter.export`, against the original `nn.Module`:

| #     | Stage                                                                | Section in [exporter_dynamo.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_dynamo.py) | What it does                                                                                                                                                                                                                                                                                                | How to extend                                                                                                                                                                                                                                                                |
| ----- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Forward-signature patch** (`patch_forward_signature`)              | `# ── Stage 1: Model signature patch ──`                                                                                             | Replaces `model.forward` with an explicit flat-arg signature derived from the inputs dict, so `torch.export` doesn't bundle `**kwargs` into a single tuple. This is the entry contract `torch.export` reads before tracing.                                                                                | Internal — no extension knob.                                                                                                                                                                                                                                                |
| **2** | **Model patches** (`_PATCHES["dynamo"]` via `apply_patches("dynamo")`) | `# ── Stage 2: Model patches ──`                                                                                                     | Reversible class-attribute swaps applied during tracing. Each `_patch_*(original) → replacement` factory targets one or more `Class.method` paths and replaces a non-exportable model pattern (data-dependent loops, in-place ops, mask checks, chunked-attention `split → zip → cat`) with an export-safe equivalent. | Define `_patch_*(original)` and decorate with `@register_patch("dynamo", *dotted_paths)`. Pass multiple paths to share the same factory across classes (e.g. `_update_mamba_mask` on Jamba/Bamba/…). Examples: mamba/linear-attn mask, NLLB classifier cast, chunked-vision attention. |
| **3** | **Pytree registration** (`register_cache_pytrees_for_model`)         | `# ── Stage 3: Pytree registration ──`                                                                                               | Registers flatten/unflatten via `torch.utils._pytree.register_pytree_node` for every captured `Cache` / `ModelOutput`. Reflection-driven, tuned for tensor containers (not a general serialiser).                                                                                                          | Usually automatic. If a type isn't reflectable, add a branch to `_flatten_to_context` / `_unflatten_from_context`.                                                                                                                                                            |
| **4** | **Dynamic shapes** (`get_auto_dynamic_shapes`)                       | `# ── Stage 4: Dynamic shapes ──`                                                                                                    | Auto-assigns `Dim.AUTO` to every tensor and cache leaf when `DynamoConfig.dynamic=True` and the user did not pass `dynamic_shapes` explicitly.                                                                                                                                                              | Override per-export via `DynamoConfig.dynamic_shapes`.                                                                                                                                                                                                                       |
| **5** | **State cleanup** (`reset_model_state` / `_STATEFUL_CACHE_ATTRS`)    | `# ── Stage 5: Model state cleanup ──`                                                                                               | Resets non-`Cache` tensor attributes set inside `forward` (e.g. glm_moe_dsa `_cached_keys`, wav2vec2_bert `cached_rotary_positional_embedding`) that `torch.export` leaves as FakeTensors, so a follow-up eager forward is safe.                                                                            | Append the attribute name to `_STATEFUL_CACHE_ATTRS`.                                                                                                                                                                                                                        |

### `OnnxExporter`

`OnnxExporter` extends `DynamoExporter` with five numbered stages applied around
`torch.onnx.export`. The labels match the `# ── Stage N: … ──` headers in the source:

| #     | Stage                                                       | Section in [exporter_onnx.py](https://github.com/huggingface/transformers/blob/main/src/transformers/exporters/exporter_onnx.py) | When it runs                                     | Lifecycle                            | What it does                                                                                                                                                                                                                                                                | How to extend                                                                                |
| ----- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **1** | **Torch patches** (`_PATCHES["onnx"]`)                      | `# ── Stage 1: Torch patches ──`                                                                                                 | During `torch.export` / `torch.onnx.export`      | Reversible (`apply_patches("onnx")`) | Reversible swaps of `torch` ops (`where`, `unsqueeze`, `scaled_dot_product_attention`, `searchsorted`, …) that the ONNX decomposer can't lower as-is. Each `_patch_*(original)` closes over the original.                                                                   | Define `_patch_*(original)` and decorate with `@register_patch("onnx", "dotted.path")`.       |
| **2** | **ONNX patches** (`_PATCHES["onnx"]`)                       | `# ── Stage 2: ONNX patches ──`                                                                                                  | During `torch.onnx.export`                       | Reversible (`apply_patches("onnx")`) | Hooks the private `_prepare_exported_program_for_export` step so the FX node fixes (stage 3) run again right after `run_decompositions` — any new symbolic-guard nodes the ONNX decomposition introduces get repaired before the FX → ONNX lowering picks them up.          | Same registry as stage 1 — define `_patch_*(original)` and decorate with `@register_patch("onnx", "dotted.path")`. |
| **3** | **FX node fixes** (`_FX_NODE_FIXES["onnx"]`)                | `# ── Stage 3: FX node fixes ──`                                                                                                 | After `torch.export`, again after `run_decompositions` | In-place (`apply_fx_node_fixes("onnx", gm)`) | Per-node rewrites on the `GraphModule` to drop or replace nodes the ONNX decomposer can't lower (alias ops, in-place views, `_assert_*`, dead comparisons, in-place `triu_`, `fill_diagonal_`, `sort(stable=True)`). DCE runs automatically at the end of the walk.   | Define `_fix_*(gm, node) → bool` (return `True` to consume) and decorate with `@register_fx_node_fix("onnx")`. |
| **4** | **ONNX translations** (`_ONNX_TRANSLATION_TABLE`)           | `# ── Stage 4: ONNX translations ──`                                                                                             | During FX → ONNX lowering                        | n/a (translation table)              | Overrides `torchlib`'s default lowering for specific aten ops where the default is buggy or missing. Currently `aten.index_put` (bool-mask path), `aten.bincount` (`OneHot + ReduceSum`), and `aten._grouped_mm` / `transformers.grouped_mm_fallback` (MoE grouped-matmul → unrolled `Slice + MatMul + Concat`). | Implement an `_aten_*` onnxscript function and add it to `_ONNX_TRANSLATION_TABLE`.          |
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

A second list, [`EXPORT_SKIP_MODEL_CLASSES`](https://github.com/huggingface/transformers/blob/main/tests/exporters/test_utils.py),
opts a handful of model classes out of the entire export sweep when the model itself is
fundamentally non-exportable as-is (data-dependent control flow that can't be vectorised,
modules treated as forward arguments, …). Same expectations: every entry carries a TODO
naming the underlying model change needed; the list should shrink, not grow.

</details>

## API reference

### Exporter classes

[[autodoc]] transformers.exporters.exporter_dynamo.DynamoExporter
    - export

[[autodoc]] transformers.exporters.exporter_onnx.OnnxExporter
    - export

[[autodoc]] transformers.exporters.exporter_executorch.ExecutorchExporter
    - export

### Configuration

[[autodoc]] transformers.exporters.configs.DynamoConfig

[[autodoc]] transformers.exporters.configs.OnnxConfig

[[autodoc]] transformers.exporters.configs.ExecutorchConfig

### Utilities

[[autodoc]] transformers.exporters.utils.get_leaf_tensors

[[autodoc]] transformers.exporters.utils.prepare_for_export

[[autodoc]] transformers.exporters.utils.decompose_prefill_decode

[[autodoc]] transformers.exporters.utils.decompose_multimodal

[[autodoc]] transformers.exporters.utils.decompose_for_generation

[[autodoc]] transformers.exporters.utils.is_multimodal
