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

Export any [`PreTrainedModel`] to ONNX, ExecuTorch, or a standalone PyTorch program with the
same two lines of code, regardless of the target runtime.

```python
exporter = DynamoExporter()  # or OnnxExporter, ExecutorchExporter
config = DynamoConfig(dynamic=True)
exported = exporter.export(model, inputs, config=config)
```

The exporters live inside Transformers instead of a downstream library, so architecture changes,
new attention patterns, and custom cache types are supported at export time as soon as they land
in the modeling code.

> [!WARNING]
> The exporters are experimental. Many of the patches in this module work around specific
> upstream bugs (Torch, ONNX Script, ONNX Runtime, ExecuTorch) and will be removed as soon as the
> fix lands upstream. Until the API stabilizes, treat the patches as tied to the versions used in
> the test suite. Pin those versions in production tooling, and expect new patches to appear and
> old ones to disappear as upstream changes land.

| Exporter               | Output                     | Runtime                                       |
| ---------------------- | -------------------------- | --------------------------------------------- |
| [`DynamoExporter`]     | `ExportedProgram`          | Any PyTorch runtime, AOT compilation          |
| [`OnnxExporter`]       | `ONNXProgram`              | Any ONNX runtime (ORT, TensorRT, OpenVINO) |
| [`ExecutorchExporter`] | `ExecutorchProgramManager` | Mobile and edge devices (ExecuTorch)          |

[`AutoHfExporter`] picks the right exporter from a config, and [`AutoExportConfig`] picks the
right config class from a dict. Both follow the same auto-class pattern in Transformers, which
is useful when the backend is selected at runtime instead of hardcoded at the call site.

```python
from transformers.exporters import AutoExportConfig, AutoHfExporter

export_config_dict = {"export_format": "onnx", "dynamic": True}
config = AutoExportConfig.from_dict(export_config_dict)
exporter = AutoHfExporter.from_config(config)

onnx_program = exporter.export(model, inputs, config=config)
```

## Installation

Install the dependencies for the backend you plan to export to.

> [!TIP]
> The versions below are the ones the exporter test suite is pinned against. Newer or older
> releases often work, but the exporter patches target a specific API surface, so for production
> tooling pin these and expect [`HfExporter`] to log a warning when it detects drift.

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
</hfoptions>

## Export a model

All exporters share the same interface. Create an exporter with a config, and call [`~exporters.HfExporter.export`].
Switch between runtimes by swapping the exporter class.

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
</hfoptions>

## Dynamic shapes

The examples above pass `dynamic=True`, which marks every tensor
dimension as dynamic so the exported graph accepts inputs of any size at runtime without
retracing.

For fine-grained control over which dimensions are dynamic, pass explicit `dynamic_shapes`
instead. This is forwarded directly to [torch.export.export](https://pytorch.org/docs/stable/export.html).

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
    # guard wouldn't hold across the explicit symbolic range. Most LLMs need this under fine-grained
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
    # guard wouldn't hold across the explicit symbolic range. Most LLMs need this under fine-grained
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
    # guard wouldn't hold across the explicit symbolic range. Most LLMs need this under fine-grained
    # ``Dim(min=, max=)`` bounds. Not needed with ``dynamic=True`` / ``Dim.AUTO``, where torch.export
    # infers shape relations instead of verifying them against user-stated bounds.
    prefer_deferred_runtime_asserts_over_guards=True,
)
et_program = exporter.export(model, inputs, config=config)
```

</hfoption>
</hfoptions>

## Generative models

For autoregressive generation, the model's `forward` has different shapes at the prefill step
(full prompt, no KV cache) versus the decode step (single token, populated KV cache). Exporters
expose [`~HfExporter.export_for_generation`], which splits both stages and exports each.

For multi-modal generative models, the prefill additionally splits into an image or audio
encoder, the language model, and `lm_head`. Encoder and language-model discovery uses the
canonical [`~PreTrainedModel.get_encoder`] (`modality="image"` / `"audio"`) and
[`~PreTrainedModel.get_decoder`] accessors, so any new architecture using these
works out of the box.

A projector component appears only when the model exposes one
under a canonical attribute name (`multi_modal_projector`, `connector`, `embed_vision`,
`embed_audio`). Qwen2-VL below folds its projector into the vision tower, so its component dict
has no separate `multi_modal_projector` key. New architectures must align their projector
attribute to one of these canonical names instead of growing the list.

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
</hfoptions>

> [!WARNING]
> The exported components are independent graphs, not a ready-to-run inference pipeline. The
> caller is responsible for running each encoder, projecting embeddings, and orchestrating the
> generation loop.

<details>
<summary>What <code>export_for_generation</code> does under the hood</summary>

[`~exporters.utils.decompose_for_generation`] runs `model.generate(**inputs, max_new_tokens=2)`
once and hooks `model.forward` to capture the real prefill and decode kwargs (and the
per-submodule kwargs via hooks on each encoder / projector / language model if the model is
multi-modal). That's why it works for any architecture, including decoder-only, SSM,
encoder-decoder, and multi-modal models, without per-model glue. `export_for_generation` is a
one-liner over it.

The capture runs the model eagerly on `inputs`, so pass small but representative values like a
short prompt, a single small image, a few audio frames. The exported program isn't tied to
those sizes (dynamic shapes still flow through), but smaller capture inputs make
`decompose_for_generation` cheaper and keep symbolic-shape inference tractable.

Call `decompose_for_generation` directly when you want to do something between decomposing and
exporting like running an eager forward for verification, swapping a submodule's inputs, or skipping a stage.

```python
from transformers.exporters.utils import decompose_for_generation

components = decompose_for_generation(model, inputs)
# {"image_encoder": (submodel, fwd_kwargs), "language_model": (...), ..., "decode": (...)}

exported = {}
for name, (submodel, subinputs) in components.items():
    eager_outputs = submodel(**subinputs)  # sanity-check the eager forward before exporting
    exported[name] = exporter.export(submodel, subinputs, config=config)
```

</details>

## Limitations and workarounds

`torch.export`, `torch.onnx.export`, and ExecuTorch each have rough edges around specific
PyTorch patterns. The exporters work around these with a small set of reversible patches
and FX-level fixes applied at well-defined points in the export flow. None of this is
visible from the public `export` API, but the most common things to know:

- Flash-attention and flex-attention are not exportable on any backend; `sdpa` is the preferred
  setting and `eager` also works (slower). Set one of them on the model before calling `export()`
  if it's using something else.
- `grouped_mm` traces fine through `DynamoExporter` and is auto-translated for `OnnxExporter`;
  for `ExecutorchExporter` with the XNNPACK backend, the exporter swaps MoE experts to
  `batched_mm` because XNNPACK has no `_grouped_mm.out` kernel.
- A short list of models (`EXPORT_SKIP_MODEL_CLASSES`) is skipped from the export sweep when
  the model itself is fundamentally non-exportable; each entry carries a TODO with the
  model-side change needed.

## Next steps

- Add export support for a new architecture or backend with the patch and fix registries in
  [Extending the exporters](./exporters_extend).
