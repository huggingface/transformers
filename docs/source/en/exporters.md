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

# Exporters

Export any [`PreTrainedModel`] to ONNX, ExecuTorch, or a standalone PyTorch program, regardless of the target runtime.

```python
exporter = DynamoExporter()  # or OnnxExporter, ExecutorchExporter
config = DynamoConfig(dynamic=True)
exported = exporter.export(model, inputs, config=config)
```

The exporters live inside Transformers instead of a downstream library, so architecture changes,
new attention patterns, and custom cache types are supported at export time as soon as they land
in the modeling code.

> [!WARNING]
> The exporters are experimental. Many of the patches in this module work around specific upstream bugs (Torch, ONNX Script, ONNX Runtime, ExecuTorch) and will be removed as soon as the fix lands upstream. Until the API stabilizes, treat the patches as tied to the versions used in the test suite. Pin those versions in production tooling, and expect new patches to appear and old ones to disappear as upstream changes land.

Every exporter returns an [`~exporters.ExportArtifacts`]; `artifact` is the backend's own program object.

| Exporter               | `artifact`                 | Runtime                                    |
| ---------------------- | -------------------------- | ------------------------------------------ |
| [`DynamoExporter`]     | `ExportedProgram`          | Any PyTorch runtime, AOT compilation       |
| [`OnnxExporter`]       | `ONNXProgram`              | Any ONNX runtime (ORT, TensorRT, OpenVINO) |
| [`ExecutorchExporter`] | `ExecutorchProgramManager` | Mobile and edge devices (ExecuTorch)       |

[`AutoHfExporter`] picks the right exporter from a config, and [`AutoExportConfig`] picks the
right config class from a dict. Both follow the same auto-class pattern in Transformers, which
is useful when the backend is selected at runtime instead of hardcoded at the call site.

```python
from transformers.exporters import AutoExportConfig, AutoHfExporter

export_config_dict = {"export_format": "onnx", "dynamic": True}
config = AutoExportConfig.from_dict(export_config_dict)
exporter = AutoHfExporter.from_config(config)

exported = exporter.export(model, inputs, config=config)
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

All exporters share the same interface. Create an exporter with a config, and call
[`~exporters.HfExporter.export`]. It returns an [`~exporters.ExportArtifacts`]: the exported graph, what
the trace recorded about it, and the configs it was traced with — everything needed to run it or save it.

Switch between runtimes by swapping the exporter class; nothing else in the flow changes.

<hfoptions id="exporters-quickstart">
<hfoption id="Dynamo">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import DynamoExporter, DynamoConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exported = DynamoExporter().export(model, inputs, config=DynamoConfig(dynamic=True))
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exported = OnnxExporter().export(model, inputs, config=OnnxConfig(dynamic=True))
```

</hfoption>
<hfoption id="ExecuTorch">

[`~exporters.ExecutorchConfig#backend`] defaults to `xnnpack` which targets the CPU and works on CPU-only installations. `cuda` targets the GPU and requires a CUDA-enabled environment. Requesting it without CUDA raises a `RuntimeError`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exported = ExecutorchExporter().export(model, inputs, config=ExecutorchConfig(backend="xnnpack", dynamic=True))
```

</hfoption>
</hfoptions>

### Run it

`runner()` binds the graph to the runtime that runs it — an unlifted `torch.export` module, an ONNX Runtime
session, a loaded `.pte` — and returns its outputs as named tensors, whichever backend produced it.

```python
outputs = exported.runner()(**inputs)
logits = outputs["logits"]
```

`runtime()` goes one level up and gives something that behaves like the model: an
[`~exporters.ExportedModel`] for a single graph, an [`~exporters.ExportedGenerator`] for an export that
`generate` drives (see [Export for generation](#export-for-generation)).

```python
outputs = exported.runtime()(**inputs)   # -> ModelOutput, so outputs.logits works
```

To reach the backend's own program object — for tooling that speaks ONNX or ExecuTorch directly — use
`artifact`:

```python
exported.artifact                        # ONNXProgram / ExportedProgram / ExecutorchProgramManager
```

### Save and load it

`save_pretrained` writes the graph, the configs, and a manifest describing both. The manifest is what makes
the directory loadable: it records the backend, which file each component is, and what the trace recorded
about each graph — the precision it computes in, the cache it was traced against, the shapes it saw. A
runner without that would have to infer them from tensor names and shapes, and get them wrong quietly.

```python
exported.save_pretrained("qwen3-export")
```

```
qwen3-export/
├── config.json
├── export.json          # backend, components, and per-graph metadata
└── model.onnx           # or model.pt2 / model.pte
```

[`AutoExportedModel`] loads it back as whatever it was exported as, no need to remember which backend
wrote it:

```python
from transformers.exporters import AutoExportedModel

exported_model = AutoExportedModel.from_pretrained("qwen3-export")
outputs = exported_model(**inputs)
```

> [!TIP]
> Loading refuses a directory whose manifest has no recorded metadata rather than falling back to
> inference, because a runner that guesses the precision or the cache layout still runs — and produces
> quietly wrong numbers. Re-save with `save_pretrained` if you hit this.

## Dynamic shapes

Passing `dynamic=True` marks every tensor
dimension as dynamic so the exported graph accepts inputs of any size at runtime without
retracing.

For fine-grained control over which dimensions are dynamic, pass explicit `dynamic_shapes`
instead, which is forwarded directly to [torch.export.export](https://pytorch.org/docs/stable/export.html).

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
exported = exporter.export(model, inputs, config=config)
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
exported = exporter.export(model, inputs, config=config)
```

</hfoption>
</hfoptions>

## Generative models

For autoregressive generation, the model's `forward` has different shapes at the prefill step
(full prompt, no KV cache) versus the decode step (single token, populated KV cache). Exporters
expose [`~HfExporter.export_for_generation`], which splits both stages and exports each.

For multi-modal generative models, the prefill additionally splits into an image or audio
encoder, the language model, and `lm_head`. Encoder and language-model discovery uses
[`~PreTrainedModel.get_encoder`] (`modality="image"` or `"audio"`) and
[`~PreTrainedModel.get_decoder`] accessors, so any new architecture using these
work out of the box.

A projector component appears only when the model exposes one
under an attribute name (`multi_modal_projector`, `connector`, `embed_vision`,
`embed_audio`). Qwen2-VL below folds its projector into the vision tower, so its component dict
has no separate `multi_modal_projector` key. New architectures must align their projector
attribute to one of these names instead of growing the list.

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

### How `export_for_generation` works

[`~exporters.utils.decompose_for_generation`] runs `model.generate(**inputs, max_new_tokens=2)`
once and hooks `model.forward` to capture the real prefill and decode kwargs (and the
per-submodule kwargs via hooks on each encoder/projector/language model if the model is
multi-modal). That's why it works for any architecture, including decoder-only, SSM,
encoder-decoder, and multi-modal models, without per-model glue. `export_for_generation` is a
one-liner over it.

The capture runs the model eagerly on `inputs`, so pass small but representative values, such as a
short prompt, a single small image, or a few audio frames. The exported program isn't tied to
those sizes (dynamic shapes still flow through), but smaller capture inputs make
`decompose_for_generation` cheaper and keep symbolic-shape inference tractable.

Call `decompose_for_generation` directly to act between decomposing and exporting, such as
running an eager forward for verification, swapping a submodule's inputs, or skipping a stage.

```python
from transformers.exporters.utils import decompose_for_generation

components = decompose_for_generation(model, inputs)
# {"image_encoder": (submodel, fwd_kwargs), "language_model": (...), ..., "decode": (...)}

artifacts, metadata = {}, {}
for name, (submodel, subinputs) in components.items():
    eager_outputs = submodel(**subinputs)  # sanity-check the eager forward before exporting
    artifacts[name], metadata[name] = exporter.export_artifact(submodel, subinputs, config=config)
```

`export_for_generation` is this loop plus the [`~exporters.ExportArtifacts`] it wraps the results in.

### Multi-token decode

By default the `decode` component is a **single-token** step — one query token against the KV cache —
so `torch.export` specializes its query-sequence axis to 1. Pass `multi_token_decode=True` to capture
`decode` as a **multi-token** decode instead: [`~exporters.utils.decompose_for_generation`] merges two
consecutive decode steps (it captures with `max_new_tokens=3`) into one forward, so that axis stays
symbolic. A single graph then serves every query length — one token (ordinary decoding), many tokens
at once (continuation-from-past, e.g. accepting a chunk of speculative tokens), and a plain prefill
when the cache is empty.

<hfoptions id="multi-token-decode">
<hfoption id="Dynamo">

```python
from transformers.exporters import DynamoExporter, DynamoConfig

exporter = DynamoExporter()
config = DynamoConfig(dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config, multi_token_decode=True)
# components["decode"] now accepts a variable number of query tokens
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers.exporters import OnnxExporter, OnnxConfig

exporter = OnnxExporter()
config = OnnxConfig(dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config, multi_token_decode=True)
# components["decode"] now accepts a variable number of query tokens
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

exporter = ExecutorchExporter()
config = ExecutorchConfig(backend="xnnpack", dynamic=True)
components = exporter.export_for_generation(model, inputs, config=config, multi_token_decode=True)
# components["decode"] now accepts a variable number of query tokens
```

</hfoption>
</hfoptions>

The query axis only stays symbolic under a dynamic-shape export (`dynamic=True`); a static export
freezes it at the captured length, giving a fixed multi-token graph. It composes with the static KV
cache below — the merged decode writes each step's tokens into the fixed-size cache in place, and the
cache handles where they land internally.

### Static KV cache

`generate()` grows a `DynamicCache` by default, reallocating as the sequence extends — a moving target
for an exported graph. A **static** cache is a fixed-size buffer, allocated once and written in place at
the current position each step. Combined with a [multi-token decode](#multi-token-decode) it collapses
generation into a single exported graph: the `decode` graph takes a fixed-size cache and a *variable*
number of query tokens, so one graph serves both the prompt (empty cache → prefill) and each generated
token (populated cache → decode). Export it by forwarding a `GenerationConfig` with
`cache_implementation="static"` (and a `max_cache_len`) alongside `multi_token_decode=True`:

<hfoptions id="static-cache">
<hfoption id="Dynamo">

```python
from transformers import GenerationConfig
from transformers.exporters import DynamoExporter, DynamoConfig

exporter = DynamoExporter()
gen_config = GenerationConfig(cache_implementation="static", max_cache_len=2048)
components = exporter.export_for_generation(
    model, inputs, config=DynamoConfig(dynamic=True), generation_config=gen_config, multi_token_decode=True
)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import GenerationConfig
from transformers.exporters import OnnxExporter, OnnxConfig

exporter = OnnxExporter()
gen_config = GenerationConfig(cache_implementation="static", max_cache_len=2048)
components = exporter.export_for_generation(
    model, inputs, config=OnnxConfig(dynamic=True), generation_config=gen_config, multi_token_decode=True
)
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers import GenerationConfig
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

exporter = ExecutorchExporter()
gen_config = GenerationConfig(cache_implementation="static", max_cache_len=2048)
components = exporter.export_for_generation(
    model, inputs, config=ExecutorchConfig(backend="xnnpack", dynamic=True), generation_config=gen_config, multi_token_decode=True
)
```

</hfoption>
</hfoptions>

The `decode` graph now has two symbolic axes — the query length (how many tokens you feed) and the cache
length (`max_cache_len`, resizable at load time). `dynamic=True` marks these (and every other axis)
`Dim.AUTO`, so the exported graph accepts any prompt length and cache size at load time.

#### Zero-copy in-place updates

The static cache is passed in and mutated in place, so one buffer carries state across decode steps with no
host copies — as long as the runtime binds the caller's buffers rather than copying through its own arena.
[`~exporters.ExportedGenerator`] and the runners under it do this for you: `torch.export` records the cache
write as a `USER_INPUT_MUTATION` so the tensors passed in are updated directly, and
[`OnnxModelRunner`] binds each matched `input.<name>` / `output.<name>` pair to one device buffer, so the
cache is read and updated in place across the loop with no per-step allocation.

ExecuTorch needs one thing from you: turn off the memory-planning allocations on [`ExecutorchConfig`] so the
in-place write can land in the caller's own tensor (see the reference for what each flag does):

  ```python
  config = ExecutorchConfig(
      backend="xnnpack",
      dynamic=True,
      alloc_graph_input=False,
      alloc_graph_output=False,
      alloc_mutable_buffers=False,
  )
  ```

  > [!NOTE]
  > The zero-copy in-place write also needs the caller to bind output buffers at runtime via
  > `Method::set_output_data_ptr` — **not surfaced by the Python runtime** (`executorch.runtime.Method`
  > exposes only `execute`/`set_inputs`/`get_outputs`). The flags above set it up, but the in-place
  > write is a **C++-only** path. From Python, read the updated cache back from the method outputs each
  > step.

### Generate from an export

The components an export produces are not much use one at a time: generation needs a loop that grows a
cache, advances positions, rebuilds the mask each step, and — on ONNX Runtime — binds the cache in and out
of one device buffer so nothing is reallocated per token. [`~exporters.ExportedGenerator`] is that loop. It
takes the exported components and drives them through the ordinary `generate` API.

```python
from transformers import GenerationConfig
from transformers.exporters import OnnxExporter, OnnxConfig

gen_config = GenerationConfig(cache_implementation="static", max_cache_len=2048)
exported = OnnxExporter().export_for_generation(
    model, inputs, config=OnnxConfig(dynamic=True), generation_config=gen_config, multi_token_decode=True
)
```

There are two ways to run it, and neither needs per-backend code — the same calls drive `torch.export`,
ONNX Runtime and ExecuTorch.

Straight from the export, without touching disk:

```python
runtime = exported.runtime()
ids = runtime.generate(**inputs, max_new_tokens=32)
```

Or save it and load it back, which is the deployment path:

```python
from transformers.exporters import AutoExportedModel

exported.save_pretrained("qwen3-generate")

runtime = AutoExportedModel.from_pretrained("qwen3-generate")   # a local directory or a Hub repo
ids = runtime.generate(**inputs, max_new_tokens=32)
```

The `generation_config` travels with the artifacts, which matters here: it declares the cache the graphs
were traced against, so a load that guessed a different one would build the wrong cache.

This covers decoder-only text, VLMs (including the multi-axis M-RoPE position ids, which the runtime
builds by running the model class's own `get_rope_index` on the saved config, with no weights loaded),
and encoder-decoder models.

<details>

<summary>Driving the steps yourself</summary>

`runners()` gives the graphs bound to their runtimes, keyed by component, for a loop you write yourself —
custom serving, speculative decoding, anything `generate` does not cover. Each runner takes and returns
named tensors whatever the backend produced it.

```python
runners = exported.runners()
outputs = runners["decode"](input_ids=..., attention_mask=..., position_ids=..., past_key_values=...)
logits = outputs["logits"]
```

The cache is whatever the graphs were traced against — a `StaticCache` for `torch.export`, device buffers
for ONNX Runtime, caller arrays in C++ for ExecuTorch.

> [!NOTE]
> ExecuTorch's zero-copy in-place cache write needs `Method::set_output_data_ptr`, which its Python runtime
> does not expose (`executorch.runtime.Method` offers only `execute`/`set_inputs`/`get_outputs`), so from
> Python read the updated cache back from the method outputs each step. The in-place path is C++-only.

</details>

## Limitations and workarounds

`torch.export`, `torch.onnx.export`, and ExecuTorch each have rough edges around specific
PyTorch patterns. The exporters work around these with a small set of reversible patches
and FX-level fixes applied at well-defined points in the export flow. None of this is
visible from the public `export` API, but the most common things to know:

- FlashAttention and FlexAttention are not exportable on any backend. `sdpa` is the preferred
setting and `eager` also works (slower). Set one of them on the model before calling `export`
if it's using something else.
- `grouped_mm` traces fine through `DynamoExporter` and is auto-translated for `OnnxExporter`.
For `ExecutorchExporter` with the XNNPACK backend, the exporter swaps MoE experts to
`batched_mm` because XNNPACK has no `_grouped_mm.out` kernel.

## Next steps

- Add export support for a new architecture or backend with the patch and fix registries in
[Extending the exporters](./exporters_extend).
