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

Export any [`PreTrainedModel`] to ONNX, ExecuTorch, OpenVINO, or a standalone PyTorch program, regardless of the target runtime.

```python
exporter = DynamoExporter()  # or OnnxExporter, ExecutorchExporter, OpenVINOExporter
config = DynamoConfig(dynamic=True)
exported_artifacts = exporter.export(model, inputs, config=config)
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
| [`OpenVINOExporter`]   | `openvino.Model`           | OpenVINO runtime (Intel CPU/GPU/NPU)       |

[`AutoHfExporter`] picks the right exporter from a config, and [`AutoExportConfig`] picks the
right config class from a dict. Both follow the same auto-class pattern in Transformers, which
is useful when the backend is selected at runtime instead of hardcoded at the call site.

```python
from transformers.exporters import AutoExportConfig, AutoHfExporter

export_config_dict = {"export_format": "onnx", "dynamic": True}
config = AutoExportConfig.from_dict(export_config_dict)
exporter = AutoHfExporter.from_config(config)

exported_artifacts = exporter.export(model, inputs, config=config)
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
<hfoption id="ExecuTorch XNNPACK">

```bash
pip install transformers "torch==2.12.0" "executorch==1.3.1"
```

</hfoption>
<hfoption id="ExecuTorch MLX">

Requires macOS 14 or later on Apple Silicon. MLX is included in the macOS ARM64 ExecuTorch
nightly wheels; it is not a separate package or extra. The following nightly pair was validated
with the MLX exporter tests. Install it from the standard ExecuTorch nightly registry:

```bash
pip install transformers
pip install \
  "executorch==1.6.0.dev20260924" \
  "torch==2.15.0.dev20260924" \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Install `torch` explicitly because nightly ExecuTorch wheels do not declare it as a dependency.
For source builds, see the [MLX installation guide](https://docs.pytorch.org/executorch/main/backends/mlx/mlx-overview.html).

</hfoption>
<hfoption id="OpenVINO">

```bash
pip install transformers "torch==2.12.0" "openvino==2026.3.1"
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

exported_artifacts = DynamoExporter().export(model, inputs, config=DynamoConfig(dynamic=True))
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exported_artifacts = OnnxExporter().export(model, inputs, config=OnnxConfig(dynamic=True))
```

</hfoption>
<hfoption id="ExecuTorch">

[`~exporters.ExecutorchConfig#backend`] selects the target: `xnnpack` (the default) for CPU,
`mlx` for Apple Silicon GPU, or `cuda` for a CUDA-enabled GPU. Choose the corresponding
[installation option](#installation). MLX execution requires the MLX delegate and its Metal libraries;
requesting CUDA without a CUDA-enabled environment raises a `RuntimeError`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exported_artifacts = ExecutorchExporter().export(
    model, inputs, config=ExecutorchConfig(backend="xnnpack", dynamic=True)  # Use "mlx" on Apple Silicon.
)
```

</hfoption>
</hfoptions>

### Run it

`runtime()` gives something that behaves like the model, whichever backend produced the graph: an
[`~exporters.ExportedModel`] for a single graph, an [`~exporters.ExportedGenerator`] for an export that
`generate` drives (see [Generative models](#generative-models)).

```python
outputs = exported_artifacts.runtime()(**inputs)   # -> ModelOutput, so outputs.logits works
```

To reach the backend's own program object — for tooling that speaks ONNX or ExecuTorch directly — use
`artifact`:

```python
exported_artifacts.artifact                        # ONNXProgram / ExportedProgram / ExecutorchProgramManager / openvino.Model
```

### Save and load it

`save_pretrained` writes the graph, the configs, and a manifest describing both. The manifest is what makes
the directory loadable: it records the backend, which file each component is, and what the trace recorded
about each graph — the precision it computes in, the cache it was traced against, the shapes it saw. A
runner without that would have to infer them from tensor names and shapes, and get them wrong quietly.

```python
exported_artifacts.save_pretrained("qwen3-export")
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

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(["Hello, world!", "Hi"], padding=True, return_tensors="pt")

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

config = OnnxConfig(
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    # Emit data-dependent shape guards as runtime asserts instead of failing the export when a guard
    # wouldn't hold across the explicit range. Not needed with `dynamic=True`, where torch.export infers
    # shape relations instead of verifying them against user-stated bounds.
    prefer_deferred_runtime_asserts_over_guards=True,
)
exported_artifacts = OnnxExporter().export(model, inputs, config=config)
```

Every config accepts the same `dynamic_shapes` and `prefer_deferred_runtime_asserts_over_guards`, so the
other exporters take this unchanged.

## Generative models

For autoregressive generation, the model's `forward` has different shapes at the prefill step
(full prompt, no KV cache) versus the decode step (single token, populated KV cache). Exporters
expose [`~HfExporter.export_for_generation`], which splits both stages and exports each.

For multi-modal generative models the prompt splits further: one graph per modality from the model's own
`get_<modality>_features` method (`image_encoder`, `audio_encoder` — the tower *and* its projector, since
that method runs both), and `embed_tokens` for `input_ids -> inputs_embeds`. The text stack stays whole as
`decode`, taking `inputs_embeds`, so the runtime scatters each modality's features into the embeddings
before running it. Any architecture exposing those methods works without further wiring.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.exporters import OnnxExporter, OnnxConfig

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=messages[0]["content"][0]["url"], return_tensors="pt").to(model.device)

exported_artifacts = OnnxExporter().export_for_generation(model, inputs, config=OnnxConfig(dynamic=True))
# components: "image_encoder", "embed_tokens", "decode"; reach a backend program through
# `exported_artifacts["decode"].artifact`, or run them all with `exported_artifacts.runtime().generate(...)`
```

Swap in any other exporter and its config; the components are the same.

### How `export_for_generation` works

[`~exporters.decompose.decompose_for_generation`] runs `model.generate(**inputs, max_new_tokens=2)`
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
from transformers.exporters.decompose import decompose_for_generation

components = decompose_for_generation(model, inputs)
# {"image_encoder": Component, "embed_tokens": Component, "decode": Component}

artifacts, metadata = {}, {}
for name, component in components.items():
    eager_outputs = component.module(**component.inputs)  # sanity-check the eager forward before exporting
    artifacts[name], metadata[name] = exporter.export_artifact(component.module, component.inputs, config=config)
```

`export_for_generation` is this loop plus the [`~exporters.ExportArtifacts`] it wraps the results in.

### Multi-token decode

A dynamic export (`dynamic=True`) captures `decode` as a **multi-token** decode:
[`~exporters.decompose.decompose_for_generation`] merges two consecutive decode steps (it captures with
`max_new_tokens=3`) into one forward, so the query-sequence axis stays symbolic. A single graph then serves
every query length — one token (ordinary decoding), many tokens at once (continuation-from-past, e.g.
accepting a chunk of speculative tokens), and a plain prefill when the cache is empty — so the export ships
one text stack instead of two. A prompt graph is kept only where the decode graph provably cannot stand in.
Pass `multi_token_decode=False` to keep a single-token `decode` beside a separate `prefill` graph anyway — for
a runtime that wants a fixed one-token decode shape.

A static export cannot keep the query axis symbolic, so it captures a **single-token** `decode` step
instead, with a separate `prefill` graph for the prompt (asking it for `multi_token_decode=True` is refused). The multi-token decode composes with the static KV
cache below — the merged decode writes each step's tokens into the fixed-size cache in place, and the cache
handles where they land internally.

### Static KV cache

> [!NOTE]
> The ExecuTorch examples in this section, including zero-copy updates and the decode loop,
> are **XNNPACK-only**. The MLX exporter rejects `StaticCache`; use `DynamicCache` with MLX.

`generate()` grows a `DynamicCache` by default, reallocating as the sequence extends — a moving target
for an exported graph. A **static** cache is a fixed-size buffer, allocated once and written in place at
the current position each step. Combined with a [multi-token decode](#multi-token-decode) it collapses
generation into a single exported graph: the `decode` graph takes a fixed-size cache and a *variable*
number of query tokens, so one graph serves both the prompt (empty cache → prefill) and each generated
token (populated cache → decode). Export it by forwarding a `GenerationConfig` with
`cache_implementation="static"` (and a `max_cache_len`) with a dynamic export:

```python
from transformers import GenerationConfig
from transformers.exporters import OnnxExporter, OnnxConfig

gen_config = GenerationConfig(cache_implementation="static", max_cache_len=2048)
exported_artifacts = OnnxExporter().export_for_generation(
    model, inputs, config=OnnxConfig(dynamic=True), generation_config=gen_config
)
```

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

ExecuTorch (XNNPACK) needs one thing from you: turn off the memory-planning allocations on [`ExecutorchConfig`] so the
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

[`ExecutorchModelRunner`] then binds each cache output to the cache tensor it updates. That needs an ExecuTorch
runtime whose `Method` has `set_output`; on an older one the runner reads the updated cache back from the
outputs each step instead.

### Generate from an export

The components an export produces are not much use one at a time: generation needs a loop that grows a
cache, advances positions, rebuilds the mask each step, and — on ONNX Runtime — binds the cache in and out
of one device buffer so nothing is reallocated per token. [`~exporters.ExportedGenerator`] is that loop. It
takes the exported components and drives them through the ordinary `generate` API.

```python
from transformers import GenerationConfig
from transformers.exporters import OnnxExporter, OnnxConfig

gen_config = GenerationConfig(cache_implementation="static", max_cache_len=2048)
exported_artifacts = OnnxExporter().export_for_generation(
    model, inputs, config=OnnxConfig(dynamic=True), generation_config=gen_config
)
```

There are two ways to run it, and neither needs per-backend code — the same calls drive `torch.export`,
ONNX Runtime and ExecuTorch.

Straight from the export, without touching disk:

```python
runtime = exported_artifacts.runtime()
ids = runtime.generate(**inputs, max_new_tokens=32)
```

Or save it and load it back, which is the deployment path:

```python
from transformers.exporters import AutoExportedModel

exported_artifacts.save_pretrained("qwen3-generate")

runtime = AutoExportedModel.from_pretrained("qwen3-generate")   # a local directory or a Hub repo
ids = runtime.generate(**inputs, max_new_tokens=32)
```

The `generation_config` travels with the artifacts, which matters here: it declares the cache the graphs
were traced against, so a load that guessed a different one would build the wrong cache.

This covers decoder-only text, VLMs (including the multi-axis M-RoPE position ids, which the runtime
builds by running the model class's own `get_rope_index` on the saved config, with no weights loaded),
and encoder-decoder models.

## Limitations and workarounds

`torch.export`, `torch.onnx.export`, and ExecuTorch each have rough edges around specific
PyTorch patterns. The exporters work around these with a small set of reversible patches
and FX-level fixes applied at well-defined points in the export flow. None of this is
visible from the public `export` API, but the most common things to know:

- FlashAttention and FlexAttention are not exportable on any backend. `sdpa` is the preferred
setting and `eager` also works (slower). Set one of them on the model before calling `export`
if it's using something else.
- `grouped_mm` traces fine through `DynamoExporter` and is auto-translated for `OnnxExporter`.
For `ExecutorchExporter` with either XNNPACK or MLX, the exporter swaps MoE experts to
`batched_mm` before export.

## Next steps

- Add export support for a new architecture or backend with the patch and fix registries in
[Extending the exporters](./exporters_extend).
