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

| Exporter               | Output                     | Runtime                                    |
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

[`~exporters.ExecutorchConfig#backend`] defaults to `xnnpack` which targets the CPU and works on CPU-only installations. `cuda` targets the GPU and requires a CUDA-enabled environment. Requesting it without CUDA raises a `RuntimeError`.

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

exported = {}
for name, (submodel, subinputs) in components.items():
    eager_outputs = submodel(**subinputs)  # sanity-check the eager forward before exporting
    exported[name] = exporter.export(submodel, subinputs, config=config)
```

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

The static cache is passed in and mutated in place, so one buffer carries state across decode steps
with no host copies — as long as the runtime binds the caller's buffers rather than copying through its
own arena. What that takes is the only per-backend part left:

- **Dynamo** — the exported program models the cache write as a `USER_INPUT_MUTATION`, so calling
  `components["decode"].module()(...)` updates the cache tensors you pass in directly. Reuse the same
  tensors each step; nothing to configure.

- **ONNX Runtime** — the decode graph exposes the cache as matched `input.<name>` / `output.<name>`
  pairs. ORT's `CudaSession.set_buffer_sharing` (`onnxruntime.transformers.io_binding_helper`) binds
  each pair to one device buffer, so the cache is read and updated in place across the loop with no host
  round-trips.

- **ExecuTorch** — turn off the memory-planning allocations on [`ExecutorchConfig`] so the in-place
  write can land in the caller's own tensor (see the reference for what each flag does):

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
  > write is a **C++-only** path (see the ExecuTorch decode-loop example below). From Python, read the
  > updated cache back from the method outputs each step.

<details>

<summary>Decode-loop inference examples</summary>

The loop is the same shape on every backend — it's the *same* graph throughout. Start from an empty
fixed-size cache, feed the whole prompt once (empty cache → prefill), then one token at a time
(populated cache → decode). Each call passes `input_ids`, a causal `attention_mask`, and `position_ids`
(advanced by the number of new tokens each step), plus the cache, and gets back logits for every query
position. Where each token lands in the cache is tracked internally by the static cache, so there's
nothing extra to thread through the call. How the cache is set up differs per runtime (a `StaticCache`
object for Dynamo, raw device buffers for ONNX Runtime, caller arrays in C++ for ExecuTorch), so each
tab builds its own below. The Dynamo and ONNX Runtime tabs update the cache in place; ExecuTorch's
in-place path is C++ (its Python runtime can't, as noted above).

<hfoptions id="decode-loop">
<hfoption id="Dynamo">

torch.export records the static-cache write as a `USER_INPUT_MUTATION`, so the loaded graph's `module()`
updates the `StaticCache` you pass in **directly** — one cache carries state across the whole loop with
nothing to bind or thread back out. `register_pytree_node(StaticCache)` lets `torch.export.load` unflatten
the `StaticCache` input. The cache has to be **initialized up front** (torch.export bakes the allocated K/V
into the input spec, so a lazy blank cache won't match) — but the saved program carries its own
`example_inputs`, so reuse that already-initialized `StaticCache` template, reset to empty:

```python
import copy
import torch
from transformers import StaticCache
from transformers.exporters.exporter_dynamo import register_pytree_node

register_pytree_node(StaticCache)
exported = torch.export.load("decode.pt2")
decode = exported.module()   # runs on the device its inputs / cache live on (CUDA here)

# the artifact carries an initialized StaticCache template — reuse it (reset to empty)
_, example_kwargs = exported.example_inputs
past_key_values = copy.deepcopy(example_kwargs["past_key_values"])
past_key_values.reset()

def causal_mask(positions, cache_len):   # [1, 1, len(positions), cache_len]
    return (torch.arange(cache_len, device="cuda")[None, :] <= positions[:, None])[None, None]

# prefill: the whole prompt in one call
positions = torch.arange(prompt_len, device="cuda")
logits = decode(input_ids=prompt_ids, attention_mask=causal_mask(positions, max_cache_len),
                position_ids=positions[None], past_key_values=past_key_values).logits
next_token = logits[:, -1:].argmax(-1)

# decode: query=1 buffers reused in place
input_ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
position_ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
attention_mask = torch.empty((1, 1, 1, max_cache_len), dtype=torch.bool, device="cuda")
slots = torch.arange(max_cache_len, device="cuda")
for position in range(prompt_len, max_cache_len):
    input_ids.copy_(next_token)
    position_ids.fill_(position)
    attention_mask[0, 0, 0].copy_(slots <= position)
    logits = decode(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values).logits
    next_token = logits[:, -1:].argmax(-1)
```

</hfoption>
<hfoption id="ONNX Runtime">

ONNX Runtime runs the graph as-is; the in-place cache update is done with ORT's `CudaSession`
(`onnxruntime.transformers.io_binding_helper`), a thin wrapper over ORT io-binding. `set_buffer_sharing`
binds a cache `input.<name>` and its matching `output.<name>` to **one** device buffer, so the mutated
K/V/counter are written straight back into the input; `allocate_buffers` allocates the remaining
(non-shared) outputs — here just `logits`; and `infer(feed_dict)` binds your CUDA tensors by pointer
and runs. The cache buffers come straight from the graph's own input metadata (`get_inputs()` shape and
type), so no model config is needed — the one symbolic axis (cache length) becomes `max_cache_len`:

```python
import torch
import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import CudaSession, TypeHelper

def causal_mask(positions, cache_len):
    return (torch.arange(cache_len, device="cuda")[None, :] <= positions[:, None])[None, None]

session = ort.InferenceSession("decode.onnx", providers=["CUDAExecutionProvider"])
cuda = CudaSession(session, torch.device("cuda"))

# fresh device cache buffers built from each cache input's own shape/dtype; share each
# input.<name>/output.<name> pair on one buffer so the update lands in place
cache = {}
for i in session.get_inputs():
    if not i.name.startswith("input."):
        continue
    name = i.name[len("input.") :]
    dims = [max_cache_len if isinstance(d, str) and not d.isdigit() else int(d) for d in i.shape]
    cache[name] = torch.zeros(dims, dtype=TypeHelper.ort_type_to_torch_type(i.type), device="cuda")
    cuda.set_buffer_sharing(f"input.{name}", f"output.{name}")
cache_feed = {f"input.{name}": buf for name, buf in cache.items()}

vocab_size = next(o.shape[-1] for o in session.get_outputs() if o.name.endswith("logits"))

# prefill: the whole prompt in one call
positions = torch.arange(prompt_len, device="cuda")
cuda.allocate_buffers({"logits": (1, prompt_len, vocab_size)})
out = cuda.infer({"input_ids": prompt_ids, "attention_mask": causal_mask(positions, max_cache_len),
                  "position_ids": positions[None], **cache_feed})
next_token = out["logits"][:, -1:].argmax(-1)

# decode: query=1 buffers reused in place
cuda.allocate_buffers({"logits": (1, 1, vocab_size)})
input_ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
position_ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
attention_mask = torch.empty((1, 1, 1, max_cache_len), dtype=torch.bool, device="cuda")
slots = torch.arange(max_cache_len, device="cuda")
for position in range(prompt_len, max_cache_len):
    input_ids.copy_(next_token)
    position_ids.fill_(position)
    attention_mask[0, 0, 0].copy_(slots <= position)
    out = cuda.infer({"input_ids": input_ids, "attention_mask": attention_mask,
                      "position_ids": position_ids, **cache_feed})
    next_token = out["logits"][:, -1:].argmax(-1)
```

</hfoption>
<hfoption id="ExecuTorch">

ExecuTorch's on-device runtime is C++, and the in-place cache update relies on
`Method::set_output_data_ptr` — **not surfaced by the Python runtime** (`executorch.runtime.Method`
exposes only `execute`/`set_inputs`/`get_outputs`), so the zero-copy decode is a C++-only path. Bind
each mutated-cache **output** onto its matching cache **input** buffer, and the new K/V/counter land in
the caller's `StaticCache` buffers with no copies. As in the other tabs the shapes and sizes come from the
artifact itself — here the program's `method_meta` (`input_tensor_meta`/`output_tensor_meta` →
`TensorInfo::nbytes()`), the C++ has no Python runtime to query:

```cpp
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>

using namespace executorch::runtime;
using executorch::extension::FileDataLoader;
using executorch::extension::make_tensor_ptr;

// load the exported decode and its `forward` method (Result error-checks elided for brevity)
auto loader = FileDataLoader::from("decode.pte");
auto program = Program::load(&loader.get());

// method-execution memory: a fixed arena for bookkeeping + one buffer per the method's memory plan
std::array<uint8_t, 4 * 1024 * 1024> arena;
MemoryAllocator method_allocator(arena.size(), arena.data());
auto meta = program->method_meta("forward");
std::vector<std::vector<uint8_t>> planned(meta->num_memory_planned_buffers());
std::vector<Span<uint8_t>> planned_spans;
for (size_t i = 0; i < planned.size(); ++i) {
    planned[i].resize(meta->memory_planned_buffer_size(i).get());
    planned_spans.push_back({planned[i].data(), planned[i].size()});
}
HierarchicalAllocator planned_allocator({planned_spans.data(), planned_spans.size()});
MemoryManager memory_manager(&method_allocator, &planned_allocator);
auto decode = std::move(program->load_method("forward", &memory_manager).get());

// shapes and sizes come from method_meta — no external config. Inputs are [ids, mask, position_ids,
// cache×N]; outputs are [returned cache×N, logits, mutated cache×N].
const size_t num_cache_tensors = meta->num_inputs() - 3;
const size_t logits_out_idx = num_cache_tensors;
std::vector<size_t> cache_nbytes(num_cache_tensors), cache_out_idx(num_cache_tensors);
for (size_t i = 0; i < num_cache_tensors; ++i) {
    cache_nbytes[i] = meta->input_tensor_meta(3 + i)->nbytes();
    cache_out_idx[i] = num_cache_tensors + 1 + i;   // mutated cache = the last N outputs
}
const size_t logits_nbytes = meta->output_tensor_meta(logits_out_idx)->nbytes();

// one set of StaticCache buffers (K/V + per-layer counters) is reused across steps, bound in place each call
auto forward = [&](const TensorPtr& input_ids, const TensorPtr& mask, const TensorPtr& position_ids) {
    decode.set_input(EValue(*input_ids), 0);
    decode.set_input(EValue(*mask), 1);
    decode.set_input(EValue(*position_ids), 2);
    for (int i = 0; i < num_cache_tensors; ++i)
        decode.set_input(EValue(*cache_tensor[i]), 3 + i);
    // bind each mutated-cache output onto that same input tensor's data → the write lands in place, zero copies
    for (int i = 0; i < num_cache_tensors; ++i)
        decode.set_output_data_ptr(cache_tensor[i]->mutable_data_ptr(), cache_nbytes[i], cache_out_idx[i]);
    decode.set_output_data_ptr(logits_data, logits_nbytes, logits_out_idx);
    decode.execute();                 // cache updated in place; logits written to logits_data
    return argmax_last(logits_data);  // greedy pick
};

// prefill the whole prompt, then decode one token per step — the cache carries in place across all calls
int64_t next_token = forward(prompt_ids, prompt_mask, prompt_positions);
for (int64_t position = prompt_len; position < max_cache_len; ++position) {
    next_token = forward(make_tensor_ptr({1, 1}, &next_token, ScalarType::Long),
                         causal_mask(position),  // [1, 1, 1, max_cache_len] bool
                         make_tensor_ptr({1, 1}, &position, ScalarType::Long));
}
```

</hfoption>
</hfoptions>

</details>

## Quantization

Every export config accepts a `quantizer`. Set it to any PT2E
[`Quantizer`](https://docs.pytorch.org/ao/main/pt2e_quantization/index.html) and the exporter runs post-training quantization on
the traced graph (`prepare_pt2e` → calibrate → `convert_pt2e`) before the program is returned or
lowered. Quantization happens on the graph rather than the modeling code, so a single `quantizer` works
across every backend and architecture without per-model handling.

Quantize through [`~HfExporter.export_for_generation`], which exports the decomposed generation components. Their attention mask is a precomputed graph input, which keeps PT2E away from the in-graph mask construction that trips its `make_fx` retrace on a full model forward.

```python
from transformers import LlamaForCausalLM
from transformers.exporters import DynamoExporter, DynamoConfig
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").eval()
inputs = ...  # forward kwargs

quantizer = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config())
config = DynamoConfig(dynamic=True, quantizer=quantizer, calibration_dataset=[inputs])
exported = DynamoExporter().export(model, inputs, config)  # quantize/dequantize ops folded into the graph
```

### Choosing a quantizer

Each target runtime expects its own quantizer. Whichever you pass, the quantized graph is portable from there. It runs on inductor as int8, translates to ONNX `QuantizeLinear`/`DequantizeLinear` (per-channel included), or lowers to an ExecuTorch `.pte`.

| Where you'll run | Quantizer to pass |
| --- | --- |
| PyTorch inductor, or ONNX Runtime (QDQ) | `X86InductorQuantizer` (torchao) |
| ExecuTorch XNNPACK backend | `XNNPACKQuantizer` |
| ExecuTorch QNN backend (Qualcomm HTP) | `QnnQuantizer` |

### Calibration

`calibration_dataset` is a list of forward-kwarg dicts run through the prepared graph to gather
observer statistics. Omit it and the exporter falls back to a single pass over the export's own sample
inputs, with a warning (one sample can skew the observed ranges).

For generative models, set `calibration_dataset` on the config you pass to [`~HfExporter.export_for_generation`] and give it generate-style kwargs. Each sample runs through a short `generate`, and every component (`prefill`, `decode`, the encoders) is calibrated on the activations captured for it.

### A different recipe per component

Pass a `{component: config}` dict (instead of a single config) to
[`~HfExporter.export_for_generation`], and each component is quantized on its own terms. The common
multimodal case is static int8 on the vision tower, dynamic int8 on the language decoder, and a
full-precision `lm_head`:

```python
def x86(dynamic):  # same quantizer family, static (per-channel) vs dynamic int8
    return X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config(is_dynamic=dynamic))

config = {
    "image_encoder": DynamoConfig(dynamic=True, quantizer=x86(dynamic=False)),         # static int8
    "multi_modal_projector": DynamoConfig(dynamic=True, quantizer=x86(dynamic=False)),
    "language_model": DynamoConfig(dynamic=True, quantizer=x86(dynamic=True)),          # dynamic int8
    "decode": DynamoConfig(dynamic=True, quantizer=x86(dynamic=True)),
    "lm_head": DynamoConfig(dynamic=True),                                              # no quantizer → fp32
}
components = DynamoExporter().export_for_generation(model, inputs, config, multi_token_decode=True)
```

The dict must name every component [`~exporters.utils.decompose_for_generation`] produces; a component
whose config sets no `quantizer` is left in full precision.

> [!NOTE]
> Quantization always runs on these decomposed components, whose attention mask is a precomputed input.
> That keeps PT2E away from in-graph mask construction, which otherwise trips its `make_fx` retrace.

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
