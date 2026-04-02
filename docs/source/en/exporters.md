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

| Exporter               | Output                         | Runtime                              |
| ---------------------- | ------------------------------ | ------------------------------------ |
| [`DynamoExporter`]     | `torch.export.ExportedProgram` | Any PyTorch runtime, AOT compilation |
| [`OnnxExporter`]       | `torch.onnx.ONNXProgram`       | ONNX Runtime, TensorRT, OpenVINO, …  |
| [`ExecutorchExporter`] | `ExecutorchProgramManager`     | Mobile and edge devices (ExecuTorch) |

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer("Hello, world!", return_tensors="pt"))

exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
exported = exporter.export(model, inputs)

# run the exported graph directly
outputs = exported.module()(**inputs)
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
import numpy as np

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

exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend="xnnpack"))
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
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> exported = exporter.export(model, inputs)

>>> # works with any sequence length
>>> exported.module()(**dict(tokenizer("Hi", return_tensors="pt")))
>>> exported.module()(**dict(tokenizer("A much longer input sequence.", return_tensors="pt")))
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
onnx_program = exporter.export(model, inputs)

# works with any sequence length
onnx_program(**dict(tokenizer("Hi", return_tensors="pt")))
onnx_program(**dict(tokenizer("A much longer input sequence.", return_tensors="pt")))
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
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> batch = torch.export.Dim("batch", min=1, max=32)
>>> seq = torch.export.Dim("seq", min=1, max=2048)

>>> exporter = DynamoExporter(export_config=DynamoConfig(
...     dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}}
... ))
>>> exported = exporter.export(model, inputs)
```

</hfoption>
<hfoption id="ONNX">

```python
import torch
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("seq", min=1, max=2048)

exporter = OnnxExporter(export_config=OnnxConfig(
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}}
))
onnx_program = exporter.export(model, inputs)
```

</hfoption>
</hfoptions>

## Generative models (prefill / decode)

For autoregressive generation, the model's `forward` has different shapes at the prefill step
(full prompt, no KV cache) versus the decode step (single token, populated KV cache).
[`decompose_prefill_decode`] runs `model.generate()` for two tokens and captures both.

<hfoptions id="generate">
<hfoption id="Dynamo">

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig
>>> from transformers.exporters.utils import decompose_prefill_decode

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> inputs = dict(tokenizer("Hello, world!", return_tensors="pt"))

>>> stages = decompose_prefill_decode(model, inputs)
>>> # stages = [("prefill", model_copy, prefill_inputs), ("decode", model_copy, decode_inputs)]

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> for name, stage_model, stage_inputs in stages:
...     exported = exporter.export(stage_model, stage_inputs)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig
from transformers.exporters.utils import decompose_prefill_decode

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = dict(tokenizer("Hello, world!", return_tensors="pt"))

stages = decompose_prefill_decode(model, inputs)
# stages = [("prefill", model_copy, prefill_inputs), ("decode", model_copy, decode_inputs)]

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
for name, stage_model, stage_inputs in stages:
    onnx_program = exporter.export(stage_model, stage_inputs)
```

</hfoption>
</hfoptions>

## Vision-language models (VLMs)

VLMs are exported as separate components — vision encoder, projector, language model — each as
an independent graph. [`decompose_vlm`] detects VLM submodules automatically and captures their
inputs via a single forward pass with hooks.

For generative VLMs, first decompose into prefill/decode, then decompose the prefill stage
into its submodules. The decode stage stays as a single graph.

<hfoptions id="multicomponent">
<hfoption id="Dynamo">

```python
>>> from transformers import AutoModelForVision2Seq, AutoProcessor
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig
>>> from transformers.exporters.utils import decompose_vlm, decompose_prefill_decode, is_vlm

>>> model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

>>> # step 1: split into prefill and decode stages
>>> stages = decompose_prefill_decode(model, inputs)
>>> _, prefill_model, prefill_inputs = stages[0]

>>> # step 2: decompose the prefill into vision encoder, projector, language model
>>> components = decompose_vlm(prefill_model, prefill_inputs)
>>> components += stages[1:]  # add the decode stage

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> for name, submodel, subinputs in components:
...     exported = exporter.export(submodel, subinputs)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig
from transformers.exporters.utils import decompose_vlm, decompose_prefill_decode, is_vlm

model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# step 1: split into prefill and decode stages
stages = decompose_prefill_decode(model, inputs)
_, prefill_model, prefill_inputs = stages[0]

# step 2: decompose the prefill into vision encoder, projector, language model
components = decompose_vlm(prefill_model, prefill_inputs)
components += stages[1:]  # add the decode stage

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
for name, submodel, subinputs in components:
    onnx_program = exporter.export(submodel, subinputs)
```

</hfoption>
</hfoptions>

Supported submodule attribute names are listed in [`~transformers.exporters.utils._VLM_SUBMODULE_NAMES`].
If a new architecture uses a different attribute name, add it to that tuple.

<Tip warning={true}>

The exported components are **independent graphs**, not a turnkey inference pipeline.
The caller is responsible for running each encoder, projecting embeddings, and orchestrating
the generation loop. We are actively working to reduce the glue required between components.

</Tip>

## API reference

### Exporter classes

[[autodoc]] transformers.exporters.exporter_dynamo.DynamoExporter
    - export

[[autodoc]] transformers.exporters.exporter_onnx.OnnxExporter
    - export

[[autodoc]] transformers.exporters.exporter_executorch.ExecutorchExporter
    - export

### Configuration

Each exporter accepts a typed config dataclass. All configs inherit from [`DynamoConfig`], so the
`dynamic`, `strict`, `dynamic_shapes`, and `prefer_deferred_runtime_asserts_over_guards` fields
are available on every exporter.

### Utilities

[[autodoc]] transformers.exporters.utils.get_leaf_tensors

[[autodoc]] transformers.exporters.utils.prepare_for_export

[[autodoc]] transformers.exporters.utils.decompose_prefill_decode

[[autodoc]] transformers.exporters.utils.decompose_vlm

[[autodoc]] transformers.exporters.utils.is_vlm
