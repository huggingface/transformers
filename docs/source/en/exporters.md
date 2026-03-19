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

Transformers ships three built-in exporters that convert any [`PreTrainedModel`] to a portable,
runtime-optimised format without leaving the library.

| Exporter               | Output                         | Target                               |
| ---------------------- | ------------------------------ | ------------------------------------ |
| [`DynamoExporter`]     | `torch.export.ExportedProgram` | Any PyTorch runtime, AOT compilation |
| [`OnnxExporter`]       | `torch.onnx.ONNXProgram`       | ONNX Runtime, TensorRT, OpenVINO, …  |
| [`ExecutorchExporter`] | `ExecutorchProgramManager`     | Mobile and edge devices (ExecuTorch) |

All three share the same calling convention — create an exporter with a config, call `.export(model, inputs)` —
and all inherit from a common base so multicomponent and generative decomposition works identically across backends.

> [!TIP]
> The exporters described here operate directly on `PreTrainedModel` instances and require no additional
> libraries beyond `torch` (and `onnxscript`/`executorch` for the respective backends).
> For the production-ready Optimum-based exporters that support quantization, graph optimisation, and Hub upload,
> see [Exporting to production](./serialization).

## Installation

```bash
pip install transformers torch
```

For ONNX export:

```bash
pip install onnxscript onnxruntime
```

For ExecuTorch export:

```bash
pip install executorch
```

## Quick start

<hfoptions id="exporters-quickstart">
<hfoption id="torch.export (Dynamo)">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
exported = exporter.export(model, dict(inputs))

# run the exported graph
outputs = exported.module()(**dict(inputs))
print(outputs.logits.shape)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
onnx_program = exporter.export(model, dict(inputs))

# run with the built-in ORT session
outputs = onnx_program(**dict(inputs))
```

</hfoption>
<hfoption id="ExecuTorch">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend="xnnpack"))
exporter.export(model, dict(inputs))
```

</hfoption>
</hfoptions>

## Configuration

Each exporter accepts a typed config dataclass. All configs inherit from [`DynamoConfig`], so the
`dynamic`, `strict`, `dynamic_shapes`, and `prefer_deferred_runtime_asserts_over_guards` fields
are available on every exporter. See the [Exporters API reference](./main_classes/exporters) for
the full field descriptions.

## Dynamic shapes

Set `dynamic=True` on any config to export with symbolic (dynamic) input shapes. All tensor dimensions
are automatically marked as `Dim.AUTO`, which means the exported graph accepts inputs of any size at
runtime without retracing.

<hfoptions id="dynamic-shapes">
<hfoption id="torch.export (Dynamo)">

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
<hfoption id="torch.export (Dynamo)">

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

## Multicomponent models

Vision-language models (VLMs) and encoder-decoder models are best exported as separate components.
The exporters detect multicomponent models automatically via [`is_multicomponent`] and decompose them
into individual submodules — each exported as an independent graph.

<hfoptions id="multicomponent">
<hfoption id="torch.export (Dynamo)">

```python
>>> from transformers import AutoModelForVision2Seq, AutoProcessor
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig
>>> from transformers.exporters.utils import decompose_encoder_decoder, is_multicomponent

>>> model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

>>> components = decompose_encoder_decoder(model, inputs) if is_multicomponent(model) else [("model", model, inputs)]
>>> # components = [("vision_model", vit, vit_inputs), ("language_model", llm, llm_inputs), ...]

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> for name, submodel, subinputs in components:
...     exported = exporter.export(submodel, subinputs)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig
from transformers.exporters.utils import decompose_encoder_decoder, is_multicomponent

model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

components = decompose_encoder_decoder(model, inputs) if is_multicomponent(model) else [("model", model, inputs)]
# components = [("vision_model", vit, vit_inputs), ("language_model", llm, llm_inputs), ...]

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
for name, submodel, subinputs in components:
    onnx_program = exporter.export(submodel, subinputs)
```

</hfoption>
</hfoptions>

The decomposition is done via a single forward pass with hooks — it captures the exact inputs each
submodule receives, so the exported graphs have correct signatures without any manual wiring.

Supported submodule attribute names are listed in [`~transformers.exporters.utils._SUBMODULE_NAMES`].
If a new architecture uses a different attribute name, add it to that tuple.

<Tip warning={true}>

The exported components are **independent graphs**, not a turnkey inference pipeline.
The caller is responsible for running each encoder, fusing the multimodal embeddings, projecting
them into the language model's input space, and orchestrating the generation loop.
We are actively working to reduce the amount of glue required between components.

</Tip>

## Generative models (prefill / decode)

For autoregressive generation, the model's `forward` has different shapes at the prefill step
(full prompt, no KV cache) versus the decode step (single token, populated KV cache).
[`decompose_prefill_decode`] runs `model.generate()` for two tokens and captures both.

<hfoptions id="generate">
<hfoption id="torch.export (Dynamo)">

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

For multicomponent generative models (VLMs), combine both decompositions: decompose the prefill
stage into its submodules and keep the decode stage as a single graph.

<hfoptions id="generate-vlm">
<hfoption id="torch.export (Dynamo)">

```python
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig
>>> from transformers.exporters.utils import decompose_encoder_decoder, decompose_prefill_decode, is_multicomponent

>>> stages = decompose_prefill_decode(model, inputs)
>>> _, prefill_model, prefill_inputs = stages[0]

>>> components = decompose_encoder_decoder(prefill_model, prefill_inputs) if is_multicomponent(prefill_model) else [("prefill", prefill_model, prefill_inputs)]
>>> components += stages[1:]  # add the decode stage

>>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
>>> for name, submodel, subinputs in components:
...     exported = exporter.export(submodel, subinputs)
```

</hfoption>
<hfoption id="ONNX">

```python
from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig
from transformers.exporters.utils import decompose_encoder_decoder, decompose_prefill_decode, is_multicomponent

stages = decompose_prefill_decode(model, inputs)
_, prefill_model, prefill_inputs = stages[0]

components = decompose_encoder_decoder(prefill_model, prefill_inputs) if is_multicomponent(prefill_model) else [("prefill", prefill_model, prefill_inputs)]
components += stages[1:]  # add the decode stage

exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
for name, submodel, subinputs in components:
    onnx_program = exporter.export(submodel, subinputs)
```

</hfoption>
</hfoptions>

## API reference

### Exporter classes

[[autodoc]] transformers.exporters.exporter_dynamo.DynamoExporter
    - export

[[autodoc]] transformers.exporters.exporter_onnx.OnnxExporter
    - export

[[autodoc]] transformers.exporters.exporter_executorch.ExecutorchExporter
    - export

### Utilities

[[autodoc]] transformers.exporters.utils.prepare_for_export

[[autodoc]] transformers.exporters.utils.decompose_prefill_decode

[[autodoc]] transformers.exporters.utils.decompose_encoder_decoder

[[autodoc]] transformers.exporters.utils.is_multicomponent

[[autodoc]] transformers.exporters.utils.get_leaf_tensors
