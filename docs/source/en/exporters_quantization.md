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

# Quantizing exports

Every [export](./exporters) config can take a `quantizer`, which quantizes the model during the export. The quantizers
live in `transformers.exporters.quantizers`, and each one runs at a fixed point of the export: on the `torch.export`
FX graph, before any conversion, or on the model a backend converts it into.

| Quantizer | Quantizes | Dynamo | ONNX | ExecuTorch | OpenVINO |
| --- | --- | --- | --- | --- | --- |
| [`~exporters.quantizers.PT2EQuantizer`] | the FX graph, with PyTorch 2 Export (PT2E) quantization | 🟢 | 🟢 | 🟢 | 🟢 |
| [`~exporters.quantizers.NNCFTorchFXQuantizer`] | the FX graph, with [NNCF](https://github.com/openvinotoolkit/nncf) | 🟢 | 🟢 | 🟢 (weight-only) | 🟢 |
| [`~exporters.quantizers.OnnxRuntimeQuantizer`] | the ONNX model, with `onnxruntime.quantization` | 🔴 | 🟢 | 🔴 | 🔴 |
| [`~exporters.quantizers.NNCFOnnxQuantizer`] | the ONNX model, with [NNCF](https://github.com/openvinotoolkit/nncf) | 🔴 | 🟢 | 🔴 | 🔴 |
| [`~exporters.quantizers.NNCFOpenVINOQuantizer`] | the OpenVINO model, with [NNCF](https://github.com/openvinotoolkit/nncf) | 🔴 | 🔴 | 🔴 | 🟢 |

The export checks the quantizer against its backend and installed packages before tracing, so a mismatch fails early.
Like the exporters, each quantizer lists the package versions it is tested on, and warns when yours differ.

The example below quantizes a sentiment classifier to int8 with ONNX Runtime, then runs it.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.exporters import OnnxConfig, OnnxExporter, OnnxRuntimeQuantizer

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

texts = [
    "A wonderful, heartfelt film.",
    "The plot was dull and the acting worse.",
    "I would watch it again tomorrow.",
    "Two hours I will never get back.",
]
calibration_dataset = [tokenizer(text, return_tensors="pt") for text in texts]
inputs = tokenizer("An absolute delight from start to finish.", return_tensors="pt")

config = OnnxConfig(dynamic=True, quantizer=OnnxRuntimeQuantizer(calibration_dataset=calibration_dataset))
onnx_program = OnnxExporter().export(model, inputs, config)

logits = onnx_program(**inputs)[0]
print(model.config.id2label[int(logits.argmax())])  # POSITIVE
```

## PT2E

[`~exporters.quantizers.PT2EQuantizer`] wraps a PT2E quantizer, from torchao or from the target runtime, and runs it
on the FX graph (`prepare_pt2e` → calibrate → `convert_pt2e`), so it works with every backend. Each PT2E quantizer
injects its own quantize/dequantize ops, so pick the one your runtime supports:

| Where you'll run | PT2E quantizer | Import from |
| --- | --- | --- |
| PyTorch inductor, ONNX Runtime (QDQ), OpenVINO | `X86InductorQuantizer` | `torchao.quantization.pt2e.quantizer.x86_inductor_quantizer` |
| ExecuTorch XNNPACK backend | `XNNPACKQuantizer` | `executorch.backends.xnnpack.quantizer.xnnpack_quantizer` |

For ExecuTorch, wrap ExecuTorch's own `XNNPACKQuantizer`:

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from transformers.exporters import ExecutorchConfig, PT2EQuantizer

xnnpack = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
config = ExecutorchConfig(backend="xnnpack", quantizer=PT2EQuantizer(xnnpack, calibration_dataset=calibration_dataset))
```

Quantize generative models through [`~HfExporter.export_for_generation`]. Here `calibration_dataset` holds generate
kwargs, like `inputs`, and each component calibrates on the inputs captured for it. A text-only model has a `prefill`
and a `decode` component; a multimodal model splits its prefill further, into its modality encoders (such as
`image_encoder`), its projector, its `language_model` and its `lm_head` (see
[How `export_for_generation` works](./exporters#how-exportforgeneration-works)).

```python
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import DynamoConfig, DynamoExporter, PT2EQuantizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompts = ["The capital of France is", "Photosynthesis turns sunlight into", "The first moon landing was in"]
calibration_dataset = [tokenizer(prompt, return_tensors="pt") for prompt in prompts]
inputs = tokenizer("Once upon a time", return_tensors="pt")

x86 = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config())
config = DynamoConfig(dynamic=True, quantizer=PT2EQuantizer(x86, calibration_dataset=calibration_dataset))
components = DynamoExporter().export_for_generation(model, inputs, config)  # {"prefill": ..., "decode": ...}
```

On OpenVINO, PT2E weights stay in full precision behind quantize/dequantize pairs, which OpenVINO converts to
`FakeQuantize` and compresses to int8 itself.

## ONNX Runtime

[`~exporters.quantizers.OnnxRuntimeQuantizer`] runs `quantize_static` by default (int8 QDQ activations and weights,
calibrated on `calibration_dataset`), as in the example above, with the arguments of an
[`~exporters.quantizers.OnnxRuntimeStaticConfig`]. With an [`~exporters.quantizers.OnnxRuntimeDynamicConfig`] it runs
`quantize_dynamic` (int8 weights, activations quantized at runtime, no calibration).

```python
from transformers.exporters import OnnxRuntimeDynamicConfig, OnnxRuntimeStaticConfig

config = OnnxConfig(
    quantizer=OnnxRuntimeQuantizer(OnnxRuntimeStaticConfig(per_channel=True), calibration_dataset=calibration_dataset)
)
config = OnnxConfig(quantizer=OnnxRuntimeQuantizer(OnnxRuntimeDynamicConfig()))
```

## NNCF

The three NNCF quantizers in the table above run `nncf.quantize` by default (int8 activations and weights, calibrated
on `calibration_dataset`, with `model_type=nncf.ModelType.TRANSFORMER` unless you pass another), with the arguments of
an [`~exporters.quantizers.NNCFQuantizationConfig`]. With an [`~exporters.quantizers.NNCFWeightCompressionConfig`] they
run `nncf.compress_weights` (weight-only int8 or int4, with AWQ and scale estimation for int4, plus GPTQ and LoRA
correction on OpenVINO). ExecuTorch lowers only [`~exporters.quantizers.NNCFTorchFXQuantizer`]'s weight-only result.

```python
import nncf
from transformers.exporters import (
    DynamoConfig,
    NNCFOnnxQuantizer,
    NNCFOpenVINOQuantizer,
    NNCFTorchFXQuantizer,
    NNCFWeightCompressionConfig,
    OnnxConfig,
    OpenVINOConfig,
    OpenVINOExporter,
)

int4 = NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT4_SYM, group_size=128)

# OpenVINO: int8 activations and weights, or weight-only int4
openvino_int8 = OpenVINOConfig(quantizer=NNCFOpenVINOQuantizer(calibration_dataset=calibration_dataset))
openvino_int4 = OpenVINOConfig(quantizer=NNCFOpenVINOQuantizer(int4))
# ONNX: weight-only int4, as ONNX Runtime's `MatMulNBits`
onnx_int4 = OnnxConfig(quantizer=NNCFOnnxQuantizer(int4))
# Dynamo: int8 on the FX graph
dynamo_int8 = DynamoConfig(quantizer=NNCFTorchFXQuantizer(calibration_dataset=calibration_dataset))

ov_model = OpenVINOExporter().export(model, inputs, openvino_int4)
```

Int8 activations run int8 matmuls, which saturate on CPUs without VNNI (AVX2-only Intel and AMD Zen 2 and older), so
expect lower accuracy there. Weight-only compression keeps the matmuls in floating point.

## Calibration

A quantizer's `calibration_dataset` is an iterable of model inputs, such as a list of tokenizer outputs or a
`DataLoader` whose batches collate to model inputs. It is a plain attribute, so you can also set it after creating
the quantizer (`quantizer.calibration_dataset = samples`). Omit it and a quantizer that needs data calibrates on the
export's own sample inputs, with a warning, because one sample can skew the observed ranges. Quantizers that need no
data (dynamic ONNX Runtime quantization, int8 weight compression) never read it.

## A different recipe per component

Pass a `{component: config}` dict to [`~HfExporter.export_for_generation`] to quantize each component on its own
terms. For the model above, static int8 on `prefill` and the lighter dynamic int8 on `decode`:

```python
dynamic_x86 = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config(is_dynamic=True))

config = {
    "prefill": DynamoConfig(dynamic=True, quantizer=PT2EQuantizer(x86)),
    "decode": DynamoConfig(dynamic=True, quantizer=PT2EQuantizer(dynamic_x86)),
}
components = DynamoExporter().export_for_generation(model, inputs, config)
```

The dict must name every component [`~exporters.utils.decompose_for_generation`] produces (a multimodal model adds its
encoders, and may split out `language_model` and `lm_head`). A component whose config sets no quantizer stays in full
precision. In this dict, a quantizer's `calibration_dataset` holds the component's forward kwargs, not generate
kwargs.

## Add a quantizer

Quantizers live in `transformers.exporters.quantizers`, one class per toolchain and representation. Subclass
[`~exporters.quantizers.ExportQuantizer`] and declare:

- `quantizes_fx_graph`: `True` for the `torch.export` FX graph, before any conversion (so every backend takes the
  result), or `False` (the default) for the model a backend converts it into (`onnx_ir.Model`, `openvino.Model`).
- `supported_formats`: the export formats it quantizes. The export rejects any other format before tracing.
- `required_packages`, and optionally `min_versions` and `tested_versions`: what it imports. The export checks
  they're installed before tracing, raises below a minimum version, and warns when a version differs from the tested
  one.

Then implement `quantize(model, sample_inputs, export_format)`. `model` is the FX graph or the backend model, and
`sample_inputs` are the export's own inputs. Calibrate on a [`~exporters.quantizers.CalibrationSet`]: it reads the
quantizer's `calibration_dataset` (or `sample_inputs` when there is none) and maps each sample to the model's own
inputs with a feed from `transformers.exporters.quantizers.utils` (`traced_inputs` for the FX graph, `onnx_feed` for ONNX,
`openvino_feed` for OpenVINO).

```python
from transformers.exporters import ExportFormat
from transformers.exporters.quantizers import CalibrationSet, ExportQuantizer
from transformers.exporters.quantizers.utils import onnx_feed


class MyOnnxQuantizer(ExportQuantizer):
    supported_formats = (ExportFormat.ONNX,)
    required_packages = ("my_toolkit",)
    tested_versions = {"my_toolkit": "1.0.0"}

    def quantize(self, model, sample_inputs, export_format):
        import my_toolkit

        names = [graph_input.name for graph_input in model.graph.inputs]
        calibration = CalibrationSet(self.calibration_dataset, sample_inputs, lambda sample: onnx_feed(names, sample))
        return my_toolkit.quantize(model, calibration_feeds=list(calibration))
```
