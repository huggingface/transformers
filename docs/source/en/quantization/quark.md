<!--Copyright 2024 Advanced Micro Devices, Inc. and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quark

[Quark](https://quark.docs.amd.com/latest/) is a deep learning quantization toolkit designed to be agnostic to specific data types, algorithms, and hardware. Different pre-processing strategies, algorithms and data-types can be combined in Quark.

The PyTorch support integrated through 🤗 Transformers primarily targets AMD CPUs and GPUs, and is primarily meant to be used for evaluation purposes.

Users interested in Quark can refer to its [documentation](https://quark.docs.amd.com/latest/) to get started quantizing models and using them in supported open-source libraries!

Although Quark has its own checkpoint / [configuration format](https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test/blob/main/config.json#L26), the library also supports producing models with a serialization layout compliant with other quantization/runtime implementations ([AutoAWQ](https://huggingface.co/docs/transformers/quantization/awq), [native fp8 in 🤗 Transformers](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)).

## Support matrix

Models quantized through Quark support a large range of features, that can be combined together. All quantized models independently of their configuration can seamlessly be reloaded through `PretrainedModel.from_pretrained`.

| **Feature**                     | **Supported subset in Quark**                                                                             |   |
|---------------------------------|-----------------------------------------------------------------------------------------------------------|---|
| Data types                      | int8, int4, int2, bfloat16, float16, fp8_e5m2, fp8_e4m3, fp6_e3m2, fp6_e2m3, fp4, OCP MX, MX6, MX9, bfp16 |   |
| Pre-quantization transformation | SmoothQuant, QuaRot, SpinQuant, AWQ                                                                       |   |
| Quantization algorithm          | GPTQ                                                                                                      |   |
| Supported operators             | ``nn.Linear``, ``nn.Conv2d``, ``nn.ConvTranspose2d``, ``nn.Embedding``, ``nn.EmbeddingBag``               |   |
| Granularity                     | per-tensor, per-channel, per-block, per-layer, per-layer type                                             |   |
| KV cache                        | fp8                                                                                                       |   |
| Activation calibration          | MinMax / Percentile / MSE                                                                                 |   |
| Quantization strategy           | weight-only, static, dynamic, with or without output quantization                                         |   |

## Models on Hugging Face Hub

Public models using Quark native serialization can be found at https://huggingface.co/models?other=quark.

Although Quark also supports [models using `quant_method="fp8"`](https://huggingface.co/models?other=fp8) and [models using `quant_method="awq"`](https://huggingface.co/models?other=awq), Transformers loads these models rather through [AutoAWQ](https://huggingface.co/docs/transformers/quantization/awq) or uses the [native fp8 support in 🤗 Transformers](https://huggingface.co/docs/transformers/quantization/finegrained_fp8).

## Using Quark models in Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EmbeddedLLM/Llama-3.1-8B-Instruct-w_fp8_per_channel_sym")
model = model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("EmbeddedLLM/Llama-3.1-8B-Instruct-w_fp8_per_channel_sym")

inp = tokenizer("Where is a good place to cycle around Tokyo?", return_tensors="pt").to("cuda")

res = model.generate(**inp)

print(tokenizer.batch_decode(res))
```