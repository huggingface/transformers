<!--Copyright 2025 Advanced Micro Devices, Inc. and The HuggingFace Team. All rights reserved.

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

The PyTorch support integrated through 🤗 Transformers primarily targets AMD CPUs and GPUs, and is primarily meant to be used for evaluation purposes. For example, it is possible to use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with 🤗 Transformers backend and evaluate a wide range of models quantized through Quark seamlessly.

Users interested in Quark can refer to its [documentation](https://quark.docs.amd.com/latest/) to get started quantizing models and using them in supported open-source libraries!

Although Quark has its own checkpoint / [configuration format](https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test/blob/main/config.json#L26), the library also supports producing models with a serialization layout compliant with other quantization/runtime implementations ([AutoAWQ](https://huggingface.co/docs/transformers/quantization/awq), [native fp8 in 🤗 Transformers](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)).

To be able to load Quark quantized models in Transformers, the library first needs to be installed:

```bash
pip install amd-quark
```

## Support matrix

Models quantized through Quark support a large range of features, that can be combined together. All quantized models independently of their configuration can seamlessly be reloaded through `PretrainedModel.from_pretrained`.

The table below shows a few features supported by Quark:

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

Here is an example of how one can load a Quark model in Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "EmbeddedLLM/Llama-3.1-8B-Instruct-w_fp8_per_channel_sym"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

print(model.model.layers[0].self_attn.q_proj)
# QParamsLinear(
#   (weight_quantizer): ScaledRealQuantizer()
#   (input_quantizer): ScaledRealQuantizer()
#   (output_quantizer): ScaledRealQuantizer()
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
inp = tokenizer("Where is a good place to cycle around Tokyo?", return_tensors="pt")
inp = inp.to(model.device)

res = model.generate(**inp, min_new_tokens=50, max_new_tokens=100)

print(tokenizer.batch_decode(res)[0])
# <|begin_of_text|>Where is a good place to cycle around Tokyo? There are several places in Tokyo that are suitable for cycling, depending on your skill level and interests. Here are a few suggestions:
# 1. Yoyogi Park: This park is a popular spot for cycling and has a wide, flat path that's perfect for beginners. You can also visit the Meiji Shrine, a famous Shinto shrine located in the park.
# 2. Imperial Palace East Garden: This beautiful garden has a large, flat path that's perfect for cycling. You can also visit the
```
