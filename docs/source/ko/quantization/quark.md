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

# Quark[[quark]]

[Quark](https://quark.docs.amd.com/latest/)는 특정 데이터 타입, 알고리즘, 하드웨어에 구애받지 않도록 설계된 딥러닝 양자화 툴킷입니다. Quark에서는 다양한 전처리 전략, 알고리즘, 데이터 타입을 조합하여 사용할 수 있습니다.

🤗 Transformers를 통해 통합된 PyTorch 지원은 주로 AMD CPU 및 GPU를 대상으로 하며, 주로 평가 목적으로 사용됩니다. 예를 들어, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)를 🤗 Transformers 백엔드와 함께 사용하여 Quark로 양자화된 다양한 모델을 원활하게 평가할 수 있습니다.

Quark에 관심이 있는 사용자는 [문서](https://quark.docs.amd.com/latest/)를 참고하여 모델 양자화를 시작하고 지원되는 오픈 소스 라이브러리에서 사용할 수 있습니다!

Quark는 자체 체크포인트/[설정 포맷](https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test/blob/main/config.json#L26)를 가지고 있지만, 다른 양자화/런타임 구현체 ([AutoAWQ](https://huggingface.co/docs/transformers/quantization/awq), [네이티브 fp8](https://huggingface.co/docs/transformers/quantization/finegrained_fp8))와 호환되는 직렬화 레이아웃으로 모델을 생성하는 것도 지원합니다.

Transformer에서 Quark 양자화 모델을 로드하려면 먼저 라이브러리를 설치해야 합니다:

```bash
pip install amd-quark
```

## 지원 매트릭스[[Support matrix]]

Quark를 통해 양자화된 모델은 함께 조합할 수 있는 광범위한 기능을 지원합니다. 구성에 관계없이 모든 양자화된 모델은 `PretrainedModel.from_pretrained`를 통해 원활하게 다시 로드할 수 있습니다.

아래 표는 Quark에서 지원하는 몇 가지 기능을 보여줍니다:

| **기능**                        | **Quark에서 지원하는 항목**                                                                             |   |
|---------------------------------|-----------------------------------------------------------------------------------------------------------|---|
| 데이터 타입                     | int8, int4, int2, bfloat16, float16, fp8_e5m2, fp8_e4m3, fp6_e3m2, fp6_e2m3, fp4, OCP MX, MX6, MX9, bfp16 |   |
| 양자화 전 모델 변환 | SmoothQuant, QuaRot, SpinQuant, AWQ                                                                       |   |
| 양자화 알고리즘                 | GPTQ                                                                                                      |   |
| 지원 연산자                     | ``nn.Linear``, ``nn.Conv2d``, ``nn.ConvTranspose2d``, ``nn.Embedding``, ``nn.EmbeddingBag``               |   |
| 세분성(Granularity)             | per-tensor, per-channel, per-block, per-layer, per-layer type                                             |   |
| KV 캐시                         | fp8                                                                                                       |   |
| 활성화 캘리브레이션             | MinMax / Percentile / MSE                                                                                 |   |
| 양자화 전략                     | weight-only, static, dynamic, with or without output quantization                                         |   |

## Hugging Face Hub의 모델[[Models on Hugging Face Hub]]

Quark 네이티브 직렬화를 사용하는 공개 모델은 https://huggingface.co/models?other=quark 에서 찾을 수 있습니다.

Quark는 [`quant_method="fp8"`을 이용하는 모델](https://huggingface.co/models?other=fp8)과 [`quant_method="awq"`을 사용하는 모델](https://huggingface.co/models?other=awq)도 지원하지만, Transformers는 이러한 모델을 [AutoAWQ](https://huggingface.co/docs/transformers/quantization/awq)를 통해 불러오거나 
[🤗 Transformers의 네이티브 fp8 지원](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)을 사용합니다.

## Transformers에서 Quark모델 사용하기[[Using Quark models in Transformers]]

다음은 Transformers에서 Quark 모델을 불러오는 방법의 예시입니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "EmbeddedLLM/Llama-3.1-8B-Instruct-w_fp8_per_channel_sym"
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to("cuda")

print(model.model.layers[0].self_attn.q_proj)
# QParamsLinear(
#   (weight_quantizer): ScaledRealQuantizer()
#   (input_quantizer): ScaledRealQuantizer()
#   (output_quantizer): ScaledRealQuantizer()
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
inp = tokenizer("Where is a good place to cycle around Tokyo?", return_tensors="pt")
inp = inp.to("cuda")

res = model.generate(**inp, min_new_tokens=50, max_new_tokens=100)

print(tokenizer.batch_decode(res)[0])
# <|begin_of_text|>Where is a good place to cycle around Tokyo? There are several places in Tokyo that are suitable for cycling, depending on your skill level and interests. Here are a few suggestions:
# 1. Yoyogi Park: This park is a popular spot for cycling and has a wide, flat path that's perfect for beginners. You can also visit the Meiji Shrine, a famous Shinto shrine located in the park.
# 2. Imperial Palace East Garden: This beautiful garden has a large, flat path that's perfect for cycling. You can also visit the
```
