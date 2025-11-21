<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 단일 GPU에서 효율적인 추론 [[efficient-inference-on-a-single-gpu]]

이 가이드 외에도, [단일 GPU에서의 훈련 가이드](perf_train_gpu_one)와 [CPU에서의 추론 가이드](perf_infer_cpu)에서도 관련 정보를 찾을 수 있습니다.

## FP4 혼합 정밀도 추론을 위한 `bitsandbytes` 통합 [[bitsandbytes-integration-for-fp4-mixedprecision-inference]]

`bitsandbytes`를 설치하면 GPU에서 손쉽게 모델을 압축할 수 있습니다. FP4 양자화를 사용하면 원래의 전체 정밀도 버전과 비교하여 모델 크기를 최대 8배 줄일 수 있습니다. 아래에서 시작하는 방법을 확인하세요.

<Tip>

이 기능은 다중 GPU 설정에서도 사용할 수 있습니다.

</Tip>

### 요구 사항 [[requirements-for-fp4-mixedprecision-inference]]

- 최신 `bitsandbytes` 라이브러리
`pip install bitsandbytes>=0.39.0`

- 최신 `accelerate`를 소스에서 설치
`pip install git+https://github.com/huggingface/accelerate.git`

- 최신 `transformers`를 소스에서 설치
`pip install git+https://github.com/huggingface/transformers.git`

### FP4 모델 실행 - 단일 GPU 설정 - 빠른 시작 [[running-fp4-models-single-gpu-setup-quickstart]]

다음 코드를 실행하여 단일 GPU에서 빠르게 FP4 모델을 실행할 수 있습니다.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
```
`device_map`은 선택 사항입니다. 그러나 `device_map = 'auto'`로 설정하는 것이 사용 가능한 리소스를 효율적으로 디스패치하기 때문에 추론에 있어 권장됩니다.

### FP4 모델 실행 - 다중 GPU 설정 [[running-fp4-models-multi-gpu-setup]]

다중 GPU에서 혼합 4비트 모델을 가져오는 방법은 단일 GPU 설정과 동일합니다(동일한 명령어 사용):
```py
model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
```
하지만 `accelerate`를 사용하여 각 GPU에 할당할 GPU RAM을 제어할 수 있습니다. 다음과 같이 `max_memory` 인수를 사용하세요:

```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True), max_memory=max_memory_mapping
)
```
이 예에서는 첫 번째 GPU가 600MB의 메모리를 사용하고 두 번째 GPU가 1GB를 사용합니다.

### 고급 사용법 [[advanced-usage]]

이 방법의 더 고급 사용법에 대해서는 [양자화](main_classes/quantization) 문서 페이지를 참조하세요.

## Int8 혼합 정밀도 행렬 분해를 위한 `bitsandbytes` 통합 [[bitsandbytes-integration-for-int8-mixedprecision-matrix-decomposition]]

<Tip>

이 기능은 다중 GPU 설정에서도 사용할 수 있습니다.

</Tip>

[`LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale`](https://huggingface.co/papers/2208.07339) 논문에서 우리는 몇 줄의 코드로 Hub의 모든 모델에 대한 Hugging Face 통합을 지원합니다.
이 방법은 `float16` 및 `bfloat16` 가중치에 대해 `nn.Linear` 크기를 2배로 줄이고, `float32` 가중치에 대해 4배로 줄입니다. 이는 절반 정밀도에서 이상치를 처리함으로써 품질에 거의 영향을 미치지 않습니다.

![HFxbitsandbytes.png](https://cdn-uploads.huggingface.co/production/uploads/1659861207959-62441d1d9fdefb55a0b7d12c.png)

Int8 혼합 정밀도 행렬 분해는 행렬 곱셈을 두 개의 스트림으로 분리합니다: (1) fp16로 곱해지는 체계적인 특이값 이상치 스트림 행렬(0.01%) 및 (2) int8 행렬 곱셈의 일반적인 스트림(99.9%). 이 방법을 사용하면 매우 큰 모델에 대해 예측 저하 없이 int8 추론이 가능합니다.
이 방법에 대한 자세한 내용은 [논문](https://huggingface.co/papers/2208.07339)이나 [통합에 관한 블로그 글](https://huggingface.co/blog/hf-bitsandbytes-integration)에서 확인할 수 있습니다.

![MixedInt8.gif](https://cdn-uploads.huggingface.co/production/uploads/1660567469965-62441d1d9fdefb55a0b7d12c.gif)

커널은 GPU 전용으로 컴파일되어 있기 때문에 혼합 8비트 모델을 실행하려면 GPU가 필요합니다. 이 기능을 사용하기 전에 모델의 1/4(또는 모델 가중치가 절반 정밀도인 경우 절반)을 저장할 충분한 GPU 메모리가 있는지 확인하세요.
이 모듈을 사용하는 데 도움이 되는 몇 가지 참고 사항이 아래에 나와 있습니다. 또는 [Google colab](#colab-demos)에서 데모를 따라할 수도 있습니다.

### 요구 사항 [[requirements-for-int8-mixedprecision-matrix-decomposition]]

- `bitsandbytes<0.37.0`을 사용하는 경우, 8비트 텐서 코어(Turing, Ampere 또는 이후 아키텍처 - 예: T4, RTX20s RTX30s, A40-A100)를 지원하는 NVIDIA GPU에서 실행하는지 확인하세요. `bitsandbytes>=0.37.0`을 사용하는 경우, 모든 GPU가 지원됩니다.
- 올바른 버전의 `bitsandbytes`를 다음 명령으로 설치하세요:
`pip install bitsandbytes>=0.31.5`
- `accelerate`를 설치하세요
`pip install accelerate>=0.12.0`

### 혼합 Int8 모델 실행 - 단일 GPU 설정 [[running-mixedint8-models-single-gpu-setup]]

필요한 라이브러리를 설치한 후 혼합 8비트 모델을 가져오는 방법은 다음과 같습니다:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

텍스트 생성의 경우:

* `pipeline()` 함수 대신 모델의 `generate()` 메소드를 사용하는 것을 권장합니다. `pipeline()` 함수로는 추론이 가능하지만, 혼합 8비트 모델에 최적화되지 않았기 때문에 `generate()` 메소드를 사용하는 것보다 느릴 수 있습니다. 또한, nucleus 샘플링과 같은 일부 샘플링 전략은 혼합 8비트 모델에 대해 `pipeline()` 함수에서 지원되지 않습니다.
* 입력을 모델과 동일한 GPU에 배치하는 것이 좋습니다.

다음은 간단한 예입니다:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```


### 혼합 Int8 모델 실행 - 다중 GPU 설정 [[running-mixedint8-models-multi-gpu-setup]]

다중 GPU에서 혼합 8비트 모델을 로드하는 방법은 단일 GPU 설정과 동일합니다(동일한 명령어 사용):
```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```
하지만 `accelerate`를 사용하여 각 GPU에 할당할 GPU RAM을 제어할 수 있습니다. 다음과 같이 `max_memory` 인수를 사용하세요:

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True), max_memory=max_memory_mapping
)
```
이 예시에서는 첫 번째 GPU가 1GB의 메모리를 사용하고 두 번째 GPU가 2GB를 사용합니다.

### Colab 데모 [[colab-demos]]

이 방법을 사용하면 이전에 Google Colab에서 추론할 수 없었던 모델에 대해 추론할 수 있습니다.
Google Colab에서 8비트 양자화를 사용하여 T5-11b(42GB in fp32)를 실행하는 데모를 확인하세요:

[![Open In Colab: T5-11b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)

또는 BLOOM-3B에 대한 데모를 확인하세요:

[![Open In Colab: BLOOM-3b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)