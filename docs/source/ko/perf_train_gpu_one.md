<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPU[[gpu]]

GPU는 높은 메모리 대역폭과 병렬 처리 능력 덕분에 딥러닝 모델 학습에 널리 사용됩니다. GPU와 모델 크기에 따라 수십억 개의 매개변수를 가진 모델도 학습할 수 있습니다. 핵심은 GPU 메모리 사용량(데이터 처리량/학습 시간)과 학습 속도 간의 균형을 잘 맞추는 것입니다.

이 가이드는 GPU에서 효율적으로 모델을 학습하기 위해 Transformers와 PyTorch에서 제공하는 기능을 설명합니다. 대부분의 경우, 여러 기능을 함께 사용해 학습을 최적화하는 것이 좋습니다.

아래 표를 참고하시면 학습 상황에 적합한 기능을 빠르게 확인하실 수 있습니다.

| 기능                              | 학습 속도 | 메모리 사용 |
| --------------------------------- | --------- | ----------- |
| 배치 크기                         | 예        | 예          |
| 그래디언트 누적                   | 아니요    | 예          |
| 그래디언트 체크포인팅             | 아니요    | 예          |
| 혼합 정밀도                       | 예        | 경우에 따라 |
| 옵티마이저                        | 예        | 예          |
| 데이터 사전 적재                  | 예        | 아니요      |
| torch_empty_cache_steps           | 아니요    | 예          |
| torch.compile                     | 예        | 아니요      |
| 스케일드 닷 프로덕션 어텐션(SDPA) | 예        | 예          |

## Trainer[[trainer]]

[Trainer](./trainer)는 [`TrainingArguments`]를 통해 설정할 수 있는 다양한 유용한 학습 기능을 지원합니다. 이 섹션에서는 학습을 최적화하는 데 특히 중요한 기능들을 소개합니다.

### 배치 크기[[batch-size]]

배치 크기는 GPU 학습 효율을 결정하는 가장 중요한 하이퍼파라미터 중 하나로, 메모리 사용량과 학습 속도에 영향을 줍니다. 더 큰 배치 크기는 GPU의 병렬 처리 능력을 활용하기 때문에 더 빠른 학습을 제공합니다. 8, 64, 128, 256, 512처럼 2의 거듭제곱 크기를 사용하는 것이 권장됩니다. 배치 크기는 GPU와 모델의 데이터 타입에 따라 달라집니다.

[`TrainingArguments`]의 [`~TrainingArguments.per_device_train_batch_size`]를 설정하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
)
```

입력 피처, 출력 뉴런 수, 배치 크기가 성능에 어떻게 영향을 미치는지에 대해서는 NVIDIA Performance 가이드를 참고하세요. 이는 GPU가 수행하는 일반 행렬 곱(GEMM)과 관련이 있으며, 매개변수가 클수록 병렬화와 효율성이 높아집니다.

Tensore Core Requirements도 데이터 타입과 GPU에 따라 텐서 곱셈 속도를 극대화할 수 있는 배치 크기 선택에 유용합니다. 예를 들어, fp16에서는 8의 배수가, A100 GPU에서는 64의 배수가 권장됩니다.

마지막으로, 작은 매개변수에서는 Dimension Quantization Effects를 고려하세요. 행렬 차원이 GPU의 스레드 블록 타일 크기로 나누어떨어지지 않으면 GPU 자원이 충분히 활용되지 못할 수 있습니다. 적절한 배치 크기를 선택해 행렬이 타일 크기로 나누어지도록 하면 학습 속도를 크게 높일 수 있습니다.

### 그래디언트 누적[[gradient-accumulation]]

그래디언트 누적은 메모리 제약을 극복해 단일 GPU에 맞지 않는 큰 모델을 학습할 수 있게 합니다. 여러 미니 배치에서 그래디언트를 누적한 뒤 파라미터를 업데이트하기 때문에 메모리를 절약하고 더 큰 유효 배치 크기로 학습할 수 있습니다. 다만, 추가적인 순전파 및 역전파가 필요하므로 학습 속도는 느려질 수 있습니다.

[TrainingArguments]의 [~TrainingArguments.per_device_train_batch_size]를 설정해 활성화하세요.

```python
from transformers import TrainingArguments

# 유효 배치 크기 64
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
)
```

### 그래디언트 체크포인팅[[gradient-checkpointing]]

그래디언트 체크포인팅은 역전파 시 일부 중간 활성화만 저장하고 나머지는 다시 계산해 메모리 사용량을 줄입니다. 이렇게 하면 순전파의 모든 중간 활성화를 저장하지 않아도 되어 메모리 오버헤드를 줄일 수 있습니다. 단, 약 20% 정도 학습 속도가 느려집니다.

[TrainingArguments]의 [~TrainingArguments.gradient_checkpointing]를 설정해 활성화하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
)
```

### 혼합 정밀도[[mixed-precision]]

혼합 정밀도는 일부 연산을 반정밀도(fp16)로, 일부를 전정밀도(fp32)로 처리해 학습 속도를 높입니다. 반정밀도 연산은 전정밀도에 비해 연산 비용이 적어 빠르고, 일부를 전정밀도로 유지해 정확도를 보장합니다.

혼합 정밀도 학습에는 여러 데이터 타입이 있습니다.

<hfoptions id="mixed-precision"> <hfoption id="fp16">
혼합 정밀도 학습의 주요 이점은 활성화를 fp16으로 저장하는 것입니다.

[TrainingArguments]의 [~TrainingArguments.fp16]를 설정해 fp16 데이터 타입으로 혼합 정밀도 학습을 활성화하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True.
)
```

fp16은 메모리 최적화가 되지 않을 수 있습니다. 최적화 단계에서 fp16으로 계산된 그래디언트가 fp32로 변환되므로 특히 작은 배치 크기에서는 GPU 메모리가 더 많이 사용될 수 있습니다.

</hfoption> <hfoption id="bf16">
bf16은 일부 정밀도를 희생해 더 넓은 동적 범위를 제공하여 오버플로/언더플로 오류를 방지합니다. fp16과 달리 손실 스케일링 없이 사용할 수 있습니다. bf16은 NVIDIA Ampere 이상의 아키텍처에서 지원됩니다.

[TrainingArguments]의 [~TrainingArguments.bf16]를 설정해 bf16 데이터 타입으로 혼합 정밀도 학습을 활성화하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
)
```

</hfoption> <hfoption id="tf32">
tf32는 NVIDIA Ampere GPU에서 합성곱과 행렬 곱 입력을 tf32로 변환합니다. 나머지 연산과 저장은 fp32로 유지됩니다. tf32는 fp32와 동일한 범위를 유지하면서 fp16의 정밀도를 제공하고 bf16보다 정밀합니다. tf32를 fp16이나 bf16과 혼합하면 처리량을 최대 16배까지 높일 수 있습니다.

NVIDIA Ampere GPU에서는 기본적으로 활성화되어 있지만 아래와 같이 명시적으로 설정할 수도 있습니다.

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

[TrainingArguments]의 tf32()를 설정해 tf32 모드로 혼합 정밀도 학습을 활성화하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True.
    tf32=True,
)
```

</hfoption> </hfoptions>

### 옵티마이저[[optimizers]]

Transformers는 기본적으로 PyTorch의 AdamW (adamw_torch) 옵티마이저를 사용합니다. 과거 그래디언트의 가중 평균을 저장하기 때문에 모델 매개변수 수에 비례해 메모리가 추가로 필요합니다. 큰 모델을 학습할 때는 다른 옵티마이저를 고려하세요. 예를 들어, Apex를 설치했다면 adamw_apex_fused를 사용해 가장 빠른 학습 속도를 얻을 수 있습니다.

[TrainingArguments]의 [~TrainingArguments.optim]을 설정해 옵티마이저를 선택하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit"
)
```

학습 시나리오에 따라 다양한 옵티마이저가 있습니다. (전체 목록은 OptimizerNames를 참고하세요.) 예를 들어, Adafactor는 메모리를 절약하지만 수렴 속도가 느립니다. 또 다른 예로 8-bit AdamW는 옵티마이저 상태를 양자화해 메모리를 줄입니다.

더 많은 특화된 옵티마이저에 대해서는 optimizer 가이드를 참고하세요.

### 데이터 사전 적재[[data-preloading]]

데이터 사전 적재는 CPU에서 데이터를 미리 준비해 GPU가 지속적으로 작업하도록 하여 유휴 시간을 줄이고 활용도를 높입니다. 이를 위해 두 가지 방법이 있습니다.

1. CPU에 핀 메모리를 할당해 데이터를 직접 GPU로 전송합니다.
2. CPU 스레드나 작업자(worker) 수를 늘려 데이터를 더 빠르게 적재합니다.

[TrainingArguments]의 [~TrainingArguments.dataloader_pin_memory]와 [~TrainingArguments.dataloader_num_workers]를 설정해 핀 메모리와 작업자 수를 조절하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)
```

## PyTorch[[pytorch]]

PyTorch는 메모리 사용량을 줄이고 학습 속도를 높이는 여러 기능을 제공합니다. 이들은 Transformers에서 몇 줄의 코드만 추가해도 사용할 수 있습니다.

### torch.empty_cache_steps[[torch-empty-cache-steps]]

torch.cuda.empty_cache 함수는 사용하지 않는 캐시 메모리를 해제해 OOM(메모리 부족) 오류를 방지하지만 학습 속도가 약 10% 느려집니다.

[TrainingArguments]의 torch_empty_cache_steps()를 설정해 일정 스텝마다 실행하도록 설정하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    torch_empty_cache_steps=4,
)
```

### torch.compile[[torch-compile]]

torch.compile은 PyTorch 코드를 최적화된 커널로 컴파일해 학습 속도를 높입니다. TorchDynamo가 Frame Evaluation API를 이용해 그래프를 캡처한 뒤, 이를 백엔드별 최적화된 커널로 컴파일합니다.

[TrainingArguments]의 [~TrainingArguments.torch_compile]를 설정해 활성화하고, torch_compile_backend()로 백엔드를 선택하세요.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    torch_empty_cache_steps=4,
    torch_compile=True,
    torch_compile_backend="inductor"
)
```

아래 표를 참고해 학습 상황에 맞는 백엔드를 선택하세요.

| 백엔드         | 설명                                                                                                                         | 목적         |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------ |
| eager          | PyTorch로 GraphModule 실행                                                                                                   | 디버깅       |
| aot_eager      | AOTAutograd 그래프를 PyTorch eager 모드로 실행                                                                               | 디버깅       |
| inductor       | TorchInductor와 CUDA Graphs 사용                                                                                             | 학습 및 추론 |
| nvfuser        | nvFuser와 TorchScript 사용                                                                                                   | 학습 및 추론 |
| aot_nvfuser    | AOTAutograd와 nvFuser 사용                                                                                                   | 학습 및 추론 |
| aot_cudagraphs | AOTAutograd와 CUDA Graphs 사용                                                                                               | 학습 및 추론 |
| ofi            | TorchScript의 [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html) 사용 | 추론         |
| fx2trt         | [Torch-TensorRT](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html) 사용                              | 추론         |
| onnxrt         | [ONNX-RT](https://onnxruntime.ai/) 사용                                                                                      | 추론         |
| ipex           | [IPEX](https://github.com/intel/intel-extension-for-pytorch) 사용                                                            | 추론         |

### 스케일드 닷 프로덕션 어텐션[[scaled-dot-production-attention]]

torch.nn.functional.scaled_dot_product_attention (SDPA)는 스케일드 닷 프로덕션 어텐션 메커니즘의 PyTorch 네이티브 구현입니다. SDPA는 기존 어텐션 메커니즘보다 효율적이고 최적화되어 있으며, 세 가지 구현을 지원합니다.

FlashAttention2: fp16 또는 bf16으로 모델을 캐스팅하면 자동으로 활성화됩니다.

xFormers 또는 메모리 효율적인 어텐션: fp32를 지원합니다.

C++로 구현된 스케일드 닷 프로덕션 어텐션.

PyTorch 2.1.1+에서는 기본으로 활성화되어 있지만, [~PreTrainedModel.from_pretrained]에서 attn_implementation="sdpa"로 명시할 수도 있습니다.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", attn_implementation="sdpa")
```
