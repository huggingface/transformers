<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 단일 GPU에서의 효율적인 훈련 [[efficient-training-on-a-single-gpu]]

이 가이드는 단일 GPU에서 대규모 모델을 효율적으로 훈련하는 데 초점을 맞춥니다. 이러한 방법은 여러 GPU가 있는 컴퓨터에 액세스할 수 있는 경우에도 여전히 유효하지만 [다중-GPU 섹션](perf_train_gpu_many)에 설명된 추가 방법에도 액세스할 수 있습니다.

이 섹션에서는 대규모 모델의 메모리 사용량을 줄이고 훈련 속도를 높이는 몇 가지 트릭과 이들이 [`Trainer`] 및 [🤗 Accelerate](https://huggingface.co/docs/accelerate/)에 통합된 방식을 살펴봅니다. 각 방법은 속도 또는 메모리 사용량을 개선할 수 있으며, 아래 표에서 요약되어 있습니다:

|Method|Speed|Memory|
|:-----|:----|:-----|
| Gradient accumulation | No | Yes |
| Gradient checkpointing | No| Yes |
| Mixed precision training | Yes | (No) |
| Batch size | Yes | Yes |
| Optimizer choice | Yes | Yes |
| DataLoader | Yes | No |
| DeepSpeed Zero | No | Yes |

괄호는 엄격하게는 해당하지 않을 수 있지만 일반적으로 주요 관심사가 아니거나 무시 가능한 경우입니다. 시작하기 전에 다음 라이브러리를 설치했는지 확인하세요:

```bash
pip install transformers datasets accelerate nvidia-ml-py3
```

`nvidia-ml-py3` 라이브러리를 사용하면 Python 내에서 모델의 메모리 사용량을 모니터링할 수 있습니다. 터미널에서 `nvidia-smi` 명령에 익숙할 수 있습니다. 이 라이브러리를 사용하면 Python에서 동일한 정보에 액세스할 수 있습니다.

그런 다음 일부 더미 데이터를 생성합니다. 100에서 30000 사이의 임의의 토큰 ID와 이진 레이블을 생성합니다. 총 512개의 시퀀스가 있으며 각각 길이가 512이고, 이를 PyTorch 형식으로 [`~datasets.Dataset`]에 저장합니다.


```py
import numpy as np
from datasets import Dataset


seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")
```

[`Trainer`]를 사용하여 GPU 이용률 및 훈련 실행에 대한 요약 통계를 출력하려고 합니다. 이를 위해 두 개의 도우미 함수를 설정합니다:

```py
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
```

GPU 메모리에 빈 공간이 있는지 확인해 보겠습니다:

```py
>>> print_gpu_utilization()
GPU memory occupied: 0 MB.
```

잘 보입니다. GPU 메모리가 모델을 로드하기 전에 예상대로 비어 있습니다. 해당 사항이 컴퓨터에서 적용되지 않는 경우 GPU 메모리를 사용하는 모든 프로세스를 중지했는지 확인하세요. 그러나 모든 빈 GPU 메모리를 사용자가 사용할 수 있는 것은 아닙니다. 모델이 GPU에 로드되면 커널도 로드되며, 이는 1-2GB의 메모리를 차지할 수 있습니다. 얼마나 차지하는지 확인하기 위해 작은 텐서를 GPU에 로드하여 커널도 함께 로드합니다.

```py
>>> import torch


>>> torch.ones((1, 1)).to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 1343 MB.
```

커널만으로 GPU 메모리가 1.3GB 차지하는 것을 알 수 있습니다. 이제 모델이 사용하는 공간을 확인해 보겠습니다.

## 모델 로드 [[load-model]]

먼저, `bert-large-uncased` 모델을 로드합니다. 모델 가중치를 직접 GPU에 로드하여 가중치만이 차지하는 공간을 확인할 수 있습니다.


```py
>>> from transformers import AutoModelForSequenceClassification


>>> model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 2631 MB.
```

모델 가중치만으로 GPU 메모리가 1.3GB 차지하는 것을 볼 수 있습니다. 정확한 숫자는 사용하는 특정 GPU에 따라 다를 수 있습니다. 새로운 GPU에서는 모델이 최적화된 방식으로 로드되어 메모리 사용을 빠르게 할 수 있으므로 더 많은 공간을 차지할 수도 있습니다. 이제 `nvidia-smi` CLI와 동일한 결과를 빠르게 확인해 볼 수도 있습니다:


```bash
nvidia-smi
```

```bash
Tue Jan 11 08:58:05 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    39W / 300W |   2631MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3721      C   ...nvs/codeparrot/bin/python     2629MiB |
+-----------------------------------------------------------------------------+
```

이전과 동일한 숫자를 얻으며, 16GB 메모리를 가진 V100 GPU를 사용 중임을 알 수 있습니다. 이제 모델 훈련을 시작하고 GPU 메모리 사용량이 어떻게 변화하는지 확인해 보겠습니다. 먼저, 모든 실험에서 사용할 몇 가지 표준 훈련 인수를 설정합니다:

```py
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```

<Tip>

 참고: 실험 이후 메모리를 제대로 지우기 위해 실험 간에 Python 커널을 재시작해야 합니다. 위의 모든 단계를 실행한 후 아래 실험 중 하나만 실행하세요.

</Tip>

## 일반 훈련 [[vanilla-training]]

첫 번째 실험으로 [`Trainer`]를 사용하여 수정 없이 모델을 훈련하고 배치 크기는 4로 설정합니다:

```py
from transformers import TrainingArguments, Trainer, logging

logging.set_verbosity_error()


training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```

비교적 작은 배치 크기로도 GPU의 전체 메모리를 거의 채웁니다. 그러나 더 큰 배치 크기는 종종 모델 수렴 속도를 높이거나 최종 성능을 향상시킬 수 있습니다. 따라서 이상적으로는 모델의 요구에 맞게 배치 크기를 조정하고 GPU 제한에 맞추지 않는 것이 좋습니다. 흥미로운 점은 모델의 크기보다 훨씬 더 많은 메모리를 사용한다는 것입니다. 이 경우에 대해 좀 더 자세히 이해하기 위해 모델의 작업과 메모리 요구 사항을 살펴보겠습니다.

## 모델 작업의 요소 [[anatomy-of-models-operations]]

Transformer 아키텍처에는 다음과 같이 3가지 주요 작업 그룹이 있습니다.

1. **텐서 연산**

    선형 레이어와 Multi-Head Attention 구성 요소는 모두 배치 **행렬-행렬 곱셈**을 수행합니다. 이러한 작업은 Transformer를 훈련하는 데 가장 계산 집약적인 부분입니다.

2. **통계적 정규화**

    Softmax와 레이어 정규화는 텐서 연산보다 계산 집약적인 작업이 덜하며, 하나 이상의 **축소 연산**을 수행하고 결과를 맵을 통해 적용합니다.

3. **원소별 연산자**

    이러한 연산자에는 **바이어스, 드롭아웃, 활성화 및 잔차 연결**이 포함됩니다. 이러한 연산은 가장 계산 집약적인 작업입니다.

이러한 정보는 성능 병목 현상을 분석할 때 유용할 수 있습니다.

이 요약은 [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)에서 파생되었습니다.


## 모델 메모리의 구조 [[anatomy-of-models-memory]]
모델을 훈련하는 것은 모델을 GPU에 넣는 것보다 훨씬 더 많은 메모리를 사용한다는 것을 알아봤습니다. 이는 훈련 중에 GPU 메모리를 사용하는 여러 구성 요소가 있기 때문입니다. GPU 메모리의 구성 요소는 다음과 같습니다:
1. 모델 가중치
2. 옵티마이저 상태
3. 그래디언트
4. 그래디언트 계산을 위해 저장된 순방향 활성화
5. 임시 버퍼
6. 기능별 메모리

AdamW로 훈련된 일반적인 모델은 활성화 메모리를 포함한 모델 파라미터 당 18바이트가 필요합니다. 추론 과정에서는 옵티마이저 상태와 그래디언트가 없으므로 이를 제외할 수 있습니다. 따라서 혼합 정밀도 추론에는 모델 파라미터 당 6바이트와 활성화 메모리가 필요합니다.

세부 사항을 살펴보겠습니다.

**모델 가중치:**

- fp32 훈련의 경우 파라미터 수 * 4바이트
- 혼합 정밀도 훈련의 경우 파라미터 수 * 6바이트 (메모리에 fp32와 fp16로 된 모델을 유지)

**옵티마이저 상태:**

- 일반 AdamW의 경우 파라미터 수 * 8바이트 (2개의 상태를 유지)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)와 같은 8비트 AdamW 옵티마이저의 경우 파라미터 수 * 2바이트
- 모멘텀이 있는 SGD와 같은 옵티마이저의 경우 파라미터 수 * 4바이트 (상태는 1개만 유지)

**그래디언트**

- fp32 또는 혼합 정밀도 훈련의 경우 파라미터 수 * 4바이트 (그래디언트는 항상 fp32로 유지)

**순방향 활성화**

- 크기는 여러 요소에 따라 다릅니다. 주요한 요소로는 시퀀스 길이, 히든 크기, 배치 크기가 있습니다.

순방향 함수와 역방향 함수에 전달되고 반환되는 입력과 출력, 그리고 그래디언트 계산을 위해 저장된 순방향 활성화가 있습니다.

**임시 메모리**

또한 계산이 완료된 후에 해제되는 모든 종류의 임시 변수가 있지만, 해당 변수는 추가적인 메모리를 필요로 할 수 있으며 OOM으로 이어질 수도 있습니다. 따라서 코딩할 때는 이러한 임시 변수에 대해 전략적으로 생각하고 더 이상 필요하지 않은 경우에 명시적으로 해제하는 것이 중요합니다.

**기능별 메모리**

또한 소프트웨어에는 특수한 메모리 요구 사항이 있을 수 있습니다. 예를 들어, 빔 서치를 사용하여 텍스트를 생성하는 경우 소프트웨어는 입력과 출력의 여러 복사본을 유지해야 합니다.

**`forward` 대 `backward` 실행 속도**

합성곱과 선형 레이어의 경우 역방향에서는 순방향에 비해 2배의 flop이 필요하며, 일반적으로 순방향보다 약 2배 느립니다(크기가 더 어색하기 때문에 때로는 더 느릴 수도 있음). 활성화는 일반적으로 대역폭 제한이며, 역방향에서는 순방향보다 더 많은 데이터를 읽어야 하는 것이 일반적입니다(예: 활성화 순방향은 한 번 읽고 한 번 씁니다. 활성화 역방향은 순방향의 gradOutput 및 output을 두 번 읽고 한 번 씁니다. gradInput).

따라서 GPU 메모리를 절약하거나 작업을 더 빠르게 할 수 있는 몇 가지 장소가 있을 수 있습니다. 첫 번째 간단한 최적화인 적절한 배치 크기 선택부터 시작해 보겠습니다.

## 배치 크기 [[batch-sizes]]

배치 크기와 입력/출력 뉴런 수가 특정한 숫자로 나누어질 때(일반적으로 8에서 시작하지만 훨씬 더 높을 수도 있음), 가장 효율적인 성능을 얻을 수 있습니다. 이 숫자는 사용하는 특정 하드웨어와 모델의 dtype에 따라 많이 다릅니다.

예를 들어, 완전 연결 계층(즉, GEMM에 해당하는 계층)의 경우, NVIDIA는 [입력/출력 뉴런 수](
https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features)와 [배치 크기](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)에 대한 권장 사항을 제공합니다.

[Tensor Core 요구 사항](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)은 dtype과 하드웨어에 따라 곱셈기를 정의합니다. 예를 들어, fp16의 경우 8의 배수가 권장되지만, A100에서는 64가 권장됩니다!

작은 매개변수의 경우 [차원 양자화 효과](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization)도 고려해야 합니다. 여기서 타일링이 발생하고 적절한 곱셈기는 상당한 속도 향상을 가져올 수 있습니다.

## 그래디언트 누적 [[gradient-accumulation]]

그래디언트 누적은 전체 배치에 대한 그래디언트를 한 번에 계산하는 대신 작은 단계로 나누어 그래디언트를 계산하는 아이디어입니다. 이를 위해 모델을 순방향 및 역방향으로 반복적으로 작은 배치로 계산하고 그 과정에서 그래디언트를 누적합니다. 충분한 그래디언트가 누적되면 모델의 최적화 단계를 실행합니다. 이렇게 함으로써 GPU 메모리에 들어갈 수 없는 수의 전체 배치 크기를 쉽게 증가시킬 수 있습니다. 그러나 반복적인 순방향 및 역방향 계산은 훈련을 약간 느리게 할 수 있습니다.

우리는 [`Trainer`]에서 간단히 [`TrainingArguments`]에 `gradient_accumulation_steps` 인자를 추가함으로써 그래디언트 누적을 사용할 수 있습니다. 모델의 메모리 풋프린트에 어떤 영향을 미치는지 확인해 보겠습니다:

```py
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 66.03
Samples/second: 7.75
GPU memory occupied: 8681 MB.
```

메모리 풋프린트가 크게 감소한 반면, 일반 실행보다 약간 느리게 실행되는 것을 볼 수 있습니다. 물론 누적 단계 수를 늘릴수록 이는 변경될 것입니다. 일반적으로 가능한 한 GPU 사용량을 최대로 활용하는 것이 좋습니다. 따라서 우리의 경우, 배치 크기 4는 이미 GPU의 한계에 근접했습니다. 배치 크기 64로 훈련하려면 `per_device_train_batch_size=1` 및 `gradient_accumulation_steps=64` 대신 `per_device_train_batch_size=4` 및 `gradient_accumulation_steps=16`를 사용해야 하며, 이는 동일한 효과적인 배치 크기를 가지면서 사용 가능한 GPU 리소스를 더 잘 활용합니다.

자세한 내용은 [RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537) 및 [A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957)의 벤치마크를 참조하세요.


다음으로는 그래디언트 체크포인팅이라는 조금 더 GPU 메모리를 절약하는 다른 방법을 살펴보겠습니다.

## 그래디언트 체크포인팅 [[gradient-checkpointing]]

배치 크기를 1로 설정하고 그래디언트 누적을 사용하더라도 대형 모델을 처리할 때 메모리 부족 문제가 발생할 수 있습니다. 역전파 도중 그래디언트를 계산하기 위해 순전파의 모든 활성화가 일반적으로 저장됩니다. 이로 인해 큰 메모리 오버헤드가 발생할 수 있습니다. 그러나 순전파 도중 모든 활성화를 잊고 역전파 도중에 필요할 때 다시 계산할 수도 있습니다. 그러나 이는 상당한 계산 오버헤드를 추가하고 훈련을 느리게 할 수 있습니다.

그래디언트 체크포인팅은 이 두 가지 접근법 사이의 타협점을 찾아 계산 그래프 전체에서 전략적으로 선택된 활성화를 저장하여 그래디언트를 위해 다시 계산해야 할 활성화의 일부만 다시 계산합니다. 그래디언트 체크포인팅의 아이디어를 설명하는 [이 훌륭한 글](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)을 참조하세요.

[`Trainer`]에서 그래디언트 체크포인팅을 활성화하려면 단순히 [`TrainingArguments`]에 플래그로 전달하면 됩니다. 그 외의 모든 것은 내부적으로 처리됩니다:

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 85.47
Samples/second: 5.99
GPU memory occupied: 6775 MB.
```

이렇게 하면 메모리를 더 절약할 수 있지만 훈련이 약간 느려집니다. 일반적인 경험적 규칙은 그래디언트 체크포인팅이 훈련을 약 20% 정도 느리게 한다는 것입니다. 이제 속도를 약간 회복시킬 수 있는 다른 방법을 살펴보겠습니다: 혼합 정밀도 훈련.


## 부동 소수점 데이터 유형 [[floating-data-types]]

혼합 정밀도 훈련의 아이디어는 모든 변수를 전체(32비트) 부동 소수점 정밀도로 저장할 필요가 없다는 것입니다. 정밀도를 줄일 수 있다면 변수 및 계산이 더 빠릅니다. 다음은 메모리 사용량과 처리량에 영향을 미치는 일반적으로 사용되는 부동 소수점 데이터 유형입니다:

- fp32 (`float32`)
- fp16 (`float16`)
- bf16 (`bfloat16`)
- tf32 (CUDA 내부 데이터 유형)

이 다이어그램은 이러한 데이터 유형이 어떻게 서로 관련되는지 보여줍니다.

![data types](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tf32-bf16-fp16-fp32.png)
(출처: [NVIDIA 블로그](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/))

fp16과 fp32는 꽤 오랫동안 사용되어 왔지만, bf16과 tf32는 Ampere 아키텍처 GPU에서만 사용할 수 있습니다. TPUs도 bf16을 지원합니다. 우선 가장 일반적으로 사용되는 FP16 훈련부터 시작해 보겠습니다.


### FP16 훈련 [[fp16-training]]

혼합 정밀도 훈련의 아이디어는 모든 변수를 전체(32비트) 부동 소수점 정밀도로 저장할 필요가 없다는 것입니다. 정밀도를 줄일 수 있다면 변수 및 계산이 더 빠릅니다. 주된 이점은 활성화를 절반(16비트) 정밀도로 저장하는 것에서 나옵니다. 그래디언트도 절반 정밀도로 계산되지만 최적화 단계에서 다시 전체 정밀도로 변환되므로 여기에서는 메모리가 절약되지 않습니다. 모델은 GPU에 16비트와 32비트 정밀도 모두로 존재하기 때문에 GPU 메모리를 더 사용할 수 있습니다(GPU에 원래 모델의 1.5배). 특히 작은 배치 크기에 대해서는 그렇습니다. 전체와 절반 정밀도로 계산하는 일부 계산이 섞여 있기 때문에 이 접근법은 혼합 정밀도 훈련이라고도 합니다. 혼합 정밀도 훈련을 활성화하려면 `fp16` 플래그를 `True`로 설정하기만 하면 됩니다:
```py
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 27.46
Samples/second: 18.64
GPU memory occupied: 13939 MB.
```


이렇게 하면 일반 훈련보다 거의 2배 빠릅니다. 이를 이전 방법과 함께 사용해 보겠습니다:


```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 50.76
Samples/second: 10.09
GPU memory occupied: 7275 MB.
```

시작 시점과 비교하여 GPU 메모리를 약 절반 정도 사용하면서 약간 더 빠릅니다.

### BF16 [[bf16]]
Ampere 이상의 하드웨어에 액세스할 수 있다면 훈련 및 평가에 bf16을 사용할 수 있습니다. bf16은 fp16보다 정밀도가 낮지만 훨씬 더 큰 동적 범위를 가지고 있습니다. 따라서 이전에 모델을 훈련하는 동안 오버플로 문제가 발생했다면 bf16은 대부분의 경우 이를 방지합니다. fp16의 경우 최대 숫자는 `65535`이며 그 이상의 숫자는 오버플로됩니다. bf16 숫자는 `3.39e+38` 정도까지 크게 될 수 있으며, 이는 fp32와 거의 동일한 수치 범위입니다 - 각각 숫자 범위에 8비트가 사용됩니다.

🤗 Trainer에서 BF16을 활성화하려면 다음을 사용할 수 있습니다:

```python
TrainingArguments(bf16=True)
```

### TF32 [[tf32]]
Ampere 하드웨어는 tf32라는 마법 같은 데이터 유형을 사용합니다. fp32와 동일한 숫자 범위(8비트)를 가지지만 정밀도는 23비트가 아닌 10비트(즉, fp16과 동일)만 사용하며 총 19비트를 사용합니다.

tf32는 일반적인 fp32 훈련 및/또는 추론 코드를 사용할 수 있으며 tf32 지원을 활성화함으로써 최대 3배의 처리량 향상을 얻을 수 있습니다. 코드에 다음을 추가하기만 하면 됩니다:

```
import torch
torch.backends.cuda.matmul.allow_tf32 = True
```

이렇게 하면 CUDA는 가능한 경우 tf32 대신 fp32를 사용하도록 자동으로 전환합니다. 이는 Ampere 시리즈 GPU를 사용한다는 가정하에 이루어집니다.

모든 정밀도 감소 케이스와 마찬가지로 이는 귀하의 요구에 만족하는지 여부에 따라 다를 수 있으므로 실험을 진행해야 합니다. [NVIDIA 연구](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)에 따르면 대부분의 머신 러닝 훈련은 영향을 받지 않으며, perplexity 및 수렴도 fp32 훈련과 동일하게 나타났습니다.

이미 fp16이나 bf16 혼합 정밀도를 사용하고 있다면 처리량에도 도움이 될 수 있습니다.

🤗 Trainer에서 이 모드를 활성화할 수 있습니다:
```python
TrainingArguments(tf32=True)
```
기본적으로 PyTorch의 기본값을 사용합니다.

참고: tf32 모드는 CUDA 내부적으로 사용되는 것이므로 `tensor.to(dtype=torch.tf32)`와 같이 직접 액세스할 수 없습니다(`torch.tf32`가 존재하지 않음).

참고: 이 기능을 사용하려면 `torch>=1.7`이 필요합니다.

tf32와 다른 정밀도 간의 다양한 벤치마크를 확인할 수도 있습니다:
[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803) 및
[A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004543189).

지금까지 부동 소수점 유형을 변경하여 처리량을 증가시킬 수 있는 방법을 살펴보았지만, 아직 끝나지 않았습니다! GPU 메모리를 더 절약할 수 있는 또 다른 영역이 있습니다: 옵티마이저.

## 옵티마이저 [[optimizer]]

Transformer 모델을 훈련시키는 데 가장 일반적으로 사용되는 옵티마이저는 Adam 또는 AdamW(가중치 감쇠를 적용한 Adam)입니다. Adam은 이전 그래디언트의 롤링 평균을 저장함으로써 좋은 수렴을 달성하지만, 이는 모델 파라미터의 수에 비례하여 추가적인 메모리 사용량이 발생합니다. 이를 해결하기 위한 한 가지 방법은 Adafactor와 같은 대체 옵티마이저를 사용하는 것입니다. Adafactor는 일부 모델에 대해 잘 작동하지만 종종 불안정성 문제가 있을 수 있습니다.

HF Trainer는 사용 가능한 다양한 옵티마이저를 내장하고 있습니다. 원하는 옵티마이저를 활성화하려면 간단히 명령줄에 `--optim` 플래그를 전달하면 됩니다.

현재 지원되는 옵티마이저를 확인하려면:

```bash
$ python examples/pytorch/translation/run_translation.py -h | grep "\-optim"
         [--optim {adamw_hf,adamw_torch,adamw_torch_xla,adamw_apex_fused,adafactor}]
```

예를 들어, [NVIDIA/apex](https://github.com/NVIDIA/apex)가 설치되어 있다면 `--optim adamw_apex_fused`는 지원되는 모든 AdamW 옵티마이저 중 가장 빠른 훈련 경험을 제공합니다.

반면에 [8bit BNB optimizer](https://github.com/TimDettmers/bitsandbytes)는 모든 옵티마이저 상태를 양자화하도록 구성하면 일반적인 AdamW 옵티마이저가 사용하는 메모리의 3/4를 절약할 수 있지만, 일부 상황에서는 일부 옵티마이저 상태만 양자화되어 더 많은 메모리가 사용될 수 있습니다.

이제 숫자를 확인하고 `t5-3b`와 같이 30억 개의 파라미터를 가진 모델을 예로 사용해 보겠습니다. 기가바이트는 10억 바이트와 대응되므로 파라미터 수(십억 개)에 파라미터 당 필요한 바이트 수를 곱하여 GPU 메모리 사용량(기가바이트)을 얻을 수 있습니다:

- 표준 AdamW는 각 파라미터당 8바이트를 사용하므로, 여기서 옵티마이저는 (`8*3`) 24GB의 GPU 메모리를 필요로 합니다.
- Adafactor는 약 4바이트 이상을 사용하므로, (`4*3`) 12GB 이상입니다.
- 8bit BNB 양자화된 옵티마이저는 모든 옵티마이저 상태가 양자화되는 경우에만 (`2*3`) 6GB를 사용합니다.

먼저 Adafactor를 살펴보겠습니다.

### Adafactor [[adafactor]]

Adafactor는 가중치 행렬의 각 요소에 대한 롤링 평균을 유지하는 대신 집계된 정보(롤링 평균의 행 및 열 합계)만 저장하여 메모리 사용량을 크게 줄입니다. Adafactor의 단점 중 하나는 일부 경우에 Adam보다 수렴 속도가 느릴 수 있다는 점이므로 여기에서는 몇 가지 실험을 권장합니다. Adafactor를 사용하려면 간단히 `optim="adafactor"`로 설정하면 됩니다:


```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 64.31
Samples/second: 7.96
GPU memory occupied: 12295 MB.
```

이로써 GPU에서 몇 GB를 더 절약할 수 있습니다. 앞서 소개한 다른 방법들에 이것을 추가해 보겠습니다:


```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    optim="adafactor",
    **default_args,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 56.54
Samples/second: 9.06
GPU memory occupied: 4847 MB.
```

15GB의 메모리 사용량에서 5GB로 줄어들었습니다. 즉, 처리량을 유지하면서 3배의 향상이 이루어졌습니다! 그러나 앞에서 언급했듯이 Adafactor의 수렴은 Adam보다 더 나쁠 수 있습니다. Adafactor의 대체제인 8-bit Adam이라는 대안이 있습니다.

### 8-bit Adam [[8bit-adam]]

Adafactor와 달리 8-bit Adam은 옵티마이저 상태를 집계하는 대신 전체 상태를 유지하고 양자화합니다. 양자화는 낮은 정밀도로 상태를 저장하고 최적화를 위해서만 상태를 역양자화하는 것을 의미합니다. 이는 FP16 훈련의 아이디어와 유사하며 낮은 정밀도의 변수를 사용하여 메모리를 절약하는 것입니다.

이전 접근 방식과 달리 이 방식은 단순한 플래그로 [`Trainer`]에 통합되지 않습니다. 8-bit Adam 옵티마이저를 설치한 다음 이를 커스텀 옵티마이저로 [`Trainer`]에 전달해야 합니다. 8-bit Adam 옵티마이저를 구현한 `bitsandbytes` 라이브러리를 설치하기 위해 Github [repo](https://github.com/TimDettmers/bitsandbytes)의 설치 가이드를 따르십시오.

설치한 후 옵티마이저를 초기화하기만 하면 됩니다. 이는 상당한 작업량처럼 보이지만 사실상 두 단계만 필요합니다: 첫 번째로 모델의 파라미터를 가중치 감쇠를 적용할 그룹과 적용하지 않을 그룹으로 나눠야 합니다. 보통 바이어스와 레이어 정규화 파라미터는 가중치 감쇠되지 않습니다. 그런 다음 두 번째 단계에서는 이전에 사용한 AdamW 옵티마이저와 동일한 파라미터를 사용하기 위해 몇 가지 인수 설정을 수행합니다.

<Tip>
기존 사전 훈련된 모델에 8-bit 옵티마이저를 사용하려면 임베딩 레이어에 변경이 필요합니다.
자세한 내용은 [이 이슈](https://github.com/huggingface/transformers/issues/14819)를 참조하세요.
</Tip>

```py
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
```

이제 커스텀 옵티마이저를 `Trainer`의 인수로 전달할 수 있습니다:
```py
trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
result = trainer.train()
print_summary(result)
```

```
Time: 55.95
Samples/second: 9.15
GPU memory occupied: 13085 MB.
```

Adafactor와 유사한 메모리 개선이 이루어지면서 그래디언트의 전체 롤링 평균을 유지하는 것을 확인할 수 있습니다. 전체 설정으로 실험을 반복해 보겠습니다:

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
result = trainer.train()
print_summary(result)
```

```
Time: 49.46
Samples/second: 10.35
GPU memory occupied: 5363 MB.
```

다시한번 3배 정도의 메모리 개선과 심지어 Adafactor보다 약간 더 높은 처리량을 얻었습니다. 이렇게 우리는 대형 모델의 메모리 사용량을 최적화하는 방법을 살펴보았습니다. 다음 그래프는 우리의 모든 실험을 요약한 것입니다:

![png](https://huggingface.co/datasets/lvwerra/repo-images/raw/main/gpu-memory-savings.png)

### `_multi_tensor` [[multitensor]]
pytorch-nightly에서는 작은 특성 텐서가 많은 상황에 대한 옵티마이저의 속도를 크게 향상시킬 것으로 기대되는 `torch.optim._multi_tensor`가 도입되었습니다. 기본값이 될 예정이지만 더 일찍 실험해 보고 최신 버전을 사용해도 상관없다면 다음을 참조하세요: https://github.com/huggingface/transformers/issues/9965


## Using 🤗 Accelerate [[using-accelerate]]

지금까지 우리는 [`Trainer`]를 사용하여 실험을 실행했지만, 더 유연한 대안은 🤗 Accelerate를 사용하는 것입니다. 🤗 Accelerate를 사용하면 훈련 루프를 완전한 PyTorch로 작성하고 일부 작은 수정을 통해 원하는 대로 제어할 수 있습니다. 또한 코드를 변경하지 않고 CPU, GPU, TPU 또는 분산 다중 GPU 설정과 같은 다양한 인프라에서 손쉽게 확장할 수 있습니다. 이제 🤗 Accelerate에서 위에서 소개한 모든 최적화를 구현하는 데 필요한 내용을 살펴보겠습니다. 여전히 [`TrainingArguments`]를 사용하여 훈련 설정을 래핑할 수 있습니다:


```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)
```

🤗 Accelerate를 사용한 전체 예제 훈련 루프는 몇 줄의 코드로 이루어져 있습니다:


```py
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

model.train()
for step, batch in enumerate(dataloader, start=1):
    loss = model(**batch).loss
    loss = loss / training_args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

먼저 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)에서 데이터셋을 래핑합니다. 그런 다음 모델의 [`~PreTrainedModel.gradient_checkpointing_enable`] 메소드를 호출하여 그래디언트 체크포인팅을 활성화할 수 있습니다. [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator)를 초기화할 때, 혼합 정밀도 훈련을 사용하려는지 지정할 수 있으며, [`prepare`] 호출에서 자동으로 처리해줍니다. [`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare) 호출 중에는 데이터로더도 여러 GPU를 사용하는 경우 워커들 간에 분산되어 처리됩니다. 앞서 실험에서 사용한 8-bit 옵티마이저를 사용합니다.

마지막으로, 주요 훈련 루프를 작성할 수 있습니다. `backward` 호출은 🤗 Accelerate가 처리합니다. 그리고 그래디언트 누적이 어떻게 작동하는지 확인할 수 있습니다: 누적 동안 손실을 정규화하여 누적 끝에 평균을 얻고, 충분한 스텝이 모이면 최적화를 실행합니다. 이제 질문은 이전 단계와 같은 메모리 양을 사용하는지 확인해야 합니다. 확인해 보겠습니다:


```py
>>> print_gpu_utilization()
GPU memory occupied: 5363 MB.
```

실제로 그렇습니다. 🤗 Accelerate를 사용하여 이러한 최적화 기법을 구현하는 것은 몇 줄의 코드만 필요하며, 훈련 루프에서 더 유연성을 제공하는 이점이 있습니다. 모든 기능에 대한 자세한 설명은 [Accelerate 문서](https://huggingface.co/docs/accelerate/index)를 참조하세요.

## DataLoader [[dataloader]]

훌륭한 훈련 속도를 달성하기 위한 중요한 요구 사항 중 하나는 GPU가 처리할 수 있는 최대 속도로 데이터를 공급하는 능력입니다. 기본적으로 모든 작업이 주 프로세스에서 수행되며, 데이터를 디스크에서 충분히 빠르게 읽지 못할 수 있고 이로 인해 병목 현상이 발생하여 GPU의 활용도가 떨어질 수 있습니다.

- `DataLoader(pin_memory=True, ...)`는 데이터를 CPU의 고정된 메모리에 미리로드하므로 CPU에서 GPU 메모리로의 전송이 훨씬 빨라집니다.
- `DataLoader(num_workers=4, ...)` - 여러 워커를 생성하여 데이터를 더 빨리 미리로드합니다 - 훈련 중 GPU 활용도 통계를 확인하고 100%에 가깝지 않은 경우 워커 수를 늘리는 실험을 진행합니다. 물론, 문제가 다른 곳에 있을 수 있으므로 많은 워커 수는 반드시 더 나은 성능을 보장하지 않습니다.

## DeepSpeed ZeRO [[deepspeed-zero]]

Deepspeed를 사용하는 방법에 대한 자세한 내용은 [여기](main_classes/deepspeed)에서 찾을 수 있습니다.

먼저, 간단한 결정 트리입니다:

1. 모델이 단일 GPU에 맞고 작은 배치 크기를 수용할 공간이 충분한 경우 - 이 경우 Deepspeed를 사용할 필요가 없으며 오히려 성능이 저하될 수 있습니다.
2. 모델이 단일 GPU에 맞지 않거나 작은 배치를 수용할 수 없는 경우 - DeepSpeed ZeRO + CPU Offload 및 더 큰 모델의 경우 NVMe Offload를 사용하세요.

이제 결정 트리가 DeepSpeed를 사용하라고 제안했다면, 먼저 [설치](main_classes/deepspeed#installation)하고, 구성 파일을 생성하고 DeepSpeed를 실행하는 다음 가이드 중 하나를 따르세요.

활성화:

- HF Trainer-based examples: see this [guide](main_classes/deepspeed#deployment-with-one-gpu).
- Custom HF Trainer-based program: Same as above, but pass:

    ```python
    TrainingArguments(deepspeed="/path/to/ds_config.json")
    ```
- 노트북에서 배포: 이 [가이드](main_classes/deepspeed#deployment-in-notebooks)를 참조하세요.

- 사용자 지정 훈련 루프: 이는 다소 복잡하지만 [HF Trainer](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py)에서 구현되는 방법을 공부할 수 있습니다. 
코드에서 `deepspeed`를 검색하면 됩니다.


## GPU 선택 [[choice-of-gpu]]
가끔씩 위에서 소개한 모든 최적화를 적용해도 특정 GPU에서의 처리량이 여전히 충분하지 않을 수 있습니다. 간단한 해결책은 GPU의 유형을 변경하는 것입니다. 예를 들어, Google Colab에서 얻는 것과 같이 K80에서 더 고급스러운 GPU인 V100 또는 A100으로 전환하는 것입니다. 비록 더 비싸지만, 큰 메모리와 빠른 아키텍처로 인해 일반적으로 더 비용 효율적입니다.

이제, 큰 모델의 훈련을 확장할 때 최적화해야 할 사항에 대해 되돌아보고 알아보겠습니다.

## 확장 방법 [[how-to-scale]]

모델을 훈련할 때 동시에 최적화해야 할 두 가지 측면이 있습니다:

- 데이터 처리량 / 훈련 시간
- 모델 성능

각각의 방법은 메모리 사용량과 처리량을 변경시킵니다. 일반적으로 우리는 처리량(샘플/초)을 최대화하여 훈련 비용을 최소화하려고 합니다. 이는 GPU를 최대한 활용하여 GPU 메모리를 최대한 활용하는 것으로 일반적으로 달성됩니다. 예를 들어, 앞서 언급한대로 우리는 GPU 메모리의 크기를 넘어서는 배치 크기를 사용하려고 할 때에만 그래디언트 누적을 사용합니다. 원하는 배치 크기가 메모리에 맞는 경우에는 그래디언트 누적을 적용할 필요가 없으며, 이는 훈련을 느리게만 할 것입니다.

두 번째 목표는 모델 성능입니다. 우리가 할 수 있다고 해서 반드시 큰 배치 크기를 사용해야 하는 것은 아닙니다. 하이퍼파라미터 튜닝의 일부로 최상의 결과를 얻는 배치 크기를 결정하고, 그에 따라 처리량을 최적화해야 합니다.


## 효율적인 소프트웨어 빌드 [[efficient-software-prebuilds]]

PyTorch의 [pip 및 conda 빌드](https://pytorch.org/get-started/locally/#start-locally)는 PyTorch를 실행하는 데 충분한 cuda toolkit과 함께 사전 빌드되어 있습니다. 

그러나 cuda 확장 프로그램을 빌드해야 하는 경우에는 추가 노력이 필요할 수 있습니다. 예를 들어, `apex`와 같이 사전 컴파일되지 않은 라이브러리를 사용하는 경우에는 몇 가지 구성해야 할 사항이 있을 수 있습니다. 또한, 올바른 cuda toolkit을 시스템 전체에 설치하는 방법을 찾는 것은 복잡할 수 있습니다. 이러한 사용자의 요구를 충족시키기 위해 PyTorch와 NVIDIA는 NGC 도커 컨테이너의 새 버전을 릴리스합니다. 이 도커 컨테이너는 모든 것이 사전 빌드되어 있으며, 프로그램을 설치하면 그대로 실행할 수 있습니다.

이 접근 방식은 PyTorch 소스를 수정하거나 사용자 정의 빌드를 만들려는 경우에도 유용합니다.

원하는 도커 이미지 버전을 찾으려면 [여기](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)에서 시작하여 최신 월간 릴리스 중 하나를 선택하세요. 원하는 릴리스의 릴리스 노트로 이동하고 환경 구성 요소가 필요한 사항과 일치하는지 확인한 다음 해당 문서의 맨 위로 이동하여 해당 NGC 페이지로 이동하세요. 길을 잃는 경우를 대비해 [PyTorch NGC 이미지의 인덱스](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)도 제공됩니다.

다음은 도커 이미지를 다운로드하고 배포하는 지침을 따르면 됩니다.

## 희소성 [[sparsity]]

### Mixture of Experts [[mixture-of-experts]]

최근 논문들 중 상당수가 Mixture of Experts(MoE)를 Transformer 모델에 통합하여 4-5배의 훈련 속도 향상과 빠른 추론을 보고했습니다.

더 많은 매개변수가 더 좋은 성능을 낸다는 것을 발견한 이 기술은 훈련 비용을 증가시키지 않고 매개변수 수를 10배로 늘릴 수 있도록 합니다.


이 접근 방식에서는 각 FFN(Feed-Forward Network) 레이어를 MoE 레이어로 대체합니다. MoE 레이어는 많은 전문가로 구성되어 있으며, 각 전문가는 입력 토큰의 위치에 따라 균형 잡힌 방식으로 훈련됩니다.

![MoE Transformer 2x block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf-moe-transformer.png)

(출처: [GLAM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html))

이 섹션 끝에 나열된 논문들에서 상세한 내용과 비교 표를 찾을 수 있습니다.

이 접근 방식의 주요 단점은 거대한 양의 GPU 메모리를 필요로 한다는 것입니다. 거의 밀도가 있는 동등한 모델보다 한 차원 더 큽니다. 훨씬 높은 메모리 요구 사항을 극복하기 위한 다양한 증류(distillation) 및 접근 방식이 제안되었습니다.

그러나 직접적인 트레이드오프가 있으며, 수십 개 또는 수백 개의 전문가가 아닌 몇 개의 전문가와 2-3배 작은 기본 모델을 사용할 수 있으므로 전체 모델 크기를 5배 줄일 수 있으며, 훈련 속도를 적당히 늘리고 메모리 요구 사항도 적당히 증가시킬 수 있습니다.

대부분의 관련 논문과 구현은 Tensorflow/TPU를 기반으로 합니다:

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model (GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

PyTorch에서도 DeepSpeed가 구현한 [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596), [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)와 관련된 블로그 포스트와 큰 Transformer 기반 자연어 생성 모델을 위한 특정 배포: [블로그 포스트](https://www.deepspeed.ai/news/2021/12/09/deepspeed-moe-nlg.html), [Megatron-Deepspeed 브랜치](Thttps://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training)도 있습니다.


## 단일 GPU를 초과한 확장 [[scaling-beyond-a-single-gpu]]

대규모 언어 모델의 사전 훈련과 같은 일부 애플리케이션에서는 위에서 언급한 모든 방법을 적용해도 여전히 충분히 빠르지 않을 수 있습니다. 이 경우 실험을 여러 GPU로 확장해야 합니다.

또 다른 다중 GPU에서 훈련하는 사용 사례는 모델이 위에서 언급한 모든 기법을 사용하여 단일 GPU에 맞지 않는 경우입니다. 이 경우 더 많은 방법을 적용할 수 있지만 약간 더 복잡해집니다. 일반적으로 모델 자체가 여러 GPU에 분산되는 파이프라인 또는 텐서 병렬성의 형태로 이루어지며, 이러한 병렬성 전략과 메모리 풋프린트를 줄이기 위한 몇 가지 추가적인 최적화를 구현한 DeepSpeed를 활용할 수도 있습니다. DeepSpeed에 대해서는 ["다중 GPU 훈련" 섹션](perf_train_gpu_many)에서 자세히 읽어볼 수 있습니다.

## PyTorch 네이티브 어텐션 사용 [[using-pytorch-native-attention]]

PyTorch 2.0에서는 네이티브 [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA)를 발표했습니다. 이를 사용하면 GPU에 퓨즈드 커널을 사용하는 [메모리 효율적인 어텐션](https://arxiv.org/abs/2112.05682) 및 [플래시 어텐션](https://arxiv.org/abs/2205.14135)을 사용할 수 있습니다.

[`optimum`](https://github.com/huggingface/optimum) 패키지를 설치한 후, 관련 내부 모듈을 다음과 같이 교체하여 PyTorch의 네이티브 어텐션을 사용할 수 있습니다:

```python
model = model.to_bettertransformer()
```

그런 다음 훈련은 일반적인 방식으로 진행할 수 있습니다.

## torch.compile 사용 [[using-torchcompile]]

PyTorch 2.0에서는 새로운 컴파일 함수를 도입했으며, 이에 대한 자세한 내용은 [공식 문서](https://pytorch.org/get-started/pytorch-2.0/)에서 확인할 수 있습니다. 이 함수는 Python의 프레임 평가 API를 사용하여 기존 PyTorch 프로그램으로부터 그래프를 자동으로 생성합니다. 그래프를 캡처한 후에는 다양한 백엔드를 배포하여 그래프를 최적화된 엔진으로 변환할 수 있습니다. 다음 중 하나의 옵션을 선택하여 성능을 향상시킬 수 있습니다.

`torch.compile`은 점점 늘어나는 백엔드 목록을 가지고 있으며, 이는 [backends.py](https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/backends.py)에서 찾을 수 있습니다. 또는 `torchdynamo.list_backends()`를 사용하여 선택적 종속성과 함께 각 백엔드를 확인할 수 있습니다.


가장 일반적으로 사용되는 일부 백엔드는 다음과 같습니다.

**디버깅 백엔드**:
* `dynamo.optimize("eager")` - 추출된 GraphModule을 실행하기 위해 PyTorch를 사용합니다. 이는 TorchDynamo 문제를 디버깅하는 데 유용합니다.
* `dynamo.optimize("aot_eager")` - 컴파일러 없이 AotAutograd와 함께 PyTorch eager를 사용하여 AotAutograd의 추출된 forward 및 backward 그래프를 실행합니다. 이는 디버깅에 유용하지만 속도 향상은 기대하기 어렵습니다.

**훈련 및 추론 백엔드**:
* `dynamo.optimize("inductor")` - codegened Triton 커널을 활용하여 AotAutograd 및 cudagraphs와 함께 TorchInductor 백엔드를 사용합니다. [더 알아보기](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `dynamo.optimize("nvfuser")` - TorchScript를 사용한 nvFuser입니다. [더 알아보기](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_nvfuser")` - AotAutograd를 사용한 nvFuser입니다. [더 알아보기](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_cudagraphs")` - AotAutograd를 사용한 cudagraphs입니다. [더 알아보기](https://github.com/pytorch/torchdynamo/pull/757)

**추론 전용 백엔드**:
* `dynamo.optimize("ofi")` - Torchscript의 optimize_for_inference를 사용합니다. [더 알아보기](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `dynamo.optimize("fx2trt")` - 추론 최적화를 위해 Nvidia TensorRT를 사용합니다. [더 알아보기](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
* `dynamo.optimize("onnxrt")` - CPU/GPU에서 추론을 위해 ONNXRT를 사용합니다. [더 알아보기](https://onnxruntime.ai/)
* `dynamo.optimize("ipex")` - CPU에서 추론을 위해 IPEX를 사용합니다. [더 알아보기](https://github.com/intel/intel-extension-for-pytorch)


$hf_i18n_placeholder168