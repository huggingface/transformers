<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 모델 학습 해부하기 [[model-training-anatomy]]

모델 훈련 속도와 메모리 활용의 효율성을 향상시키기 위해 적용할 수 있는 성능 최적화 기술을 이해하려면 GPU가 훈련 중에 어떻게 활용되는지, 그리고 수행되는 연산에 따라 연산 강도가 어떻게 변하는지에 익숙해져야 합니다.

먼저 GPU 활용과 모델 훈련 실행에 대한 예시를 살펴보겠습니다. 데모를 위해 몇몇 라이브러리를 설치해야 합니다:

```bash
pip install transformers datasets accelerate nvidia-ml-py3
```

`nvidia-ml-py3` 라이브러리는 Python 내부에서 모델의 메모리 사용량을 모니터링할 수 있게 해줍니다. 터미널의 `nvidia-smi` 명령어에 익숙할 수 있는데, 이 라이브러리는 Python에서 직접 동일한 정보에 접근할 수 있게 해줍니다.

그 다음, 100과 30000 사이의 무작위 토큰 ID와 분류기를 위한 이진 레이블인 더미 데이터를 생성합니다.
길이가 각각 512인 총 512개의 시퀀스를 가져와 PyTorch 형식의 [`~datasets.Dataset`]에 저장합니다.


```py
>>> import numpy as np
>>> from datasets import Dataset


>>> seq_len, dataset_size = 512, 512
>>> dummy_data = {
...     "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
...     "labels": np.random.randint(0, 1, (dataset_size)),
... }
>>> ds = Dataset.from_dict(dummy_data)
>>> ds.set_format("pt")
```

GPU 활용 및 [`Trainer`]로 실행한 훈련 과정에 대한 요약 통계를 출력하기 위해 두 개의 도우미 함수를 정의하겠습니다:

```py
>>> from pynvml import *


>>> def print_gpu_utilization():
...     nvmlInit()
...     handle = nvmlDeviceGetHandleByIndex(0)
...     info = nvmlDeviceGetMemoryInfo(handle)
...     print(f"GPU memory occupied: {info.used//1024**2} MB.")


>>> def print_summary(result):
...     print(f"Time: {result.metrics['train_runtime']:.2f}")
...     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
...     print_gpu_utilization()
```

시작할 때 GPU 메모리가 비어 있는지 확인해 봅시다:

```py
>>> print_gpu_utilization()
GPU memory occupied: 0 MB.
```

좋습니다. 모델을 로드하기 전에는 예상대로 GPU 메모리가 점유되지 않았습니다. 그렇지 않다면 사용자의 기기에서 GPU 메모리를 사용하는 모든 프로세스를 중단해야 합니다. 그러나 사용자는 모든 여유 GPU 메모리를 사용할 수는 없습니다. 모델이 GPU에 로드될 때 커널도 로드되므로 1-2GB의 메모리를 차지할 수 있습니다. 얼마나 되는지 확인하기 위해 GPU에 작은 텐서를 로드하여 커널이 로드되도록 트리거합니다.

```py
>>> import torch


>>> torch.ones((1, 1)).to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 1343 MB.
```

커널만으로도 GPU 메모리의 1.3GB를 차지합니다. 이제 모델이 얼마나 많은 공간을 사용하는지 확인해 보겠습니다.

## 모델 로드 [[load-model]]

우선, `google-bert/bert-large-uncased` 모델을 로드합니다. 모델의 가중치를 직접 GPU에 로드해서 가중치만이 얼마나 많은 공간을 차지하는지 확인할 수 있습니다.


```py
>>> from transformers import AutoModelForSequenceClassification


>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased").to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 2631 MB.
```

모델의 가중치만으로도 GPU 메모리를 1.3 GB 차지하는 것을 볼 수 있습니다. 정확한 숫자는 사용하는 GPU에 따라 다릅니다. 최신 GPU에서는 모델 사용 속도를 높이는 최적화된 방식으로 가중치가 로드되므로, 모델이 더 많은 공간을 차지할 수 있습니다. 이제 `nvidia-smi` CLI와 동일한 결과를 얻는지 빠르게 확인할 수 있습니다:


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

이전과 동일한 숫자가 출력되고 16GB 메모리를 가진 V100 GPU를 사용하고 있다는 것도 볼 수 있습니다. 그러므로 이제 모델 훈련을 시작하여 GPU 메모리 사용량이 어떻게 달라지는지 볼 수 있습니다. 우선 몇몇 표준 훈련 인수를 설정합니다:

```py
default_args = {
    "output_dir": "tmp",
    "eval_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```

<Tip>

여러 실험을 실행할 계획이라면, 실험 간에 메모리를 제대로 비우기 위해서 Python 커널을 실험 사이마다 재시작해야 합니다.

</Tip>

## 기본 훈련에서의 메모리 활용 [[memory-utilization-at-vanilla-training]]

[`Trainer`]를 사용하여, GPU 성능 최적화 기술을 사용하지 않고 배치 크기가 4인 모델을 훈련시키겠습니다:

```py
>>> from transformers import TrainingArguments, Trainer, logging

>>> logging.set_verbosity_error()


>>> training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
>>> trainer = Trainer(model=model, args=training_args, train_dataset=ds)
>>> result = trainer.train()
>>> print_summary(result)
```

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```

우리는 비교적 작은 배치 크기로도 전체 GPU 메모리를 거의 다 차지하는 것을 볼 수 있습니다. 그러나 배치 크기가 클수록 모델 수렴 속도가 빨라지고 최종 성능이 향상되는 경우가 많습니다. 그래서 이상적으로는 GPU 제한이 아닌 우리 모델의 요구사항에 맞게 배치 크기를 조정하려고 합니다. 흥미롭게도 우리는 모델의 크기보다 훨씬 더 많은 메모리를 사용합니다. 왜 이런 현상이 발생하는지 조금 더 잘 이해하기 위해 모델의 연산과 메모리 요구 사항을 살펴보겠습니다.

## 모델의 연산 해부하기 [[anatomy-of-models-operations]]

트랜스포머 아키텍처에는 연산 강도(compute-intensity)에 따라 그룹화된 3가지 주요 연산 그룹이 있습니다.

1. **텐서 축약(Tensor Contractions)**

    선형 레이어와 멀티헤드 어텐션의 구성 요소는 모두 **행렬-행렬 곱셈(matrix-matrix multiplications)**을 일괄적으로 처리합니다. 이 연산은 트랜스포머 훈련에서 가장 연산 강도가 높은 부분입니다.

2. **통계 정규화(Statistical Normalizations)**

    소프트맥스와 레이어 정규화는 텐서 축약보다 연산 강도가 낮습니다. 하나 이상의 **감소 연산(reduction operations)**을 포함하며, 그 결과는 map을 통해 적용됩니다.

3. **원소별 연산자(Element-wise Operators)**

    그 외 연산자들, **편향(biases), 드롭아웃(dropout), 활성화 함수(activations), 잔차 연결(residual connections)**이 여기에 해당합니다. 이 연산들은 연산 강도가 가장 낮습니다.

이러한 지식은 성능 병목 현상을 분석할 때 도움이 될 수 있습니다.

이 내용은 [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)을 참고하였습니다.


## 모델의 메모리 구조 [[anatomy-of-models-memory]]

모델을 훈련시키는 데는 단순히 GPU에 모델을 올리는 것보다 훨씬 더 많은 메모리를 사용한다는 것을 보았습니다. 이는 훈련 중 GPU 메모리를 사용하는 많은 구성 요소가 있기 때문입니다. GPU 메모리의 구성 요소는 다음과 같습니다:

1. 모델 가중치
2. 옵티마이저 상태
3. 그라디언트
4. 그라디언트 계산을 위해 저장된 순방향 활성화
5. 임시 버퍼
6. 기능별 메모리

AdamW를 사용하여 혼합 정밀도로 훈련된 일반적인 모델은 모델 파라미터당 18 바이트와 활성화 메모리가 필요합니다. 추론 단계에서는 옵티마이저와 그라디언트가 필요하지 않으므로 이들은 제외합니다. 따라서 혼합 정밀도 추론의 경우 모델 매개변수당 6 바이트와 활성화 메모리가 필요합니다.

자세히 살펴보겠습니다.

**모델 가중치:**

- fp32 훈련의 경우 매개 변수 수 * 4 바이트
- 혼합 정밀도 훈련의 경우 매개 변수 수 * 6 바이트 (메모리에 fp32와 fp16 두 가지 모델을 유지)

**옵티마이저 상태:**

- 일반 AdamW의 경우 매개 변수 수 * 8 바이트 (2가지 상태 유지)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)와 같은 8비트 AdamW 옵티마이저의 경우 매개 변수 수 * 2 바이트
- Momentum을 가진 SGD와 같은 옵티마이저의 경우 매개 변수 수 * 4 바이트 (하나의 상태만 유지)

**그라디언트**

- fp32 또는 혼합 정밀도 훈련의 경우 매개 변수 수 * 4 바이트 (그라디언트는 항상 fp32으로 유지됩니다.)

**순방향 활성화**

- 크기는 여러 요인에 따라 달라지며, 주요 요인은 시퀀스 길이, 은닉 상태의 크기 및 배치 크기입니다.

순방향 및 역방향 함수에서 전달 및 반환되는 입력과 출력이 있으며, 그라디언트 계산을 위해 저장된 순방향 활성화가 있습니다.

**임시 메모리**

더불어 모든 종류의 임시 변수는 연산이 완료되면 곧바로 해제되지만, 그 순간에는 추가 메모리가 필요할 수 있고 OOM을 유발할 수 있습니다. 따라서 코딩할 때 이러한 임시 변수에 대해 전략적으로 생각하고 때로는 더 이상 필요 없는 임시 변수를 즉시 명시적으로 메모리에서 제거하는 것이 중요합니다.

**기능별 메모리**

그런 다음, 소프트웨어에는 특별한 메모리 요구 사항이 있을 수 있습니다. 예를 들어, 빔 검색을 사용하여 텍스트를 생성할 때 소프트웨어는 입력과 출력 사본을 여러 개 유지해야 합니다.

**`forward` vs `backward` 실행 속도**

합성곱과 선형 레이어의 경우 순방향에 비해 역방향에서는 2배의 플롭스가 필요하므로 일반적으로 2배 정도 느리게 변환됩니다(역방향의 경우 사이즈가 부자연스럽기 때문에, 때로는 더욱 느릴 수도 있습니다). 활성화는 일반적으로 대역폭이 제한되어 있으며, 일반적으로 순방향보다 역방향에서 더 많은 데이터를 읽어야 합니다. (예를 들어, 순방향 활성화 시 한 번 씩 읽고 쓰지만, 역방향 활성화에서는 순방향 gradOutput과 출력에 대해 총 두 번 읽고 gradInput에 대해 한 번 씁니다.)

보다시피, GPU 메모리를 절약하거나 작업 속도를 높일 수 있는 몇 가지 방법이 있습니다.
이제 GPU 활용과 계산 속도에 영향을 주는 것이 무엇인지를 이해했으므로, [Methods and tools for efficient training on a single GPU](perf_train_gpu_one) 문서 페이지를 참조하여 성능 최적화 기법에 대해 알아보세요.