<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 다중 GPU에서 효율적인 훈련 [[efficient-training-on-multiple-gpus]]

단일 GPU에서의 훈련이 너무 느리거나 모델 가중치가 단일 GPU의 메모리에 맞지 않는 경우, 다중-GPU 설정을 사용합니다. 단일 GPU에서 다중 GPU로 전환하기 위해서는 작업을 분산해야 합니다. 데이터, 텐서 또는 파이프라인과 같은 병렬화 기법을 사용하여 작업을 병렬로 처리할 수 있습니다. 그러나 이러한 설정을 모두에게 적용할 수 있는 완벽한 해결책은 없으며, 어떤 설정이 가장 적합한지는 사용하는 하드웨어에 따라 달라집니다. 이 문서는 주로 PyTorch 기반의 구현을 중심으로 설명하며, 대부분의 개념은 다른 프레임워크에도 적용될 수 있을 것으로 예상됩니다.

<Tip>

 참고: [단일 GPU 섹션](perf_train_gpu_one)에서 소개된 전략(혼합 정밀도 훈련 또는 그래디언트 누적 등)은 일반적으로 모델 훈련에 적용되며, 다중-GPU 또는 CPU 훈련과 같은 다음 섹션으로 진입하기 전에 해당 섹션을 참고하는 것이 좋습니다.

</Tip>

먼저 1D 병렬화 기술에 대해 자세히 논의한 후, 이러한 기술을 결합하여 2D 및 3D 병렬화를 구현하여 더 빠른 훈련과 더 큰 모델을 지원하는 방법을 살펴볼 것입니다. 또한 다른 효과적인 대안 방식도 소개될 예정입니다.

## 개념 [[concepts]]

다음은 이 문서에서 자세히 설명될 주요 개념에 대한 간단한 설명입니다.

1. **DataParallel (DP)** - 동일한 설정이 여러 번 복제되고, 각 설정에 데이터 일부를 받습니다. 처리는 병렬로 수행되며 모든 설정은 각 훈련 단계의 끝날 때 동기화됩니다.
2. **TensorParallel (TP)** - 각 텐서는 여러 개의 묶음으로 분할되기에, 전체 텐서가 단일 GPU에 상주하는 대신 텐서의 각 샤드가 지정된 GPU에 상주합니다. 처리하는 동안 각 샤드는 서로 다른 GPU에서 개별적으로 병렬 처리되며 결과는 단계가 끝날 때 동기화됩니다. 분할이 수평 수준에서 이루어지기 때문에 이를 수평 병렬 처리라고 부를 수 있습니다.
3. **PipelineParallel (PP)** - 모델이 수직으로 (레이어 수준) 여러 GPU에 분할되어 모델의 단일 GPU에는 하나 또는 여러 레이어가 배치됩니다. 각 GPU는 파이프라인의 서로 다른 단계를 병렬로 처리하며 작은 배치 묶음에서 작동합니다.
4. **Zero Redundancy Optimizer (ZeRO)** - TP와 유사하게 텐서를 샤딩하지만, 전체 텐서는 순방향 또는 역방향 계산을 위해 재구성되므로 모델을 수정할 필요가 없습니다. 또한 제한된 GPU 메모리를 보완하기 위해 다양한 오프로드 기술을 지원합니다.
5. **Sharded DDP** - ZeRO의 기본 개념으로 다른 ZeRO 구현에서도 사용되는 용어입니다.

각 개념의 구체적인 내용에 대해 자세히 들어가기 전에 대규모 인프라에서 대규모 모델을 훈련하는 경우의 대략적인 결정 과정을 살펴보겠습니다.

## 확장성 전략 [[scalability-strategy]]

**⇨ 단일 노드 / 다중-GPU**
* 모델이 단일 GPU에 맞는 경우:

    1. DDP - 분산 DP
    2. ZeRO - 상황과 구성에 따라 더 빠를 수도 있고 그렇지 않을 수도 있음

* 모델이 단일 GPU에 맞지 않는 경우:

    1. PP
    2. ZeRO
    3. TP

    노드 내 연결 속도가 매우 빠른 NVLINK 또는 NVSwitch의 경우 세 가지 방법은 대부분 비슷한 성능을 보여야 하며, PP가 없는 경우 TP 또는 ZeRO보다 빠를 것입니다. TP의 정도도 차이를 만들 수 있습니다. 특정 설정에서 승자를 찾기 위해 실험하는 것이 가장 좋습니다.

    TP는 거의 항상 단일 노드 내에서 사용됩니다. 즉, TP 크기 <= 노드당 GPU 수입니다.

* 가장 큰 레이어가 단일 GPU에 맞지 않는 경우:

    1. ZeRO를 사용하지 않는 경우 - PP만으로는 맞지 않으므로 TP를 반드시 사용해야 함
    2. ZeRO를 사용하는 경우에는 위의 "단일 GPU" 항목과 동일


**⇨ 다중 노드 /  다중 GPU**

* 노드 간 연결 속도가 빠른 경우:

    1. ZeRO - 모델에 대부분의 수정을 필요로 하지 않음
    2. PP+TP+DP - 통신이 적지만 모델에 대대적인 변경이 필요함

* 노드 간 연결 속도가 느리며, GPU 메모리가 여전히 부족한 경우:

    1. DP+PP+TP+ZeRO-1
	


## 데이터 병렬화 [[data-parallelism]]

2개의 GPU만으로도 대부분의 사용자들은 `DataParallel` (DP)과 `DistributedDataParallel` (DDP)을 통해 향상된 훈련 속도를 누릴 수 있습니다. 이는 PyTorch의 내장 기능입니다. 일반적으로 DDP를 사용하는 것이 좋으며, DP는 일부 모델에서 작동하지 않을 수 있으므로 주의해야 합니다. [PyTorch 문서](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)에서도 DDP의 사용을 권장합니다.

### DP vs DDP [[dp-vs-ddp]]

`DistributedDataParallel` (DDP)은 일반적으로 `DataParallel` (DP)보다 빠르지만, 항상 그렇지는 않습니다:
* DP는 파이썬 스레드 기반인 반면, DDP는 다중 프로세스 기반이기 때문에 GIL과 같은 파이썬 스레드 제한이 없습니다.
* 그러나 GPU 카드 간의 느린 상호 연결성은 DDP로 인해 실제로 느린 결과를 낼 수 있습니다.

이 두 모드 간의 GPU 간 통신 오버헤드의 주요 차이점은 다음과 같습니다:

[DDP](https://pytorch.org/docs/master/notes/ddp.html):

- 시작할 때, 주 프로세스가 모델을 gpu 0에서 다른 모든 gpu로 복제합니다.
- 그런 다음 각 배치에 대해:
   1. 각 gpu는 자체 미니 배치 데이터를 직접 사용합니다.
   2. `backward` 동안 로컬 그래디언트가 준비되면, 모든 프로세스에 평균화됩니다.

[DP](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html):

각 배치에 대해:
   1. gpu 0은 데이터 배치를 읽고 각 gpu에 미니 배치를 보냅니다.
   2. 업데이트된 모델을 gpu 0에서 각 gpu로 복제합니다.
   3. `forward`를 실행하고 각 gpu의 출력을 gpu 0으로 보내고 손실을 계산합니다.
   4. gpu 0에서 모든 gpu로 손실을 분산하고 `backward`를 실행합니다.
   5. 각 gpu에서 그래디언트를 gpu 0으로 보내고 이를 평균화합니다.

DDP는 각 배치마다 그래디언트를 보내는 통신만을 수행하며, DP는 배치마다 5개의 다른 데이터 교환을 수행합니다.

DP는 파이썬 스레드를 통해 프로세스 내에서 데이터를 복제하며, DDP는 [torch.distributed](https://pytorch.org/docs/master/distributed.html)를 통해 데이터를 복제합니다.

DP에서는 gpu 0이 다른 gpu보다 훨씬 더 많은 작업을 수행하므로, gpu의 활용도가 낮아집니다.

DDP는 여러 대의 컴퓨터에서 사용할 수 있지만, DP의 경우는 그렇지 않습니다.

DP와 DDP 사이에는 다른 차이점이 있지만, 이 토론과는 관련이 없습니다.

이 2가지 모드를 깊게 이해하고 싶다면, [이 문서](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)를 강력히 추천합니다. 이 문서는 멋진 다이어그램을 포함하고 있으며, 다양한 하드웨어에서 여러 벤치마크와 프로파일러 출력을 설명하여 필요한 세부 사항을 모두 설명합니다.

실제 벤치마크를 살펴보겠습니다:

| Type   | NVlink | Time |
| :----- | -----  | ---: |
| 2:DP   | Y      | 110s |
| 2:DDP  | Y      | 101s |
| 2:DDP  | N      | 131s |


분석:

여기서 DP는 NVlink가 있는 DDP보다 약 10% 느립니다. 그러나 NVlink가 없는 DDP보다 약 15% 빠릅니다.

실제 차이는 각 GPU가 다른 GPU와 동기화해야 하는 데이터 양에 따라 달라질 것입니다. 동기화할 데이터가 많을수록 느린 링크가 총 실행 시간을 늦출 수 있습니다.

다음은 전체 벤치마크 코드와 출력입니다:

해당 벤치마크에서 `NCCL_P2P_DISABLE=1`을 사용하여 NVLink 기능을 비활성화했습니다.

```

# DP
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}

# DDP w/ NVlink
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVlink
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

하드웨어: 각각 24GB의 TITAN RTX 2개 + NVlink과 2개의 NVLink (`nvidia-smi topo -m`에서 `NV2`입니다.)
소프트웨어: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`

## ZeRO 데이터 병렬화 [[zero-data-parallelism]]

ZeRO를 기반으로 한 데이터 병렬화 (ZeRO-DP)는 다음 [블로그 글](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)의 다음 다이어그램에서 설명되고 있습니다.
![DeepSpeed-Image-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

이 개념은 이해하기 어려울 수 있지만, 실제로는 매우 간단한 개념입니다. 이는 일반적인 `DataParallel` (DP)과 동일하지만, 전체 모델 매개변수, 그래디언트 및 옵티마이저 상태를 복제하는 대신 각 GPU는 그 중 일부만 저장합니다. 그리고 실행 시간에는 주어진 레이어에 대해 전체 레이어 매개변수가 필요할 때 각 GPU가 서로에게 필요한 부분을 제공하기 위해 동기화됩니다 - 그게 전부입니다.

각각 3개의 레이어와 3개의 매개변수가 있는 간단한 모델을 생각해 봅시다:
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
레이어 La에는 가중치 a0, a1 및 a2가 있습니다.

3개의 GPU가 있는 경우, Sharded DDP (= Zero-DP)는 다음과 같이 모델을 3개의 GPU에 분할합니다:

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

일반적인 DNN 다이어그램을 상상해보면 이는 텐서 병렬 처리와 같은 수평 슬라이싱입니다. 수직 슬라이싱은 전체 레이어 그룹을 다른 GPU에 배치하는 것입니다. 이는 시작에 불과합니다.

이제 이러한 각각의 GPU는 DP에서 작동하는 것과 마찬가지로 일반적인 미니 배치를 받습니다:
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

입력은 수정되지 않은 상태로 일반 모델에 의해 처리될 것으로 간주합니다.

먼저, 입력은 레이어 La에 도달합니다.

GPU0에만 집중해 보겠습니다. x0은 순방향 경로를 수행하기 위해 a0, a1, a2 파라미터가 필요하지만 GPU0에는 a0만 있습니다. GPU1에서 a1을, GPU2에서 a2를 전송받아 모델의 모든 조각을 하나로 모읍니다.

병렬적으로, GPU1은 미니 배치 x1을 받고 a1만 가지고 있지만, a0 및 a2 매개변수가 필요합니다. 따라서 GPU0 및 GPU2에서 이를 가져옵니다.

GPU2도 동일한 작업을 수행합니다. 입력 x2를 받고 GPU0 및 GPU1에서 각각 a0과 a1을, 그리고 자신의 a2와 함께 전체 텐서를 복원합니다.

3개의 GPU는 복원된 전체 텐서를 받고 forward가 수행됩니다.

계산이 완료되면 더 이상 필요하지 않은 데이터는 삭제되고, 해당 데이터는 계산 중에만 사용됩니다. 복원은 사전 패치를 통해 효율적으로 수행됩니다.

그리고 전체 프로세스는 레이어 Lb에 대해 반복되고, 그 다음 Lc로 순방향으로, 그다음은 역방향으로 Lc -> Lb -> La로 반복됩니다.

개인적으로 이것은 효율적인 그룹 배낭 여행자의 중량 분배 전략처럼 들립니다:

1. 사람 A가 텐트를 운반합니다.
2. 사람 B가 난로를 운반합니다.
3. 사람 C가 도끼를 운반합니다.

이제 매일 밤 각자 가진 것을 다른 사람들과 공유하고, 가지지 않은 것은 다른 사람들로부터 받고, 아침에는 할당된 유형의 장비를 싸고 계속해서 여행을 진행합니다. 이것이 Sharded DDP / Zero DP입니다.

이 전략을 각각 자신의 텐트, 난로 및 도끼를 개별적으로 운반해야 하는 단순한 전략과 비교해보면 훨씬 비효율적일 것입니다. 이것이 Pytorch의 DataParallel (DP 및 DDP)입니다.

이 주제에 대해 논문을 읽을 때 다음 동의어를 만날 수 있습니다: Sharded, Partitioned.

ZeRO가 모델 가중치를 분할하는 방식을 자세히 살펴보면, 텐서 병렬화와 매우 유사한 것을 알 수 있습니다. 이는 이후에 설명될 수직 모델 병렬화와는 달리 각 레이어의 가중치를 분할/분할하기 때문입니다.

구현:

- [DeepSpeed](https://www.deepspeed.ai/features/#the-zero-redundancy-optimizer)는 1단계 + 2단계 + 3단계의 ZeRO-DP를 제공합니다.
- [Fairscale](https://github.com/facebookresearch/fairscale/#optimizer-state-sharding-zero)은 1단계 + 2단계 + 3단계의 ZeRO-DP를 제공합니다.
- [`transformers` 통합](main_classes/trainer#trainer-integrations)

## 네이티브 모델 병렬 처리(수직적) 및 파이프라인 병렬 처리[[naive-model-parallelism-vertical-and-pipeline-parallelism]]

Naive Model Parallelism (MP)은 모델 레이어 그룹을 다중 GPU에 분산하는 방식입니다. 메커니즘은 상대적으로 간단합니다. 원하는 레이어를 `.to()`를 사용하여 원하는 장치로 전환하면 데이터가 해당 레이어로 들어오고 나갈 때 데이터도 레이어와 동일한 장치로 전환되고 나머지는 수정되지 않습니다.

대부분의 모델이 그려지는 방식이 레이어를 세로로 슬라이스하기 때문에 이를 수직 모델 병렬화라고 부릅니다. 예를 들어 다음 다이어그램은 8레이어 모델을 보여줍니다:

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```
우리는 모델을 수직으로 2개로 분할하여 레이어 0-3을 GPU0에 배치하고 레이어 4-7을 GPU1에 배치했습니다.

이제 데이터가 레이어 0에서 1로, 1에서 2로, 2에서 3으로 이동하는 동안에는 일반적인 모델입니다. 그러나 데이터가 레이어 3에서 레이어 4로 전달되어야 할 때는 GPU0에서 GPU1로 이동해야 하므로 통신 오버헤드가 발생합니다. 참여하는 GPU가 동일한 컴퓨팅 노드(예: 동일한 물리적인 기계)에 있는 경우 이 복사는 매우 빠릅니다. 그러나 GPU가 서로 다른 컴퓨팅 노드(예: 여러 기계)에 위치한 경우 통신 오버헤드는 상당히 크게 될 수 있습니다.

그런 다음 레이어 4부터 5로, 6으로, 7로 진행되는 것은 일반적인 모델과 동일하게 진행되고, 7번째 레이어가 완료되면 데이터를 다시 레이어 0으로 보내거나 또는 레이블을 마지막 레이어로 보내야 할 필요가 있습니다. 이제 손실을 계산하고 옵티마이저가 작동할 수 있습니다.

문제점:
- 이 방식을 "naive" MP라고 부르는 이유는 주어진 상황에 하나의 GPU를 제외한 모든 GPU가 유휴 상태라는 점입니다. 따라서 4개의 GPU를 사용하는 경우 단일 GPU의 메모리 양을 4배로 늘리고 나머지 하드웨어는 무시하는 것과 거의 동일합니다. 또한 장치 간 데이터 복사의 오버헤드도 있습니다. 따라서 4개의 6GB 카드는 naive MP를 사용하여 1개의 24GB 카드와 동일한 크기를 수용할 수 있지만, 후자는 데이터 복사의 오버헤드가 없으므로 훈련을 더 빨리 완료합니다. 그러나 예를 들어 40GB 카드가 있고 45GB 모델을 맞추어야 할 경우 4개의 40GB 카드로 맞출 수 있습니다 (하지만 그래디언트와 옵티마이저 상태 때문에 가까스로 가능합니다).
- 공유 임베딩은 GPU 간에 복사해야 할 수도 있습니다.

파이프라인 병렬화 (PP)은 거의 naive MP와 동일하지만 GPU 유휴 상태 문제를 해결하기 위해 들어오는 배치를 마이크로 배치로 나누고 인공적으로 파이프라인을 생성하여 서로 다른 GPU가 동시에 계산에 참여할 수 있게 합니다.

[GPipe 논문](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)에서 가져온 그림은 상단에 naive MP를, 하단에는 PP를 보여줍니다:

![mp-pp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)

하단 다이어그램에서 PP가 유휴 영역이 적은 것을 쉽게 볼 수 있습니다. 유휴 부분을 "bubble"이라고 합니다.

다이어그램의 양쪽 부분은 참여하는 GPU가 4개인 병렬성을 보여줍니다. 즉, 4개의 GPU가 파이프라인에 참여합니다. 따라서 4개의 파이프 단계 F0, F1, F2 및 F3의 순방향 경로와 B3, B2, B1 및 B0의 역방향 경로가 있습니다.

PP는 조정해야 할 새로운 하이퍼파라미터인 `chunks`를 도입합니다. 이는 동일한 파이프 단계를 통해 일련의 데이터를 묶어서 보내는 방식을 정의합니다. 예를 들어, 아래 다이어그램에서 `chunks=4`를 볼 수 있습니다. GPU0은 0, 1, 2 및 3 (F0,0, F0,1, F0,2, F0,3) 묶음에서 동일한 순방향 경로를 수행하고, 다른 GPU가 작업을 수행하기 시작하고 완료가 시작될 때만 GPU0이 묶음의 역순으로 3, 2, 1 및 0 (B0,3, B0,2, B0,1, B0,0) 경로를 수행합니다.

개념적으로 이는 그래디언트 누적 단계 (GAS)와 동일한 개념입니다. 파이토치에서는 `chunks`를 사용하고 DeepSpeed에서는 동일한 하이퍼파라미터를 GAS로 참조합니다.

묶음으로 인해 PP는 마이크로 배치 (MBS)의 개념을 도입합니다. DP는 전역 데이터 배치 크기를 미니 배치로 나눕니다. 따라서 DP 차수가 4이고 전역 배치 크기가 1024이면 256씩 4개의 미니 배치로 분할됩니다 (1024/4). 그리고 `chunks` (또는 GAS)의 수가 32이면 마이크로 배치 크기는 8이 됩니다 (256/32). 각 파이프라인 단계는 한 번에 하나의 마이크로 배치와 함께 작동합니다.

DP + PP 설정의 전역 배치 크기를 계산하려면 `mbs*chunks*dp_degree` (`8*32*4=1024`)를 수행합니다.

다이어그램으로 돌아가 보겠습니다.

`chunks=1`로 설정하면 매우 비효율적인 naive MP가 생성되며, 매우 큰 `chunks` 값으로 설정하면 아주 작은 마이크로 배치 크기가 생성되어 효율적이지 않을 수 있습니다. 따라서 가장 효율적인 GPU 활용을 위해 어떤 값이 가장 적절한지 실험을 해야 합니다.

다이어그램에서 보이는 것처럼 "dead" 시간의 버블이 존재하여 마지막 `forward` 단계가 `backward` 단계가 파이프라인을 완료하기를 기다려야 하는 상황이 발생하지만, `chunks`의 가장 적절한 값을 찾는 것의 목적은 모든 참여하는 GPU에서 동시에 고도로 활용되는 GPU 활용을 가능하게 하여 버블의 크기를 최소화하는 것입니다.

해결책은 전통적인 파이프라인 API와 더 현대적인 솔루션으로 나뉩니다. 전통적인 파이프라인 API 솔루션과 현대적인 솔루션에 대해 알아보겠습니다.

전통적인 파이프라인 API 솔루션:
- 파이토치
- FairScale
- DeepSpeed
- Megatron-LM

현대적인 솔루션:
- Varuna
- Sagemaker

전통적인 파이프라인 API 솔루션의 문제점:
- 모델을 상당히 수정해야 한다는 점이 문제입니다. 파이프라인은 모듈의 정상적인 흐름을 `nn.Sequential` 시퀀스로 다시 작성해야 하므로 모델의 설계를 변경해야 할 수 있습니다.
- 현재 파이프라인 API는 매우 제한적입니다. 파이프라인의 매우 첫 번째 단계에서 전달되는 많은 파이썬 변수가 있는 경우 이를 해결해야 합니다. 현재 파이프라인 인터페이스는 하나의 텐서 또는 텐서의 튜플을 유일한 입력 및 출력으로 요구합니다. 이러한 텐서는 마이크로 배치로 미니 배치로 묶을 것이므로 첫 번째 차원으로 배치 크기가 있어야 합니다. 가능한 개선 사항은 여기에서 논의되고 있습니다. https://github.com/pytorch/pytorch/pull/50693
- 파이프 단계 수준에서 조건부 제어 흐름은 불가능합니다. 예를 들어, T5와 같은 인코더-디코더 모델은 조건부 인코더 단계를 처리하기 위해 특별한 해결책이 필요합니다.
- 각 레이어를 정렬하여 하나의 모델의 출력이 다른 모델의 입력이 되도록해야 합니다.

우리는 아직 Varuna와 SageMaker로 실험하지 않았지만, 해당 논문들은 위에서 언급한 문제들의 목록을 극복했고 사용자의 모델에 대한 변경 사항이 훨씬 적게 필요하다고 보고하고 있습니다.

구현:
- [파이토치](https://pytorch.org/docs/stable/pipeline.html) (파이토치-1.8에서 초기 지원, 1.9에서 점진적으로 개선되고 1.10에서 더 개선됨). [예제](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)도 참고하세요.
- [FairScale](https://fairscale.readthedocs.io/en/latest/tutorials/pipe.html)
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)은 내부 구현을 가지고 있습니다 - API 없음.
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972) - 이는 AWS에서만 사용할 수 있는 소유 솔루션입니다.
- [OSLO](https://github.com/tunib-ai/oslo) - 이는 Hugging Face Transformers를 기반으로 구현된 파이프라인 병렬화입니다.

🤗 Transformers 상태: 이 작성 시점에서 모델 중 어느 것도 완전한 PP를 지원하지 않습니다. GPT2와 T5 모델은 naive MP를 지원합니다. 주요 장애물은 모델을 `nn.Sequential`로 변환하고 모든 입력을 텐서로 가져와야 하는 것을 처리할 수 없기 때문입니다. 현재 모델에는 이러한 변환을 매우 복잡하게 만드는 많은 기능이 포함되어 있어 제거해야 합니다.

기타 접근 방법:

DeepSpeed, Varuna 및 SageMaker는 [교차 파이프라인(Interleaved Pipeline)](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html) 개념을 사용합니다.
![interleaved-pipeline-execution](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-sagemaker-interleaved-pipeline.png)

여기서는 버블(유휴 시간)을 역방향 패스에 우선순위를 부여하여 최소화합니다.

Varuna는 가장 효율적인 스케줄링을 찾기 위해 시뮬레이션을 사용하여 스케줄을 개선하려고 합니다.

OSLO는 `nn.Sequential`로 변환하지 않고 Transformers를 기반으로 한 파이프라인 병렬화를 구현했습니다.

## 텐서 병렬 처리 [[tensor-parallelism]]

텐서 병렬 처리에서는 각 GPU가 텐서의 일부분만 처리하고 전체 텐서가 필요한 연산에 대해서만 전체 텐서를 집계합니다.

이 섹션에서는 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 논문인 [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)에서의 개념과 다이어그램을 사용합니다.

Transformer의 주요 구성 요소는 fully connected `nn.Linear`와 비선형 활성화 함수인 `GeLU`입니다.

Megatron 논문의 표기법을 따라 행렬의 점곱 부분을 `Y = GeLU(XA)`로 표현할 수 있습니다. 여기서 `X`와 `Y`는 입력 및 출력 벡터이고 `A`는 가중치 행렬입니다.

행렬 형태로 계산을 살펴보면, 행렬 곱셈을 다중 GPU로 분할할 수 있는 방법을 쉽게 알 수 있습니다:
![Parallel GEMM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_gemm.png)

가중치 행렬 `A`를 `N`개의 GPU에 대해 열별로 분할하고 병렬로 행렬 곱셈 `XA_1`에서 `XA_n`까지 수행하면 `N`개의 출력 벡터 `Y_1, Y_2, ..., Y_n`가 생성되며 독립적으로 `GeLU`에 전달될 수 있습니다:
![independent GeLU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-independent-gelu.png)

이 원리를 사용하여 동기화가 필요하지 않은 GPU 간의 임의 깊이의 MLP를 업데이트할 수 있습니다. 그러나 결과 벡터를 샤드로부터 재구성해야 하는 마지막 단계까지는 GPU 간의 동기화가 필요합니다. Megatron-LM 논문의 저자들은 이에 대한 유용한 그림을 제공합니다:
![parallel shard processing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_shard_processing.png)

다중 헤드 어텐션 레이어의 병렬화는 더욱 간단합니다. 이미 독립적인 다중 헤드를 가지고 있기 때문에 이미 병렬화되어 있습니다!
![parallel self-attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_self_attention.png)

특별 고려사항: TP는 매우 빠른 네트워크가 필요하므로 한 개 이상의 노드에서 TP를 수행하는 것은 권장되지 않습니다. 실제로 노드에 4개의 GPU가 있는 경우 TP의 최대 차수는 4입니다. TP 차수가 8인 경우 최소한 8개의 GPU가 있는 노드를 사용해야 합니다.

이 섹션은 원래의 [더 자세한 TP 개요](https://github.com/huggingface/transformers/issues/10321#issuecomment-783543530)를 기반으로 합니다. 
작성자는 [@anton-l](https://github.com/anton-l)입니다.

SageMaker는 더 효율적인 처리를 위해 TP와 DP를 결합합니다.

대체 이름:
- DeepSpeed는 이를 [텐서 슬라이싱](https://www.deepspeed.ai/features/#model-parallelism)이라고 부릅니다.

구현:
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)은 내부 구현을 가지고 있으므로 모델에 매우 특화되어 있습니다.
- [parallelformers](https://github.com/tunib-ai/parallelformers) (현재는 추론에만 해당)
- [SageMaker](https://arxiv.org/abs/2111.05972) - 이는 AWS에서만 사용할 수 있는 소유 솔루션입니다.
- [OSLO](https://github.com/tunib-ai/oslo)은 Transformers를 기반으로 한 텐서 병렬 처리 구현을 가지고 있습니다.

🤗 Transformers 현황:
- core: 아직 핵심 부분에 구현되지 않음
- 그러나 추론을 하려면 [parallelformers](https://github.com/tunib-ai/parallelformers)가 대부분의 모델을 지원합니다. 따라서 핵심 부분에 구현되기 전까지 그들의 것을 사용할 수 있습니다. 그리고 훈련 모드도 지원될 예정입니다.
- Deepspeed-Inference는 CUDA 커널을 기반으로 하는 매우 빠른 추론 모드에서 BERT, GPT-2 및 GPT-Neo 모델을 지원합니다. 자세한 내용은 [여기](https://www.deepspeed.ai/tutorials/inference-tutorial/)를 참조하세요.

## DP+PP [[dppp]]

DeepSpeed [pipeline tutorial](https://www.deepspeed.ai/tutorials/pipeline/)에서 다음 다이어그램은 DP와 PP를 결합하는 방법을 보여줍니다.

![dp-pp-2d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

여기서 DP 랭크 0은 GPU2를 보지 못하고, DP 랭크 1은 GPU3을 보지 못하는 것이 중요합니다. DP에게는 딱 2개의 GPU인 것처럼 데이터를 공급합니다. GPU0은 PP를 사용하여 GPU2에게 일부 작업을 "비밀리에" 할당합니다. 그리고 GPU1도 GPU3을 도움으로 삼아 같은 방식으로 작업합니다.

각 차원마다 적어도 2개의 GPU가 필요하므로 최소한 4개의 GPU가 필요합니다.

구현:
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers 현황: 아직 구현되지 않음

## DP+PP+TP [[dppptp]]

더 효율적인 훈련을 위해 PP와 TP 및 DP를 결합하여 3D 병렬 처리를 사용합니다. 다음 다이어그램에서 이를 확인할 수 있습니다.

![dp-pp-tp-3d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

이 다이어그램은 [3D parallelism: Scaling to trillion-parameter models](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)이라는 블로그 글에서 확인할 수 있습니다.

각 차원마다 적어도 2개의 GPU가 필요하므로 최소한 8개의 GPU가 필요합니다.

구현:
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed는 더욱 효율적인 DP인 ZeRO-DP라고도 부릅니다.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers 현황: 아직 구현되지 않음. PP와 TP가 없기 때문입니다.

## ZeRO DP+PP+TP [[zero-dppptp]]

DeepSpeed의 주요 기능 중 하나는 DP의 확장인 ZeRO입니다. ZeRO-DP에 대해 이미 [ZeRO Data Parallelism](#zero-data-parallelism)에서 논의되었습니다. 일반적으로 이는 PP나 TP를 필요로하지 않는 독립적인 기능입니다. 그러나 PP와 TP와 결합할 수도 있습니다.

ZeRO-DP가 PP와 (선택적으로 TP와) 결합되면 일반적으로 ZeRO 단계 1(옵티마이저 분할)만 활성화됩니다.

이론적으로는 ZeRO 단계 2(그라디언트 분할)를 파이프라인 병렬 처리와 함께 사용할 수도 있지만, 이는 성능에 나쁜 영향을 미칠 것입니다. 각 마이크로 배치마다 그라디언트를 샤딩하기 전에 추가적인 리듀스-스캐터 컬렉티브가 필요하며, 이는 잠재적으로 상당한 통신 오버헤드를 추가합니다. 파이프라인 병렬 처리의 특성상 작은 마이크로 배치가 사용되며, 산술 연산 강도(마이크로 배치 크기)를 균형 있게 유지하면서 파이프라인 버블(마이크로 배치 수)을 최소화하는 것에 중점을 둡니다. 따라서 해당 통신 비용은 문제가 될 것입니다.

또한, PP로 인해 정상보다 적은 수의 레이어가 있으므로 메모리 절약은 크지 않을 것입니다. PP는 이미 그래디언트 크기를 ``1/PP``로 줄이기 때문에 그래디언트 샤딩의 절약 효과는 순수 DP보다는 미미합니다.

ZeRO 단계 3도 같은 이유로 좋은 선택이 아닙니다 - 더 많은 노드 간 통신이 필요합니다.

그리고 ZeRO가 있기 때문에 다른 이점은 ZeRO-Offload입니다. 이는 단계 1이므로 옵티마이저 상태를 CPU로 오프로드할 수 있습니다.

구현:
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 및 [BigScience의 Megatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed), 이전 저장소의 포크입니다.
- [OSLO](https://github.com/tunib-ai/oslo)

중요한 논문:

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](
https://arxiv.org/abs/2201.11990)

🤗 Transformers 현황: 아직 구현되지 않음, PP와 TP가 없기 때문입니다.

## FlexFlow [[flexflow]]

[FlexFlow](https://github.com/flexflow/FlexFlow)는 약간 다른 방식으로 병렬화 문제를 해결합니다.

논문: ["Beyond Data and Model Parallelism for Deep Neural Networks" by Zhihao Jia, Matei Zaharia, Alex Aiken](https://arxiv.org/abs/1807.05358)

이는 Sample-Operator-Attribute-Parameter를 기반으로 하는 일종의 4D 병렬화를 수행합니다.

1. Sample = 데이터 병렬화 (샘플별 병렬)
2. Operator = 단일 연산을 여러 하위 연산으로 병렬화
3. Attribute = 데이터 병렬화 (길이별 병렬)
4. Parameter = 모델 병렬화 (수평 또는 수직과 관계없이)

예시:
* Sample

512 길이의 10개의 배치를 가정해 봅시다. 이를 sample 차원으로 2개의 장치에 병렬화하면, 10 x 512는 5 x 2 x 512가 됩니다.

* Operator

레이어 정규화를 수행한다면, 우선 std를 계산하고 두 번째로 mean을 계산한 다음 데이터를 정규화할 수 있습니다. Operator 병렬화는 std와 mean을 병렬로 계산할 수 있도록 합니다. 따라서 operator 차원으로 2개의 장치 (cuda:0, cuda:1)에 병렬화하면, 먼저 입력 데이터를 두 장치로 복사한 다음 cuda:0에서 std를 계산하고 cuda:1에서 동시에 mean을 계산합니다.

* Attribute

512 길이의 10개의 배치가 있습니다. 이를 attribute 차원으로 2개의 장치에 병렬화하면, 10 x 512는 10 x 2 x 256이 됩니다.

* Parameter

이는 tensor 모델 병렬화 또는 naive layer-wise 모델 병렬화와 유사합니다.

![flex-flow-soap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-flexflow.jpeg)

이 프레임워크의 중요한 점은 (1) GPU/TPU/CPU 대 (2) RAM/DRAM 대 (3) 빠른 인트라-커넥트 대 느린 인터-커넥트와 같은 리소스를 고려하여 어디에서 어떤 병렬화를 사용할지를 알고리즘적으로 자동으로 최적화한다는 것입니다.

하나 매우 중요한 측면은 FlexFlow가 정적이고 고정된 워크로드를 가진 모델에 대한 DNN 병렬화를 최적화하기 위해 설계되었다는 것입니다. 동적인 동작을 가진 모델은 반복마다 다른 병렬화 전략을 선호할 수 있습니다.

따라서 이 프레임워크의 장점은 선택한 클러스터에서 30분 동안 시뮬레이션을 실행하고 이 특정 환경을 최적으로 활용하기 위한 최상의 전략을 제안한다는 것입니다. 부품을 추가/제거/교체하면 실행하고 그에 대한 계획을 다시 최적화한 후 훈련할 수 있습니다. 다른 설정은 자체적인 사용자 정의 최적화를 가질 수 있습니다.

🤗 Transformers 현황: 아직 통합되지 않음. 이미 [transformers.utils.fx](https://github.com/huggingface/transformers/blob/master/src/transformers/utils/fx.py)를 통해 모델을 FX-추적할 수 있으며, 이는 FlexFlow의 선행 조건입니다. 따라서 어떤 작업을 수행해야 FlexFlow가 우리의 모델과 함께 작동할 수 있는지 파악해야 합니다.


## 어떤 전략을 사용해야 할까요? [[which-strategy-to-use-when]]

다음은 어떤 병렬화 전략을 언제 사용해야 하는지에 대한 매우 대략적인 개요입니다. 각 목록의 첫 번째 전략이 일반적으로 더 빠릅니다.

**⇨ 단일 GPU**

* 모델이 단일 GPU에 맞는 경우:

    1. 일반적인 사용

* 모델이 단일 GPU에 맞지 않는 경우:

    1. ZeRO + CPU 및 옵션으로 NVMe 언로드
    2. 위와 동일하게 사용하되, 가장 큰 레이어가 단일 GPU에 맞지 않는 경우 Memory Centric Tiling(자세한 내용은 아래 참조)을 추가적으로 사용

* 가장 큰 레이어가 단일 GPU에 맞지 않는 경우:

1. ZeRO - [Memory Centric Tiling](https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling) (MCT) 활성화. 이를 통해 크기가 매우 큰 레이어를 임의로 분할하여 순차적으로 실행할 수 있습니다. MCT는 GPU에 활성화된 매개변수의 수를 줄이지만 활성화 메모리에는 영향을 주지 않습니다. 현재 작성 기준으로 이 요구사항은 매우 드물기 때문에 사용자가 `torch.nn.Linear`를 수동으로 수정해야 합니다.

**⇨ 단일 노드 / 다중 GPU**

* 모델이 단일 GPU에 맞는 경우:

    1. DDP - 분산 DP
    2. ZeRO - 상황과 구성에 따라 빠를 수도 있고 그렇지 않을 수도 있습니다.

* 모델이 단일 GPU에 맞지 않는 경우:

    1. PP
    2. ZeRO
    3. TP

    NVLINK 또는 NVSwitch를 통한 매우 빠른 인트라-노드 연결이 있는 경우 이 세 가지 방법은 거의 동등할 것이며, 이러한 연결이 없는 경우 PP가 TP나 ZeRO보다 빠를 것입니다. 또한 TP의 차수도 영향을 줄 수 있습니다. 특정 설정에서 우승자를 찾기 위해 실험하는 것이 가장 좋습니다.

    TP는 거의 항상 단일 노드 내에서 사용됩니다. 즉, TP 크기 <= 노드당 GPU 수입니다.

* 가장 큰 레이어가 단일 GPU에 맞지 않는 경우:

    1. ZeRO를 사용하지 않을 경우 - PP만 사용할 수 없으므로 TP를 사용해야 합니다.
    2. ZeRO를 사용할 경우, "단일 GPU"의 항목과 동일한 항목 참조


**⇨ 다중 노드 / 다중 GPU**

* 빠른 노드 간 연결이 있는 경우:

    1. ZeRO - 모델에 대한 수정이 거의 필요하지 않습니다.
    2. PP+TP+DP - 통신이 적지만 모델에 대한 대규모 변경이 필요합니다.

* 느린 노드 간 연결 및 GPU 메모리 부족한 경우:

    1. DP+PP+TP+ZeRO-1
