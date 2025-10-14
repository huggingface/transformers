<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 병렬화 방법[[parallelism-methods]]

멀티 GPU 환경은 학습 속도를 높이고, 단일 GPU 메모리에는 담을 수 없는 대규모 모델을 학습할 수 있도록 해줍니다. 이는 여러 GPU에 작업을 병렬로 분산하는 방식에 기반합니다. 병렬화 방식에는 데이터 병렬화, 텐서 병렬화, 파이프라인 병렬화, 모델 병렬화 등 여러 종류가 있으며, 각각 데이터 또는 모델에 따라 작업을 다르게 분할합니다.

이 가이드에서는 다양한 병렬화 방식과 이들을 조합하는 방법, 그리고 환경에 따라 알맞은 전략을 선택하는 방법에 대해 다룹니다. 분산 학습에 대한 자세한 내용은 [Accelerate](https://hf.co/docs/accelerate/index) 문서를 참조하세요.

대규모 언어 모델 확장에 대한 종합 가이드는 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)을 확인하세요. 대규모 학습을 위한 상세한 전략과 모범 사례를 제공합니다.

## 확장성 전략[[scalability-strategy]]

[Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)를 사용하여 모델이 필요로 하는 메모리를 계산하세요. 그런 다음 아래 표를 참조하여 설정에 따라 전략을 선택하세요.

모델이 얼마나 많은 메모리를 필요로 하는지 확인하려면 [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)를 사용해보세요. 그 다음, 아래 표를 참고하여 현재 환경에 가장 적합한 병렬화 전략을 선택하세요.

| 설정 | 시나리오 | 전략 |
|---|---|---|
| 단일 노드/다중 GPU | 단일 GPU에 적재 가능 | DistributedDataParallel 또는 ZeRO |
|  | 단일 GPU에 적재 불가능 | PipelineParallel, ZeRO 또는 TensorParallel |
|  | 가장 큰 모델 레이어가 적재 불가능 | TensorParallel 또는 ZeRO |
| 다중 노드/다중 GPU | 노드 간 연결 속도가 빠름 (NVLink 또는 NVSwitch) | ZeRO 또는 3D 병렬화 (PipelineParallel, TensorParallel, DataParallel) |
|  | 노드 간 연결 속도가 느림 | ZeRO 또는 3D 병렬화 (PipelineParallel, TensorParallel, DataParallel) |

## 데이터 병렬화[[data-parallelism]]

데이터 병렬화는 데이터를 여러 GPU에 균등하게 분산하는 방식입니다. 각 GPU는 동일한 모델의 복사본을 가지고, 자신에게 할당된 일부 데이터를 처리합니다. 처리 후에는 각 GPU의 결과를 동기화하여 하나로 합칩니다.

이 방식은 데이터를 병렬로 처리하기 때문에 학습 시간을 크게 단축할 수 있으며, 사용 가능한 GPU 수에 따라 확장 가능합니다. 다만, 각 GPU의 결과를 동기화하는 과정에서 추가적인 오버헤드가 발생할 수 있습니다.

데이터 병렬화에는 DataParallel (DP)과 DistributedDataParallel (DDP) 두 가지 유형이 있습니다.

### DataParallel

[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)은 *하나의 머신*에서 여러 GPU를 활용한 분산 학습을 지원합니다. 동작 방식은 다음과 같습니다.

1. 기본 GPU인 `GPU 0`이 배치 데이터를 읽고 이를 여러 미니 배치로 나누어 다른 GPU로 전송합니다.
2. 최신 모델이 `GPU 0`에서 다른 GPU로 복제됩니다.
3. 각 GPU에서 `forward` 패스가 수행되고 그 결과가 `GPU 0`으로 전송되어 손실을 계산합니다.
4. 계산된 손실이 `GPU 0`에서 다른 GPU로 분배되어 `backward` 연산을 수행합니다.
5. 각 GPU의 그레이디언트가 `GPU 0`으로 다시 모인 뒤, 평균이 계산됩니다.

### DistributedDataParallel

[DistributedDataParallel](https://pytorch.org/docs/main/notes/ddp.html)은 *여러 대의 머신*과 여러 GPU를 활용한 분산 학습을 지원합니다. 동작 과정은 다음과 같습니다.

1. 메인 프로세스가 기본 GPU인 `GPU 0`에서 각 GPU로 모델을 복제합니다.
2. 각 GPU가 자신에게 할당된 미니배치를 직접 처리합니다.
3. `backward` 패스 동안 로컬 그레이디언트가 모든 GPU에서 평균화됩니다.

DDP는 GPU 간 통신 오버헤드를 줄이고, 각 GPU를 효율적으로 활용하며, 여러 머신으로의 확장성도 뛰어나기 때문에 가장 권장되는 방식입니다.

### ZeRO 데이터 병렬화[[zero-data-parallelism]]

[Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/)는 메모리 효율을 극대화한 데이터 병렬화 방식입니다. 매개변수, 그레이디언트, 옵티마이저 상태를 데이터 병렬 프로세스 간에 분할함으로써 메모리 사용량을 크게 줄입니다. ZeRO는 다음과 같이 세 가지 단계로 나뉩니다.


- 1단계는 옵티마이저 상태를 분할합니다
- 2단계는 옵티마이저와 그레이디언트 상태를 분할합니다
- 3단계는 옵티마이저, 그레이디언트, 매개변수를 분할합니다

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png"/>
</div>

## 모델 병렬화[[model-parallelism]]

모델 병렬화는 하나의 모델을 여러 GPU에 분산합니다. 모델을 분할하는 방법에는 여러 가지가 있지만, 모델의 레이어를 GPU별로 나누어 분산하는 것이 일반적입니다. `forward` 단계에서는 첫 번째 GPU가 데이터 배치를 처리한 뒤, 결과를 다음 GPU의 레이어로 전달합니다. `backward` 패스에서는 마지막 레이어에서 첫 번째 레이어로 데이터를 전송합니다.

모델 병렬화는 단일 GPU 메모리에 담기에는 너무 큰 모델을 학습할 때 유용한 전략입니다. 하지만 한 번에 한 GPU만 활성화되기 때문에 GPU 자원 활용이 불균형해질 수 있으며, GPU 간 결과를 주고받는 과정에서 통신 오버헤드가 발생해 병목 현상이 발생할 수 있습니다.

## 파이프라인 병렬화[[pipeline-parallelism]]

파이프라인 병렬화는 개념적으로 모델 병렬화와 매우 유사하지만, GPU의 유휴 시간을 줄여 더 효율적으로 동작합니다. 각 GPU가 전체 미니배치를 모두 처리할 때까지 기다리는 대신, 데이터를 *마이크로 배치*로 나누어 처리합니다. 하나의 마이크로 배치 처리가 끝나면 즉시 다음 GPU로 전달되고, 다른 GPU도 동시에 자신에게 할당된 마이크로 배치를 처리할 수 있습니다. 이를 통해 GPU들이 순차적으로 대기하지 않고 병렬로 작업을 진행할 수 있습니다.

파이프라인 병렬화는 모델 병렬화와 동일한 장점을 공유하지만, GPU 활용도를 최적화하고 유휴 시간을 줄일 수 있습니다. 하지만 모델을 [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 형태의 모듈로 재구성해야 하며, 마지막 `forward` 단계는 `backward` 단계가 끝날 때까지 기다려야 하기 때문에 유휴 시간을 완전히 없앨 수는 없습니다.

## 텐서 병렬화[[tensor-parallelism]]

텐서 병렬화는 큰 텐서 연산을 여러 GPU에 나누어 수행하는 방식입니다. 텐서를 가로 또는 세로로 분할하고, 각 조각을 서로 다른 GPU에 할당해 처리합니다. 각 GPU는 자신에게 할당된 텐서 조각에 대해 계산을 수행하고, 마지막에 결과를 동기화하여 최종 출력을 재구성합니다.

이 방식은 단일 GPU 메모리에 담기 어려운 대규모 모델 학습에 효과적입니다. 각 GPU가 텐서의 일부를 병렬로 처리하기 때문에 속도와 효율성 측면에서도 유리하며, 다른 병렬화 방식과 함께 사용할 수도 있습니다. 다만, 다른 병렬화 방식과 마찬가지로 GPU 간 통신 오버헤드가 발생합니다.

추론에 텐서 병렬화를 사용하는 방법을 알아보려면 [Tensor parallelism](./perf_infer_gpu_multi) 가이드를 참조하세요.

## 하이브리드 병렬화[[hybrid-parallelism]]

병렬화 방식들을 조합하면 메모리를 더욱 효율적으로 사용할 수 있으며, 수십억 개의 파라미터를 가진 대규모 모델도 효과적으로 학습할 수 있습니다.

### 데이터 병렬화와 파이프라인 병렬화[[data-parallelism-and-pipeline-parallelism]]

데이터 병렬화와 파이프라인 병렬화를 함께 사용하면, 데이터를 여러 GPU에 분산하고 각 미니배치를 마이크로 배치로 나누어 파이프라인 병렬화를 구현할 수 있습니다.

각 데이터 병렬 랭크는 마치 단일 GPU만 있는 것처럼 동작하지만, GPU 0과 1은 마이크로 배치를 GPU 2와 3으로 넘겨 유휴 시간을 줄일 수 있습니다.

이 방식은 GPU의 유휴 시간을 최소화하여 병렬 데이터 처리 효율을 높입니다.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png"/>
</div>

### ZeRO 데이터 병렬화, 파이프라인 병렬화, 모델 병렬화 (3D 병렬화)[[zero-data-parallelism-pipeline-parallelism-and-model-parallelism-3d-parallelism]]

데이터, 파이프라인, 모델 병렬화가 결합된 방식을 [3D 병렬화](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)라고 하며, 메모리와 연산 효율성을 극대화할 수 있습니다.

메모리 효율성은 모델을 여러 GPU에 분할하고, 이를 여러 단계로 나누어 파이프라인을 구성함으로써 달성됩니다. 이렇게 하면 각 GPU가 마이크로 배치 데이터를 병렬로 처리할 수 있어, 모델, 옵티마이저, 활성값의 메모리 사용량이 줄어듭니다.

연산 효율성은 ZeRO 데이터 병렬화를 통해 구현됩니다. 각 GPU는 모델, 옵티마이저, 활성값의 일부분만 저장하며, 데이터 병렬 노드 간 통신은 다른 파이프라인 단계와 독립적으로 또는 병렬로 수행될 수 있어 더 높은 통신 대역폭을 확보할 수 있습니다.

이 접근 방식은 수조 개의 파라미터를 가진 초대형 모델로 확장 가능합니다.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png"/>
</div>
