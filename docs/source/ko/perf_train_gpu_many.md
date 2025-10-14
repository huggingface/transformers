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

멀티 GPU 설정은 학습을 가속화하고 단일 GPU에는 맞지 않는 대형 모델을 메모리에 적재하는 데 효과적입니다. 이는 GPU 간에 작업을 병렬화하는 방식에 의존합니다. 데이터 병렬화, 텐서 병렬화, 파이프라인 병렬화, 모델 병렬화 등 여러 유형의 병렬화가 있습니다. 각 병렬화 유형은 데이터 또는 모델에 따라 작업을 다르게 분할합니다.

이 가이드에서는 다양한 병렬화 방법, 이를 조합하는 방법, 그리고 설정에 적합한 전략을 선택하는 방법을 설명합니다. 분산 학습에 대한 자세한 내용은 [Accelerate](https://hf.co/docs/accelerate/index) 문서를 참조하세요.

대규모 언어 모델 확장에 대한 종합 가이드는 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)을 확인하세요. 대규모 학습을 위한 상세한 전략과 모범 사례를 제공합니다.

## 확장성 전략[[scalability-strategy]]

[Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)를 사용하여 모델이 필요로 하는 메모리를 계산하세요. 그런 다음 아래 표를 참조하여 설정에 따라 전략을 선택하세요.

| 설정 | 시나리오 | 전략 |
|---|---|---|
| 단일 노드/멀티 GPU | 단일 GPU에 적재 가능 | DistributedDataParallel 또는 ZeRO |
|  | 단일 GPU에 적재 불가능 | PipelineParallel, ZeRO 또는 TensorParallel |
|  | 가장 큰 모델 레이어가 적재 불가능 | TensorParallel 또는 ZeRO |
| 멀티 노드/멀티 GPU | 빠른 노드 간 연결 (NVLink 또는 NVSwitch) | ZeRO 또는 3D 병렬화 (PipelineParallel, TensorParallel, DataParallel) |
|  | 느린 노드 간 연결 | ZeRO 또는 3D 병렬화 (PipelineParallel, TensorParallel, DataParallel) |

## 데이터 병렬화[[data-parallelism]]

데이터 병렬화는 여러 GPU에 데이터를 균등하게 분산합니다. 각 GPU는 모델의 사본을 보유하며 동시에 데이터의 일부를 처리합니다. 마지막에 각 GPU의 결과가 동기화되고 결합됩니다.

데이터 병렬화는 데이터를 병렬로 처리하여 학습 시간을 크게 단축하며, 사용 가능한 GPU 수에 따라 확장 가능합니다. 그러나 각 GPU의 결과를 동기화하는 과정에서 오버헤드가 추가될 수 있습니다.

데이터 병렬화에는 DataParallel (DP)과 DistributedDataParallel (DDP) 두 가지 유형이 있습니다.

### DataParallel

[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)은 여러 GPU를 가진 *단일 머신*에서 분산 학습을 지원합니다.

1. 기본 GPU인 `GPU 0`이 데이터 배치를 읽고 그 중 미니 배치를 다른 GPU로 전송합니다.
2. 최신 모델이 `GPU 0`에서 다른 GPU로 복제됩니다.
3. 각 GPU에서 `forward` 패스가 수행되고 그 출력이 `GPU 0`으로 전송되어 손실을 계산합니다.
4. 손실이 `GPU 0`에서 다른 GPU로 분배되어 `backward` 패스를 수행합니다.
5. 각 GPU의 그레이디언트가 `GPU 0`으로 다시 전송되어 평균화됩니다.

### DistributedDataParallel

[DistributedDataParallel](https://pytorch.org/docs/main/notes/ddp.html)은 여러 GPU를 가진 *여러 머신*에서 분산 학습을 지원합니다.

1. 메인 프로세스가 기본 GPU인 `GPU 0`에서 각 GPU로 모델을 복제합니다.
2. 각 GPU가 미니 배치 데이터를 직접 처리합니다.
3. `backward` 패스 동안 로컬 그레이디언트가 모든 GPU에서 평균화됩니다.

DDP는 GPU 간 통신 오버헤드를 줄이고, 각 GPU를 효율적으로 활용하며, 여러 머신으로 확장할 수 있기 때문에 권장됩니다.

### ZeRO 데이터 병렬화[[zero-data-parallelism]]

[Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/)는 더욱 메모리 효율적인 데이터 병렬화 유형입니다. 파라미터, 그레이디언트, 옵티마이저 상태를 데이터 병렬 프로세스 간에 분할하여 메모리 사용량을 줄임으로써 메모리 효율성을 크게 향상시킵니다. ZeRO에는 세 가지 단계가 있습니다:

- Stage 1은 옵티마이저 상태를 분할합니다
- Stage 2는 옵티마이저와 그레이디언트 상태를 분할합니다
- Stage 3은 옵티마이저, 그레이디언트, 파라미터를 분할합니다

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png"/>
</div>

## 모델 병렬화[[model-parallelism]]

모델 병렬화는 모델을 여러 GPU에 분산합니다. 모델을 분할하는 여러 방법이 있지만, 일반적인 방법은 모델 레이어를 GPU 간에 분산하는 것입니다. `forward` 패스에서 첫 번째 GPU가 데이터 배치를 처리하고 다음 GPU의 다음 레이어 그룹으로 전달합니다. `backward` 패스에서는 데이터가 마지막 레이어에서 첫 번째 레이어로 역방향으로 전송됩니다.

모델 병렬화는 단일 GPU의 메모리에 맞지 않는 대형 모델을 학습하는 데 유용한 전략입니다. 그러나 한 번에 하나의 GPU만 활성화되기 때문에 GPU 활용도가 불균형합니다. GPU 간에 결과를 전달하는 것도 통신 오버헤드를 추가하며 병목 현상이 될 수 있습니다.

## 파이프라인 병렬화[[pipeline-parallelism]]

파이프라인 병렬화는 개념적으로 모델 병렬화와 매우 유사하지만, 유휴 GPU 시간을 줄여 더 효율적입니다. 각 GPU가 데이터 배치 처리를 완료할 때까지 기다리는 대신, 파이프라인 병렬화는 데이터의 *마이크로 배치*를 생성합니다. 하나의 마이크로 배치가 완료되면 즉시 다음 GPU로 전달됩니다. 이런 방식으로 각 GPU는 다른 GPU가 미니 배치 데이터 처리를 완전히 끝낼 때까지 기다리지 않고 동시에 데이터의 일부를 처리할 수 있습니다.

파이프라인 병렬화는 모델 병렬화와 동일한 장점을 공유하지만, GPU 활용도를 최적화하고 유휴 시간을 줄입니다. 그러나 파이프라인 병렬화는 모델을 [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 모듈의 시퀀스로 재작성해야 할 수 있으며, 마지막 `forward` 패스도 `backward` 패스가 완료될 때까지 기다려야 하기 때문에 유휴 시간을 완전히 줄이는 것은 불가능하여 더 복잡할 수 있습니다.

## 텐서 병렬화[[tensor-parallelism]]

텐서 병렬화는 대형 텐서 연산을 여러 GPU에 분산합니다. 텐서는 수평 또는 수직으로 슬라이스되고 각 슬라이스는 별도의 GPU에서 처리됩니다. 각 GPU는 텐서 슬라이스에 대한 계산을 수행하고 마지막에 결과가 동기화되어 최종 결과를 재구성합니다.

텐서 병렬화는 단일 GPU의 메모리에 맞지 않는 대형 모델을 학습하는 데 효과적입니다. 또한 각 GPU가 텐서 슬라이스를 병렬로 처리할 수 있어 더 빠르고 효율적이며, 다른 병렬화 방법과 결합할 수 있습니다. 그러나 다른 병렬화 방법과 마찬가지로 텐서 병렬화도 GPU 간 통신 오버헤드를 추가합니다.

추론에 사용하는 방법을 알아보려면 [Tensor parallelism](./perf_infer_gpu_multi) 가이드를 참조하세요.

## 하이브리드 병렬화[[hybrid-parallelism]]

병렬화 방법을 결합하면 더 큰 메모리 절감을 달성하고 수십억 개의 파라미터를 가진 모델을 더욱 효율적으로 학습할 수 있습니다.

### 데이터 병렬화와 파이프라인 병렬화[[data-parallelism-and-pipeline-parallelism]]

데이터 및 파이프라인 병렬화는 GPU 간에 데이터를 분산하고 각 미니 배치 데이터를 마이크로 배치로 분할하여 파이프라인 병렬화를 달성합니다.

각 데이터 병렬 랭크는 두 개 대신 하나의 GPU만 있는 것처럼 프로세스를 처리하지만, GPU 0과 1은 마이크로 배치 데이터를 GPU 2와 3으로 오프로드하여 유휴 시간을 줄일 수 있습니다.

이 접근 방식은 유휴 GPU 활용도를 줄여 병렬 데이터 처리를 최적화합니다.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png"/>
</div>

### ZeRO 데이터 병렬화, 파이프라인 병렬화, 모델 병렬화 (3D 병렬화)[[zero-data-parallelism-pipeline-parallelism-and-model-parallelism-3d-parallelism]]

데이터, 파이프라인, 모델 병렬화가 결합되어 [3D parallelism](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)을 형성하여 메모리와 연산 효율성을 최적화합니다.

메모리 효율성은 GPU 간에 모델을 분할하고 이를 단계로 나누어 파이프라인을 생성함으로써 달성됩니다. 이를 통해 GPU가 데이터의 마이크로 배치를 병렬로 처리하여 모델, 옵티마이저, 활성화의 메모리 사용량을 줄일 수 있습니다.

연산 효율성은 각 GPU가 모델, 옵티마이저, 활성화의 슬라이스만 저장하는 ZeRO 데이터 병렬화를 통해 가능합니다. 이를 통해 통신이 다른 파이프라인 단계와 독립적으로 또는 병렬로 발생할 수 있기 때문에 데이터 병렬 노드 간 통신 대역폭이 더 높아집니다.

이 접근 방식은 수조 개의 파라미터를 가진 극대형 모델로 확장 가능합니다.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png"/>
</div>
