<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 성능 및 확장성 [[performance-and-scalability]]

더 큰 트랜스포머 모델을 학습하고 제품에 배포하는 것은 다양한 도전과제와 함께 진행됩니다. 학습 중에는 모델이 사용 가능한 GPU 메모리보다 더 많은 메모리를 필요로 하거나 학습 속도가 매우 느릴 수 있으며, 추론을 위해 배포할 때는 제품 환경에서 요구되는 처리량으로 인해 과부하를 받을 수 있습니다. 이 문서는 이러한 도전과제를 극복하고 사용 사례에 가장 적합한 설정을 찾도록 도움을 주기 위해 설계되었습니다. 학습과 추론으로 가이드를 분할했는데, 각각 다른 도전과제와 해결책이 있습니다. 그리고 그 안에는 학습에 대한 단일 GPU 대 다중 GPU 또는 추론에 대한 CPU 대 GPU와 같은 다양한 종류의 하드웨어 설정에 대한 별도의 가이드가 있습니다.

![perf_overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf_overview.png)

이 문서는 시나리오에 유용한 방법들에 대한 개요와 진입점 역할을 합니다.

## 학습 [[training]]

효율적인 트랜스포머 모델 학습에는 GPU나 TPU와 같은 가속기가 필요합니다. 가장 일반적인 경우는 단일 GPU만 사용하는 경우지만, 다중 GPU 및 CPU 학습에 대한 섹션도 있습니다(곧 더 많은 내용이 추가될 예정).

<Tip>

 참고: 단일 GPU 섹션에서 소개된 대부분의 전략(예: 혼합 정밀도 학습 또는 그라디언트 누적)은 일반적인 모델 학습에도 적용되므로, 다중 GPU나 CPU 학습과 같은 다음 섹션에 들어가기 전에 꼭 살펴보시기 바랍니다.

</Tip>

### 단일 GPU [[single-gpu]]

단일 GPU에서 큰 모델을 학습하는 것은 도전적일 수 있지만, 가능하게 만드는 도구와 방법이 있습니다. 이 섹션에서는 혼합 정밀도 학습, 그라디언트 누적 및 체크포인팅, 효율적인 옵티마이저, 최적의 배치 크기를 결정하기 위한 전략 등에 대해 논의합니다.

[단일 GPU 학습 섹션으로 이동](perf_train_gpu_one)

### 다중 GPU [[multigpu]]

일부 경우에는 단일 GPU에서의 학습이 여전히 너무 느리거나 큰 모델을 수용할 수 없는 경우입니다. 다중 GPU 설정으로 전환하는 것은 논리적인 단계이지만, 한 번에 여러 GPU에서 학습하는 것은 새로운 결정을 요구합니다. 각 GPU에는 모델의 전체 복사본이 있는지 아니면 모델 자체도 분산되는지에 대한 문제입니다. 이 섹션에서는 데이터, 텐서 및 파이프라인 병렬성에 대해 살펴봅니다.

[다중 GPU 학습 섹션으로 이동](perf_train_gpu_many)

### CPU [[cpu]]


[CPU 학습 섹션으로 이동](perf_train_cpu)


### TPU [[tpu]]

[_곧 제공될 예정_](perf_train_tpu)

### 특수한 하드웨어 [[specialized-hardware]]

[_곧 제공될 예정_](perf_train_special)

## 추론 [[inference]]

제품 환경에서 큰 모델을 사용한 효율적인 추론은 학습과 마찬가지로 도전적일 수 있습니다. 다음 섹션에서는 CPU 및 단일/다중 GPU 환경에서 추론을 실행하는 단계에 대해 알아봅니다.

### CPU [[cpu]]

[CPU 추론 섹션으로 이동](perf_infer_cpu)

### 단일 GPU [[single-gpu]]

[단일 GPU 추론 섹션으로 이동](perf_infer_gpu_one)

### 다중 GPU [[multigpu]]

[다중 GPU 추론 섹션으로 이동](perf_infer_gpu_many)

### 특수한 하드웨어 [[specialized-hardware]]

[_곧 제공될 예정_](perf_infer_special)

## 하드웨어 [[hardware]]

하드웨어 섹션에서는 자신의 딥러닝 머신을 구축할 때 유용한 팁과 요령을 찾을 수 있습니다.

[하드웨어 섹션으로 이동](perf_hardware)


## 기여하기 [[contribute]]

이 문서는 완성되지 않은 상태이며, 추가해야 할 내용이나 수정 사항이 많이 있습니다. 따라서 추가하거나 수정할 내용이 있으면 주저하지 말고 PR을 열어 주시거나, 자세한 내용을 논의하기 위해 Issue를 시작해 주시기 바랍니다.

A가 B보다 좋다고 하는 기여를 할 때는, 재현 가능한 벤치마크와/또는 해당 정보의 출처 링크를 포함해주세요(당신으로부터의 직접적인 정보가 아닌 경우).