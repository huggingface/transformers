<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Video Vision Transformer (ViViT) [[video-vision-transformer-vivit]]

## 개요 [[overview]]

Vivit 모델은 Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid가 제안한 논문 [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)에서 소개되었습니다. 이 논문은 비디오 이해를 위한 pure-transformer 기반의 모델 집합 중에서 최초로 성공한 모델 중 하나를 소개합니다. 

논문의 초록은 다음과 같습니다:

*우리는 이미지 분류에서 최근 성공을 거둔 순수 트랜스포머 기반 모델을 바탕으로 비디오 분류를 위한 모델을 제안합니다. 본 모델은 입력 비디오로부터 시공간 토큰을 추출한 후, 이를 일련의 트랜스포머 레이어로 인코딩합니다. 비디오에서 발생하는 긴 토큰 시퀀스를 처리하기 위해, 입력의 공간 및 시간 차원을 분리하는 여러 효율적인 모델 변형을 제안합니다. 트랜스포머 기반 모델은 대규모 학습 데이터셋에서만 효과적이라는 것이 일반적이지만, 우리는 학습 중 모델을 효과적으로 정규화하고, 사전 학습된 이미지 모델을 활용함으로써 상대적으로 작은 데이터셋에서도 학습할 수 있는 방법을 보여줍니다. 또한, 철저한 소거(ablation) 연구를 수행하고 Kinetics 400 및 600, Epic Kitchens, Something-Something v2, Moments in Time을 포함한 여러 비디오 분류 벤치마크에서 최첨단 성과를 달성하여, 기존의 3D 합성곱 신경망 기반 방법들을 능가합니다.*

이 모델은 [jegormeister](https://huggingface.co/jegormeister)가 기여하였습니다. 원본 코드(JAX로 작성됨)는 [여기](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)에서 확인할 수 있습니다.

## VivitConfig [[transformers.VivitConfig]]

[[autodoc]] VivitConfig

## VivitImageProcessor [[transformers.VivitImageProcessor]]

[[autodoc]] VivitImageProcessor
    - preprocess

## VivitModel [[transformers.VivitModel]]

[[autodoc]] VivitModel
    - forward

## VivitForVideoClassification [[transformers.VivitForVideoClassification]]

[[autodoc]] transformers.VivitForVideoClassification
    - forward
