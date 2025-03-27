<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Swin Transformer V2 [[swin-transformer-v2]]

## 개요 [[overview]]

Swin Transformer V2는  Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo가 제안한 논문 [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)에서 소개되었습니다.

논문의 초록은 다음과 같습니다:

*대규모 NLP 모델들은 언어 작업에서의 성능을 크게 향상하며, 성능이 포화하는 징후를 보이지 않습니다. 또한, 사람과 유사한 few-shot 학습 능력을 보여줍니다. 이 논문은 대규모 모델을 컴퓨터 비전 분야에서 탐구하고자 합니다. 대형 비전 모델을 훈련하고 적용하는 데 있어 세 가지 주요 문제를 다룹니다: 훈련 불안정성, 사전 학습과 파인튜닝 간의 해상도 차이, 그리고 레이블이 달린 데이터에 대한 높은 요구입니다. 세 가지 주요 기법을 제안합니다: 1) 훈련 안정성을 개선하기 위한 residual-post-norm 방법과 cosine attention의 결합; 2) 저해상도 이미지로 사전 학습된 모델을 고해상도 입력으로 전이할 수 있는 log-spaced continuous position bias 방법; 3) 레이블이 달린 방대한 이미지의 필요성을 줄이기 위한 self-supervised 사전 학습 방법인 SimMIM입니다. 이러한 기법들을 통해 30억 개의 파라미터를 가진 Swin Transformer V2 모델을 성공적으로 훈련하였으며, 이는 현재까지 가장 크고 고밀도의 비전 모델로, 최대 1,536×1,536 해상도의 이미지를 다룰 수 있습니다. 이 모델은 ImageNet-V2 이미지 분류, COCO 객체 탐지, ADE20K 의미론적 분할, Kinetics-400 비디오 행동 분류 등 네 가지 대표적인 비전 작업에서 새로운 성능 기록을 세웠습니다. 또한, 우리의 훈련은 Google의 billion-level 비전 모델과 비교해 40배 적은 레이블이 달린 데이터와 40배 적은 훈련 시간으로 이루어졌다는 점에서 훨씬 더 효율적입니다.*

이 모델은 [nandwalritik](https://huggingface.co/nandwalritik)이 기여하였습니다.
원본 코드는 [여기](https://github.com/microsoft/Swin-Transformer)에서 확인할 수 있습니다.

## 리소스 [[resources]]

Swin Transformer v2의 사용을 도울 수 있는 Hugging Face 및 커뮤니티(🌎로 표시)의 공식 자료 목록입니다.


<PipelineTag pipeline="image-classification"/>

- [`Swinv2ForImageClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)을 통해 지원됩니다.
- 관련 자료: [이미지 분류 작업 가이드](../tasks/image_classification)

또한:

- [`Swinv2ForMaskedImageModeling`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)를 통해 지원됩니다.

새로운 자료를 추가하고 싶으시다면, 언제든지 Pull Request를 열어주세요! 저희가 검토해 드릴게요. 이때, 추가하는 자료는 기존 자료와 중복되지 않고 새로운 내용을 보여주는 자료여야 합니다.

## Swinv2Config [[transformers.Swinv2Config]]

[[autodoc]] Swinv2Config

## Swinv2Model [[transformers.Swinv2Model]]

[[autodoc]] Swinv2Model
    - forward

## Swinv2ForMaskedImageModeling [[transformers.Swinv2ForMaskedImageModeling]]

[[autodoc]] Swinv2ForMaskedImageModeling
    - forward

## Swinv2ForImageClassification [[transformers.Swinv2ForImageClassification]]

[[autodoc]] transformers.Swinv2ForImageClassification
    - forward
