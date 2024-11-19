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

# TimeSformer [[timesformer]]

## 개요 [[overview]]

TimeSformer 모델은 Facebook Research에서 제안한 [TimeSformer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)에서 소개되었습니다. 이 연구는 첫 번째 비디오 Transformer로서, 행동 인식 분야에서 중요한 이정표가 되었습니다. 또한 Transformer 기반의 비디오 이해 및 분류 논문에 많은 영감을 주었습니다.

논문의 초록은 다음과 같습니다. 

*우리는 공간과 시간에 걸쳐 셀프 어텐션만을 사용하는 합성곱이 없는(convolution-free) 비디오 분류 방법을 제안합니다. 이 방법은 “TimeSformer”라고 불리며, 표준 Transformer 아키텍처를 비디오에 적용하여 프레임 수준 패치 시퀀스로부터 직접 시공간적 특징을 학습할 수 있게 합니다. 우리의 실험적 연구는 다양한 셀프 어텐션 방식을 비교하며, 시간적 어텐션과 공간적 어텐션을 각각의 블록 내에서 별도로 적용하는 “분할 어텐션” 방식이 고려된 설계 선택 중 가장 우수한 비디오 분류 정확도를 제공한다는 것을 시사합니다. 이 혁신적인 설계에도 불구하고, TimeSformer는 Kinetics-400 및 Kinetics-600을 포함한 여러 행동 인식 벤치마크에서 최첨단 결과를 달성했으며, 현재까지 보고된 가장 높은 정확도를 기록했습니다. 마지막으로, 3D 합성곱 네트워크와 비교했을 때, TimeSformer는 더 빠르게 학습할 수 있으며, 약간의 정확도 저하를 감수하면 테스트 효율성이 크게 향상되고, 1분 이상의 긴 비디오 클립에도 적용할 수 있습니다. 코드와 모델은 다음 링크에서 확인할 수 있습니다: [https URL 링크](https://github.com/facebookresearch/TimeSformer).*

이 모델은 [fcakyon](https://huggingface.co/fcakyon)이 기여하였습니다.
원본 코드는 [여기](https://github.com/facebookresearch/TimeSformer)에서 확인할 수 있습니다.

## 사용 팁 [[usage-tips]]

다양한 사전 학습된 모델의 변형들이 있습니다. 사용하려는 데이터셋에 맞춰 사전 학습된 모델을 선택해야 합니다. 또한, 모델 크기에 따라 클립당 입력 프레임 수가 달라지므로, 사전 학습된 모델을 선택할 때 이 매개변수를 고려해야 합니다.


## 리소스 [[resources]]

- [Video classification task guide](../tasks/video_classification)

## TimesformerConfig [[transformers.TimesformerConfig]]

[[autodoc]] TimesformerConfig

## TimesformerModel [[transformers.TimesformerModel]]

[[autodoc]] TimesformerModel
    - forward

## TimesformerForVideoClassification [[transformers.TimesformerForVideoClassification]]

[[autodoc]] TimesformerForVideoClassification
    - forward