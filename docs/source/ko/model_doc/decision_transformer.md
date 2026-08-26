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

# 결정 트랜스포머(Decision Transformer)

## 개요[[overview]]

결정 트랜스포머(Decision Transformer) 모델은 Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch가 발표한 논문 [결정 트랜스포머 : 시퀀스 모델링을 통한 강화 학습[(Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)에서 제안되었습니다.

논문의 개요:

*우리는 강화 학습(RL)을 시퀀스 모델링 문제로 추상화하는 프레임워크를 소개합니다. 
이를 통해 트랜스포머 아키텍처의 단순성과 확장성, 그리고 이와 관련된 발전 사항을 GPT-x 및 BERT와 같은 언어 모델링에서 활용할 수 있습니다.
그 중에서도, 강화 학습 문제를 조건부 시퀀스 모델링으로 해석하는 결정 트랜스포머(Decision Transformer)를 소개합니다. 
가치 함수를 맞추거나 정책 그래디언트를 계산하는 기존의 RL 접근 방식과 달리,
결정 트랜스포머는 인과적으로 마스킹된 트랜스포머를 활용해 단순히 최적의 행동을 출력합니다. 
자기회귀 모델을 원하는 보상(리턴), 과거 상태, 행동에 조건화하여 결정 트랜스포머 모델은 원하는 보상을 달성할 수 있는 미래 행동을 생성할 수 있습니다. 
이 모델은 단순함에도 불구하고, Atari, OpenAI Gym, Key-to-Door 작업에서 최신 모델 프리 오프라인 강화 학습(RL) 기준 모델들의 성능과 대등하거나 이를 능가합니다.*

이 버전의 모델은 상태가 벡터인 작업에 사용됩니다.

이 모델은 [edbeeching](https://huggingface.co/edbeeching) 에 의해 제공되었으며, 원본 코드는 [여기](https://github.com/kzl/decision-transformer)에서 확인할 수 있습니다.

## DecisionTransformerConfig

[[autodoc]] DecisionTransformerConfig


## DecisionTransformerGPT2Model

[[autodoc]] DecisionTransformerGPT2Model
    - forward

## DecisionTransformerModel

[[autodoc]] DecisionTransformerModel
    - forward
