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

# 궤적 트랜스포머[[trajectory-transformer]]

<Tip warning={true}>


이 모델은 유지 보수 모드로만 운영되며, 코드를 변경하는 새로운 PR(Pull Request)은 받지 않습니다.
이 모델을 실행하는 데 문제가 발생한다면, 이 모델을 지원하는 마지막 버전인 v4.30.0를 다시 설치해 주세요. 다음 명령어를 실행하여 재설치할 수 있습니다: `pip install -U transformers==4.30.0`.

</Tip>

## 개요[[overview]]

Trajectory Transformer 모델은 Michael Janner, Qiyang Li, Sergey Levine이 제안한 [하나의 커다란 시퀀스 모델링 문제로서의 오프라인 강화학습](https://arxiv.org/abs/2106.02039)라는 논문에서 소개되었습니다.

해당 논문의 초록입니다:

*강화학습(RL)은 일반적으로 마르코프 속성을 활용하여 시간에 따라 문제를 인수분해하면서 정적 정책이나 단일 단계 모델을 추정하는 데 중점을 둡니다. 하지만 우리는 RL을 높은 보상 시퀀스로 이어지는 행동 시퀀스를 생성하는 것을 목표로 하는 일반적인 시퀀스 모델링 문제로 볼 수도 있습니다. 이러한 관점에서, 자연어 처리와 같은 다른 도메인에서 잘 작동하는 고용량 시퀀스 예측 모델이 RL 문제에도 효과적인 해결책을 제공할 수 있는지 고려해 볼 만합니다. 이를 위해 우리는 RL을 시퀀스 모델링의 도구로 어떻게 다룰 수 있는지 탐구하며, 트랜스포머 아키텍처를 사용하여 궤적에 대한 분포를 모델링하고 빔 서치를 계획 알고리즘으로 재활용합니다. RL을 시퀀스 모델링 문제로 프레임화하면 다양한 설계 결정이 단순화되어, 오프라인 RL 알고리즘에서 흔히 볼 수 있는 많은 구성 요소를 제거할 수 있습니다. 우리는 이 접근 방식의 유연성을 장기 동역학 예측, 모방 학습, 목표 조건부 RL, 오프라인 RL에 걸쳐 입증합니다. 더 나아가, 이 접근 방식을 기존의 모델 프리 알고리즘과 결합하여 희소 보상, 장기 과제에서 최신 계획기(planner)를 얻을 수 있음을 보여줍니다.*

이 모델은 [CarlCochet](https://huggingface.co/CarlCochet)에 의해 기여되었습니다.
원본 코드는 [이곳](https://github.com/jannerm/trajectory-transformer)에서 확인할 수 있습니다.

## 사용 팁[[usage-tips]]

이 트랜스포머는 심층 강화학습에 사용됩니다. 사용하려면 이전의 모든 타임스텝에서의 행동, 상태, 보상으로부터 시퀀스를 생성해야 합니다. 이 모델은 이 모든 요소를 함께 하나의 큰 시퀀스(궤적)로 취급합니다.

## TrajectoryTransformerConfig[[transformers.TrajectoryTransformerConfig]]

[[autodoc]] TrajectoryTransformerConfig

## TrajectoryTransformerModel[[transformers.TrajectoryTransformerModel]]

[[autodoc]] TrajectoryTransformerModel
    - forward
