<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Informer[[informer]]

## 개요[[overview]]

The Informer 모델은 Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang가 제안한 [Informer: 장기 시퀀스 시계열 예측(LSTF)을 위한 더욱 효율적인 트랜스포머(Beyond Efficient Transformer)](https://arxiv.org/abs/2012.07436)라는 논문에서 소개되었습니다.

이 방법은 확률적 어텐션 메커니즘을 도입하여 "게으른" 쿼리가 아닌 "활성" 쿼리를 선택하고, 희소 트랜스포머를 제공하여 기존 어텐션의 이차적 계산 및 메모리 요구사항을 완화합니다.

해당 논문의 초록입니다:

*실제로 많은 응용프로그램에서는 장기 시퀀스 시계열 예측(LSTF)을 필요로 합니다. LSTF는 출력 - 입력 간 정확한 장기 의존성 결합도를 포착해내는 높은 예측 능력을 모델에 요구합니다. 최근 연구들은 예측 능력을 향상시킬 수 있는 트랜스포머의 잠재력을 보여주고 있습니다. 그러나, 트랜스포머를 LSTF에 직접 적용하지 못하도록 막는 몇 심각한 문제점들이 있습니다. 예로, 이차 시간 복잡도, 높은 메모리 사용량, 인코더-디코더 아키텍처의 본질적 한계를 들 수 있습니다. 이러한 문제를 해결하기 위해 LSTF를 위한 효율적인 트랜스포머 기반 모델인 Informer를 설계했습니다.

Informer의 세가지 독특한 특성:
(i) ProbSparse 셀프 어텐션 메커니즘으로, 시간 복잡도와 메모리 사용량에서 O(L logL)를 달성하며 시퀀스 의존성 정렬에서 비교 가능한 성능을 보입니다. 
(ii) 셀프 어텐션 증류는 계단식 레이어 입력을 반으로 줄여 지배적인 어텐션을 강조하고 극단적으로 긴 입력 시퀀스를 효율적으로 처리합니다. 
(iii) 생성 스타일 디코더는 개념적으로 단순하지만 장기 시계열 시퀀스를 단계별 방식이 아닌 한 번의 전방 연산으로 예측하여 장기 시퀀스 예측의 추론 속도를 크게 향상시킵니다. 4개의 대규모 데이터셋에 걸친 광범위한 실험은 Informer가 기존 방법들을 크게 능가하며 LSTF 문제에 새로운 해결책을 제공함을 보여줍니다.*

이 모델은 [elisim](https://huggingface.co/elisim)와 [kashif](https://huggingface.co/kashif)가 기여했습니다.
원본 코드는 [이곳](https://github.com/zhouhaoyi/Informer2020)에서 확인할 수 있습니다.

## 자료[[resources]]

시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

- HuggingFace 블로그에서 Informer 포스트를 확인하세요: [Informer를 활용한 다변량 확률적 시계열 예측](https://huggingface.co/blog/informer)

## InformerConfig[[transformers.InformerConfig]]

[[autodoc]] InformerConfig

## InformerModel[[transformers.InformerModel]]

[[autodoc]] InformerModel
    - forward

## InformerForPrediction[[transformers.InformerForPrediction]]

[[autodoc]] InformerForPrediction
    - forward