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

# Autoformer[[autoformer]]

## 개요[[overview]]

The Autoformer 모델은 Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long가 제안한 [오토포머: 장기 시계열 예측을 위한 자기상관 분해 트랜스포머](https://arxiv.org/abs/2106.13008) 라는 논문에서 소개 되었습니다.

이 모델은 트랜스포머를 심층 분해 아키텍처로 확장하여, 예측 과정에서 추세와 계절성 요소를 점진적으로 분해할 수 있습니다.

해당 논문의 초록입니다:

*예측 시간을 연장하는 것은 극한 기상 조기 경보 및 장기 에너지 소비 계획과 같은 실제 응용 프로그램에 중요한 요구 사항입니다. 본 논문은 시계열의 장기 예측 문제를 연구합니다. 기존의 트랜스포머 기반 모델들은 장거리 종속성을 발견하기 위해 다양한 셀프 어텐션 메커니즘을 채택합니다. 그러나 장기 미래의 복잡한 시간적 패턴으로 인해 모델이 신뢰할 수 있는 종속성을 찾기 어렵습니다. 또한, 트랜스포머는 긴 시계열의 효율성을 위해 점별 셀프 어텐션의 희소 버전을 채택해야 하므로 정보 활용의 병목 현상이 발생합니다. 우리는 트랜스포머를 넘어서 자기상관 메커니즘을 갖춘 새로운 분해 아키텍처인 Autoformer를 설계했습니다. 우리는 시계열 분해의 전처리 관행을 깨고 이를 심층 모델의 기본 내부 블록으로 혁신했습니다. 이 설계는 Autoformer에 복잡한 시계열에 대한 점진적 분해 능력을 부여합니다. 또한, 확률 과정 이론에서 영감을 받아 시계열의 주기성을 기반으로 자기상관 메커니즘을 설계했으며, 이는 하위 시계열 수준에서 종속성 발견과 표현 집계를 수행합니다. 자기상관은 효율성과 정확도 면에서 셀프 어텐션를 능가합니다. 장기 예측에서 Autoformer는 에너지, 교통, 경제, 날씨, 질병 등 5가지 실용적 응용 분야를 포괄하는 6개 벤치마크에서 38%의 상대적 개선으로 최첨단 정확도를 달성했습니다.*

이 모델은 [elisim](https://huggingface.co/elisim)과 [kashif](https://huggingface.co/kashif)에 의해 기여 되었습니다.
원본 코드는 [이곳](https://github.com/thuml/Autoformer)에서 확인할 수 있습니다.

## 자료[[resources]]

시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

- HuggingFace 블로그에서 Autoformer 포스트를 확인하세요: [네, 그렇습니다, 트랜스포머는 시계열 예측에 효과적입니다(+ Autoformer)](https://huggingface.co/blog/autoformer)

## AutoformerConfig[[transformers.AutoformerConfig]]

[[autodoc]] AutoformerConfig

## AutoformerModel[[transformers.AutoformerModel]]

[[autodoc]] AutoformerModel
    - forward

## AutoformerForPrediction[[transformers.AutoformerForPrediction]]

[[autodoc]] AutoformerForPrediction
    - forward
