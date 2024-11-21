<!--Copyright 2022 The HuggingFace Team and Microsoft. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Graphormer[[graphormer]]

<Tip warning={true}>

이 모델은 유지 보수 모드로만 운영되며, 코드를 변경하는 새로운 PR(Pull Request)은 받지 않습니다.
이 모델을 실행하는 데 문제가 발생한다면, 이 모델을 지원하는 마지막 버전인 v4.40.2를 다시 설치해 주세요. 다음 명령어를 실행하여 재설치할 수 있습니다: `pip install -U transformers==4.40.2`.

</Tip>

## 개요[[overview]]

Graphormer 모델은 Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu가 제안한 [트랜스포머가 그래프 표현에 있어서 정말 약할까?](https://arxiv.org/abs/2106.05234) 라는 논문에서 소개되었습니다. Graphormer는 그래프 트랜스포머 모델입니다. 텍스트 시퀀스 대신 그래프에서 계산을 수행할 수 있도록 수정되었으며, 전처리와 병합 과정에서 임베딩과 관심 특성을 생성한 후 수정된 어텐션을 사용합니다.

해당 논문의 초록입니다:

*트랜스포머 아키텍처는 자연어 처리와 컴퓨터 비전 등 많은 분야에서 지배적인 선택을 받고 있는 아키텍처 입니다. 그러나 그래프 수준 예측 리더보드 상에서는 주류 GNN 변형모델들에 비해 경쟁력 있는 성능을 달성하지 못했습니다. 따라서 트랜스포머가 그래프 표현 학습에서 어떻게 잘 수행될 수 있을지는 여전히 미스터리였습니다. 본 논문에서는 Graphormer를 제시함으로써 이 미스터리를 해결합니다. Graphormer는 표준 트랜스포머 아키텍처를 기반으로 구축되었으며, 특히 최근의 OpenGraphBenchmark Large-Scale Challenge(OGB-LSC)의 광범위한 그래프 표현 학습 작업에서 탁월한 결과를 얻을 수 있었습니다. 그래프에서 트랜스포머를 활용하는데 핵심은 그래프의 구조적 정보를 모델에 효과적으로 인코딩하는 것입니다. 이를 위해 우리는 Graphormer가 그래프 구조 데이터를 더 잘 모델링할 수 있도록 돕는 몇 가지 간단하면서도 효과적인 구조적 인코딩 방법을 제안합니다. 또한, 우리는 Graphormer의 표현을 수학적으로 특성화하고, 그래프의 구조적 정보를 인코딩하는 우리의 방식으로 많은 인기 있는 GNN 변형모델들이 Graphormer의 특수한 경우로 포함될 수 있음을 보여줍니다.*

이 모델은 [clefourrier](https://huggingface.co/clefourrier)가 기여했습니다. 원본 코드는 [이곳](https://github.com/microsoft/Graphormer)에서 확인할 수 있습니다.

## 사용 팁[[usage-tips]]

이 모델은 큰 그래프(100개 이상의 노드개수/엣지개수)에서는 메모리 사용량이 폭발적으로 증가하므로 잘 작동하지 않습니다. 대안으로 배치 크기를 줄이거나, RAM을 늘리거나 또는 algos_graphormer.pyx 파일의 `UNREACHABLE_NODE_DISTANCE` 매개변수를 줄이는 방법도 있지만, 700개 이상의 노드개수/엣지개수를 처리하기에는 여전히 어려울 것입니다.

이 모델은 토크나이저를 사용하지 않고, 대신 훈련 중에 특별한 콜레이터(collator)를 사용합니다.

## GraphormerConfig[[transformers.GraphormerConfig]]

[[autodoc]] GraphormerConfig

## GraphormerModel[[transformers.GraphormerModel]]

[[autodoc]] GraphormerModel
    - forward

## GraphormerForGraphClassification[[transformers.GraphormerForGraphClassification]]

[[autodoc]] GraphormerForGraphClassification
    - forward
