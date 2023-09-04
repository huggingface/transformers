<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BERTology

BERT와 같은 대규모 트랜스포머의 내부 동작을 조사하는 연구 분야가 점점 더 중요해지고 있습니다.
혹자는 "BERTology"라 칭하기도 합니다. 이 분야의 좋은 예시는 다음과 같습니다:


- BERT는 고전적인 NLP 파이프라인의 재발견 - Ian Tenney, Dipanjan Das, Ellie Pavlick:
  https://arxiv.org/abs/1905.05950
- 16개의 헤드가 정말로 1개보다 나은가? - Paul Michel, Omer Levy, Graham Neubig:
  https://arxiv.org/abs/1905.10650
- BERT는 무엇을 보는가? BERT의 어텐션 분석 - Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning:
  https://arxiv.org/abs/1906.04341
- CAT-probing: 프로그래밍 언어에 대해 사전훈련된 모델이 어떻게 코드 구조를 보는지 알아보기 위한 메트릭 기반 접근 방법:
  https://arxiv.org/abs/2210.04633

우리는 이 새로운 연구 분야의 발전을 돕기 위해, BERT/GPT/GPT-2 모델에 내부 표현을 살펴볼 수 있는 몇 가지 기능을 추가했습니다.
이 기능들은 주로 Paul Michel의 훌륭한 작업을 참고하여 개발되었습니다
(https://arxiv.org/abs/1905.10650):


- BERT/GPT/GPT-2의 모든 은닉 상태에 접근하기,
- BERT/GPT/GPT-2의 각 헤드의 모든 어텐션 가중치에 접근하기,
- 헤드의 출력 값과 그래디언트를 검색하여 헤드 중요도 점수를 계산하고 https://arxiv.org/abs/1905.10650에서 설명된 대로 헤드를 제거하는 기능을 제공합니다.

이러한 기능들을 이해하고 직접 사용해볼 수 있도록 [bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py) 예제 스크립트를 추가했습니다. 이 예제 스크립트에서는 GLUE에 대해 사전훈련된 모델에서 정보를 추출하고 모델을 가지치기(prune)해봅니다.
