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

# BARThez [[barthez]]

## 개요 [[overview]]

BARThez 모델은 2020년 10월 23일, Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis에 의해 [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://huggingface.co/papers/2010.12321)에서 제안되었습니다.

이 논문의 초록:


*자기지도 학습에 의해 가능해진 귀납적 전이 학습은 자연어 처리(NLP) 분야 전반에 걸쳐 큰 반향을 일으켰으며, 
BERT와 BART와 같은 모델들은 수많은 자연어 이해 작업에서 새로운 최첨단 성과를 기록했습니다. 일부 주목할 만한 예외가 있지만, 
대부분의 사용 가능한 모델과 연구는 영어에 집중되어 있었습니다. 본 연구에서는 BARThez를 소개합니다. 
이는 (우리가 아는 한) 프랑스어를 위한 첫 번째 BART 모델입니다. 
BARThez는 과거 연구에서 얻은 매우 큰 프랑스어 단일 언어 말뭉치로 사전훈련되었으며, 
BART의 변형 방식에 맞게 조정되었습니다. 
CamemBERT 및 FlauBERT와 같은 기존의 BERT 기반 프랑스어 모델과 달리, BARThez는 생성 작업에 특히 적합합니다. 
이는 인코더뿐만 아니라 디코더도 사전훈련되었기 때문입니다. 
우리는 FLUE 벤치마크에서의 판별 작업 외에도 이 논문과 함께 공개하는 새로운 요약 데이터셋인 OrangeSum에서 BARThez를 평가했습니다. 
또한 이미 사전훈련된 다국어 BART의 사전훈련을 BARThez의 말뭉치로 계속 진행하였으며, 
결과적으로 얻어진 모델인 mBARTHez가 기본 BARThez보다 유의미한 성능 향상을 보였고, 
CamemBERT 및 FlauBERT와 동등하거나 이를 능가함을 보였습니다.*

이 모델은 [moussakam](https://huggingface.co/moussakam)이 기여했습니다. 저자의 코드는 [여기](https://github.com/moussaKam/BARThez)에서 찾을 수 있습니다.

> [!TIP]
> BARThez 구현은 🤗 BART와 동일하나, 토큰화에서 차이가 있습니다. 구성 클래스와 그 매개변수에 대한 정보는 [BART 문서](bart)를 참조하십시오. 
> BARThez 전용 토크나이저는 아래에 문서화되어 있습니다.

## 리소스 [[resources]]

- BARThez는 🤗 BART와 유사한 방식으로 시퀀스-투-시퀀스 작업에 맞춰 미세 조정될 수 있습니다. 다음을 확인하세요:
  [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).


## BarthezTokenizer [[bartheztokenizer]]

[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast [[bartheztokenizerfast]]

[[autodoc]] BarthezTokenizerFast
