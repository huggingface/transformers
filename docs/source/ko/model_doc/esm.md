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

# ESM [[esm]]

## 개요 [[overview]]

이 페이지는 Meta AI의 Fundamental AI Research 팀에서 제공하는 Transformer 단백질 언어 모델에 대한 코드와 사전 훈련된 가중치를 제공합니다. 여기에는 최첨단인 ESMFold와 ESM-2, 그리고 이전에 공개된 ESM-1b와 ESM-1v가 포함됩니다. Transformer 단백질 언어 모델은 Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, Rob Fergus의 논문 [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/content/118/15/e2016239118)에서 소개되었습니다. 이 논문의 첫 번째 버전은 2019년에 [출판 전 논문](https://www.biorxiv.org/content/10.1101/622803v1?versioned=true) 형태로 공개되었습니다.

ESM-2는 다양한 구조 예측 작업에서 테스트된 모든 단일 시퀀스 단백질 언어 모델을 능가하며, 원자 수준의 구조 예측을 가능하게 합니다. 이 모델은 Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, Alexander Rives의 논문 [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://doi.org/10.1101/2022.07.20.500902)에서 공개되었습니다.

이 논문에서 함께 소개된 ESMFold는 ESM-2 스템을 사용하며, 최첨단의 정확도로 단백질 접힘 구조를 예측할 수 있는 헤드를 갖추고 있습니다. [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)와 달리, 이는 대형 사전 훈련된 단백질 언어 모델 스템의 토큰 임베딩에 의존하며, 추론 시 다중 시퀀스 정렬(MSA) 단계를 수행하지 않습니다. 이는 ESMFold 체크포인트가 완전히 "독립적"이며, 예측을 위해 알려진 단백질 시퀀스와 구조의 데이터베이스, 그리고 그와 관련 외부 쿼리 도구를 필요로 하지 않는다는 것을 의미합니다. 그리고 그 결과, 훨씬 빠릅니다.

"Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences"의 초록은 다음과 같습니다:

*인공지능 분야에서는 대규모의 데이터와 모델 용량을 갖춘 비지도 학습의 조합이 표현 학습과 통계적 생성에서 주요한 발전을 이끌어냈습니다. 생명 과학에서는 시퀀싱 기술의 성장이 예상되며, 자연 시퀀스 다양성에 대한 전례 없는 데이터가 나올 것으로 기대됩니다. 진화적 단계에서 볼 때, 단백질 언어 모델링은 생물학을 위한 예측 및 생성 인공지능을 향한 논리적인 단계에 있습니다. 이를 위해 우리는 진화적 다양성을 아우르는 2억 5천만 개의 단백질 시퀀스에서 추출한 860억 개의 아미노산에 대해 심층 컨텍스트 언어 모델을 비지도 학습으로 훈련합니다. 그 결과 모델은 그 표현에서 생물학적 속성에 대한 정보를 포함합니다. 이 표현은 시퀀스 데이터만으로 학습됩니다. 학습된 표현 공간은 아미노산의 생화학적 특성 수준에서부터 단백질의 원거리 상동성까지 구조를 반영하는 다중 규모의 조직을 가지고 있습니다. 이 표현에는 2차 및 3차 구조에 대한 정보가 인코딩되어 있으며, 선형 전사에 의해 식별 될 수 있습니다. 표현 학습은 돌연변이에 의한 효과와 2차 구조의 최첨단 지도 예측을 가능하게 하고, 넓은 범위의 접촉 부위 예측을 위한 최첨단 특징을 향상시킵니다.*

"Language models of protein sequences at the scale of evolution enable accurate structure prediction"의 초록은 다음과 같습니다:

*대형 언어 모델은 최근 규모가 커짐에 따라 긴급한 기능을 개발하여 단순한 패턴 매칭을 넘어 더 높은 수준의 추론을 수행하고 생생한 이미지와 텍스트를 생성하는 것으로 나타났습니다. 더 작은 규모에서 훈련된 단백질 시퀀스의 언어 모델이 연구되었지만, 그들이 규모가 커짐에 따라 생물학에 대해 무엇을 배우는지는 거의 알려져 있지 않습니다. 이 연구에서 우리는 현재까지 평가된 가장 큰 150억 개의 매개변수를 가진 모델을 훈련합니다. 우리는 모델이 규모가 커짐에 따라 단일 아미노산의 해상도로 단백질의 3차원 구조를 예측할 수 있는 정보를 학습한다는 것을 발견했습니다. 우리는 개별 단백질 시퀀스로부터 직접 고정밀 원자 수준의 엔드-투-엔드 구조 예측을 하기 위한 ESMFold를 제시합니다. ESMFold는 언어 모델에 잘 이해되는 낮은 퍼플렉서티를 가진 시퀀스에 대해 AlphaFold2와 RoseTTAFold와 유사한 정확도를 가지고 있습니다. ESMFold의 추론은 AlphaFold2보다 한 자릿수 빠르며, 메타게놈 단백질의 구조적 공간을 실용적인 시간 내에 탐색할 수 있게 합니다.*

원본 코드는 [여기](https://github.com/facebookresearch/esm)에서 찾을 수 있으며, Meta AI의 Fundamental AI Research 팀에서 개발되었습니다. ESM-1b, ESM-1v, ESM-2는 [jasonliu](https://huggingface.co/jasonliu)와 [Matt](https://huggingface.co/Rocketknight1)에 의해 HuggingFace에 기여되었습니다.

ESMFold는 [Matt](https://huggingface.co/Rocketknight1)와 [Sylvain](https://huggingface.co/sgugger)에 의해 HuggingFace에 기여되었으며, 이 과정에서 많은 도움을 준 Nikita Smetanin, Roshan Rao, Tom Sercu에게 큰 감사를 드립니다!

## 사용 팁 [[usage-tips]]

- ESM 모델은 마스크드 언어 모델링(MLM) 목표로 훈련되었습니다.
- HuggingFace의 ESMFold 포트는 [openfold](https://github.com/aqlaboratory/openfold) 라이브러리의 일부를 사용합니다. `openfold` 라이브러리는 Apache License 2.0에 따라 라이선스가 부여됩니다.

## 리소스 [[resources]]

- [텍스트 분류 작업 가이드](../tasks/sequence_classification)
- [토큰 분류 작업 가이드](../tasks/token_classification)
- [마스킹드 언어 모델링 작업 가이드](../tasks/masked_language_modeling)

## EsmConfig [[transformers.EsmConfig]]

[[autodoc]] EsmConfig
    - all

## EsmTokenizer [[transformers.EsmTokenizer]]

[[autodoc]] EsmTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

<frameworkcontent>
<pt>

## EsmModel [[transformers.EsmModel]]

[[autodoc]] EsmModel
    - forward

## EsmForMaskedLM [[transformers.EsmForMaskedLM]]

[[autodoc]] EsmForMaskedLM
    - forward

## EsmForSequenceClassification [[transformers.EsmForSequenceClassification]]

[[autodoc]] EsmForSequenceClassification
    - forward

## EsmForTokenClassification [[transformers.EsmForTokenClassification]]

[[autodoc]] EsmForTokenClassification
    - forward

## EsmForProteinFolding [[transformers.EsmForProteinFolding]]

[[autodoc]] EsmForProteinFolding
    - forward

</pt>
<tf>

## TFEsmModel [[transformers.TFEsmModel]]

[[autodoc]] TFEsmModel
    - call

## TFEsmForMaskedLM [[transformers.TFEsmForMaskedLM]]

[[autodoc]] TFEsmForMaskedLM
    - call

## TFEsmForSequenceClassification [[transformers.TFEsmForSequenceClassification]]

[[autodoc]] TFEsmForSequenceClassification
    - call

## TFEsmForTokenClassification [[transformers.TFEsmForTokenClassification]]

[[autodoc]] TFEsmForTokenClassification
    - call

</tf>
</frameworkcontent>
