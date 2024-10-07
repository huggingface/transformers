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

# DeBERTa-v2

## 개요


DeBERTa 모델은 Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen이 작성한 [DeBERTa: 분리된 어텐션을 활용한 디코딩 강화 BERT](https://arxiv.org/abs/2006.03654)이라는 논문에서 제안되었습니다. 이 모델은 2018년 Google이 발표한 BERT 모델과 2019년 Facebook이 발표한 RoBERTa 모델을 기반으로 합니다.
DeBERTa는 RoBERTa에서 사용된 데이터의 절반만을 사용하여 분리된(disentangled) 어텐션과 향상된 마스크 디코더 학습을 통해 RoBERTa를 개선했습니다.

논문의 초록은 다음과 같습니다:

*사전 학습된 신경망 언어 모델의 최근 발전은 많은 자연어 처리(NLP) 작업의 성능을 크게 향상시켰습니다. 본 논문에서는 두 가지 새로운 기술을 사용하여 BERT와 RoBERTa 모델을 개선한 새로운 모델 구조인 DeBERTa를 제안합니다. 첫 번째는 분리된 어텐션 메커니즘으로, 각 단어가 내용과 위치를 각각 인코딩하는 두 개의 벡터로 표현되며, 단어들 간의 어텐션 가중치는 내용과 상대적 위치에 대한 분리된 행렬을 사용하여 계산됩니다. 두 번째로, 모델 사전 학습을 위해 마스킹된 토큰을 예측하는 출력 소프트맥스 층을 대체하는 향상된 마스크 디코더가 사용됩니다. 우리는 이 두 가지 기술이 모델 사전 학습의 효율성과 다운스트림 작업의 성능을 크게 향상시킨다는 것을 보여줍니다. RoBERTa-Large와 비교했을 때, 절반의 학습 데이터로 학습된 DeBERTa 모델은 광범위한 NLP 작업에서 일관되게 더 나은 성능을 보여주며, MNLI에서 +0.9%(90.2% vs 91.1%), SQuAD v2.0에서 +2.3%(88.4% vs 90.7%), RACE에서 +3.6%(83.2% vs 86.8%)의 성능 향상을 달성했습니다. DeBERTa 코드와 사전 학습된 모델은 https://github.com/microsoft/DeBERTa 에서 공개될 예정입니다.*


다음 정보들은 [원본 구현 저장소](https://github.com/microsoft/DeBERTa)에서 보실 수 있습니다. DeBERTa v2는 DeBERTa의 두번째 모델입니다. 
DeBERTa v2는 SuperGLUE 단일 모델 제출에 사용된 1.5B 모델을 포함하며, 인간 기준점(베이스라인) 89.8점 대비 89.9점을 달성했습니다. 저자의 
[블로그](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/)에서 더 자세한 정보를 확인할 수 있습니다.

v2의 새로운 점:

- **어휘(Vocabulary)** v2에서는 학습 데이터로부터 구축된 128K 크기의 새로운 어휘를 사용하도록 토크나이저가 변경되었습니다. GPT2 기반 토크나이저 대신, 이제는 [센텐스피스 기반](https://github.com/google/sentencepiece) 토크나이저를 사용합니다.
- **nGiE[n그램 유도(Induced) 입력 인코딩]** DeBERTa-v2 모델은 입력 토큰들의 지역적 의존성을 더 잘 학습하기 위해 첫 번째 트랜스포머 층과 함께 추가적인 합성곱 층을 사용합니다.
- **어텐션 층에서 위치 투영 행렬과 내용 투영 행렬 공유** 이전 실험들을 기반으로, 이는 성능에 영향을 주지 않으면서 매개변수를 절약할 수 있습니다.
- **상대적 위치를 인코딩하기 위한 버킷 적용** DeBERTa-v2 모델은 T5와 유사하게 상대적 위치를 인코딩하기 위해 로그 버킷을 사용합니다.
- **900M 모델 & 1.5B 모델** 900M과 1.5B, 두 가지 추가 모델 크기가 제공되며, 이는 다운스트림 작업의 성능을 크게 향상시킵니다.

[DeBERTa](https://huggingface.co/DeBERTa) 모델의 텐서플로 2.0 구현은 [kamalkraj](https://huggingface.co/kamalkraj)가 기여했습니다. 원본 코드는 [이곳](https://github.com/microsoft/DeBERTa)에서 확인하실 수 있습니다.

## 자료

- [텍스트 분류 작업 가이드](../tasks/sequence_classification)
- [토큰 분류 작업 가이드](../tasks/token_classification)
- [질의응답 작업 가이드](../tasks/question_answering)
- [마스크 언어 모델링 작업 가이드](../tasks/masked_language_modeling)
- [다중 선택 작업 가이드](../tasks/multiple_choice)

## DebertaV2Config

[[autodoc]] DebertaV2Config

## DebertaV2Tokenizer

[[autodoc]] DebertaV2Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaV2TokenizerFast

[[autodoc]] DebertaV2TokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

<frameworkcontent>
<pt>

## DebertaV2Model

[[autodoc]] DebertaV2Model
    - forward

## DebertaV2PreTrainedModel

[[autodoc]] DebertaV2PreTrainedModel
    - forward

## DebertaV2ForMaskedLM

[[autodoc]] DebertaV2ForMaskedLM
    - forward

## DebertaV2ForSequenceClassification

[[autodoc]] DebertaV2ForSequenceClassification
    - forward

## DebertaV2ForTokenClassification

[[autodoc]] DebertaV2ForTokenClassification
    - forward

## DebertaV2ForQuestionAnswering

[[autodoc]] DebertaV2ForQuestionAnswering
    - forward

## DebertaV2ForMultipleChoice

[[autodoc]] DebertaV2ForMultipleChoice
    - forward

</pt>
<tf>

## TFDebertaV2Model

[[autodoc]] TFDebertaV2Model
    - call

## TFDebertaV2PreTrainedModel

[[autodoc]] TFDebertaV2PreTrainedModel
    - call

## TFDebertaV2ForMaskedLM

[[autodoc]] TFDebertaV2ForMaskedLM
    - call

## TFDebertaV2ForSequenceClassification

[[autodoc]] TFDebertaV2ForSequenceClassification
    - call

## TFDebertaV2ForTokenClassification

[[autodoc]] TFDebertaV2ForTokenClassification
    - call

## TFDebertaV2ForQuestionAnswering

[[autodoc]] TFDebertaV2ForQuestionAnswering
    - call

## TFDebertaV2ForMultipleChoice

[[autodoc]] TFDebertaV2ForMultipleChoice
    - call

</tf>
</frameworkcontent>
