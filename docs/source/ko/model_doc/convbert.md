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

# ConvBERT [[convbert]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=convbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-convbert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/conv-bert-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 개요 [[overview]]

ConvBERT 모델은 Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan에 의해 제안되었으며, 제안 논문 제목은 [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)입니다.

논문의 초록은 다음과 같습니다:

*BERT와 그 변형 모델과 같은 사전 학습된 언어 모델들은 최근 다양한 자연어 이해 과제에서 놀라운 성과를 이루었습니다. 그러나 BERT는 글로벌 셀프 어텐션 블록에 크게 의존하기 때문에 메모리 사용량이 많고 계산 비용이 큽니다. 모든 어텐션 헤드가 글로벌 관점에서 어텐션 맵을 생성하기 위해 입력 시퀀스 전체를 탐색하지만, 일부 헤드는 로컬 종속성만 학습할 필요가 있다는 것을 발견했습니다. 이는 불필요한 계산이 포함되어 있음을 의미합니다. 따라서 우리는 이러한 self-attention 헤드들을 대체하여 로컬 종속성을 직접 모델링하기 위해 새로운 span 기반 동적 컨볼루션을 제안합니다. 새로운 컨볼루션 헤드와 나머지 self-attention 헤드들이 결합하여 글로벌 및 로컬 문맥 학습에 더 효율적인 혼합 어텐션 블록을 구성합니다. 우리는 BERT에 이 혼합 어텐션 설계를 적용하여 ConvBERT 모델을 구축했습니다. 실험 결과, ConvBERT는 다양한 다운스트림 과제에서 BERT 및 그 변형 모델보다 더 우수한 성능을 보였으며, 훈련 비용과 모델 파라미터 수가 더 적었습니다. 특히 ConvBERTbase 모델은 GLUE 스코어 86.4를 달성하여 ELECTRAbase보다 0.7 높은 성과를 보이며, 훈련 비용은 1/4 이하로 줄었습니다. 코드와 사전 학습된 모델은 공개될 예정입니다.*

이 모델은 [abhishek](https://huggingface.co/abhishek)에 의해 기여되었으며, 원본 구현은 여기에서 찾을 수 있습니다 : https://github.com/yitu-opensource/ConvBert



## 사용 팁 [[usage-tips]]
ConvBERT 훈련 팁은 BERT와 유사합니다. 사용 팁은 [BERT 문서](bert).를 참고하십시오.


## 리소스 [[resources]]

- [텍스트 분류 작업 가이드 (Text classification task guide)](../tasks/sequence_classification)
- [토큰 분류 작업 가이드 (Token classification task guide)](../tasks/token_classification)
- [질의응답 작업 가이드 (Question answering task guide)](../tasks/question_answering)
- [마스킹된 언어 모델링 작업 가이드 (Masked language modeling task guide)](../tasks/masked_language_modeling)
- [다중 선택 작업 가이드 (Multiple choice task guide)](../tasks/multiple_choice)

## ConvBertConfig [[transformers.ConvBertConfig]]

[[autodoc]] ConvBertConfig

## ConvBertTokenizer [[transformers.ConvBertTokenizer]]

[[autodoc]] ConvBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## ConvBertTokenizerFast [[transformers.ConvBertTokenizerFast]]

[[autodoc]] ConvBertTokenizerFast

<frameworkcontent>
<pt>

## ConvBertModel [[transformers.ConvBertModel]]

[[autodoc]] ConvBertModel
    - forward

## ConvBertForMaskedLM [[transformers.ConvBertForMaskedLM]]

[[autodoc]] ConvBertForMaskedLM
    - forward

## ConvBertForSequenceClassification [[transformers.ConvBertForSequenceClassification]]

[[autodoc]] ConvBertForSequenceClassification
    - forward

## ConvBertForMultipleChoice [[transformers.ConvBertForMultipleChoice]]

[[autodoc]] ConvBertForMultipleChoice
    - forward

## ConvBertForTokenClassification [[transformers.ConvBertForTokenClassification]]

[[autodoc]] ConvBertForTokenClassification
    - forward

## ConvBertForQuestionAnswering [[transformers.ConvBertForQuestionAnswering]]

[[autodoc]] ConvBertForQuestionAnswering
    - forward

</pt>
<tf>

## TFConvBertModel [[transformers.TFConvBertModel]]

[[autodoc]] TFConvBertModel
    - call

## TFConvBertForMaskedLM [[transformers.TFConvBertForMaskedLM]]

[[autodoc]] TFConvBertForMaskedLM 
    - call

## TFConvBertForSequenceClassification [[transformers.TFConvBertForSequenceClassification]]

[[autodoc]] TFConvBertForSequenceClassification
    - call

## TFConvBertForMultipleChoice [[transformers.TFConvBertForMultipleChoice]]

[[autodoc]] TFConvBertForMultipleChoice
    - call

## TFConvBertForTokenClassification [[transformers.TFConvBertForTokenClassification]]

[[autodoc]] TFConvBertForTokenClassification
    - call

## TFConvBertForQuestionAnswering [[transformers.TFConvBertForQuestionAnswering]]

[[autodoc]] TFConvBertForQuestionAnswering
    - call

</tf>
</frameworkcontent>
