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

# DeBERTa[[deberta]]

## 개요[[overview]]


DeBERTa 모델은 Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen이 작성한 [DeBERTa: 분리된 어텐션을 활용한 디코딩 강화 BERT](https://arxiv.org/abs/2006.03654)이라는 논문에서 제안되었습니다. 이 모델은 2018년 Google이 발표한 BERT 모델과 2019년 Facebook이 발표한 RoBERTa 모델을 기반으로 합니다.
DeBERTa는 RoBERTa에서 사용된 데이터의 절반만을 사용하여 분리된(disentangled) 어텐션과 향상된 마스크 디코더 학습을 통해 RoBERTa를 개선했습니다.

논문의 초록은 다음과 같습니다:

*사전 학습된 신경망 언어 모델의 최근 발전은 많은 자연어 처리(NLP) 작업의 성능을 크게 향상시켰습니다. 본 논문에서는 두 가지 새로운 기술을 사용하여 BERT와 RoBERTa 모델을 개선한 새로운 모델 구조인 DeBERTa를 제안합니다. 첫 번째는 분리된 어텐션 메커니즘으로, 각 단어가 내용과 위치를 각각 인코딩하는 두 개의 벡터로 표현되며, 단어들 간의 어텐션 가중치는 내용과 상대적 위치에 대한 분리된 행렬을 사용하여 계산됩니다. 두 번째로, 모델 사전 학습을 위해 마스킹된 토큰을 예측하는 출력 소프트맥스 층을 대체하는 향상된 마스크 디코더가 사용됩니다. 우리는 이 두 가지 기술이 모델 사전 학습의 효율성과 다운스트림 작업의 성능을 크게 향상시킨다는 것을 보여줍니다. RoBERTa-Large와 비교했을 때, 절반의 학습 데이터로 학습된 DeBERTa 모델은 광범위한 NLP 작업에서 일관되게 더 나은 성능을 보여주며, MNLI에서 +0.9%(90.2% vs 91.1%), SQuAD v2.0에서 +2.3%(88.4% vs 90.7%), RACE에서 +3.6%(83.2% vs 86.8%)의 성능 향상을 달성했습니다. DeBERTa 코드와 사전 학습된 모델은 https://github.com/microsoft/DeBERTa 에서 공개될 예정입니다.*

[DeBERTa](https://huggingface.co/DeBERTa) 모델의 텐서플로 2.0 구현은 [kamalkraj](https://huggingface.co/kamalkraj)가 기여했습니다. 원본 코드는 [이곳](https://github.com/microsoft/DeBERTa)에서 확인하실 수 있습니다.

## 리소스[[resources]]


DeBERTa를 시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰해 드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.


<PipelineTag pipeline="text-classification"/>

- DeBERTa와 [DeepSpeed를 이용해서 대형 모델 학습을 가속시키는](https://huggingface.co/blog/accelerate-deepspeed) 방법에 대한 포스트.
- DeBERTa와 [머신러닝으로 한층 향상된 고객 서비스](https://huggingface.co/blog/supercharge-customer-service-with-machine-learning)에 대한 블로그 포스트.
- [`DebertaForSequenceClassification`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)에서 지원됩니다.
- [`TFDebertaForSequenceClassification`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)에서 지원됩니다.
- [텍스트 분류 작업 가이드](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification" />

- [`DebertaForTokenClassification`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)에서 지원합니다.
- [`TFDebertaForTokenClassification`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)에서 지원합니다.
- 🤗 Hugging Face 코스의 [토큰 분류](https://huggingface.co/course/chapter7/2?fw=pt) 장.
- 🤗 Hugging Face 코스의 [BPE(Byte-Pair Encoding) 토큰화](https://huggingface.co/course/chapter6/5?fw=pt) 장.
- [토큰 분류 작업 가이드](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`DebertaForMaskedLM`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)에서 지원합니다.
- [`TFDebertaForMaskedLM`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)에서 지원합니다.
- 🤗 Hugging Face 코스의 [마스크 언어 모델링](https://huggingface.co/course/chapter7/3?fw=pt) 장.
- [마스크 언어 모델링 작업 가이드](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`DebertaForQuestionAnswering`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)에서 지원합니다.
- [`TFDebertaForQuestionAnswering`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)에서 지원합니다.
- 🤗 Hugging Face 코스의 [질의응답(Question answering)](https://huggingface.co/course/chapter7/7?fw=pt) 장.
- [질의응답 작업 가이드](../tasks/question_answering)

## DebertaConfig[[transformers.DebertaConfig]]

[[autodoc]] DebertaConfig

## DebertaTokenizer[[transformers.DebertaTokenizer]]

[[autodoc]] DebertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaTokenizerFast[[transformers.DebertaTokenizerFast]]

[[autodoc]] DebertaTokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

<frameworkcontent>
<pt>

## DebertaModel[[transformers.DebertaModel]]

[[autodoc]] DebertaModel
    - forward

## DebertaPreTrainedModel[[transformers.DebertaPreTrainedModel]]

[[autodoc]] DebertaPreTrainedModel

## DebertaForMaskedLM[[transformers.DebertaForMaskedLM]]

[[autodoc]] DebertaForMaskedLM
    - forward

## DebertaForSequenceClassification[[transformers.DebertaForSequenceClassification]]

[[autodoc]] DebertaForSequenceClassification
    - forward

## DebertaForTokenClassification[[transformers.DebertaForTokenClassification]]

[[autodoc]] DebertaForTokenClassification
    - forward

## DebertaForQuestionAnswering[[transformers.DebertaForQuestionAnswering]]

[[autodoc]] DebertaForQuestionAnswering
    - forward

</pt>
<tf>

## TFDebertaModel[[transformers.TFDebertaModel]]

[[autodoc]] TFDebertaModel
    - call

## TFDebertaPreTrainedModel[[transformers.TFDebertaPreTrainedModel]]

[[autodoc]] TFDebertaPreTrainedModel
    - call

## TFDebertaForMaskedLM[[transformers.TFDebertaForMaskedLM]]

[[autodoc]] TFDebertaForMaskedLM
    - call

## TFDebertaForSequenceClassification[[transformers.TFDebertaForSequenceClassification]]

[[autodoc]] TFDebertaForSequenceClassification
    - call

## TFDebertaForTokenClassification[[transformers.TFDebertaForTokenClassification]]

[[autodoc]] TFDebertaForTokenClassification
    - call

## TFDebertaForQuestionAnswering[[transformers.TFDebertaForQuestionAnswering]]

[[autodoc]] TFDebertaForQuestionAnswering
    - call

</tf>
</frameworkcontent>

