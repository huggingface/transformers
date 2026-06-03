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

# RoBERTa[[roberta]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[overview]]

RoBERTa 모델은 Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov가 제안한 논문 [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://huggingface.co/papers/1907.11692)에서 소개되었습니다. 이 모델은 2018년에 구글에서 발표한 BERT 모델을 기반으로 합니다.

RoBERTa는 BERT를 기반으로 하며, 주요 하이퍼파라미터를 수정하고, 사전 학습 단계에서 다음 문장 예측(Next Sentence Prediction)을 제거했으며, 훨씬 더 큰 미니 배치 크기와 학습률을 사용하여 학습을 진행했습니다.

해당 논문의 초록입니다:

*언어 모델 사전 학습은 성능을 크게 향상시켰지만, 서로 다른 접근 방식을 면밀히 비교하는 것은 어렵습니다. 학습은 계산 비용이 많이 들고, 종종 크기가 서로 다른 비공개 데이터셋에서 수행되며, 본 논문에서 보여주듯이 하이퍼파라미터 선택이 최종 성능에 큰 영향을 미칩니다. 우리는 BERT 사전 학습(Devlin et al., 2019)에 대한 재현 연구를 수행하여, 여러 핵심 하이퍼파라미터와 학습 데이터 크기의 영향을 면밀히 측정하였습니다. 그 결과, BERT는 충분히 학습되지 않았으며, 이후 발표된 모든 모델의 성능을 맞추거나 능가할 수 있음을 발견했습니다. 우리가 제안한 최상의 모델은 GLUE, RACE, SQuAD에서 최고 성능(state-of-the-art)을 달성했습니다. 이 결과는 지금까지 간과되어 온 설계 선택의 중요성을 강조하며, 최근 보고된 성능 향상의 근원이 무엇인지에 대한 의문을 제기합니다. 우리는 본 연구에서 사용한 모델과 코드를 공개합니다.*

이 모델은 [julien-c](https://huggingface.co/julien-c)가 기여하였습니다. 원본 코드는 [여기](https://github.com/pytorch/fairseq/tree/master/examples/roberta)에서 확인할 수 있습니다.

## 사용 팁[[usage-tips]]

- 이 구현은 [`BertModel`]과 동일하지만, 임베딩 부분에 약간의 수정이 있으며 RoBERTa 사전학습 모델에 맞게 설정되어 있습니다.
- RoBERTa는 BERT와 동일한 아키텍처를 가지고 있지만, 토크나이저로 바이트 수준 BPE(Byte-Pair Encoding, GPT-2와 동일)를 사용하고, 사전학습 방식이 다릅니다.
- RoBERTa는 `token_type_ids`를 사용하지 않기 때문에, 어떤 토큰이 어떤 문장(segment)에 속하는지 별도로 표시할 필요가 없습니다. 문장 구분은 분리 토큰 `tokenizer.sep_token`(또는 `</s>`)을 사용해 나누면 됩니다.
- RoBERTa는 BERT와 유사하지만, 더 나은 사전학습 기법을 사용합니다:

    * 동적 마스킹: RoBERTa는 매 에폭마다 토큰을 다르게 마스킹하는 반면, BERT는 한 번만 마스킹합니다.
    * 문장 패킹: 여러 문장을 최대 512 토큰까지 함께 패킹하여, 문장이 여러 문서에 걸쳐 있을 수도 있습니다.
    * 더 큰 배치 사이즈: 학습 시 더 큰 미니배치를 사용합니다.
    * 바이트 수준 BPE 어휘: 문자를 단위로 하지 않고 바이트 단위로 BPE를 적용하여 유니코드 문자를 더 유연하게 처리할 수 있습니다.

- [CamemBERT](camembert)은 RoBERTa를 기반으로 한 래퍼 모델입니다. 사용 예제는 해당 모델 페이지를 참고하세요.

## 자료[[resources]]

RoBERTa를 처음 다룰 때 도움이 되는 Hugging Face 공식 자료와 커뮤니티 자료(🌎 아이콘으로 표시됨) 목록입니다. 이 목록에 자료를 추가하고 싶다면 언제든지 Pull Request를 보내주세요! 저희가 검토 후 반영하겠습니다. 추가하려는 자료는 기존 자료를 단순히 복제하는 것이 아닌, 새롭거나 유의미한 내용을 포함하고 있는 것이 좋습니다.

<PipelineTag pipeline="text-classification"/>

- RoBERTa와 [Inference API](https://huggingface.co/inference-api)를 활용한 [트위터 감성 분석 시작하기](https://huggingface.co/blog/sentiment-analysis-twitter) 블로그 포스트.
- RoBERTa를 활용한 [Kili 및 Hugging Face AutoTrain을 이용한 의견 분류](https://huggingface.co/blog/opinion-classification-with-kili)에 관한 블로그 포스트.
- [감성 분석을 위한 RoBERTa 미세조정](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb)을 하는 방법에 대한 노트북.🌎
- ['RobertaForSequenceClassification']은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)에서 지원됩니다.
- [`TFRobertaForSequenceClassification`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)에서 지원됩니다.
- [`FlaxRobertaForSequenceClassification`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)에서 지원됩니다.
- [텍스트 분류 작업 가이드](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`RobertaForTokenClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)에서 지원됩니다.
- [`TFRobertaForTokenClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)에서 지원됩니다.
- [`FlaxRobertaForTokenClassification`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification)에서 지원됩니다.
- 🤗 Hugging Face 코스의 [토큰 분류 챕터](https://huggingface.co/course/chapter7/2?fw=pt)
- [토큰 분류 작업 가이드](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- RoBERTa를 활용한 [Transformers와 Tokenizers를 활용한 새로운 언어 모델을 처음부터 학습하는 방법](https://huggingface.co/blog/how-to-train)에 대한 블로그 포스트.
- [`RobertaForMaskedLM`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)에서 지원됩니다.
- [`TFRobertaForMaskedLM`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)에서 지원됩니다.
- [`FlaxRobertaForMaskedLM`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)에서 지원됩니다.
- 🤗 Hugging Face 코스의 [마스킹 언어 모델링 챕터](https://huggingface.co/course/chapter7/3?fw=pt)
- [마스킹 언어 모델링 작업 가이드](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- RoBERTa를 활용한 질문 응답 작업에서의 [Optimum과 Transformers 파이프라인을 이용한 추론 가속화](https://huggingface.co/blog/optimum-inference)에 대한 블로그 포스트.
- [`RobertaForQuestionAnswering`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)에서 지원됩니다.
- [`TFRobertaForQuestionAnswering`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)에서 지원됩니다.
- [`FlaxRobertaForQuestionAnswering`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering)에서 지원됩니다.
- 🤗 Hugging Face 코스의 [질의응답 챕터](https://huggingface.co/course/chapter7/7?fw=pt)
- [질의응답 작업 가이드](../tasks/question_answering)

**다중 선택**
- [`RobertaForMultipleChoice`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)에서 지원됩니다.
- [`TFRobertaForMultipleChoice`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)에서 지원됩니다.
- [다중 선택 작업 가이드](../tasks/multiple_choice)

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast
    - build_inputs_with_special_tokens

<frameworkcontent>
<pt>

## RobertaModel

[[autodoc]] RobertaModel
    - forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM
    - forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM
    - forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification
    - forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice
    - forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification
    - forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering
    - forward

</pt>
<tf>

## TFRobertaModel

[[autodoc]] TFRobertaModel
    - call

## TFRobertaForCausalLM

[[autodoc]] TFRobertaForCausalLM
    - call

## TFRobertaForMaskedLM

[[autodoc]] TFRobertaForMaskedLM
    - call

## TFRobertaForSequenceClassification

[[autodoc]] TFRobertaForSequenceClassification
    - call

## TFRobertaForMultipleChoice

[[autodoc]] TFRobertaForMultipleChoice
    - call

## TFRobertaForTokenClassification

[[autodoc]] TFRobertaForTokenClassification
    - call

## TFRobertaForQuestionAnswering

[[autodoc]] TFRobertaForQuestionAnswering
    - call

</tf>
<jax>

## FlaxRobertaModel

[[autodoc]] FlaxRobertaModel
    - __call__

## FlaxRobertaForCausalLM

[[autodoc]] FlaxRobertaForCausalLM
    - __call__

## FlaxRobertaForMaskedLM

[[autodoc]] FlaxRobertaForMaskedLM
    - __call__

## FlaxRobertaForSequenceClassification

[[autodoc]] FlaxRobertaForSequenceClassification
    - __call__

## FlaxRobertaForMultipleChoice

[[autodoc]] FlaxRobertaForMultipleChoice
    - __call__

## FlaxRobertaForTokenClassification

[[autodoc]] FlaxRobertaForTokenClassification
    - __call__

## FlaxRobertaForQuestionAnswering

[[autodoc]] FlaxRobertaForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>
