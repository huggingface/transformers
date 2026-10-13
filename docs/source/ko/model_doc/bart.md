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

# BART[[bart]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bart">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bart-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bart-large-mnli">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 개요 [[overview]]

Bart 모델은 2019년 10월 29일 Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer가 발표한 [BART: 자연어 생성, 번역, 이해를 위한 잡음 제거 seq2seq 사전 훈련](https://huggingface.co/papers/1910.13461)이라는 논문에서 소개되었습니다.

논문의 초록에 따르면,

- Bart는 양방향 인코더(BERT와 유사)와 왼쪽에서 오른쪽으로 디코딩하는 디코더(GPT와 유사)를 사용하는 표준 seq2seq/기계 번역 아키텍처를 사용합니다.
- 사전 훈련 작업은 원래 문장의 순서를 무작위로 섞고, 텍스트의 일부 구간을 단일 마스크 토큰으로 대체하는 새로운 인필링(in-filling) 방식을 포함합니다.
- BART는 특히 텍스트 생성을 위한 미세 조정에 효과적이지만 이해 작업에도 잘 작동합니다. GLUE와 SQuAD에서 비슷한 훈련 리소스로 RoBERTa의 성능과 일치하며, 추상적 대화, 질의응답, 요약 작업 등에서 최대 6 ROUGE 점수의 향상을 보이며 새로운 최고 성능을 달성했습니다.

이 모델은 [sshleifer](https://huggingface.co/sshleifer)에 의해 기여 되었습니다. 저자의 코드는 [이곳](https://github.com/pytorch/fairseq/tree/master/examples/bart)에서 확인할 수 있습니다.

## 사용 팁:[[usage-tips]]

- BART는 절대 위치 임베딩을 사용하는 모델이므로 일반적으로 입력을 왼쪽보다는 오른쪽에 패딩하는 것이 좋습니다.
- 인코더와 디코더가 있는 seq2seq 모델입니다. 인코더에는 손상된 토큰이(corrupted tokens) 입력되고, 디코더에는 원래 토큰이 입력됩니다(단, 일반적인 트랜스포머 디코더처럼 미래 단어를 숨기는 마스크가 있습니다). 사전 훈련 작업에서 인코더에 적용되는 변환들의 구성은 다음과 같습니다:

사전 훈련 작업에서 인코더에 적용되는 변환들의 구성은 다음과 같습니다:

  * 무작위 토큰 마스킹 (BERT 처럼)
  * 무작위 토큰 삭제
  * k개 토큰의 범위를 단일 마스크 토큰으로 마스킹 (0개 토큰의 범위는 마스크 토큰의 삽입을 의미)
  * 문장 순서 뒤섞기
  * 특정 토큰에서 시작하도록 문서 회전

## 구현 노트[[implementation-notes]]

- Bart는 시퀀스 분류에 `token_type_ids`를 사용하지 않습니다. 적절하게 나누기 위해서 [`BartTokenizer`]나
  [`~BartTokenizer.encode`]를 사용합니다.
- [`BartModel`]의 정방향 전달은 `decoder_input_ids`가 전달되지 않으면 `decoder_input_ids`를 자동으로 생성할 것입니다. 이는 다른 일부 모델링 API와 다른 점입니다. 이 기능의 일반적인 사용 사례는 마스크 채우기(mask filling)입니다.
-  모델 예측은 `forced_bos_token_id=0`일 때 기존 구현과 동일하게 작동하도록 의도되었습니다. 하지만, [`fairseq.encode`]에 전달하는 문자열이 공백으로 시작할 때만 이 기능이 작동합니다.
- [`~generation.GenerationMixin.generate`]는 요약과 같은 조건부 생성 작업에 사용되어야 합니다. 자세한 내용은 해당 문서의 예제를 참조하세요.
- *facebook/bart-large-cnn* 가중치를 로드하는 모델은 `mask_token_id`가 없거나, 마스크 채우기 작업을 수행할 수 없습니다.

## 마스크 채우기[[mask-filling]]

`facebook/bart-base`와 `facebook/bart-large` 체크포인트는 멀티 토큰 마스크를 채우는데 사용될 수 있습니다.

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
]
```

## 자료[[resources]]

BART를 시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

<PipelineTag pipeline="summarization"/>

- [분산형 학습: 🤗 Transformers와 Amazon SageMaker를 이용하여 요약하기 위한 BART/T5 학습](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq)에 대한 블로그 포스트.
- [blurr를 이용하여 fastai로 요약하기 위한 BART를 미세 조정](https://colab.research.google.com/github/ohmeow/ohmeow_website/blob/master/posts/2021-05-25-mbart-sequence-classification-with-blurr.ipynb)하는 방법에 대한 노트북. 🌎
- [Trainer 클래스를 사용하여 두 가지 언어로 요약하기 위한 BART 미세 조정](https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb)하는 방법에 대한 노트북. 🌎
- 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)에서는  [`BartForConditionalGeneration`]이 지원됩니다.
- [`TFBartForConditionalGeneration`]는 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)에서 지원됩니다.
- Hugging Face `datasets` 객체를 활용하여 [`BartForConditionalGeneration`]을 학습시키는 방법의 예는 이 [포럼 토론](https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904)에서 찾을 수 있습니다.
- 🤗 Hugging Face 코스의 [요약](https://huggingface.co/course/chapter7/5?fw=pt#summarization)장.
- [요약 작업 가이드](../tasks/summarization)

<PipelineTag pipeline="fill-mask"/>

- [`BartForConditionalGeneration`]는 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)을 참고하세요.
- [`TFBartForConditionalGeneration`]는 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)을 참고하세요.
- [`FlaxBartForConditionalGeneration`]는 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)을 참고 하세요.
- 🤗 Hugging Face 코스의 [마스크 언어 모델링](https://huggingface.co/course/chapter7/3?fw=pt) 챕터.
- [마스크 언어 모델링 작업 가이드](../tasks/masked_language_modeling)

<PipelineTag pipeline="translation"/>

- [Seq2SeqTrainer를 이용하여 인도어를 영어로 번역하는 mBART를 미세 조정](https://colab.research.google.com/github/vasudevgupta7/huggingface-tutorials/blob/main/translation_training.ipynb)하는 방법에 대한 가이드. 🌎
- [`BartForConditionalGeneration`]는 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)을 참고하세요.
- [`TFBartForConditionalGeneration`]는 이 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)을 참고 하세요.
- [번역 작업 가이드](../tasks/translation)

추가적으로 볼 것들:
- [텍스트 분류 작업 가이드](../tasks/sequence_classification)
- [질문 답변 작업 가이드](../tasks/question_answering)
- [인과적 언어 모델링 작업 가이드](../tasks/language_modeling)
- 이 [논문](https://huggingface.co/papers/2010.13002)은 [증류된 체크포인트](https://huggingface.co/models?search=distilbart)에 대해 설명합니다.

## BartConfig[[transformers.BartConfig]]

[[autodoc]] BartConfig
    - all

## BartTokenizer[[transformers.BartTokenizer]]

[[autodoc]] BartTokenizer
    - all

## BartTokenizerFast[[transformers.BartTokenizerFast]]

[[autodoc]] BartTokenizerFast
    - all



## BartModel[[transformers.BartModel]]

[[autodoc]] BartModel
    - forward

## BartForConditionalGeneration[[transformers.BartForConditionalGeneration]]

[[autodoc]] BartForConditionalGeneration
    - forward

## BartForSequenceClassification[[transformers.BartForSequenceClassification]]

[[autodoc]] BartForSequenceClassification
    - forward

## BartForQuestionAnswering[[transformers.BartForQuestionAnswering]]

[[autodoc]] BartForQuestionAnswering
    - forward

## BartForCausalLM[[transformers.BartForCausalLM]]

[[autodoc]] BartForCausalLM
    - forward




