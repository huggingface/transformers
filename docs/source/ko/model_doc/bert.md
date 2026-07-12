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

# BERT[[BERT]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bert-base-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 개요[[Overview]]

BERT 모델은 Jacob Devlin. Ming-Wei Chang, Kenton Lee, Kristina Touranova가 제안한 논문 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://huggingface.co/papers/1810.04805)에서 소개되었습니다. BERT는 사전 학습된 양방향 트랜스포머로,  Toronto Book Corpus와 Wikipedia로 구성된 대규모 코퍼스에서 마스킹된 언어 모델링과 다음 문장 예측(Next Sentence Prediction) 목표를 결합해 학습되었습니다.

해당 논문의 초록입니다:

*우리는 BERT(Bidirectional Encoder Representations from Transformers)라는 새로운 언어 표현 모델을 소개합니다. 최근의 다른 언어 표현 모델들과 달리, BERT는 모든 계층에서 양방향으로 양쪽 문맥을 조건으로 사용하여 비지도 학습된 텍스트에서 깊이 있는 양방향 표현을 사전 학습하도록 설계되었습니다. 그 결과, 사전 학습된 BERT 모델은 추가적인 출력 계층 하나만으로 질문 응답, 언어 추론과 같은 다양한 작업에서 미세 조정될 수 있으므로, 특정 작업을 위해 아키텍처를 수정할 필요가 없습니다.*

*BERT는 개념적으로 단순하면서도 실증적으로 강력한 모델입니다. BERT는 11개의 자연어 처리 과제에서 새로운 최고 성능을 달성했으며, GLUE 점수를 80.5% (7.7% 포인트 절대 개선)로, MultiNLI 정확도를 86.7% (4.6% 포인트 절대 개선), SQuAD v1.1 질문 응답 테스트에서 F1 점수를 93.2 (1.5% 포인트 절대 개선)로, SQuAD v2.0에서 F1 점수를 83.1 (5.1% 포인트 절대 개선)로 향상시켰습니다.*

이 모델은 [thomwolf](https://huggingface.co/thomwolf)가 기여하였습니다. 원본 코드는 [여기](https://github.com/google-research/bert)에서 확인할 수 있습니다.

## 사용 팁[[Usage tips]]

- BERT는 절대 위치 임베딩을 사용하는 모델이므로 입력을 왼쪽이 아니라 오른쪽에서 패딩하는 것이 일반적으로 권장됩니다.
- BERT는 마스킹된 언어 모델(MLM)과 Next Sentence Prediction(NSP) 목표로 학습되었습니다. 이는 마스킹된 토큰 예측과 전반적인 자연어 이해(NLU)에 뛰어나지만, 텍스트 생성에는 최적화되어있지 않습니다.
- BERT의 사전 학습 과정에서는 입력 데이터를 무작위로 마스킹하여 일부 토큰을 마스킹합니다. 전체 토큰 중 약 15%가 다음과 같은 방식으로 마스킹됩니다:

    * 80% 확률로 마스크 토큰으로 대체
    * 10% 확률로 임의의 다른 토큰으로 대체
    * 10% 확률로 원래 토큰 그대로 유지

- 모델의 주요 목표는 원본 문장을 예측하는 것이지만, 두 번째 목표가 있습니다: 입력으로 문장 A와 B (사이에는 구분 토큰이 있음)가 주어집니다. 이 문장 쌍이 연속될 확률은 50%이며, 나머지 50%는 서로 무관한 문장들입니다. 모델은 이 두 문장이 아닌지를 예측해야 합니다.

### Scaled Dot Product Attention(SDPA) 사용하기 [[Using Scaled Dot Product Attention (SDPA)]]

Pytorch는 `torch.nn.functional`의 일부로 Scaled Dot Product Attention(SDPA) 연산자를 기본적으로 제공합니다. 이 함수는 입력과 하드웨어에 따라 여러 구현 방식을 사용할 수 있습니다. 자세한 내용은 [공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)나 [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)에서 확인할 수 있습니다.

`torch>=2.1.1`에서는 구현이 가능한 경우 SDPA가 기본적으로 사용되지만, `from_pretrained()`함수에서 `attn_implementation="sdpa"`를 설정하여 SDPA를 명시적으로 사용하도록 지정할 수도 있습니다.

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased", dtype=torch.float16, attn_implementation="sdpa")
...
```

최적 성능 향상을 위해 모델을 반정밀도(예: `torch.float16` 또는 `torch.bfloat16`)로 불러오는 것을 권장합니다.

로컬 벤치마크 (A100-80GB, CPUx12, RAM 96.6GB, PyTorch 2.2.0, OS Ubuntu 22.04)에서 `float16`을 사용해 학습 및 추론을 수행한 결과, 다음과 같은 속도 향상이 관찰되었습니다.

#### 학습 [[Training]]

|batch_size|seq_len|Time per batch (eager - s)|Time per batch (sdpa - s)|Speedup (%)|Eager peak mem (MB)|sdpa peak mem (MB)|Mem saving (%)|
|----------|-------|--------------------------|-------------------------|-----------|-------------------|------------------|--------------|
|4         |256    |0.023                     |0.017                    |35.472     |939.213            |764.834           |22.800        |
|4         |512    |0.023                     |0.018                    |23.687     |1970.447           |1227.162          |60.569        |
|8         |256    |0.023                     |0.018                    |23.491     |1594.295           |1226.114          |30.028        |
|8         |512    |0.035                     |0.025                    |43.058     |3629.401           |2134.262          |70.054        |
|16        |256    |0.030                     |0.024                    |25.583     |2874.426           |2134.262          |34.680        |
|16        |512    |0.064                     |0.044                    |46.223     |6964.659           |3961.013          |75.830        |

#### 추론 [[Inference]]

|batch_size|seq_len|Per token latency eager (ms)|Per token latency SDPA (ms)|Speedup (%)|Mem eager (MB)|Mem BT (MB)|Mem saved (%)|
|----------|-------|----------------------------|---------------------------|-----------|--------------|-----------|-------------|
|1         |128    |5.736                       |4.987                      |15.022     |282.661       |282.924    |-0.093       |
|1         |256    |5.689                       |4.945                      |15.055     |298.686       |298.948    |-0.088       |
|2         |128    |6.154                       |4.982                      |23.521     |314.523       |314.785    |-0.083       |
|2         |256    |6.201                       |4.949                      |25.303     |347.546       |347.033    |0.148        |
|4         |128    |6.049                       |4.987                      |21.305     |378.895       |379.301    |-0.107       |
|4         |256    |6.285                       |5.364                      |17.166     |443.209       |444.382    |-0.264       |



## 자료[[Resources]]

BERT를 시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

<PipelineTag pipeline="text-classification"/>

- [BERT 텍스트 분류 (다른 언어로)](https://www.philschmid.de/bert-text-classification-in-a-different-language)에 대한 블로그 포스트.
- [다중 레이블 텍스트 분류를 위한 BERT (및 관련 모델) 미세 조정](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)에 대한 노트북.
- [PyTorch를 이용해 BERT를 다중 레이블 분류를 위해 미세 조정하는 방법](htt기ps://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)에 대한 노트북. 🌎
- [BERT로 EncoderDecoder 모델을 warm-start하여 요약하기](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb)에 대한 노트북.
- [`BertForSequenceClassification`]이  [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)에서 지원됩니다.
- [`TFBertForSequenceClassification`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForSequenceClassification`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)에서 지원됩니다.
- [텍스트 분류 작업 가이드](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [Keras와 함께 Hugging Face Transformers를 사용하여 비영리 BERT를 개체명 인식(NER)용으로 미세 조정하는 방법](https://www.philschmid.de/huggingface-transformers-keras-tf)에 대한 블로그 포스트.
- [BERT를 개체명 인식을 위해 미세 조정하기](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb)에 대한 노트북. 각 단어의 첫 번째 wordpiece에만 레이블을 지정하여 학습하는 방법을 설명합니다. 모든 wordpiece에 레이블을 전파하는 방법은 [이 버전](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb)에서 확인할 수 있습니다.
- [`BertForTokenClassification`]이  [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)와  [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)에서 지원됩니다.
- [`TFBertForTokenClassification`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForTokenClassification`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification)에서 지원됩니다.
- 🤗 Hugging Face 코스의 [토큰 분류 챕터](https://huggingface.co/course/chapter7/2?fw=pt).
- [토큰 분류 작업 가이드](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`BertForMaskedLM`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)에서 지원됩니다.
- [`TFBertForMaskedLM`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) 와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForMaskedLM`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)에서 지원됩니다.
- 🤗 Hugging Face 코스의 [마스킹된 언어 모델링 챕터](https://huggingface.co/course/chapter7/3?fw=pt).
- [마스킹된 언어 모델링 작업 가이드](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`BertForQuestionAnswering`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)에서 지원됩니다.
- [`TFBertForQuestionAnswering`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) 와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForQuestionAnswering`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering)에서 지원됩니다.
- 🤗 Hugging Face 코스의 [질문 답변 챕터](https://huggingface.co/course/chapter7/7?fw=pt).
- [질문 답변 작업 가이드](../tasks/question_answering)

**다중 선택**
- [`BertForMultipleChoice`]이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)에서 지원됩니다.
- [`TFBertForMultipleChoice`]이 [에제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)에서 지원됩니다.
- [다중 선택 작업 가이드](../tasks/multiple_choice)

⚡️ **추론**
- [Hugging Face Transformers와 AWS Inferentia를 사용하여 BERT 추론을 가속화하는 방법](https://huggingface.co/blog/bert-inferentia-sagemaker)에 대한 블로그 포스트.
- [GPU에서 DeepSpeed-Inference로 BERT 추론을 가속화하는 방법](https://www.philschmid.de/bert-deepspeed-inference)에 대한 블로그 포스트.

⚙️ **사전 학습**
- [Hugging Face Optimum으로 Transformers를 ONMX로 변환하는 방법](https://www.philschmid.de/pre-training-bert-habana)에 대한 블로그 포스트.

🚀 **배포**
- [Hugging Face Optimum으로 Transformers를 ONMX로 변환하는 방법](https://www.philschmid.de/convert-transformers-to-onnx)에 대한 블로그 포스트.
- [AWS에서 Hugging Face Transformers를 위한 Habana Gaudi 딥러닝 환경 설정 방법](https://www.philschmid.de/getting-started-habana-gaudi#conclusion)에 대한 블로그 포스트.
- [Hugging Face Transformers, Amazon SageMaker 및 Terraform 모듈을 이용한 BERT 자동 확장](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker-advanced)에 대한 블로그 포스트.
- [Hugging Face, AWS Lambda, Docker를 활용하여 서버리스 BERT 설정하는 방법](https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker)에 대한 블로그 포스트.
- [Amazon SageMaker와 Training Compiler를 사용하여 Hugging Face Transformers에서 BERT 미세 조정하는 방법](https://www.philschmid.de/huggingface-amazon-sagemaker-training-compiler)에 대한 블로그.
- [Amazon SageMaker를 사용한 Transformers와 BERT의 작업별 지식 증류](https://www.philschmid.de/knowledge-distillation-bert-transformers)에 대한 블로그 포스트.

## BertConfig

[[autodoc]] BertConfig
    - all

## BertTokenizer

[[autodoc]] BertTokenizer
    - get_special_tokens_mask
    - save_vocabulary

## BertTokenizerLegacy

[[autodoc]] BertTokenizerLegacy

## BertTokenizerFast

[[autodoc]] BertTokenizerFast


## Bert specific outputs

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput


## BertModel

[[autodoc]] BertModel
    - forward

## BertForPreTraining

[[autodoc]] BertForPreTraining
    - forward

## BertLMHeadModel

[[autodoc]] BertLMHeadModel
    - forward

## BertForMaskedLM

[[autodoc]] BertForMaskedLM
    - forward

## BertForNextSentencePrediction

[[autodoc]] BertForNextSentencePrediction
    - forward

## BertForSequenceClassification

[[autodoc]] BertForSequenceClassification
    - forward

## BertForMultipleChoice

[[autodoc]] BertForMultipleChoice
    - forward

## BertForTokenClassification

[[autodoc]] BertForTokenClassification
    - forward

## BertForQuestionAnswering

[[autodoc]] BertForQuestionAnswering
    - forward



