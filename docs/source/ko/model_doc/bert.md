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

# BERT[[bert]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bert-base-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 개요[[overview]]

BERT 모델은 Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova가 발표한 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 논문에서 제안되었습니다. 이 모델은 마스크드 언어 모델링과 다음 문장 예측을 결합하여 Toronto Book Corpus와 Wikipedia로 구성된 대규모 코퍼스로부터 사전훈련된 양방향 트랜스포머입니다.

논문의 초록은 다음과 같습니다:

*우리는 BERT(Bidirectional Encoder Representations from Transformers)라는 새로운 언어 표현 모델을 소개합니다. 최근의 언어 표현 모델과 달리, BERT는 모든 레이어에서 왼쪽과 오른쪽 컨텍스트를 동시에 고려하여 비지도 텍스트로부터 깊은 양방향 표현을 사전훈련하도록 설계되었습니다. 그 결과, 사전훈련된 BERT 모델은 하나의 추가 출력 레이어만으로 질의 응답 및 언어 추론과 같은 다양한 작업에 대해 최신 모델을 구축할 수 있으며, 별도의 작업별 아키텍처 수정 없이도 가능합니다.*

*BERT는 개념적으로 단순하면서도 실질적으로 강력합니다. 이 모델은 11개의 자연어 처리 작업에서 새로운 최고 성능을 달성했으며, GLUE 점수를 80.5%로 끌어올리며(수치상으로 7.7%포인트 향상), MultiNLI 정확도를 86.7%(수치상으로 4.6%포인트 향상), SQuAD v1.1 질의 응답 테스트 F1 점수를 93.2(수치상으로 1.5포인트 향상), SQuAD v2.0 테스트 F1 점수를 83.1(수치상으로 5.1포인트 향상)로 끌어올렸습니다.*

이 모델은 [thomwolf](https://huggingface.co/thomwolf)가 기여하였습니다. 원본 코드는 [여기](https://github.com/google-research/bert)에서 찾을 수 있습니다.

## 사용 팁[[usage-tips]]

- BERT는 절대적 위치 임베딩을 사용하는 모델이므로 입력을 왼쪽이 아닌 오른쪽에서 패딩하는 것을 일반적으로 권장합니다.
- BERT는 마스크드 언어 모델링(MLM)과 다음 문장 예측(NSP) 목표로 학습되었습니다. 마스킹된 토큰을 예측하는 작업과 일반적인 자연어 이해(NLU)에 효율적이지만, 텍스트 생성에는 최적화되어 있지 않습니다.
- 랜덤 마스킹을 사용해 입력을 변형합니다. 더 구체적으로 말하자면, 사전훈련 중에는 주어진 토큰의 일정 비율(일반적으로 15%)이 다음과 같이 마스킹됩니다:

    * 80% 확률로 특수 마스크 토큰을 사용
    * 10% 확률로 마스킹된 토큰과 다른 임의의 토큰을 사용
    * 10% 확률로 동일한 토큰을 유지

- 모델은 원래 문장을 예측해야 하며, 동시에 두 번째 목표도 있습니다: 입력은 A와 B라는 두 문장(사이에 구분 토큰이 있음)으로 구성됩니다. 50% 확률로 이 문장들은 코퍼스에서 연속적이며, 나머지 50%는 서로 관련이 없습니다. 모델은 문장들이 연속적인지 아닌지를 예측해야 합니다.

### Scaled Dot Product Attention(SDPA) 사용[[using-scaled-dot-product-attention-sdpa]]

PyTorch는 `torch.nn.functional`의 일부로 네이티브 Scaled Dot Product Attention(SDPA) 연산자를 포함하고 있습니다. 이 함수는 입력과 사용 중인 하드웨어에 따라 여러 구현을 적용할 수 있습니다. 자세한 내용은 [공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 또는 [GPU 추론](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) 페이지를 참조하세요.

`torch>=2.1.1`에서는 가능한 경우 기본적으로 SDPA가 사용되지만, `from_pretrained()`에서 `attn_implementation="sdpa"`를 설정하여 명시적으로 SDPA를 사용할 수도 있습니다.

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
...
```

최상의 속도 향상을 위해 모델을 반정밀도(e.g. `torch.float16` 또는 `torch.bfloat16`)로 로드하는 것을 권장합니다.

로컬 벤치마크(A100-80GB, CPUx12, RAM 96.6GB, PyTorch 2.2.0, OS Ubuntu 22.04)에서 `float16`을 사용하여 학습 및 추론 중 다음과 같은 속도 향상을 확인할 수 있었습니다.

#### 학습[[training]]

|배치 크기|시퀀스 길이|배치당 시간 (eager - 초)|배치당 시간 (sdpa - 초)|속도 향상 (%)|Eager 피크 메모리 (MB)|SDPA 피크 메모리 (MB)|메모리 절약 (%)|
|----------|-------|--------------------------|-------------------------|-----------|-------------------|------------------|--------------|
|4         |256    |0.023                     |0.017                    |35.472     |939.213            |764.834           |22.800        |
|4         |512    |0.023                     |0.018                    |23.687     |1970.447           |1227.162          |60.569        |
|8         |256    |0.023                     |0.018                    |23.491     |1594.295           |1226.114          |30.028        |
|8         |512    |0.035                     |0.025                    |43.058     |3629.401           |2134.262          |70.054        |
|16        |256    |0.030                     |0.024                    |25.583     |2874.426           |2134.262          |34.680        |
|16        |512    |0.064                     |0.044                    |46.223     |6964.659           |3961.013          |75.830        |

#### 추론[[inference]]

|배치 크기|시퀀스 길이|토큰당 지연 시간 (eager - ms)|토큰당 지연 시간 (SDPA - ms)|속도 향상 (%)|Eager 메모리 (MB)|BT 메모리 (MB)|메모리 절약 (%)|
|----------|-------|----------------------------|---------------------------|-----------|--------------|-----------|-------------|
|1         |128    |5.736                       |4.987                      |15.022     |282.661       |282.924    |-0.093       |
|1         |256    |5.689                       |4.945                      |15.055     |298.686       |298.948    |-0.088       |
|2         |128    |6.154                       |4.982                      |23.521     |314.523       |314.785    |-0.083       |
|2         |256    |6.201                       |4.949                      |25.303     |347.546       |347.033    |0.148        |
|4         |128    |6.049                       |4.987                      |21.305     |378.895       |379.301    |-0.107       |
|4         |256    |6.285                       |5.364                      |17.166     |443.209       |444.382    |-0.264       |


## 관련 자료[[resources]]

BERT를 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티(🌎로 표시된) 자료 목록입니다. 이곳에 포함될 자료를 제출하고 싶다면 자유롭게 Pull Request를 열어주세요! 자료는 기존 자료와 중복된 내용보다는 새로운 내용을 다루는 것이 좋습니다.

<PipelineTag pipeline="text-classification"/>

- [다른 언어로 BERT 텍스트 분류](https://www.philschmid.de/bert-text-classification-in-a-different-language)에 관한 블로그 게시물.
- [다중 레이블 텍스트 분류를 위한 BERT(및 다른 모델) 미세 조정](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb) 노트북.
- PyTorch를 사용해 [BERT를 다중 레이블 분류에 미세 조정](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb)하는 방법을 설명한 노트북. 🌎
- [요약을 위해 BERT로 EncoderDecoder 모델을 웜 스타트](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb)하는 방법에 관한 노트북.
- [`BertForSequenceClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)에서 지원됩니다.
- [`TFBertForSequenceClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForSequenceClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)에서 지원됩니다.
- [텍스트 분류 작업 가이드](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [Hugging Face Transformers와 Keras를 사용해 비영어 BERT를 미세 조정하여 개체명 인식 수행](https://www.philschmid.de/huggingface-transformers-keras-tf)에 관한 블로그 게시물.
- 첫 번째 워드피스만 사용해 [BERT로 개체명 인식 미세 조정](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb)하는 노트북. 모든 워드피스에 단어의 레이블을 전달하려면 이 [버전](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb)을 참고하세요.
- [`BertForTokenClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)에서 지원됩니다.
- [`TFBertForTokenClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForTokenClassification`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification)에서 지원됩니다.
- 🤗 Hugging Face 강좌의 [토큰 분류](https://huggingface.co/course/chapter7/2?fw=pt) 챕터.
- [토큰 분류 작업 가이드](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`BertForMaskedLM`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)에서 지원됩니다.
- [`TFBertForMaskedLM`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForMaskedLM`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)에서 지원됩니다.
- 🤗 Hugging Face 강좌의 [마스크드 언어 모델링](https://huggingface.co/course/chapter7/3?fw=pt) 챕터.
- [마스크드 언어 모델링 작업 가이드](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`BertForQuestionAnswering`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)에서 지원됩니다.
- [`TFBertForQuestionAnswering`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)에서 지원됩니다.
- [`FlaxBertForQuestionAnswering`]은 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering)에서 지원됩니다.
- 🤗 Hugging Face 강좌의 [질의 응답](https://huggingface.co/course/chapter7/7?fw=pt) 챕터.
- [질의 응답 작업 가이드](../tasks/question_answering)

**다중 선택 문제**
- [`BertForMultipleChoice`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)에서 지원됩니다.
- [`TFBertForMultipleChoice`]는 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) 및 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)에서 지원됩니다.
- [다중 선택 문제 작업 가이드](../tasks/multiple_choice)

⚡️ **추론(Inference)**
- [Hugging Face Transformers와 AWS Inferentia로 BERT 추론 가속화](https://huggingface.co/blog/bert-inferentia-sagemaker)에 관한 블로그 게시물.
- [GPU에서 DeepSpeed-Inference를 사용해 BERT 추론 가속화](https://www.philschmid.de/bert-deepspeed-inference)에 관한 블로그 게시물.

⚙️ **사전훈련(Pretraining)**
- [Hugging Face Transformers와 Habana Gaudi로 BERT 사전훈련](https://www.philschmid.de/pre-training-bert-habana)에 관한 블로그 게시물.

🚀 **배포(Deploy)**
- [Hugging Face Optimum으로 Transformers를 ONNX로 변환](https://www.philschmid.de/convert-transformers-to-onnx)에 관한 블로그 게시물.
- [AWS에서 Habana Gaudi로 Hugging Face Transformers 딥러닝 환경 설정](https://www.philschmid.de/getting-started-habana-gaudi#conclusion)에 관한 블로그 게시물.
- [Hugging Face Transformers, Amazon SageMaker, Terraform 모듈로 BERT 자동 확장](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker-advanced)에 관한 블로그 게시물.
- [Hugging Face, AWS Lambda, Docker를 사용한 서버리스 BERT](https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker)에 관한 블로그 게시물.
- [Amazon SageMaker 및 Training Compiler로 Hugging Face Transformers BERT 미세 조정](https://www.philschmid.de/huggingface-amazon-sagemaker-training-compiler)에 관한 블로그 게시물.
- [Transformers와 Amazon SageMaker를 사용한 BERT의 작업별 지식 증류](https://www.philschmid.de/knowledge-distillation-bert-transformers)에 관한 블로그 게시물.

## BertConfig[[transformers.BertConfig]]

[[autodoc]] BertConfig
    - all

## BertTokenizer[[transformers.BertTokenizer]]

[[autodoc]] BertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

<frameworkcontent>
<pt>

## BertTokenizerFast[[transformers.BertTokenizerFast]]

[[autodoc]] BertTokenizerFast

</pt>
<tf>

## TFBertTokenizer[[transformers.TFBertTokenizer]]

[[autodoc]] TFBertTokenizer

</tf>
</frameworkcontent>

## Bert에 특화된 출력[[transformers.models.bert.modeling_bert.BertForPreTrainingOutput]]

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput

[[autodoc]] models.bert.modeling_tf_bert.TFBertForPreTrainingOutput

[[autodoc]] models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput


<frameworkcontent>
<pt>

## BertModel[[transformers.BertModel]]

[[autodoc]] BertModel
    - forward

## BertForPreTraining[[transformers.BertForPreTraining]]

[[autodoc]] BertForPreTraining
    - forward

## BertLMHeadModel[[transformers.BertLMHeadModel]]

[[autodoc]] BertLMHeadModel
    - forward

## BertForMaskedLM[[transformers.BertForMaskedLM]]

[[autodoc]] BertForMaskedLM
    - forward

## BertForNextSentencePrediction[[transformers.BertForNextSentencePrediction]]

[[autodoc]] BertForNextSentencePrediction
    - forward

## BertForSequenceClassification[[transformers.BertForSequenceClassification]]

[[autodoc]] BertForSequenceClassification
    - forward

## BertForMultipleChoice[[transformers.BertForMultipleChoice]]

[[autodoc]] BertForMultipleChoice
    - forward

## BertForTokenClassification[[transformers.BertForTokenClassification]]

[[autodoc]] BertForTokenClassification
    - forward

## BertForQuestionAnswering[[transformers.BertForQuestionAnswering]]

[[autodoc]] BertForQuestionAnswering
    - forward

</pt>
<tf>

## TFBertModel[[transformers.TFBertModel]]

[[autodoc]] TFBertModel
    - call

## TFBertForPreTraining[[transformers.TFBertForPreTraining]]

[[autodoc]] TFBertForPreTraining
    - call

## TFBertModelLMHeadModel[[transformers.TFBertLMHeadModel]]

[[autodoc]] TFBertLMHeadModel
    - call

## TFBertForMaskedLM[[transformers.TFBertForMaskedLM]]

[[autodoc]] TFBertForMaskedLM
    - call

## TFBertForNextSentencePrediction[[transformers.TFBertForNextSentencePrediction]]

[[autodoc]] TFBertForNextSentencePrediction
    - call

## TFBertForSequenceClassification[[transformers.TFBertForSequenceClassification]]

[[autodoc]] TFBertForSequenceClassification
    - call

## TFBertForMultipleChoice[[transformers.TFBertForMultipleChoice]]

[[autodoc]] TFBertForMultipleChoice
    - call

## TFBertForTokenClassification[[transformers.TFBertForTokenClassification]]

[[autodoc]] TFBertForTokenClassification
    - call

## TFBertForQuestionAnswering[[transformers.TFBertForQuestionAnswering]]

[[autodoc]] TFBertForQuestionAnswering
    - call

</tf>
<jax>

## FlaxBertModel[[transformers.FlaxBertModel]]

[[autodoc]] FlaxBertModel
    - __call__

## FlaxBertForPreTraining[[transformers.FlaxBertForPreTraining]]

[[autodoc]] FlaxBertForPreTraining
    - __call__

## FlaxBertForCausalLM[[transformers.FlaxBertForCausalLM]]

[[autodoc]] FlaxBertForCausalLM
    - __call__

## FlaxBertForMaskedLM[[transformers.FlaxBertForMaskedLM]]

[[autodoc]] FlaxBertForMaskedLM
    - __call__

## FlaxBertForNextSentencePrediction[[transformers.FlaxBertForNextSentencePrediction]]

[[autodoc]] FlaxBertForNextSentencePrediction
    - __call__

## FlaxBertForSequenceClassification[[transformers.FlaxBertForSequenceClassification]]

[[autodoc]] FlaxBertForSequenceClassification
    - __call__

## FlaxBertForMultipleChoice[[transformers.FlaxBertForMultipleChoice]]

[[autodoc]] FlaxBertForMultipleChoice
    - __call__

## FlaxBertForTokenClassification[[transformers.FlaxBertForTokenClassification]]

[[autodoc]] FlaxBertForTokenClassification
    - __call__

## FlaxBertForQuestionAnswering[[transformers.FlaxBertForQuestionAnswering]]

[[autodoc]] FlaxBertForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>


