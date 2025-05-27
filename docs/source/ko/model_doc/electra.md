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

# ELECTRA[[electra]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAtCAMAAAANxBKoAAAC7lBMVEUAAADg5vYHPVgAoJH+/v76+v39/f9JbLP///9+AIgAnY3///+mcqzt8fXy9fgkXa3Ax9709fr+///9/f8qXq49qp5AaLGMwrv8/P0eW60VWawxYq8yqJzG2dytt9Wyu9elzci519Lf3O3S2efY3OrY0+Xp7PT///////+dqNCexMc6Z7AGpJeGvbenstPZ5ejQ1OfJzOLa7ejh4+/r8fT29vpccbklWK8PVa0AS6ghW63O498vYa+lsdKz1NDRt9Kw1c672tbD3tnAxt7R6OHp5vDe7OrDyuDn6vLl6/EAQKak0MgATakkppo3ZK/Bz9y8w9yzu9jey97axdvHzeG21NHH4trTwthKZrVGZLSUSpuPQJiGAI+GAI8SWKydycLL4d7f2OTi1+S9xNzL0ePT6OLGzeEAo5U0qJw/aLEAo5JFa7JBabEAp5Y4qZ2QxLyKmsm3kL2xoMOehrRNb7RIbbOZgrGre68AUqwAqZqNN5aKJ5N/lMq+qsd8kMa4pcWzh7muhLMEV69juq2kbKqgUaOTR5uMMZWLLZSGAI5VAIdEAH+ovNDHuNCnxcy3qcaYx8K8msGplrx+wLahjbYdXrV6vbMvYK9DrZ8QrZ8tqJuFms+Sos6sw8ecy8RffsNVeMCvmb43aLltv7Q4Y7EZWK4QWa1gt6meZKUdr6GOAZVeA4xPAISyveLUwtivxtKTpNJ2jcqfvcltiMiwwcfAoMVxhL+Kx7xjdrqTe60tsaNQs6KaRKACrJ6UTZwkqpqTL5pkHY4AloSgsd2ptNXPvNOOncuxxsqFl8lmg8apt8FJcr9EbryGxLqlkrkrY7dRa7ZGZLQ5t6iXUZ6PPpgVpZeJCJFKAIGareTa0+KJod3H0deY2M+esM25usmYu8d2zsJOdcBVvrCLbqcAOaaHaKQAMaScWqKBXqCXMJ2RHpiLF5NmJZAdAHN2kta11dKu1M+DkcZLdb+Mcql3TppyRJdzQ5ZtNZNlIY+DF4+voCOQAAAAZ3RSTlMABAT+MEEJ/RH+/TP+Zlv+pUo6Ifz8+fco/fz6+evr39S9nJmOilQaF/7+/f38+smmoYp6b1T+/v7++vj189zU0tDJxsGzsrKSfv34+Pf27dDOysG9t6+n/vv6+vr59uzr1tG+tZ6Qg9Ym3QAABR5JREFUSMeNlVVUG1EQhpcuxEspXqS0SKEtxQp1d3d332STTRpIQhIISQgJhODu7lAoDoUCpe7u7u7+1puGpqnCPOyZvffbOXPm/PsP9JfQgyCC+tmTABTOcbxDz/heENS7/1F+9nhvkHePG0wNDLbGWwdXL+rbLWvpmZHXD8+gMfBjTh+aSe6Gnn7lwQIOTR0c8wfX3PWgv7avbdKwf/ZoBp1Gp/PvuvXW3vw5ib7emnTW4OR+3D4jB9vjNJ/7gNvfWWeH/TO/JyYrsiKCRjVEZA3UB+96kON+DxOQ/NLE8PE5iUYgIXjFnCOlxEQMaSGVxjg4gxOnEycGz8bptuNjVx08LscIgrzH3umcn+KKtiBIyvzOO2O99aAdR8cF19oZalnCtvREUw79tCd5sow1g1UKM6kXqUx4T8wsi3sTjJ3yzDmmhenLXLpo8u45eG5y4Vvbk6kkC4LLtJMowkSQxmk4ggVJEG+7c6QpHT8vvW9X7/o7+3ELmiJi2mEzZJiz8cT6TBlanBk70cB5GGIGC1gRDdZ00yADLW1FL6gqhtvNXNG5S9gdSrk4M1qu7JAsmYshzDS4peoMrU/gT7qQdqYGZaYhxZmVbGJAm/CS/HloWyhRUlknQ9KYcExTwS80d3VNOxUZJpITYyspl0LbhArhpZCD9cRWEQuhYkNGMHToQ/2Cs6swJlb39CsllxdXX6IUKh/H5jbnSsPKjgmoaFQ1f8wRLR0UnGE/RcDEjj2jXG1WVTwUs8+zxfcrVO+vSsuOpVKxCfYZiQ0/aPKuxQbQ8lIz+DClxC8u+snlcJ7Yr1z1JPqUH0V+GDXbOwAib931Y4Imaq0NTIXPXY+N5L18GJ37SVWu+hwXff8l72Ds9XuwYIBaXPq6Shm4l+Vl/5QiOlV+uTk6YR9PxKsI9xNJny31ygK1e+nIRC1N97EGkFPI+jCpiHe5PCEy7oWqWSwRrpOvhFzcbTWMbm3ZJAOn1rUKpYIt/lDhW/5RHHteeWFN60qo98YJuoq1nK3uW5AabyspC1BcIEpOhft+SZAShYoLSvnmSfnYADUERP5jJn2h5XtsgCRuhYQqAvwTwn33+YWEKUI72HX5AtfSAZDe8F2DtPPm77afhl0EkthzuCQU0BWApgQIH9+KB0JhopMM7bJrdTRoleM2JAVNMyPF+wdoaz+XJpGoVAQ7WXUkcV7gT3oUZyi/ISIJAVKhgNp+4b4veCFhYVJw4locdSjZCp9cPUhLF9EZ3KKzURepMEtCDPP3VcWFx4UIiZIklIpFNfHpdEafIF2aRmOcrUmjohbT2WUllbmRvgfbythbQO3222fpDJoufaQPncYYuqoGtUEsCJZL6/3PR5b4syeSjZMQG/T2maGANlXT2v8S4AULWaUkCxfLyW8iW4kdka+nEMjxpL2NCwsYNBp+Q61PF43zyDg9Bm9+3NNySn78jMZUUkumqE4Gp7JmFOdP1vc8PpRrzj9+wPinCy8K1PiJ4aYbnTYpCCbDkBSbzhu2QJ1Gd82t8jI8TH51+OzvXoWbnXUOBkNW+0mWFwGcGOUVpU81/n3TOHb5oMt2FgYGjzau0Nif0Ss7Q3XB33hjjQHjHA5E5aOyIQc8CBrLdQSs3j92VG+3nNEjbkbdbBr9zm04ruvw37vh0QKOdeGIkckc80fX3KH/h7PT4BOjgCty8VZ5ux1MoO5Cf5naca2LAsEgehI+drX8o/0Nu+W0m6K/I9gGPd/dfx/EN/wN62AhsBWuAAAAAElFTkSuQmCC
">
</div>

## 개요[[overview]]

ELECTRA 모델은 [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
Generators](https://openreview.net/pdf?id=r1xMH1BtvB) 논문에서 제안되었습니다. ELECTRA는 두가지 트랜스포머 모델인 생성 모델과 판별 모델을 학습시키는 새로운 사전학습 접근법입니다. 생성 모델의 역할은 시퀀스에 있는 토큰을 대체하는 것이며 마스킹된 언어 모델로 학습됩니다. 우리가 관심을 가진 판별 모델은 시퀀스에서 어떤 토큰이 생성 모델에 의해 대체되었는지 식별합니다. 

논문의 초록은 다음과 같습니다:

*BERT와 같은 마스킹된 언어 모델(MLM) 사전학습 방법은 일부 토큰을 [MASK] 토큰으로 바꿔 손상시키고 난 뒤, 모델이 다시 원본 토큰을 복원하도록 학습합니다. 이런 방식은 다운스트림 NLP 작업을 전이할 때 좋은 성능을 내지만, 효과적으로 사용하기 위해서는 일반적으로 많은 양의 연산이 필요합니다. 따라서 대안으로, 대체 토큰 탐지라고 불리는 샘플-효과적인 사전학습을 제안합니다. 우리의 방법론은 입력에 마스킹을 하는 대신에 소형 생성 모델의 그럴듯한 대안 토큰으로 손상시킵니다. 그리고 나서, 모델이 손상된 토큰의 원래 토큰을 예측하도록 훈련시키는 대신, 판별 모델을 각각의 토큰이 생성 모델의 샘플로 손상되었는지 아닌지 학습합니다. 실험들은 통해 이 새로운 사전학습 방식은 마스킹된 일부 토큰에만 적용되는 기존 방식과 달리 모든 입력 토큰에 대해 학습이 이뤄지기 때문에 마스킹된 언어 모델(MLM)보다 더 효율적임을 입증하였습니다. 결과적으로 소개된 방식이 같은 모델 크기, 데이터, 연산량을 가진 BERT모델로 학습한 결과를 압도하는 문맥 표현 학습을 할 수 있다는 것을 확인했습니다. 특히 작은 모델에서 성능 향상이 두드러지며, 예를 들어 GPU 한 대로 4일간 학습한 모델이 30배 더 많은 계산 자원을 사용한 GPT보다 GLUE 자연어 이해 벤치마크에서 더 나은 성능을 보입니다. 대규모 환경에서도 유효하며 더 적은 연산량으로 RoBERTa와 XLNet과 비슷한 성능을 낼 수 있으며, 동일한 연산량을 가질 경우 이들의 성능을 능가합니다.*


이 모델은 [lysandre](https://huggingface.co/lysandre)이 기여했습니다. 원본 코드는 [이곳](https://github.com/google-research/electra)에서 찾아보실 수 있습니다.

## 사용 팁[[usage-tips]]

- ELECTRA는 사전학습 방법으로 기본 모델인 BERT의 구조와 거의 차이가 없습니다. 유일한 차이는 임베딩 크기와 히든 크기를 구분했다는 점입니다. 임베딩 크기는 일반적으로 더 작고, 히든 크기는 더 큽니다. 임베딩에서 임베딩 크기를 히든 크기로 변환하기 위해 추가로 선형 변환 층이 사용됩니다. 임베딩 크기와 히든 크기가 동일할 경우에는 이 선형 변환 층이 필요하지 않습니다. 
- ELECTRA는 또 다른 (작은) 마스킹된 언어 모델을 사용해 사전학습 된 트랜스포머 모델입니다.  작은 언어 모델이 입력 텍스트의 일부를 무작위로 마스킹하고, 그 자리에 새로운 토큰을 삽입합니다. ELECTRA는 원래 토큰과 대체된 토큰을 구분하는 역할을 수행합니다. GAN 훈련과 비슷하지만, 생성 모델은 ELECTRA 모델을 속이는 것이 아니라 원래 텍스트를 복원하는 목표로 몇 단계 학습합니다. 그 후 ELECTRA가 학습을 하게 됩니다.
- [구글 리서치의 구현](https://github.com/google-research/electra)으로 저장된 ELECTRA checkpoints는 생성 모델과 판별 모델을 포함합니다. 변환 스크립트에서는 사용자가 어떤 모델을 어떤 아키텍처로 내보낼지 명시해야 합니다. 일단 Hugging Face 포맷으로 변환되면, 이 체크포인트들은 모든 ELECTRA 모델에서 불러올 수 있습니다. 즉, 판별 모델은 [`ElectraForMaskedLM`] 모델에, 생성 모델은 [`ElectraForPreTraining`]모델에 불러올 수 있다는 의미입니다. (단, 생성 모델에는 분류 헤드가 존재하지 않기 때문에, 해당 부분은 무작위로 초기화됩니다.)

## 참고 자료[[resources]]

- [텍스트 분류 가이드](../tasks/sequence_classification)
- [토큰 분류 가이드](../tasks/token_classification)
- [질의 응답 가이드](../tasks/question_answering)
- [인과 언어 모델링 가이드](../tasks/language_modeling)
- [마스킹된 언어 모델링 가이드](../tasks/masked_language_modeling)
- [객관식 문제 가이드](../tasks/multiple_choice)

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## Electra specific outputs

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

[[autodoc]] models.electra.modeling_tf_electra.TFElectraForPreTrainingOutput

<frameworkcontent>
<pt>

## ElectraModel

[[autodoc]] ElectraModel
    - forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining
    - forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM
    - forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM
    - forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification
    - forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice
    - forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification
    - forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering
    - forward

</pt>
<tf>

## TFElectraModel

[[autodoc]] TFElectraModel
    - call

## TFElectraForPreTraining

[[autodoc]] TFElectraForPreTraining
    - call

## TFElectraForMaskedLM

[[autodoc]] TFElectraForMaskedLM
    - call

## TFElectraForSequenceClassification

[[autodoc]] TFElectraForSequenceClassification
    - call

## TFElectraForMultipleChoice

[[autodoc]] TFElectraForMultipleChoice
    - call

## TFElectraForTokenClassification

[[autodoc]] TFElectraForTokenClassification
    - call

## TFElectraForQuestionAnswering

[[autodoc]] TFElectraForQuestionAnswering
    - call

</tf>
<jax>

## FlaxElectraModel

[[autodoc]] FlaxElectraModel
    - __call__

## FlaxElectraForPreTraining

[[autodoc]] FlaxElectraForPreTraining
    - __call__

## FlaxElectraForCausalLM

[[autodoc]] FlaxElectraForCausalLM
    - __call__

## FlaxElectraForMaskedLM

[[autodoc]] FlaxElectraForMaskedLM
    - __call__

## FlaxElectraForSequenceClassification

[[autodoc]] FlaxElectraForSequenceClassification
    - __call__

## FlaxElectraForMultipleChoice

[[autodoc]] FlaxElectraForMultipleChoice
    - __call__

## FlaxElectraForTokenClassification

[[autodoc]] FlaxElectraForTokenClassification
    - __call__

## FlaxElectraForQuestionAnswering

[[autodoc]] FlaxElectraForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>
