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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
        <img alt= "TensorFlow" src= "https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" >
        <img alt= "Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style…Nu+W0m6K/I9gGPd/dfx/EN/wN62AhsBWuAAAAAElFTkSuQmCC">
        <img alt="SDPA" src= "https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white" > 
    </div>
</div>

# ALBERT[[albert]]

[ALBERT](https://huggingface.co/papers/1909.11942)는 [BERT](./bert)의 확장성과 학습 시 메모리 한계를 해결하기 위해 설계된 모델입니다. 이 모델은 두 가지 파라미터 감소 기법을 도입합니다. 첫 번째는 임베딩 행렬 분해(factorized embedding parametrization)로, 큰 어휘 임베딩 행렬을 두 개의 작은 행렬로 분해하여 히든 사이즈를 늘려도 파라미터 수가 크게 증가하지 않도록 합니다. 두 번째는 계층 간 파라미터 공유(cross-layer parameter sharing)로, 여러 계층이 파라미터를 공유하여 학습해야 할 파라미터 수를 줄입니다.

ALBERT는 BERT에서 발생하는 GPU/TPU 메모리 한계, 긴 학습 시간, 갑작스런 성능 저하 문제를 해결하기 위해 만들어졌습니다. ALBERT는 파라미터를 줄이기 위해 두 가지 기법을 사용하여 메모리 사용량을 줄이고 BERT의 학습 속도를 높입니다:

- **임베딩 행렬 분해:** 큰 어휘 임베딩 행렬을 두 개의 더 작은 행렬로 분해하여 메모리 사용량을 줄입니다.
- **계층 간 파라미터 공유:** 각 트랜스포머 계층마다 별도의 파라미터를 학습하는 대신, 여러 계층이 파라미터를 공유하여 학습해야 할 가중치 수를 더욱 줄입니다.

ALBERT는 BERT와 마찬가지로 절대 위치 임베딩(absolute position embeddings)을 사용하므로, 입력 패딩은 오른쪽에 적용해야 합니다. 임베딩 크기는 128이며, BERT의 768보다 작습니다. ALBERT는 한 번에 최대 512개의 토큰을 처리할 수 있습니다.

모든 공식 ALBERT 체크포인트는 [ALBERT 커뮤니티](https://huggingface.co/albert) 조직에서 확인하실 수 있습니다.

> [!TIP]
> 오른쪽 사이드바의 ALBERT 모델을 클릭하시면 다양한 언어 작업에 ALBERT를 적용하는 예시를 더 확인하실 수 있습니다.

아래 예시는 [`Pipeline`], [`AutoModel`] 그리고 커맨드라인에서 `[MASK]` 토큰을 예측하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="albert-base-v2",
    dtype=torch.float16,
    device=0
)
pipeline("식물은 광합성이라고 알려진 과정을 통해 [MASK]를 생성합니다.", top_k=5)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForMaskedLM.from_pretrained(
    "albert/albert-base-v2",
    dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto"
)

prompt = "식물은 [MASK]이라고 알려진 과정을 통해 에너지를 생성합니다."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    predictions = outputs.logits[0, mask_token_index]

top_k = torch.topk(predictions, k=5).indices.tolist()
for token_id in top_k[0]:
    print(f"예측: {tokenizer.decode([token_id])}")
```

</hfoption>

</hfoptions>

## 참고 사항[[notes]]

- BERT는 절대 위치 임베딩을 사용하므로, 오른쪽에 입력이 패딩돼야 합니다.
- 임베딩 크기 `E`는 히든 크기 `H`와 다릅니다. 임베딩은 문맥에 독립적(각 토큰마다 하나의 임베딩 벡터)이고, 은닉 상태는 문맥에 의존적(토큰 시퀀스마다 하나의 은닉 상태)입니다. 임베딩 행렬은 `V x E`(V: 어휘 크기)이므로, 일반적으로 `H >> E`가 더 논리적입니다. `E < H`일 때 모델 파라미터가 더 적어집니다.

## 참고 자료[[resources]]

아래 섹션의 자료들은 공식 Hugging Face 및 커뮤니티(🌎 표시) 자료로, AlBERT를 시작하는 데 도움이 됩니다. 여기에 추가할 자료가 있다면 Pull Request를 보내주세요! 기존 자료와 중복되지 않고 새로운 내용을 담고 있으면 좋습니다.

<PipelineTag pipeline="text-classification"/>

- [`AlbertForSequenceClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)에서 지원됩니다.

- [`TFAlbertForSequenceClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)에서 지원됩니다.

- [`FlaxAlbertForSequenceClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)에서 지원됩니다.
- [텍스트 분류 작업 가이드](../tasks/sequence_classification)에서 모델 사용법을 확인하세요.

<PipelineTag pipeline="token-classification"/>

- [`AlbertForTokenClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)에서 지원됩니다.

- [`TFAlbertForTokenClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)에서 지원됩니다.

- [`FlaxAlbertForTokenClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification)에서 지원됩니다.
- 🤗 Hugging Face의 [토큰 분류](https://huggingface.co/course/chapter7/2?fw=pt) 강좌
- [토큰 분류 작업 가이드](../tasks/token_classification)에서 모델 사용법을 확인하세요.

<PipelineTag pipeline="fill-mask"/>

- [`AlbertForMaskedLM`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)에서 지원됩니다.
- [`TFAlbertForMaskedLM`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)에서 지원됩니다.
- [`FlaxAlbertForMaskedLM`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)에서 지원됩니다.
- 🤗 Hugging Face의 [마스킹 언어 모델링](https://huggingface.co/course/chapter7/3?fw=pt) 강좌
- [마스킹 언어 모델링 작업 가이드](../tasks/masked_language_modeling)에서 모델 사용법을 확인하세요.

<PipelineTag pipeline="question-answering"/>

- [`AlbertForQuestionAnswering`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)에서 지원됩니다.
- [`TFAlbertForQuestionAnswering`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)에서 지원됩니다.
- [`FlaxAlbertForQuestionAnswering`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering)에서 지원됩니다.
- [질의응답](https://huggingface.co/course/chapter7/7?fw=pt) 🤗 Hugging Face 강좌의 챕터.
- [질의응답 작업 가이드](../tasks/question_answering)에서 모델 사용법을 확인하세요.

**다중 선택(Multiple choice)**

- [`AlbertForMultipleChoice`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)에서 지원됩니다.
- [`TFAlbertForMultipleChoice`]는 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)에서 지원됩니다.

- [다중 선택 작업 가이드](../tasks/multiple_choice)에서 모델 사용법을 확인하세요.

## AlbertConfig[[albertconfig]]

[[autodoc]] AlbertConfig

## AlbertTokenizer[[alberttokenizer]]

[[autodoc]] AlbertTokenizer 
    - get_special_tokens_mask 
    - save_vocabulary

## AlbertTokenizerFast[[alberttokenizerfast]]

[[autodoc]] AlbertTokenizerFast

## Albert 특화 출력[[albert-specific-outputs]]

[[autodoc]] models.albert.modeling_albert.AlbertForPreTrainingOutput


## AlbertModel[[albertmodel]]

[[autodoc]] AlbertModel 
    - forward

## AlbertForPreTraining[[albertforpretraining]]

[[autodoc]] AlbertForPreTraining 
    - forward

## AlbertForMaskedLM[[albertformaskedlm]]

[[autodoc]] AlbertForMaskedLM 
    - forward

## AlbertForSequenceClassification[[albertforsequenceclassification]]

[[autodoc]] AlbertForSequenceClassification 
    - forward

## AlbertForMultipleChoice[[albertformultiplechoice]]

[[autodoc]] AlbertForMultipleChoice

## AlbertForTokenClassification[[albertfortokenclassification]]

[[autodoc]] AlbertForTokenClassification 
    - forward

## AlbertForQuestionAnswering[[albertforquestionanswering]]

[[autodoc]] AlbertForQuestionAnswering 
    - forward

