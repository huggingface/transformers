<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 둘러보기[[quick-tour]]

[[open-in-colab]]
🤗 Transformer를 시작해봐요! 둘러보기는 개발자와 일반 사용자 모두를 위해 쓰여졌습니다. [`pipeline`]으로 추론하는 방법, [AutoClass](./model_doc/auto)로 사전학습된 모델과 전처리기를 적재하는 방법과 PyTorch 또는 TensorFlow로 신속하게 모델을 훈련시키는 방법을 보여줍니다. 기본을 배우고 싶다면 튜토리얼이나 [course](https://huggingface.co/course/chapter1/1)에서 여기 소개된 개념에 대한 자세한 설명을 확인하시길 권장합니다.

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하고,

```bash
!pip install transformers datasets
```

좋아하는 머신러닝 프레임워크도 설치해야 합니다.

<frameworkcontent>
<pt>
```bash
pip install torch
```
</pt>
<tf>
```bash
pip install tensorflow
```
</tf>
</frameworkcontent>

## Pipeline (파이프라인)

<Youtube id="tiZFewofSLM"/>

[`pipeline`]은 사전학습된 모델을 사용해 추론할 때 제일 쉬운 방법입니다. 여러 모달리티의 수많은 태스크에 [`pipeline`]을 즉시 사용할 수 있습니다. 지원하는 태스크의 예시는 아래 표를 참고하세요.

| **태스크**     | **설명**                                                            | **모달리티**     | **파이프라인 ID**                             |
|----------------|---------------------------------------------------------------------|------------------|-----------------------------------------------|
| 텍스트 분류    | 텍스트에 알맞은 라벨 붙이기                                         | 자연어 처리(NLP) | pipeline(task="sentiment-analysis")           |
| 텍스트 생성    | 주어진 문자열 입력과 이어지는 텍스트 생성하기                       | 자연어 처리(NLP) | pipeline(task="text-generation")              |
| 개체명 인식    | 문자열의 각 토큰마다 알맞은 라벨 붙이기 (인물, 조직, 장소 등등)     | 자연어 처리(NLP) | pipeline(task="ner")                          |
| 질의응답       | 주어진 문맥과 질문에 따라 올바른 대답하기                           | 자연어 처리(NLP) | pipeline(task="question-answering")           |
| 빈칸 채우기    | 문자열의 빈칸에 알맞은 토큰 맞추기                                  | 자연어 처리(NLP) | pipeline(task="fill-mask")                    |
| 요약           | 텍스트나 문서를 요약하기                                            | 자연어 처리(NLP) | pipeline(task="summarization")                |
| 번역           | 텍스트를 한 언어에서 다른 언어로 번역하기                           | 자연어 처리(NLP) | pipeline(task="translation")                  |
| 이미지 분류    | 이미지에 알맞은 라벨 붙이기                                         | 컴퓨터 비전(CV)  | pipeline(task="image-classification")         |
| 이미지 분할    | 이미지의 픽셀마다 라벨 붙이기(시맨틱, 파놉틱 및 인스턴스 분할 포함) | 컴퓨터 비전(CV)  | pipeline(task="image-segmentation")           |
| 객체 탐지      | 이미지 속 객체의 경계 상자를 그리고 클래스를 예측하기               | 컴퓨터 비전(CV)  | pipeline(task="object-detection")             |
| 오디오 분류    | 오디오 파일에 알맞은 라벨 붙이기                                    | 오디오           | pipeline(task="audio-classification")         |
| 자동 음성 인식 | 오디오 파일 속 음성을 텍스트로 바꾸기                               | 오디오           | pipeline(task="automatic-speech-recognition") |
| 시각 질의응답  | 주어진 이미지와 이미지에 대한 질문에 따라 올바르게 대답하기         | 멀티모달         | pipeline(task="vqa")                          |

먼저 [`pipeline`]의 인스턴스를 만들어 적용할 태스크를 고르세요. 위 태스크들은 모두 [`pipeline`]을 사용할 수 있고, 지원하는 태스크의 전체 목록을 보려면 [pipeline API 레퍼런스](./main_classes/pipelines)를 확인해주세요. 간단한 예시로 감정 분석 태스크에 [`pipeline`]를 적용해 보겠습니다.

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

[`pipeline`]은 기본 [사전학습된 모델(영어)](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)와 감정 분석을 하기 위한 tokenizer를 다운로드하고 캐시해놓습니다. 이제 원하는 텍스트에 `classifier`를 사용할 수 있습니다.

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

입력이 여러 개라면, 입력을 [`pipeline`]에 리스트로 전달해서 딕셔너리로 된 리스트를 받을 수 있습니다.

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

[`pipeline`]은 특정 태스크용 데이터셋를 전부 순회할 수도 있습니다. 자동 음성 인식 태스크에 적용해 보겠습니다.

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

이제 순회할 오디오 데이터셋를 적재하겠습니다. (자세한 내용은 🤗 Datasets [시작하기](https://huggingface.co/docs/datasets/quickstart#audio)를 참고해주세요) [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 데이터셋로 해볼까요?

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

데이터셋의 샘플링 레이트가 [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)의 훈련 당시 샘플링 레이트와 일치해야만 합니다.

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

오디오 파일은 `"audio"` 열을 호출할 때 자동으로 적재되고 다시 샘플링됩니다.
처음 4개 샘플에서 음성을 추출하여 파이프라인에 리스트 형태로 전달해보겠습니다.

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FODING HOW I'D SET UP A JOIN TO HET WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE AP SO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AND I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I THURN A JOIN A COUNT']
```

(음성이나 비전처럼) 입력이 큰 대규모 데이터셋의 경우, 메모리에 적재시키기 위해 리스트 대신 제너레이터로 입력을 모두 전달할 수 있습니다. 자세한 내용은 [pipeline API 레퍼런스](./main_classes/pipelines)를 확인해주세요.

### 파이프라인에서 다른 모델이나 tokenizer 사용하는 방법[[use-another-model-and-tokenizer-in-the-pipeline]]

[`pipeline`]은 [Hub](https://huggingface.co/models) 속 모든 모델을 사용할 수 있어, 얼마든지 [`pipeline`]을 사용하고 싶은대로 바꿀 수 있습니다. 예를 들어 프랑스어 텍스트를 다룰 수 있는 모델을 만드려면, Hub의 태그로 적절한 모델을 찾아보세요. 상위 검색 결과로 뜬 감정 분석을 위해 파인튜닝된 다국어 [BERT 모델](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)이 프랑스어를 지원하는군요.

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
[`AutoModelForSequenceClassification`]과 [`AutoTokenizer`]로 사전학습된 모델과 함께 연관된 토크나이저를 불러옵니다. (`AutoClass`에 대한 내용은 다음 섹션에서 살펴보겠습니다)

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
[`TFAutoModelForSequenceClassification`]과 [`AutoTokenizer`]로 사전학습된 모델과 함께 연관된 토크나이저를 불러옵니다. (`TFAutoClass`에 대한 내용은 다음 섹션에서 살펴보겠습니다)

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

[`pipeline`]에서 사용할 모델과 토크나이저를 입력하면 이제 (감정 분석기인) `classifier`를 프랑스어 텍스트에 적용할 수 있습니다.

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

하고싶은 것에 적용할 마땅한 모델이 없다면, 가진 데이터로 사전학습된 모델을 파인튜닝해야 합니다. 자세한 방법은 [파인튜닝 튜토리얼](./training)을 참고해주세요. 사전학습된 모델의 파인튜닝을 마치셨으면, 누구나 머신러닝을 할 수 있도록 [공유](./model_sharing)하는 것을 고려해주세요. 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

내부적으로 들어가면 위에서 사용했던 [`pipeline`]은 [`AutoModelForSequenceClassification`]과 [`AutoTokenizer`] 클래스로 작동합니다. [AutoClass](./model_doc/auto)란 이름이나 경로를 받으면 그에 알맞는 사전학습된 모델을 가져오는 '바로가기'라고 볼 수 있는데요. 원하는 태스크와 전처리에 적합한 `AutoClass`를 고르기만 하면 됩니다.

전에 사용했던 예시로 돌아가서 `AutoClass`로 [`pipeline`]과 동일한 결과를 얻을 수 있는 방법을 알아보겠습니다.

### AutoTokenizer

토크나이저는 전처리를 담당하며, 텍스트를 모델이 받을 숫자 배열로 바꿉니다. 토큰화 과정에는 단어를 어디에서 끊을지, 얼만큼 나눌지 등을 포함한 여러 규칙이 있습니다. 자세한 내용은 [토크나이저 요약](./tokenizer_summary)를 확인해주세요. 제일 중요한 점은 모델이 훈련됐을 때와 동일한 토큰화 규칙을 쓰도록 동일한 모델 이름으로 토크나이저 인스턴스를 만들어야 합니다.

[`AutoTokenizer`]로 토크나이저를 불러오고,

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

토크나이저에 텍스트를 제공하세요.

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

그러면 다음을 포함한 딕셔너리가 반환됩니다.

* [input_ids](./glossary#input-ids): 숫자로 표현된 토큰들
* [attention_mask](.glossary#attention-mask): 주시할 토큰들

토크나이저는 입력을 리스트로도 받을 수 있으며, 텍스트를 패드하거나 잘라내어 균일한 길이의 배치를 반환할 수도 있습니다.

<frameworkcontent>
<pt>
```py
>>> pt_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="pt",
... )
```
</pt>
<tf>
```py
>>> tf_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="tf",
... )
```
</tf>
</frameworkcontent>

<Tip>

[전처리](./preprocessing) 튜토리얼을 보시면 토큰화에 대한 자세한 설명과 함께 이미지, 오디오와 멀티모달 입력을 전처리하기 위한 [`AutoFeatureExtractor`]과 [`AutoProcessor`]의 사용방법도 알 수 있습니다.

</Tip>

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers로 사전학습된 인스턴스를 간단하고 통일된 방식으로 불러올 수 있습니다. 이러면 [`AutoTokenizer`]처럼 [`AutoModel`]도 불러올 수 있게 됩니다. 유일한 차이점은 태스크에 적합한 [`AutoModel`]을 선택해야 한다는 점입니다. 텍스트(또는 시퀀스) 분류의 경우 [`AutoModelForSequenceClassification`]을 불러와야 합니다.

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

[`AutoModel`] 클래스에서 지원하는 태스크들은 [태스크 정리](./task_summary) 문서를 참고해주세요.

</Tip>

이제 전처리된 입력 배치를 모델로 직접 보내야 합니다. 아래처럼 `**`를 앞에 붙여 딕셔너리를 풀어주기만 하면 됩니다.

```py
>>> pt_outputs = pt_model(**pt_batch)
```

모델의 activation 결과는 `logits` 속성에 담겨있습니다. `logits`에 Softmax 함수를 적용해서 확률 형태로 받으세요.

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers는 사전학습된 인스턴스를 간단하고 통일된 방식으로 불러올 수 있습니다. 이러면 [`AutoTokenizer`]처럼 [`TFAutoModel`]도 불러올 수 있게 됩니다. 유일한 차이점은 태스크에 적합한 [`TFAutoModel`]를 선택해야 한다는 점입니다. 텍스트(또는 시퀀스) 분류의 경우 [`TFAutoModelForSequenceClassification`]을 불러와야 합니다.

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

[`AutoModel`] 클래스에서 지원하는 태스크들은 [태스크 정리](./task_summary) 문서를 참고해주세요.

</Tip>

이제 전처리된 입력 배치를 모델로 직접 보내야 합니다. 딕셔너리의 키를 텐서에 직접 넣어주기만 하면 됩니다.

```py
>>> tf_outputs = tf_model(tf_batch)
```

모델의 activation 결과는 `logits` 속성에 담겨있습니다. `logits`에 Softmax 함수를 적용해서 확률 형태로 받으세요.

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

모든 (PyTorch 또는 TensorFlow) 🤗 Transformers 모델은 (softmax 등의) 최종 activation 함수 *이전에* 텐서를 내놓습니다. 왜냐하면 최종 activation 함수를 종종 loss 함수와 동일시하기 때문입니다. 모델 출력은 특수 데이터 클래스이므로 해당 속성은 IDE에서 자동으로 완성됩니다. 모델 출력은 튜플 또는 (정수, 슬라이스 또는 문자열로 인덱싱하는) 딕셔너리 형태로 주어지고 이런 경우 None인 속성은 무시됩니다.

</Tip>

### 모델 저장하기[[save-a-model]]

<frameworkcontent>
<pt>
모델을 파인튜닝한 뒤에는 [`PreTrainedModel.save_pretrained`]로 모델을 토크나이저와 함께 저장할 수 있습니다.

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

모델을 다시 사용할 때는 [`PreTrainedModel.from_pretrained`]로 다시 불러오면 됩니다.

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
모델을 파인튜닝한 뒤에는 [`TFPreTrainedModel.save_pretrained`]로 모델을 토크나이저와 함께 저장할 수 있습니다.

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

모델을 다시 사용할 때는 [`TFPreTrainedModel.from_pretrained`]로 다시 불러오면 됩니다.

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

🤗 Transformers 기능 중 특히 재미있는 한 가지는 모델을 저장하고 PyTorch나 TensorFlow 모델로 다시 불러올 수 있는 기능입니다. 'from_pt' 또는 'from_tf' 매개변수를 사용해 모델을 기존과 다른 프레임워크로 변환시킬 수 있습니다.

<frameworkcontent>
<pt>
```py
>>> from transformers import AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained(tf_save_directory)
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(tf_save_directory, from_tf=True)
```
</pt>
<tf>
```py
>>> from transformers import TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(pt_save_directory, from_pt=True)
```
</tf>
</frameworkcontent>

## 커스텀 모델 구축하기[[custom-model-builds]]

모델의 구성 클래스를 수정하여 모델의 구조를 바꿀 수 있습니다. 은닉층, 어텐션 헤드 수와 같은 모델의 속성을 구성에서 지정합니다. 커스텀 구성 클래스에서 모델을 만들면 처음부터 시작해야 합니다. 모델 속성은 랜덤하게 초기화되므로 의미 있는 결과를 얻으려면 먼저 모델을 훈련시킬 필요가 있습니다.

먼저 [`AutoConfig`]를 임포트하고, 수정하고 싶은 사전학습된 모델을 불러오세요. [`AutoConfig.from_pretrained`]에서 어텐션 헤드 수 같은 속성을 변경할 수 있습니다.

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
[`AutoModel.from_config`]를 사용하여 커스텀 구성대로 모델을 생성합니다.

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```
</pt>
<tf>
[`TFAutoModel.from_config`]를 사용하여 커스텀 구성대로 모델을 생성합니다.

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```
</tf>
</frameworkcontent>

커스텀 구성을 작성하는 방법에 대한 자세한 내용은 [커스텀 아키텍처 만들기](./create_a_model) 가이드를 참고하세요.

## Trainer - PyTorch에 최적화된 훈련 반복 루프[[trainer-a-pytorch-optimized-training-loop]]

모든 모델은 [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)이어서 대다수의 훈련 반복 루프에 사용할 수 있습니다. 사용자가 직접 훈련 반복 루프를 작성해도 되지만, 🤗 Transformers는 PyTorch용 [`Trainer`] 클래스를 제공합니다. 기본적인 훈련 반폭 루프가 포함되어 있고, 분산 훈련이나 혼합 정밀도 등의 추가 기능도 있습니다.

태스크에 따라 다르지만, 일반적으로 다음 매개변수를 [`Trainer`]에 전달할 것입니다.

1. [`PreTrainedModel`] 또는 [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)로 시작합니다.

   ```py
   >>> from transformers import AutoModelForSequenceClassification

   >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
   ```

2. [`TrainingArguments`]로 학습률, 배치 크기나 훈련할 epoch 수와 같이 모델의 하이퍼파라미터를 조정합니다. 기본값은 훈련 인수를 전혀 지정하지 않은 경우 사용됩니다.

   ```py
   >>> from transformers import TrainingArguments

   >>> training_args = TrainingArguments(
   ...     output_dir="path/to/save/folder/",
   ...     learning_rate=2e-5,
   ...     per_device_train_batch_size=8,
   ...     per_device_eval_batch_size=8,
   ...     num_train_epochs=2,
   ... )
   ```

3. 토크나이저, 특징추출기(feature extractor), 전처리기(processor) 클래스 등으로 전처리를 수행합니다.

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   ```

4. 데이터셋를 적재합니다.

   ```py
   >>> from datasets import load_dataset

   >>> dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
   ```

5. 데이터셋을 토큰화하는 함수를 만들고 [`~datasets.Dataset.map`]으로 전체 데이터셋에 적용시킵니다.

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])


   >>> dataset = dataset.map(tokenize_dataset, batched=True)
   ```

6. [`DataCollatorWithPadding`]로 데이터셋으로부터 표본으로 삼을 배치를 만듭니다.

   ```py
   >>> from transformers import DataCollatorWithPadding

   >>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

이제 위의 모든 클래스를 [`Trainer`]로 모으세요.

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )  # doctest: +SKIP
```

준비되었으면 [`~Trainer.train`]으로 훈련을 시작하세요.

```py
>>> trainer.train()  # doctest: +SKIP
```

<Tip>

sequence-to-sequence 모델을 사용하는 (번역이나 요약 같은) 태스크의 경우 [`Seq2SeqTrainer`]와 [`Seq2SeqTrainingArguments`] 클래스를 대신 사용하시기 바랍니다.

</Tip>

[`Trainer`] 내부의 메서드를 구현 상속(subclassing)해서 훈련 반복 루프를 개조할 수도 있습니다. 이러면 loss 함수, optimizer, scheduler 등의 기능도 개조할 수 있습니다. 어떤 메서드를 구현 상속할 수 있는지 알아보려면 [`Trainer`]를 참고하세요. 

훈련 반복 루프를 개조하는 다른 방법은 [Callbacks](./main_classes/callbacks)를 사용하는 것입니다. Callbacks로 다른 라이브러리와 통합하고, 훈련 반복 루프를 수시로 체크하여 진행 상황을 보고받거나, 훈련을 조기에 중단할 수 있습니다. Callbacks은 훈련 반복 루프 자체를 전혀 수정하지 않습니다. 만약 loss 함수 등을 개조하고 싶다면 [`Trainer`]를 구현 상속해야만 합니다.

## TensorFlow로 훈련시키기[[train-with-tensorflow]]

모든 모델은 [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)이어서 [Keras](https://keras.io/) API를 통해 TensorFlow에서 훈련시킬 수 있습니다. 🤗 Transformers에서 데이터셋를 `tf.data.Dataset` 형태로 쉽게 적재할 수 있는 [`~TFPreTrainedModel.prepare_tf_dataset`] 메서드를 제공하기 때문에, Keras의 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 및 [`fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 메서드로 즉시 훈련을 시작할 수 있습니다.

1. [`TFPreTrainedModel`] 또는 [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)로 시작합니다.

   ```py
   >>> from transformers import TFAutoModelForSequenceClassification

   >>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
   ```

2. 토크나이저, 특징추출기(feature extractor), 전처리기(processor) 클래스 등으로 전처리를 수행합니다.

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   ```

3. 데이터셋을 토큰화하는 함수를 만듭니다.

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])  # doctest: +SKIP
   ```

4. [`~datasets.Dataset.map`]으로 전체 데이터셋에 위 함수를 적용시킨 다음, 데이터셋과 토크나이저를 [`~TFPreTrainedModel.prepare_tf_dataset`]로 전달합니다. 배치 크기를 변경해보거나 데이터셋를 섞어봐도 좋습니다.

   ```py
   >>> dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
   >>> tf_dataset = model.prepare_tf_dataset(
   ...     dataset, batch_size=16, shuffle=True, tokenizer=tokenizer
   ... )  # doctest: +SKIP
   ```

5. 준비되었으면 `compile`과 `fit`으로 훈련을 시작하세요.

   ```py
   >>> from tensorflow.keras.optimizers import Adam

   >>> model.compile(optimizer=Adam(3e-5))
   >>> model.fit(dataset)  # doctest: +SKIP
   ```

## 이제 무얼 하면 될까요?[[whats-next]]

🤗 Transformers 둘러보기를 모두 읽으셨다면, 가이드를 통해 특정 기술을 배울 수 있어요. 예를 들어 커스텀 모델을 작성하는 방법, 태스크용 모델을 파인튜닝하는 방법, 스크립트로 모델을 훈련시키는 방법 등이 있습니다. 🤗 Transformers의 핵심 개념에 대해 자세히 알아보려면 커피 한 잔을 마신 뒤 개념 가이드를 살펴보셔도 좋습니다!
