<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 텍스트 분류[[text-classification]]

[[open-in-colab]]

<Youtube id="leNG9fN9FQU"/>

텍스트 분류는 자연어 처리의 일종으로, 텍스트에 레이블 또는 클래스를 지정하는 작업입니다. 많은 대기업이 다양한 실용적인 응용 분야에서 텍스트 분류를 운영하고 있습니다. 가장 인기 있는 텍스트 분류 형태 중 하나는 감성 분석으로, 텍스트 시퀀스에 🙂 긍정, 🙁 부정 또는 😐 중립과 같은 레이블을 지정합니다.

이 가이드에서 학습할 내용은:

1. [IMDb](https://huggingface.co/datasets/imdb) 데이터셋에서 [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)를 파인 튜닝하여 영화 리뷰가 긍정적인지 부정적인지 판단합니다.
2. 추론을 위해 파인 튜닝 모델을 사용합니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/text-classification)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에, 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate
```

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티에 공유하는 것을 권장합니다. 메시지가 표시되면, 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## IMDb 데이터셋 가져오기[[load-imdb-dataset]]

먼저 🤗 Datasets 라이브러리에서 IMDb 데이터셋을 가져옵니다:

```py
>>> from datasets import load_dataset

>>> imdb = load_dataset("imdb")
```

그런 다음 예시를 살펴봅시다:

```py
>>> imdb["test"][0]
{
    "label": 0,
    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
}
```

이 데이터셋에는 두 가지 필드가 있습니다:

- `text`: 영화 리뷰 텍스트
- `label`: `0`은 부정적인 리뷰, `1`은 긍정적인 리뷰를 나타냅니다.

## 전처리[[preprocess]]

다음 단계는 DistilBERT 토크나이저를 가져와서 `text` 필드를 전처리하는 것입니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

`text`를 토큰화하고 시퀀스가 DistilBERT의 최대 입력 길이보다 길지 않도록 자르기 위한 전처리 함수를 생성하세요:

```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

전체 데이터셋에 전처리 함수를 적용하려면, 🤗 Datasets [`~datasets.Dataset.map`] 함수를 사용하세요. 데이터셋의 여러 요소를 한 번에 처리하기 위해 `batched=True`로 설정함으로써 데이터셋 `map`를 더 빠르게 처리할 수 있습니다:

```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

이제 [`DataCollatorWithPadding`]를 사용하여 예제 배치를 만들어봅시다. 데이터셋 전체를 최대 길이로 패딩하는 대신, *동적 패딩*을 사용하여 배치에서 가장 긴 길이에 맞게 문장을 패딩하는 것이 효율적입니다.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## 평가하기[[evaluate]]

훈련 중 모델의 성능을 평가하기 위해 메트릭을 포함하는 것이 유용합니다. 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하여 빠르게 평가 방법을 로드할 수 있습니다. 이 작업에서는 [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 메트릭을 가져옵니다. (메트릭을 가져오고 계산하는 방법에 대해서는 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하세요):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

그런 다음 `compute_metrics` 함수를 만들어서 예측과 레이블을 계산하여 정확도를 계산하도록 [`~evaluate.EvaluationModule.compute`]를 호출합니다:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

이제 `compute_metrics` 함수는 준비되었고, 훈련 과정을 설정할 때 다시 살펴볼 예정입니다.

## 훈련[[train]]

모델을 훈련하기 전에, `id2label`와 `label2id`를 사용하여 예상되는 id와 레이블의 맵을 생성하세요:

```py
>>> id2label = {0: "NEGATIVE", 1: "POSITIVE"}
>>> label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

<frameworkcontent>
<pt>
<Tip>

[`Trainer`]를 사용하여 모델을 파인 튜닝하는 방법에 익숙하지 않은 경우, [여기](../training#train-with-pytorch-trainer)의 기본 튜토리얼을 확인하세요!

</Tip>

이제 모델을 훈련시킬 준비가 되었습니다! [`AutoModelForSequenceClassification`]로 DistilBERT를 가쳐오고 예상되는 레이블 수와 레이블 매핑을 지정하세요:

```py
>>> from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

>>> model = AutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

이제 세 단계만 거치면 끝입니다:

1. [`TrainingArguments`]에서 하이퍼파라미터를 정의하세요. `output_dir`는 모델을 저장할 위치를 지정하는 유일한 파라미터입니다. 이 모델을 Hub에 업로드하기 위해 `push_to_hub=True`를 설정합니다. (모델을 업로드하기 위해 Hugging Face에 로그인해야합니다.) 각 에폭이 끝날 때마다, [`Trainer`]는 정확도를 평가하고 훈련 체크포인트를 저장합니다.
2. [`Trainer`]에 훈련 인수와 모델, 데이터셋, 토크나이저, 데이터 수집기 및 `compute_metrics` 함수를 전달하세요.
3. [`~Trainer.train`]를 호출하여 모델은 파인 튜닝하세요.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_imdb["train"],
...     eval_dataset=tokenized_imdb["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

<Tip>

[`Trainer`]는 `tokenizer`를 전달하면 기본적으로 동적 매핑을 적용합니다. 이 경우, 명시적으로 데이터 수집기를 지정할 필요가 없습니다.

</Tip>

훈련이 완료되면, [`~transformers.Trainer.push_to_hub`] 메소드를 사용하여 모델을 Hub에 공유할 수 있습니다.

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

Keras를 사용하여 모델을 파인 튜닝하는 방법에 익숙하지 않은 경우, [여기](../training#train-a-tensorflow-model-with-keras)의 기본 튜토리얼을 확인하세요!

</Tip>
TensorFlow에서 모델을 파인 튜닝하려면, 먼저 옵티마이저 함수와 학습률 스케쥴, 그리고 일부 훈련 하이퍼파라미터를 설정해야 합니다:

```py
>>> from transformers import create_optimizer
>>> import tensorflow as tf

>>> batch_size = 16
>>> num_epochs = 5
>>> batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
>>> total_train_steps = int(batches_per_epoch * num_epochs)
>>> optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

그런 다음 [`TFAutoModelForSequenceClassification`]을 사용하여 DistilBERT를 로드하고, 예상되는 레이블 수와 레이블 매핑을 로드할 수 있습니다:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]을 사용하여 데이터셋을 `tf.data.Dataset` 형식으로 변환합니다:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_imdb["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_imdb["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)를 사용하여 훈련할 모델을 구성합니다:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

훈련을 시작하기 전에 설정해야할 마지막 두 가지는 예측에서 정확도를 계산하고, 모델을 Hub에 업로드할 방법을 제공하는 것입니다. 모두 [Keras callbacks](../main_classes/keras_callbacks)를 사용하여 수행됩니다.

[`~transformers.KerasMetricCallback`]에 `compute_metrics`를 전달하여 정확도를 높입니다.

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

[`~transformers.PushToHubCallback`]에서 모델과 토크나이저를 업로드할 위치를 지정합니다:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

그런 다음 콜백을 함께 묶습니다:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

드디어, 모델 훈련을 시작할 준비가 되었습니다! [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)에 훈련 데이터셋, 검증 데이터셋, 에폭의 수 및 콜백을 전달하여 파인 튜닝합니다:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)
```

훈련이 완료되면, 모델이 자동으로 Hub에 업로드되어 모든 사람이 사용할 수 있습니다!
</tf>
</frameworkcontent>

<Tip>

텍스트 분류를 위한 모델을 파인 튜닝하는 자세한 예제는 다음 [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) 또는 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)를 참조하세요.

</Tip>

## 추론[[inference]]

좋아요, 이제 모델을 파인 튜닝했으니 추론에 사용할 수 있습니다!

추론을 수행하고자 하는 텍스트를 가져와봅시다:

```py
>>> text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
```

파인 튜닝된 모델로 추론을 시도하는 가장 간단한 방법은 [`pipeline`]를 사용하는 것입니다. 모델로 감정 분석을 위한 `pipeline`을 인스턴스화하고, 텍스트를 전달해보세요:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
>>> classifier(text)
[{'label': 'POSITIVE', 'score': 0.9994940757751465}]
```

원한다면, `pipeline`의 결과를 수동으로 복제할 수도 있습니다.

<frameworkcontent>
<pt>
텍스트를 토큰화하고 PyTorch 텐서를 반환합니다.

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

입력을 모델에 전달하고 `logits`을 반환합니다:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

가장 높은 확률을 가진 클래스를 모델의 `id2label` 매핑을 사용하여 텍스트 레이블로 변환합니다:

```py
>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</pt>
<tf>
텍스트를 토큰화하고 TensorFlow 텐서를 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="tf")
```

입력값을 모델에 전달하고 `logits`을 반환합니다:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> logits = model(**inputs).logits
```

가장 높은 확률을 가진 클래스를 모델의 `id2label` 매핑을 사용하여 텍스트 레이블로 변환합니다:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</tf>
</frameworkcontent>
