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

# 토큰 분류[[token-classification]]

[[open-in-colab]]

<Youtube id="wVHdVlPScxA"/>

토큰 분류는 문장의 개별 토큰에 레이블을 할당합니다. 가장 일반적인 토큰 분류 작업 중 하나는 개체명 인식(Named Entity Recognition, NER)입니다. 개체명 인식은 문장에서 사람, 위치 또는 조직과 같은 각 개체의 레이블을 찾으려고 시도합니다.

이 가이드에서 학습할 내용은:

1. [WNUT 17](https://huggingface.co/datasets/wnut_17) 데이터 세트에서 [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)를 파인 튜닝하여 새로운 개체를 탐지합니다.
2. 추론을 위해 파인 튜닝 모델을 사용합니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/token-classification)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에, 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate seqeval
```

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티에 공유하는 것을 권장합니다. 메시지가 표시되면, 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## WNUT 17 데이터 세트 가져오기[[load-wnut-17-dataset]]

먼저 🤗 Datasets 라이브러리에서 WNUT 17 데이터 세트를 가져옵니다:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

다음 예제를 살펴보세요:

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

`ner_tags`의 각 숫자는 개체를 나타냅니다. 숫자를 레이블 이름으로 변환하여 개체가 무엇인지 확인합니다:

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

각 `ner_tag`의 앞에 붙은 문자는 개체의 토큰 위치를 나타냅니다:

- `B-`는 개체의 시작을 나타냅니다.
- `I-`는 토큰이 동일한 개체 내부에 포함되어 있음을 나타냅니다(예를 들어 `State` 토큰은 `Empire State Building`와 같은 개체의 일부입니다).
- `0`는 토큰이 어떤 개체에도 해당하지 않음을 나타냅니다.

## 전처리[[preprocess]]

<Youtube id="iY2AZYdZAr0"/>

다음으로 `tokens` 필드를 전처리하기 위해 DistilBERT 토크나이저를 가져옵니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

위의 예제 `tokens` 필드를 보면 입력이 이미 토큰화된 것처럼 보입니다. 그러나 실제로 입력은 아직 토큰화되지 않았으므로 단어를 하위 단어로 토큰화하기 위해 `is_split_into_words=True`를 설정해야 합니다. 예제로 확인합니다:

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

그러나 이로 인해 `[CLS]`과 `[SEP]`라는 특수 토큰이 추가되고, 하위 단어 토큰화로 인해 입력과 레이블 간에 불일치가 발생합니다. 하나의 레이블에 해당하는 단일 단어는 이제 두 개의 하위 단어로 분할될 수 있습니다. 토큰과 레이블을 다음과 같이 재정렬해야 합니다:

1. [`word_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.word_ids) 메소드로 모든 토큰을 해당 단어에 매핑합니다.
2. 특수 토큰 `[CLS]`와 `[SEP]`에 `-100` 레이블을 할당하여, PyTorch 손실 함수가 해당 토큰을 무시하도록 합니다.
3. 주어진 단어의 첫 번째 토큰에만 레이블을 지정합니다. 같은 단어의 다른 하위 토큰에 `-100`을 할당합니다.

다음은 토큰과 레이블을 재정렬하고 DistilBERT의 최대 입력 길이보다 길지 않도록 시퀀스를 잘라내는 함수를 만드는 방법입니다:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # Set the special tokens to -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

전체 데이터 세트에 전처리 함수를 적용하려면, 🤗 Datasets [`~datasets.Dataset.map`] 함수를 사용하세요. `batched=True`로 설정하여 데이터 세트의 여러 요소를 한 번에 처리하면 `map` 함수의 속도를 높일 수 있습니다:
```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

이제 [`DataCollatorWithPadding`]를 사용하여 예제 배치를 만들어봅시다. 데이터 세트 전체를 최대 길이로 패딩하는 대신, *동적 패딩*을 사용하여 배치에서 가장 긴 길이에 맞게 문장을 패딩하는 것이 효율적입니다.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## 평가[[evaluation]]

훈련 중 모델의 성능을 평가하기 위해 평가 지표를 포함하는 것이 유용합니다. 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하여 빠르게 평가 방법을 가져올 수 있습니다. 이 작업에서는 [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) 평가 지표를 가져옵니다. (평가 지표를 가져오고 계산하는 방법에 대해서는 🤗 Evaluate [빠른 둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하세요). Seqeval은 실제로 정밀도, 재현률, F1 및 정확도와 같은 여러 점수를 산출합니다.

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

먼저 NER 레이블을 가져온 다음, [`~evaluate.EvaluationModule.compute`]에 실제 예측과 실제 레이블을 전달하여 점수를 계산하는 함수를 만듭니다:

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

이제 `compute_metrics` 함수를 사용할 준비가 되었으며, 훈련을 설정하면 이 함수로 되돌아올 것입니다.

## 훈련[[train]]

모델을 훈련하기 전에, `id2label`와 `label2id`를 사용하여 예상되는 id와 레이블의 맵을 생성하세요:

```py
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<frameworkcontent>
<pt>
<Tip>

[`Trainer`]를 사용하여 모델을 파인 튜닝하는 방법에 익숙하지 않은 경우, [여기](../training#train-with-pytorch-trainer)에서 기본 튜토리얼을 확인하세요!

</Tip>

이제 모델을 훈련시킬 준비가 되었습니다! [`AutoModelForSequenceClassification`]로 DistilBERT를 가져오고 예상되는 레이블 수와 레이블 매핑을 지정하세요:

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

이제 세 단계만 거치면 끝입니다:

1. [`TrainingArguments`]에서 하이퍼파라미터를 정의하세요. `output_dir`는 모델을 저장할 위치를 지정하는 유일한 매개변수입니다. 이 모델을 허브에 업로드하기 위해 `push_to_hub=True`를 설정합니다(모델을 업로드하기 위해 Hugging Face에 로그인해야합니다.) 각 에폭이 끝날 때마다, [`Trainer`]는 seqeval 점수를 평가하고 훈련 체크포인트를 저장합니다.
2. [`Trainer`]에 훈련 인수와 모델, 데이터 세트, 토크나이저, 데이터 콜레이터 및 `compute_metrics` 함수를 전달하세요.
3. [`~Trainer.train`]를 호출하여 모델을 파인 튜닝하세요.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
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
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

훈련이 완료되면, [`~transformers.Trainer.push_to_hub`] 메소드를 사용하여 모델을 허브에 공유할 수 있습니다.

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

>>> batch_size = 16
>>> num_train_epochs = 3
>>> num_train_steps = (len(tokenized_wnut["train"]) // batch_size) * num_train_epochs
>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=2e-5,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=0.01,
...     num_warmup_steps=0,
... )
```

그런 다음 [`TFAutoModelForSequenceClassification`]을 사용하여 DistilBERT를 가져오고, 예상되는 레이블 수와 레이블 매핑을 지정합니다:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]을 사용하여 데이터 세트를 `tf.data.Dataset` 형식으로 변환합니다:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_wnut["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_wnut["validation"],
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

훈련을 시작하기 전에 설정해야할 마지막 두 가지는 예측에서 seqeval 점수를 계산하고, 모델을 허브에 업로드할 방법을 제공하는 것입니다. 모두 [Keras callbacks](../main_classes/keras_callbacks)를 사용하여 수행됩니다.

[`~transformers.KerasMetricCallback`]에 `compute_metrics` 함수를 전달하세요:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

[`~transformers.PushToHubCallback`]에서 모델과 토크나이저를 업로드할 위치를 지정합니다:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_wnut_model",
...     tokenizer=tokenizer,
... )
```

그런 다음 콜백을 함께 묶습니다:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

드디어, 모델 훈련을 시작할 준비가 되었습니다! [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)에 훈련 데이터 세트, 검증 데이터 세트, 에폭의 수 및 콜백을 전달하여 파인 튜닝합니다:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)
```

훈련이 완료되면, 모델이 자동으로 허브에 업로드되어 누구나 사용할 수 있습니다!
</tf>
</frameworkcontent>

<Tip>

토큰 분류를 위한 모델을 파인 튜닝하는 자세한 예제는 다음
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
또는 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)를 참조하세요.

</Tip>

## 추론[[inference]]

좋아요, 이제 모델을 파인 튜닝했으니 추론에 사용할 수 있습니다!

추론을 수행하고자 하는 텍스트를 가져와봅시다:

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

파인 튜닝된 모델로 추론을 시도하는 가장 간단한 방법은 [`pipeline`]를 사용하는 것입니다. 모델로 NER의 `pipeline`을 인스턴스화하고, 텍스트를 전달해보세요:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location',
  'score': 0.42658573,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856336,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.3064001,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.65523505,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668663,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```

원한다면, `pipeline`의 결과를 수동으로 복제할 수도 있습니다:

<frameworkcontent>
<pt>
텍스트를 토큰화하고 PyTorch 텐서를 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

입력을 모델에 전달하고 `logits`을 반환합니다:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

가장 높은 확률을 가진 클래스를 모델의 `id2label` 매핑을 사용하여 텍스트 레이블로 변환합니다:

```py
>>> predictions = torch.argmax(logits, dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
</pt>
<tf>
텍스트를 토큰화하고 TensorFlow 텐서를 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="tf")
```

입력값을 모델에 전달하고 `logits`을 반환합니다:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> logits = model(**inputs).logits
```

가장 높은 확률을 가진 클래스를 모델의 `id2label` 매핑을 사용하여 텍스트 레이블로 변환합니다:

```py
>>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)
>>> predicted_token_class = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
</tf>
</frameworkcontent>
