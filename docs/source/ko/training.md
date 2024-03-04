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

# 사전 학습된 모델 미세 튜닝하기[[finetune-a-pretrained-model]]

[[open-in-colab]]

사전 학습된 모델을 사용하면 상당한 이점이 있습니다. 계산 비용과 탄소발자국을 줄이고, 처음부터 모델을 학습시킬 필요 없이 최신 모델을 사용할 수 있습니다. 🤗 Transformers는 다양한 작업을 위해 사전 학습된 수천 개의 모델에 액세스할 수 있습니다. 사전 학습된 모델을 사용하는 경우, 자신의 작업과 관련된 데이터셋을 사용해 학습합니다. 이것은 미세 튜닝이라고 하는 매우 강력한 훈련 기법입니다. 이 튜토리얼에서는 당신이 선택한 딥러닝 프레임워크로 사전 학습된 모델을 미세 튜닝합니다:

* 🤗 Transformers로 사전 학습된 모델 미세 튜닝하기 [`Trainer`].
* Keras를 사용하여 TensorFlow에서 사전 학습된 모델을 미세 튜닝하기.
* 기본 PyTorch에서 사전 학습된 모델을 미세 튜닝하기.

<a id='data-processing'></a>

## 데이터셋 준비[[prepare-a-dataset]]

<Youtube id="_BZearw7f0w"/>

사전 학습된 모델을 미세 튜닝하기 위해서 데이터셋을 다운로드하고 훈련할 수 있도록 준비하세요. 이전 튜토리얼에서 훈련을 위해 데이터를 처리하는 방법을 보여드렸는데, 지금이 배울 걸 되짚을 기회입니다!

먼저 [Yelp 리뷰](https://huggingface.co/datasets/yelp_review_full) 데이터 세트를 로드합니다:

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

텍스트를 처리하고 서로 다른 길이의 시퀀스 패딩 및 잘라내기 전략을 포함하려면 토크나이저가 필요합니다. 데이터셋을 한 번에 처리하려면 🤗 Dataset [`map`](https://huggingface.co/docs/datasets/process#map) 메서드를 사용하여 전체 데이터셋에 전처리 함수를 적용하세요:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

필요한 경우 미세 튜닝을 위해 데이터셋의 작은 부분 집합을 만들어 미세 튜닝 작업 시간을 줄일 수 있습니다:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Train

여기서부터는 사용하려는 프레임워크에 해당하는 섹션을 따라야 합니다. 오른쪽 사이드바의 링크를 사용하여 원하는 프레임워크로 이동할 수 있으며, 특정 프레임워크의 모든 콘텐츠를 숨기려면 해당 프레임워크 블록의 오른쪽 상단에 있는 버튼을 사용하면 됩니다!

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

## 파이토치 Trainer로 훈련하기[[train-with-pytorch-trainer]]

🤗 Transformers는 🤗 Transformers 모델 훈련에 최적화된 [`Trainer`] 클래스를 제공하여 훈련 루프를 직접 작성하지 않고도 쉽게 훈련을 시작할 수 있습니다. [`Trainer`] API는 로깅(logging), 경사 누적(gradient accumulation), 혼합 정밀도(mixed precision) 등 다양한 훈련 옵션과 기능을 지원합니다.

먼저 모델을 가져오고 예상되는 레이블 수를 지정합니다. Yelp 리뷰 [데이터셋 카드](https://huggingface.co/datasets/yelp_review_full#data-fields)에서 5개의 레이블이 있음을 알 수 있습니다:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

사전 훈련된 가중치 중 일부가 사용되지 않고 일부 가중치가 무작위로 표시된다는 경고가 표시됩니다.
걱정마세요. 이것은 올바른 동작입니다! 사전 학습된 BERT 모델의 헤드는 폐기되고 무작위로 초기화된 분류 헤드로 대체됩니다. 이제 사전 학습된 모델의 지식으로 시퀀스 분류 작업을 위한 새로운 모델 헤드를 미세 튜닝 합니다.

</Tip>

### 하이퍼파라미터 훈련[[training-hyperparameters]]

다음으로 정할 수 있는 모든 하이퍼파라미터와 다양한 훈련 옵션을 활성화하기 위한 플래그를 포함하는 [`TrainingArguments`] 클래스를 생성합니다.

이 튜토리얼에서는 기본 훈련 [하이퍼파라미터](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)로 시작하지만, 자유롭게 실험하여 여러분들에게 맞는 최적의 설정을 찾을 수 있습니다.

훈련에서 체크포인트(checkpoints)를 저장할 위치를 지정합니다:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### 평가 하기[[evaluate]]

[`Trainer`]는 훈련 중에 모델 성능을 자동으로 평가하지 않습니다. 평가 지표를 계산하고 보고할 함수를 [`Trainer`]에 전달해야 합니다. 
[🤗 Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리는 [`evaluate.load`](https://huggingface.co/spaces/evaluate-metric/accuracy) 함수로 로드할 수 있는 간단한 [`accuracy`]함수를 제공합니다 (자세한 내용은 [둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하세요):

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

`metric`에서 [`~evaluate.compute`]를 호출하여 예측의 정확도를 계산합니다. 예측을 `compute`에 전달하기 전에 예측을 로짓으로 변환해야 합니다(모든 🤗 Transformers 모델은 로짓으로 반환한다는 점을 기억하세요):

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

미세 튜닝 중에 평가 지표를 모니터링하려면 훈련 인수에 `evaluation_strategy` 파라미터를 지정하여 각 에폭이 끝날 때 평가 지표를 확인할 수 있습니다:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

### 훈련 하기[[trainer]]

모델, 훈련 인수, 훈련 및 테스트 데이터셋, 평가 함수가 포함된 [`Trainer`] 객체를 만듭니다:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

그리고 [`~transformers.Trainer.train`]을 호출하여 모델을 미세 튜닝합니다:

```py
>>> trainer.train()
```
</pt>
<tf>
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

## Keras로 텐서플로우 모델 훈련하기[[train-a-tensorflow-model-with-keras]]

Keras API를 사용하여 텐서플로우에서 🤗 Transformers 모델을 훈련할 수도 있습니다!

### Keras용 데이터 로드[[loading-data-for-keras]]

Keras API로 🤗 Transformers 모델을 학습시키려면 데이터셋을 Keras가 이해할 수 있는 형식으로 변환해야 합니다.
데이터 세트가 작은 경우, 전체를 NumPy 배열로 변환하여 Keras로 전달하면 됩니다.
더 복잡한 작업을 수행하기 전에 먼저 이 작업을 시도해 보겠습니다.

먼저 데이터 세트를 로드합니다. [GLUE 벤치마크](https://huggingface.co/datasets/glue)의 CoLA 데이터 세트를 사용하겠습니다.
간단한 바이너리 텍스트 분류 작업이므로 지금은 훈련 데이터 분할만 사용합니다.

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # Just take the training split for now
```

다음으로 토크나이저를 로드하고 데이터를 NumPy 배열로 토큰화합니다. 레이블은 이미 0과 1로 된 리스트이기 때문에 토큰화하지 않고 바로 NumPy 배열로 변환할 수 있습니다!

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1
```

마지막으로 모델을 로드, [`compile`](https://keras.io/api/models/model_training_apis/#compile-method), [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)합니다:

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))

model.fit(tokenized_data, labels)
```

<Tip>

모델을 `compile()`할 때 손실 인수를 모델에 전달할 필요가 없습니다! 
이 인수를 비워두면 허깅 페이스 모델은 작업과 모델 아키텍처에 적합한 손실을 자동으로 선택합니다. 
원한다면 언제든지 직접 손실을 지정하여 이를 재정의할 수 있습니다!

</Tip>

이 접근 방식은 소규모 데이터 집합에서는 잘 작동하지만, 대규모 데이터 집합에서는 문제가 될 수 있습니다. 왜 그럴까요?
토큰화된 배열과 레이블을 메모리에 완전히 로드하고 NumPy는 "들쭉날쭉한" 배열을 처리하지 않기 때문에,
모든 토큰화된 샘플을 전체 데이터셋에서 가장 긴 샘플의 길이만큼 패딩해야 합니다. 이렇게 하면 배열이 훨씬 더 커지고 이 패딩 토큰으로 인해 학습 속도도 느려집니다!

### 데이터를 tf.data.Dataset으로 로드하기[[loading-data-as-a-tfdatadataset]]

학습 속도가 느려지는 것을 피하려면 데이터를 `tf.data.Dataset`으로 로드할 수 있습니다. 원한다면 직접
`tf.data` 파이프라인을 직접 작성할 수도 있지만, 이 작업을 간편하게 수행하는 수 있는 두 가지 방법이 있습니다:

- [`~TFPreTrainedModel.prepare_tf_dataset`]: 대부분의 경우 이 방법을 권장합니다. 모델의 메서드이기 때문에 모델을 검사하여 모델 입력으로 사용할 수 있는 열을 자동으로 파악하고
나머지는 버려서 더 단순하고 성능이 좋은 데이터 집합을 만들 수 있습니다.
- [`~datasets.Dataset.to_tf_dataset`]: 이 방법은 좀 더 낮은 수준이며, 포함할 '열'과 '레이블'을 정확히 지정하여
데이터셋을 생성하는 방법을 정확히 제어하고 싶을 때 유용하며, 포함할 'columns'과 'label_cols'을 정확히 지정할 수 있습니다.

[`~TFPreTrainedModel.prepare_tf_dataset`]을 사용하려면 먼저 다음 코드 샘플과 같이 토크나이저 출력을 데이터 세트에 열로 추가해야 합니다:

```py
def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

허깅 페이스 데이터셋은 기본적으로 디스크에 저장되므로 메모리 사용량을 늘리지 않는다는 점을 기억하세요! 
열이 추가되면 데이터셋에서 배치를 스트리밍하고 각 배치에 패딩을 추가할 수 있으므로 전체 데이터셋에 패딩을 추가하는 것보다 패딩 토큰의 수를 크게 줄일 수 있습니다.


```py
>>> tf_dataset = model.prepare_tf_dataset(dataset, batch_size=16, shuffle=True, tokenizer=tokenizer)
```

위의 코드 샘플에서는 배치가 로드될 때 올바르게 패딩할 수 있도록 `prepare_tf_dataset`에 토크나이저를 전달해야 합니다.
데이터셋의 모든 샘플 길이가 같고 패딩이 필요하지 않은 경우 이 인수를 건너뛸 수 있습니다.
샘플을 채우는 것보다 더 복잡한 작업(예: 마스킹된 언어의 토큰 손상 모델링)을 수행하기 위해 토큰을 손상시켜야 하는 경우, 
`collate_fn` 인수를 사용하여 샘플 목록을 배치로 변환하고 원하는 전처리를 적용할 함수를 전달할 수 있습니다. 
[예시](https://github.com/huggingface/transformers/tree/main/examples) 또는 
[노트북](https://huggingface.co/docs/transformers/notebooks)을 참조하여 이 접근 방식이 실제로 작동하는 모습을 확인하세요.

`tf.data.Dataset`을 생성한 후에는 이전과 마찬가지로 모델을 컴파일하고 훈련(fit)할 수 있습니다:

```py
model.compile(optimizer=Adam(3e-5))

model.fit(tf_dataset)
```

</tf>
</frameworkcontent>

<a id='pytorch_native'></a>

## 기본 파이토치로 훈련하기[[train-in-native-pytorch]]

<frameworkcontent>
<pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`]는 훈련 루프를 처리하며 한 줄의 코드로 모델을 미세 조정할 수 있습니다. 직접 훈련 루프를 작성하는 것을 선호하는 사용자의 경우, 기본 PyTorch에서 🤗 Transformers 모델을 미세 조정할 수도 있습니다.

이 시점에서 노트북을 다시 시작하거나 다음 코드를 실행해 메모리를 확보해야 할 수 있습니다:

```py
del model
del trainer
torch.cuda.empty_cache()
```

다음으로, '토큰화된 데이터셋'을 수동으로 후처리하여 훈련련에 사용할 수 있도록 준비합니다.

1. 모델이 원시 텍스트를 입력으로 허용하지 않으므로 `text` 열을 제거합니다:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. 모델에서 인수의 이름이 `labels`로 지정될 것으로 예상하므로 `label` 열의 이름을 `labels`로 변경합니다:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. 데이터셋의 형식을 List 대신 PyTorch 텐서를 반환하도록 설정합니다:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

그리고 앞서 표시된 대로 데이터셋의 더 작은 하위 집합을 생성하여 미세 조정 속도를 높입니다:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader[[dataloader]]

훈련 및 테스트 데이터셋에 대한 'DataLoader'를 생성하여 데이터 배치를 반복할 수 있습니다:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

예측을 위한 레이블 개수를 사용하여 모델을 로드합니다:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### 옵티마이저 및 학습 속도 스케줄러[[optimizer-and-learning-rate-scheduler]]

옵티마이저와 학습 속도 스케줄러를 생성하여 모델을 미세 조정합니다. 파이토치에서 제공하는 [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 옵티마이저를 사용해 보겠습니다:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

[`Trainer`]에서 기본 학습 속도 스케줄러를 생성합니다:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

마지막으로, GPU에 액세스할 수 있는 경우 'device'를 지정하여 GPU를 사용하도록 합니다. 그렇지 않으면 CPU에서 훈련하며 몇 분이 아닌 몇 시간이 걸릴 수 있습니다.

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

[Colaboratory](https://colab.research.google.com/) 또는 [SageMaker StudioLab](https://studiolab.sagemaker.aws/)과 같은 호스팅 노트북이 없는 경우 클라우드 GPU에 무료로 액세스할 수 있습니다.

</Tip>

이제 훈련할 준비가 되었습니다! 🥳

### 훈련 루프[[training-loop]]

훈련 진행 상황을 추적하려면 [tqdm](https://tqdm.github.io/) 라이브러리를 사용하여 트레이닝 단계 수에 진행률 표시줄을 추가하세요:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### 평가 하기[[evaluate]]

[`Trainer`]에 평가 함수를 추가한 방법과 마찬가지로, 훈련 루프를 직접 작성할 때도 동일한 작업을 수행해야 합니다. 하지만 이번에는 각 에포크가 끝날 때마다 평가지표를 계산하여 보고하는 대신, [`~evaluate.add_batch`]를 사용하여 모든 배치를 누적하고 맨 마지막에 평가지표를 계산합니다.

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```
</pt>
</frameworkcontent>

<a id='additional-resources'></a>

## 추가 자료[[additional-resources]]

더 많은 미세 튜닝 예제는 다음을 참조하세요:

- [🤗 Trnasformers 예제](https://github.com/huggingface/transformers/tree/main/examples)에는 PyTorch 및 텐서플로우에서 일반적인 NLP 작업을 훈련할 수 있는 스크립트가 포함되어 있습니다.

- [🤗 Transformers 노트북](notebooks)에는 PyTorch 및 텐서플로우에서 특정 작업을 위해 모델을 미세 튜닝하는 방법에 대한 다양한 노트북이 포함되어 있습니다.
