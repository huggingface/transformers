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

먼저 [Yelp 리뷰](https://huggingface.co/datasets/Yelp/yelp_review_full) 데이터 세트를 로드합니다:

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

<Youtube id="nvBXf7s7vTI"/>

## 파이토치 Trainer로 훈련하기[[train-with-pytorch-trainer]]

🤗 Transformers는 🤗 Transformers 모델 훈련에 최적화된 [`Trainer`] 클래스를 제공하여 훈련 루프를 직접 작성하지 않고도 쉽게 훈련을 시작할 수 있습니다. [`Trainer`] API는 로깅(logging), 경사 누적(gradient accumulation), 혼합 정밀도(mixed precision) 등 다양한 훈련 옵션과 기능을 지원합니다.

먼저 모델을 가져오고 예상되는 레이블 수를 지정합니다. Yelp 리뷰 [데이터셋 카드](https://huggingface.co/datasets/Yelp/yelp_review_full#data-fields)에서 5개의 레이블이 있음을 알 수 있습니다:

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

미세 튜닝 중에 평가 지표를 모니터링하려면 훈련 인수에 `eval_strategy` 파라미터를 지정하여 각 에폭이 끝날 때 평가 지표를 확인할 수 있습니다:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
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

<a id='pytorch_native'></a>

## 기본 파이토치로 훈련하기[[train-in-native-pytorch]]

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

<a id='additional-resources'></a>

## 추가 자료[[additional-resources]]

더 많은 미세 튜닝 예제는 다음을 참조하세요:

- [🤗 Trnasformers 예제](https://github.com/huggingface/transformers/tree/main/examples)에는 PyTorch 및 텐서플로우에서 일반적인 NLP 작업을 훈련할 수 있는 스크립트가 포함되어 있습니다.

- [🤗 Transformers 노트북](notebooks)에는 PyTorch 및 텐서플로우에서 특정 작업을 위해 모델을 미세 튜닝하는 방법에 대한 다양한 노트북이 포함되어 있습니다.
