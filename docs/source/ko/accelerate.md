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

# 🤗 Accelerate를 활용한 분산 학습[[distributed-training-with-accelerate]]

모델이 커지면서 병렬 처리는 제한된 하드웨어에서 더 큰 모델을 훈련하고 훈련 속도를 몇 배로 가속화하기 위한 전략으로 등장했습니다. Hugging Face에서는 사용자가 하나의 머신에 여러 개의 GPU를 사용하든 여러 머신에 여러 개의 GPU를 사용하든 모든 유형의 분산 설정에서 🤗 Transformers 모델을 쉽게 훈련할 수 있도록 돕기 위해 [🤗 Accelerate](https://huggingface.co/docs/accelerate) 라이브러리를 만들었습니다. 이 튜토리얼에서는 분산 환경에서 훈련할 수 있도록 기본 PyTorch 훈련 루프를 커스터마이즈하는 방법을 알아봅시다.

## 설정[[setup]]

🤗 Accelerate 설치 시작하기:

```bash
pip install accelerate
```

그 다음, [`~accelerate.Accelerator`] 객체를 불러오고 생성합니다. [`~accelerate.Accelerator`]는 자동으로 분산 설정 유형을 감지하고 훈련에 필요한 모든 구성 요소를 초기화합니다. 장치에 모델을 명시적으로 배치할 필요는 없습니다.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## 가속화를 위한 준비[[prepare-to-accelerate]]

다음 단계는 관련된 모든 훈련 객체를 [`~accelerate.Accelerator.prepare`] 메소드에 전달하는 것입니다. 여기에는 훈련 및 평가 데이터로더, 모델 및 옵티마이저가 포함됩니다:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## 백워드(Backward)[[backward]]

마지막으로 훈련 루프의 일반적인 `loss.backward()`를 🤗 Accelerate의 [`~accelerate.Accelerator.backward`] 메소드로 대체하기만 하면 됩니다:

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

다음 코드에서 볼 수 있듯이, 훈련 루프에 코드 네 줄만 추가하면 분산 학습을 활성화할 수 있습니다!

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## 학습[[train]]

관련 코드를 추가한 후에는 스크립트나 Colaboratory와 같은 노트북에서 훈련을 시작하세요.

### 스크립트로 학습하기[[train-with-a-script]]

스크립트에서 훈련을 실행하는 경우, 다음 명령을 실행하여 구성 파일을 생성하고 저장합니다:

```bash
accelerate config
```

Then launch your training with:

```bash
accelerate launch train.py
```

### 노트북으로 학습하기[[train-with-a-notebook]]

Collaboratory의 TPU를 사용하려는 경우, 노트북에서도 🤗 Accelerate를 실행할 수 있습니다. 훈련을 담당하는 모든 코드를 함수로 감싸서 [`~accelerate.notebook_launcher`]에 전달하세요:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

🤗 Accelerate 및 다양한 기능에 대한 자세한 내용은 [documentation](https://huggingface.co/docs/accelerate)를 참조하세요.