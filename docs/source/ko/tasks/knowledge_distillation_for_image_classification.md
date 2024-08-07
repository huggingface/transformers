<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# 컴퓨터 비전을 위한 지식 증류[[Knowledge-Distillation-for-Computer-Vision]]

[[open-in-colab]]

지식 증류(Knowledge distillation)는 더 크고 복잡한 모델(교사)에서 더 작고 간단한 모델(학생)로 지식을 전달하는 기술입니다. 한 모델에서 다른 모델로 지식을 증류하기 위해, 특정 작업(이 경우 이미지 분류)에 대해 학습된 사전 훈련된 교사 모델을 사용하고, 랜덤으로 초기화된 학생 모델을 이미지 분류 작업에 대해 학습합니다. 그다음, 학생 모델이 교사 모델의 출력을 모방하여 두 모델의 출력 차이를 최소화하도록 훈련합니다. 이 기법은 Hinton 등 연구진의 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)에서 처음 소개되었습니다. 이 가이드에서는 특정 작업에 맞춘 지식 증류를 수행할 것입니다. 이번에는 [beans dataset](https://huggingface.co/datasets/beans)을 사용할 것입니다.

이 가이드는 [미세 조정된 ViT 모델](https://huggingface.co/merve/vit-mobilenet-beans-224) (교사 모델)을 [MobileNet](https://huggingface.co/google/mobilenet_v2_1.4_224) (학생 모델)으로 증류하는 방법을 🤗 Transformers의 [Trainer API](https://huggingface.co/docs/transformers/en/main_classes/trainer#trainer) 를 사용하여 보여줍니다.

증류와 과정 평가를 위해 필요한 라이브러리를 설치해 봅시다.


```bash
pip install transformers datasets accelerate tensorboard evaluate --upgrade
```

이 예제에서는 `merve/beans-vit-224` 모델을 교사 모델로 사용하고 있습니다. 이 모델은 beans 데이터셋에서 파인 튜닝된 `google/vit-base-patch16-224-in21k` 기반의 이미지 분류 모델입니다. 이 모델을 무작위로 초기화된 MobileNetV2로 증류해볼 것입니다.

이제 데이터셋을 로드하겠습니다.

```python
from datasets import load_dataset

dataset = load_dataset("beans")
```

이 경우 두 모델의 이미지 프로세서가 동일한 해상도로 동일한 출력을 반환하기 때문에, 두가지를 모두 사용할 수 있습니다. 데이터셋의 모든 분할마다 전처리를 적용하기 위해 `dataset`의 `map()` 메소드를 사용할 것 입니다.


```python
from transformers import AutoImageProcessor
teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

def process(examples):
    processed_inputs = teacher_processor(examples["image"])
    return processed_inputs

processed_datasets = dataset.map(process, batched=True)
```

학생 모델(무작위로 초기화된 MobileNet)이 교사 모델(파인 튜닝된 비전 트랜스포머)을 모방하도록 할 것 입니다. 이를 위해 먼저 교사와 학생 모델의 로짓 출력값을 구합니다. 그런 다음 각 출력값을 매개변수 `temperature` 값으로 나누는데, 이 매개변수는 각 소프트 타겟의 중요도를 조절하는 역할을 합니다. 매개변수 `lambda` 는 증류 손실의 중요도에 가중치를 줍니다. 이 예제에서는 `temperature=5`와 `lambda=0.5`를 사용할 것입니다. 학생과 교사 간의 발산을 계산하기 위해 Kullback-Leibler Divergence 손실을 사용합니다. 두 데이터 P와 Q가 주어졌을 때, KL Divergence는 Q를 사용하여 P를 표현하는 데 얼만큼의 추가 정보가 필요한지를 말해줍니다. 두 데이터가 동일하다면, KL Divergence는 0이며, Q로 P를 설명하는 데 추가 정보가 필요하지 않음을 의미합니다. 따라서 지식 증류의 맥락에서 KL Divergence는 유용합니다.


```python
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss
```

이제 Hugging Face Hub에 로그인하여 `Trainer`를 통해 Hugging Face Hub에 모델을 푸시할 수 있도록 하겠습니다.


```python
from huggingface_hub import notebook_login

notebook_login()
```

이제 `TrainingArguments`, 교사 모델과 학생 모델을 설정하겠습니다.


```python
from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification

training_args = TrainingArguments(
    output_dir="my-awesome-model",
    num_train_epochs=30,
    fp16=True,
    logging_dir=f"{repo_name}/logs",
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name,
    )

num_labels = len(processed_datasets["train"].features["labels"].names)

# 모델 초기화
teacher_model = AutoModelForImageClassification.from_pretrained(
    "merve/beans-vit-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# MobileNetV2 밑바닥부터 학습
student_config = MobileNetV2Config()
student_config.num_labels = num_labels
student_model = MobileNetV2ForImageClassification(student_config)
```

`compute_metrics` 함수를 사용하여 테스트 세트에서 모델을 평가할 수 있습니다. 이 함수는 훈련 과정에서 모델의 `accuracy`와 `f1`을 계산하는 데 사용됩니다.


```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}
```

정의한 훈련 인수로 `Trainer`를 초기화해봅시다. 또한 데이터 콜레이터(data collator)를 초기화하겠습니다.

```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
trainer = ImageDistilTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    training_args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=data_collator,
    tokenizer=teacher_processor,
    compute_metrics=compute_metrics,
    temperature=5,
    lambda_param=0.5
)
```

이제 모델을 훈련할 수 있습니다.

```python
trainer.train()
```

모델을 테스트 세트에서 평가할 수 있습니다.

```python
trainer.evaluate(processed_datasets["test"])
```


테스트 세트에서 모델의 정확도는 72%에 도달했습니다. 증류의 효율성을 검증하기 위해 동일한 하이퍼파라미터로 beans 데이터셋에서 MobileNet을 처음부터 훈련하였고, 테스트 세트에서의 정확도는 63% 였습니다. 다양한 사전 훈련된 교사 모델, 학생 구조, 증류 매개변수를 시도해보시고 결과를 보고하기를 권장합니다. 증류된 모델의 훈련 로그와 체크포인트는 [이 저장소](https://huggingface.co/merve/vit-mobilenet-beans-224)에서 찾을 수 있으며, 처음부터 훈련된 MobileNetV2는 이 [저장소](https://huggingface.co/merve/resnet-mobilenet-beans-5)에서 찾을 수 있습니다.
