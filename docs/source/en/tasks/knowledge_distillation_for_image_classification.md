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
# Knowledge Distillation for Computer Vision

Knowledge distillation is a technique used to transfer knowledge from a larger, more complex model (teacher) to a smaller, simpler model (student). In the context of image classification, the goal is to train a student model which mimics the behavior of the teacher model. It was first introduced in [Distilling the Knowledge in a Neural Network by Hinton et al](https://arxiv.org/abs/1503.02531). In this notebook, we will do task-specific knowledge distillation. We will use the [beans dataset](https://huggingface.co/datasets/beans) for this.

This tutorial aims to demonstrate how to distill a [ResNet](https://huggingface.co/microsoft/resnet-50) (teacher model) to a [MobileNet](https://huggingface.co/google/mobilenet_v2_1.4_224) (student model) using the `Trainer` API of 🤗 Transformers. 

Let's install the libraries needed for distillation and evaluating the process. 

```bash
pip install transformers datasets accelerate tensorboard evaluate --upgrade
```

In this example, we are using the `microsoft/resnet-50` model, trained on the [ImageNet-1k dataset](https://huggingface.co/datasets/imagenet-1k) with a resolution of 224x224.

```python
from PIL import Image
import requests
import numpy as np

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
sample = Image.open(requests.get(url, stream=True).raw)
```

We will now load and pre-process the dataset. 

```python
from datasets import load_dataset

dataset = load_dataset("beans")
```

We will use the `map()` method of `dataset` to apply the preprocessing to every split of the dataset.

```python
from transformers import AutoImageProcessor
teacher_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

def process(examples):
    processed_inputs = teacher_processor(examples["image"])
    return processed_inputs

processed_datasets = dataset.map(process, batched=True)
```

Essentially, we want the student model (a randomly initialized MobileNet) to mimic the teacher model (pre-trained ResNet). To achieve this, we first get the logits output by the teacher and the student. Then, we divide each of them by the parameter `temperature`, which controls the importance of each soft target. We will use the KL loss to compute the divergence between the student and teacher. A parameter called `lambda` weighs the importance of the distillation loss. In this example, we will use `temperature=5` and `lambda=0.5`.


```python
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDistilTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
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

We will now login to Hugging Face Hub so we can push our model to the Hugging Face Hub through the `Trainer`. 

```python
from huggingface_hub import notebook_login

notebook_login()
```

Let's initialize the data collator and `Trainer`. We will use `DefaultDataCollator` for this. 

```python
from transformers import AutoModelForImageClassification, DefaultDataCollator, MobileNetV2Config, MobileNetV2ForImageClassification

training_args = TrainingArguments(
    output_dir="my-awesome-model",
    num_train_epochs=30,
    fp16=True,
    logging_dir=f"{repo_name}/logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name,
    )

num_labels = len(processed_datasets["train"].features["labels"].names)

# initialize models
teacher_model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# training MobileNetV2 from scratch
student_config = MobileNetV2Config()
student_config.num_labels = num_labels
student_model = MobileNetV2ForImageClassification(student_config)
```

We can use `compute_metrics` function to evaluate our model on the test set. This function will be used during the training process to compute the `accuracy` & `f1` of our model.

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}
```

Let's initialize the `Trainer` with the training arguments we defined. 

```python
data_collator = DefaultDataCollator()
trainer = ImageDistilTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    training_args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=data_collator,
    tokenizer=teacher_extractor,
    compute_metrics=compute_metrics,
    temperature=5,
    lambda_param=0.5
)
```

We can now train our model.

```python
trainer.train()
```

We can evaluate the model on the test set.

```python
trainer.evaluate(processed_datasets["test"])
```

On test set, our model reaches 65 percent accuracy. To have a sanity check over efficiency of distillation, we also trained MobileNet on the beans dataset from scratch with the same hyperparameters and observed 63 percent accuracy on the test set. We invite the readers to try different pre-trained teacher models, student architectures, distillation parameters and report their findings. The training logs can be found in [this repository](https://huggingface.co/merve/resnet-mobilenet-beans).
