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
# تقطير المعرفة للرؤية الحاسوبية

[[open-in-colab]]

تقطير المعرفة هو تقنية لنقل المعرفة من نموذج كبير ومعقد (المُعلِّم) إلى نموذج أصغر وأبسط (الطالب). لعمل التقطير، نأخذ نموذج مُعلِّم مُدرَّب مسبقًا على مهمة معينة (تصنيف الصور في هذا المثال) ونهيّئ عشوائيًا نموذج الطالب لتدريبه على نفس المهمة. ثم ندرِّب الطالب لتقليل الفارق بين مخرجاته ومخرجات المُعلِّم، بحيث يُحاكي سلوكه. طُرحت هذه الفكرة لأول مرة في [Distilling the Knowledge in a Neural Network by Hinton et al](https://huggingface.co/papers/1503.02531). في هذا الدليل سنقوم بتقطير معرفة مخصص للمهمة. سنستخدم [مجموعة بيانات الفاصوليا](https://huggingface.co/datasets/beans).

يوضح هذا الدليل كيفية تقطير [نموذج ViT مُحسَّن](https://huggingface.co/merve/vit-mobilenet-beans-224) (مُعلِّم) إلى [MobileNet](https://huggingface.co/google/mobilenet_v2_1.4_224) (طالب) باستخدام [Trainer API](https://huggingface.co/docs/transformers/en/main_classes/trainer#trainer) من 🤗 Transformers.

لنثبّت المكتبات اللازمة للتقطير وتقييم العملية.

```bash
pip install transformers datasets accelerate tensorboard evaluate --upgrade
```

في هذا المثال، نستخدم النموذج `merve/beans-vit-224` كنموذج مُعلِّم. إنه نموذج لتصنيف الصور مبني على `google/vit-base-patch16-224-in21k` ومُحسّن على مجموعة بيانات الفاصوليا. سنقطّر هذا النموذج إلى MobileNetV2 مُهيّأ من الصفر.

الآن سنحمّل مجموعة البيانات.

```python
from datasets import load_dataset

dataset = load_dataset("beans")
```

يمكننا استخدام معالج الصور (image processor) لأحد النموذجين، إذ أنهما يعيدان نفس المخرجات مع نفس الدقة. سنستخدم الدالة `map()` من `dataset` لتطبيق المعالجة المسبقة على جميع الأجزاء.

```python
from transformers import AutoImageProcessor
teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

def process(examples):
    processed_inputs = teacher_processor(examples["image"])
    return processed_inputs

processed_datasets = dataset.map(process, batched=True)
```

بالمحصلة، نريد من نموذج الطالب (MobileNet مُهيّأ عشوائيًا) أن يُحاكي نموذج المُعلِّم (Vision Transformer مُحسَّن). لتحقيق ذلك، نحصل أولًا على لوجِتس (logits) المُعلِّم والطالب، ثم نقسم كلًا منهما على معامل `temperature` الذي يتحكم بأهمية الأهداف اللينة. كما نستخدم معاملًا يُدعى `lambda` لوزن أهمية خسارة التقطير. في هذا المثال سنستخدم `temperature=5` و`lambda=0.5`. سنستخدم خسارة تباعد كولباك-لايبلر (KL Divergence) لقياس التباعد بين الطالب والمُعلِّم. إذا كان التوزيعان متطابقين فسيكون التباعد صفرًا. لذا فهو مناسب لتقطير المعرفة.

```python
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.test_utils.testing import get_backend

class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
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

سنقوم الآن بتسجيل الدخول إلى Hugging Face Hub كي نتمكن من دفع النموذج عبر `Trainer`.

```python
from huggingface_hub import notebook_login

notebook_login()
```

لنُعدّ `TrainingArguments` ونحدد نموذج المُعلِّم ونموذج الطالب.

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

# initialize models
teacher_model = AutoModelForImageClassification.from_pretrained(
    "merve/beans-vit-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# training MobileNetV2 from scratch
student_config = MobileNetV2Config()
student_config.num_labels = num_labels
student_model = MobileNetV2ForImageClassification(student_config)
```

يمكننا استخدام الدالة `compute_metrics` لتقييم النموذج على مجموعة الاختبار. ستحسب هذه الدالة `accuracy` أثناء التدريب.

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}
```

لنُهيّئ `Trainer` باستخدام إعدادات التدريب التي عرفناها، وكذلك المُجمِّع (data collator).

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
    processing_class=teacher_processor,
    compute_metrics=compute_metrics,
    temperature=5,
    lambda_param=0.5
)
```

يمكننا الآن تدريب النموذج.

```python
trainer.train()
```

ويمكننا تقييمه على مجموعة الاختبار.

```python
trainer.evaluate(processed_datasets["test"])
```

على مجموعة الاختبار حقق نموذجنا دقة تبلغ ٪72. وللتحقق من فاعلية التقطير، قمنا أيضًا بتدريب MobileNet من الصفر على نفس البيانات وبنفس الإعدادات ولاحظنا دقة قدرها ٪63. ندعو القُرّاء لتجربة نماذج مُعلِّم مختلفة وبُنى طلاب متنوعة ومعاملات تقطير مختلفة ومشاركة النتائج. سجلات التدريب ونقاط الحفظ للنموذج المقطَّر متاحة في [هذا المستودع](https://huggingface.co/merve/vit-mobilenet-beans-224)، وMobileNetV2 المدرَّب من الصفر في [هذا المستودع](https://huggingface.co/merve/resnet-mobilenet-beans-5).
