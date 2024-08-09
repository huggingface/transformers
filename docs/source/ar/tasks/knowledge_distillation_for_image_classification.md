# استخلاص المعرفة للرؤية الحاسوبية 

[[open-in-colab]]

استخلاص المعرفة هي تقنية تُستخدم لنقل المعرفة من نموذج أكبر وأكثر تعقيدًا (المعلم) إلى نموذج أصغر وأبسط (الطالب). ولاستخلاص المعرفة من نموذج إلى آخر، نستخدم نموذج المعلم مُدرب مسبقًا على مهمة معينة (تصنيف الصور في هذه الحالة) ونقوم بتطبيق نموذج الطالب الذي تم تهيئته بشكل عشوائي ليتم تدريبه على تصنيف الصور. بعد ذلك، نقوم بتدريب نموذج الطالب لتقليل الفرق بين مخرجاته ومخرجات المعلم، مما يجعله يقلد سلوكه. تم تقديم هذه التقنية لأول مرة في ورقة "استخلاص المعرفة من الشبكة العصبية" من قبل هينتون وآخرون. في هذا الدليل، سنقوم باستخلاص المعرفة الموجهة لمهمة محددة. وسنستخدم مجموعة بيانات beans للقيام بذلك.

يُظهر هذا الدليل كيفية استخلاص نموذج ViT (نموذج المعلم) الذي تم ضبطه بشكل دقيق إلى MobileNet (نموذج الطالب) باستخدام واجهة برمجة تطبيقات Trainer من مكتبة 🤗 Transformers.

دعونا نقوم بتثبيت المكتبات اللازمة لعملية الاستخلاص وتقييم العملية.

```bash
pip install transformers datasets accelerate tensorboard evaluate --upgrade
```

في هذا المثال، نستخدم نموذج `merve/beans-vit-224` كنموذج معلم. إنه نموذج تصنيف صور، يعتمد على نموذج `google/vit-base-patch16-224-in21k` الذي تم ضبطه بدقة على مجموعة بيانات beans. سنقوم باستخلاص هذا النموذج إلى MobileNetV2 الذي تم تهيئته بشكل عشوائي.

سنقوم الآن بتحميل مجموعة البيانات.

```python
from datasets import load_dataset

dataset = load_dataset("beans")
```

يمكننا استخدام معالج الصور من أي من النماذج، حيث أن كليهما يعطي نفس المخرجات بنفس الدقة. سنستخدم طريقة `map()` من مجموعة البيانات لتطبيق المعالجة المسبقة على كل جزء من مجموعة البيانات.

```python
from transformers import AutoImageProcessor
teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

def process(examples):
    processed_inputs = teacher_processor(examples["image"])
    return processed_inputs

processed_datasets = dataset.map(process, batched=True)
```

بشكل أساسي، نريد من نموذج الطالب (MobileNet الذي تم تهيئته بشكل عشوائي) أن يقلد نموذج المعلم (نموذج محول الرؤية الذي تم ضبطه بدقة). ولتحقيق ذلك، نقوم أولاً بالحصول على مخرجات logits من المعلم والطالب. بعد ذلك، نقوم بقسمة كل منهما على المعامل `temperature` الذي يتحكم في أهمية كل هدف ناعم. وهناك معامل يسمى `lambda` يزن أهمية خسارة الاستخلاص. في هذا المثال، سنستخدم `temperature=5` و `lambda=0.5`. سنستخدم خسارة Kullback-Leibler Divergence لحساب الانحراف بين الطالب والمعلم. بالنظر إلى البيانات P و Q، يوضح KL Divergence مقدار المعلومات الإضافية التي نحتاجها لتمثيل P باستخدام Q. إذا كان الاثنان متطابقين، فإن انحرافهما KL يكون صفرًا، حيث لا توجد معلومات أخرى مطلوبة لشرح P من Q. وبالتالي، في سياق استخلاص المعرفة، يكون KL divergence مفيدًا.

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

سنقوم الآن بتسجيل الدخول إلى Hugging Face Hub حتى نتمكن من دفع نموذجنا إلى Hugging Face Hub من خلال `Trainer`.

```python
from huggingface_hub import notebook_login

notebook_login()
```

دعونا نقوم بضبط `TrainingArguments`، ونموذج المعلم ونموذج الطالب.

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

يمكننا استخدام دالة `compute_metrics` لتقييم نموذجنا على مجموعة الاختبار. ستُستخدم هذه الدالة أثناء عملية التدريب لحساب الدقة (accuracy) و f1 لنموذجنا.

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}
```

دعونا نقوم بتهيئة `Trainer` باستخدام وسائط التدريب التي حددناها. سنقوم أيضًا بتهيئة مجمع البيانات الخاص بنا.

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

الآن يمكننا تدريب نموذجنا.

```python
trainer.train()
```

يمكننا تقييم النموذج على مجموعة الاختبار.

```python
trainer.evaluate(processed_datasets["test"])
```

يصل نموذجنا إلى دقة 72% على مجموعة الاختبار. ولإجراء فحص للتحقق من كفاءة عملية الاستخلاص، قمنا أيضًا بتدريب MobileNet على مجموعة بيانات beans من الصفر باستخدام نفس المعلمات ووجدنا أن الدقة على مجموعة الاختبار بلغت 63%. ندعو القراء إلى تجربة نماذج المعلم المختلفة، وهندسات الطالب، ومعلمات الاستخلاص، والإبلاغ عن النتائج التي توصلوا إليها. يمكن العثور على سجلات التدريب ونقاط التفتيش للنموذج المستخلص في [هذا المستودع](https://huggingface.co/merve/vit-mobilenet-beans-224)، ويمكن العثور على MobileNetV2 الذي تم تدريبه من الصفر في هذا [المستودع](https://huggingface.co/merve/resnet-mobilenet-beans-5).