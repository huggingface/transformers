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

# وصف الصور (Image Captioning)

[[open-in-colab]]

وصف الصور هو مهمة توليد وصفٍ نصي لصورةٍ معطاة. من التطبيقات الواقعية الشائعة لهذه المهمة مساعدة ذوي الإعاقة البصرية على التنقل في مواقف مختلفة. لذلك، يساهم وصف الصور في تحسين إتاحة المحتوى للأشخاص عبر وصف الصور لهم.

سيُرشدك هذا الدليل إلى:

- ضبط نموذج لوصف الصور على بياناتك.
- استخدام النموذج المضبوط للاستدلال.

قبل البدء، تأكد من تثبيت جميع المكتبات اللازمة:

```bash
pip install transformers datasets evaluate -q
pip install jiwer -q
```

نوصيك بتسجيل الدخول إلى حسابك على Hugging Face حتى تتمكن من رفع نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل رمزك لتسجيل الدخول:

```python
from huggingface_hub import notebook_login

notebook_login()
```

## تحميل مجموعة بيانات Pokémon BLIP للتسميات

استخدم مكتبة 🤗 Datasets لتحميل مجموعة بيانات تتكون من أزواج {صورة-تسمية}. لإنشاء مجموعة بياناتك الخاصة لوصف الصور باستخدام PyTorch، يمكنك اتباع [هذا الدفتر](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb).

```python
from datasets import load_dataset

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds
```
```bash
DatasetDict({
    train: Dataset({
        features: ['image', 'text'],
        num_rows: 833
    })
})
```

تحتوي المجموعة على ميزتين: `image` و`text`.

<Tip>

تتضمن العديد من مجموعات بيانات وصف الصور عدة تسميات لكل صورة. في هذه الحالات، يُعد الاختيار العشوائي لتسمية من بين المتاح أثناء التدريب استراتيجية شائعة.

</Tip>

قسّم جزء التدريب إلى جزأين للتدريب والاختبار باستخدام التابع [`~datasets.Dataset.train_test_split`]:

```python
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
```

لنتعرف بصريًا على بعض عينات مجموعة التدريب.

```python
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")


sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_training_images_image_cap.png" alt="عينات من صور التدريب"/>
</div>

## معالجة المجموعة مسبقًا

نظرًا لأن المجموعة تحتوي على نمطين (صورة ونص)، فسيقوم خط المعالجة المسبقة بإعداد الصور والتسميات النصية.

لفعل ذلك، حمّل صنف المُعالج المرتبط بالنموذج الذي ستقوم بضبطه دقيقًا.

```python
from transformers import AutoProcessor

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
```

سيقوم المُعالج داخليًا بإجراء معالجة مسبقة للصورة (بما في ذلك تغيير الحجم، وتحجيم البكسلات) وبتقسيم التسمية إلى رموز.

```python
def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)
```

مع تجهيز البيانات، يمكنك الآن إعداد النموذج للضبط الدقيق.

## تحميل نموذج أساسي

حمّل ["microsoft/git-base"](https://huggingface.co/microsoft/git-base) داخل كائن [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM).

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

## التقييم

يُقيَّم أداء نماذج وصف الصور غالبًا باستخدام [Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge) أو [Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer). في هذا الدليل سنستخدم معدل الخطأ في الكلمات (WER).

نستخدم مكتبة 🤗 Evaluate لهذا الغرض. للاطلاع على القيود والملاحظات الخاصة بـ WER، راجع [هذا الدليل](https://huggingface.co/spaces/evaluate-metric/wer).

```python
from evaluate import load
import torch

wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
```

## التدريب!

أنت الآن جاهز لبدء ضبط النموذج. سنستخدم 🤗 [`Trainer`].

أولًا، عرّف معاملات التدريب باستخدام [`TrainingArguments`].

```python
from transformers import TrainingArguments, Trainer

model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)
```

ثم مرّرها مع البيانات والنموذج إلى [`Trainer`].

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
```

لبدء التدريب، استدعِ [`~Trainer.train`] على كائن [`Trainer`].

```python 
trainer.train()
```

ينبغي أن تلاحظ انخفاض خسارة التدريب تدريجيًا مع تقدم التدريب.

بعد اكتمال التدريب، شارك نموذجك على Hub باستخدام التابع [`~Trainer.push_to_hub`] حتى يتمكن الجميع من استخدامه:

```python
trainer.push_to_hub()
```

## الاستدلال

اختر صورةً من `test_ds` لاختبار النموذج.

```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/test_image_image_cap.png" alt="صورة اختبار"/>
</div>

حضّر الصورة للنموذج.

```python
from accelerate.test_utils.testing import get_backend
# يكتشف تلقائيًا نوع الجهاز الأساسي (CUDA أو CPU أو XPU أو MPS ...)
device, _, _ = get_backend()
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values
```

استدعِ [`generate`] ثم فك ترميز التنبؤات.

```python
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```
```bash
a drawing of a pink and blue pokemon
```

يبدو أن النموذج المضبوط قد ولّد وصفًا جيدًا إلى حدٍ كبير!
