# توليد تعليقات توضيحية للصور

[[open-in-colab]]

تتمثل مهمة توليد تعليقات توضيحية للصور في التنبؤ بتعليق توضيحي لصورة معينة. وتشمل التطبيقات الواقعية الشائعة لهذه المهمة مساعدة الأشخاص ضعاف البصر على التنقل في مختلف المواقف. وبالتالي، تساعد تعليقات الصور التوضيحية على تحسين إمكانية وصول المحتوى للأشخاص من خلال وصف الصور لهم.

سيوضح هذا الدليل كيفية:

* ضبط نموذج توليد تعليقات توضيحية للصور بدقة.
* استخدام النموذج المضبوط بدقة للاستنتاج.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate -q
pip install jiwer -q
```

نشجعك على تسجيل الدخول إلى حسابك في Hugging Face حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتوثيق:


```python
from huggingface_hub import notebook_login

notebook_login()
```

## تحميل مجموعة بيانات Pokémon BLIP captions

استخدم مكتبة Dataset من 🤗 لتحميل مجموعة بيانات تتكون من أزواج {الصورة-التعليق التوضيحي}. لإنشاء مجموعة بيانات خاصة بك لتوليد تعليقات توضيحية للصور في PyTorch، يمكنك اتباع [هذا الدفتر](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb).


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

تحتوي مجموعة البيانات على ميزتين، `image` و`text`.

<Tip>

تحتوي العديد من مجموعات بيانات تعليقات الصور التوضيحية على تعليقات متعددة لكل صورة. وفي هذه الحالات، تتمثل إحدى الاستراتيجيات الشائعة في أخذ عينة عشوائية من التعليقات من بين التعليقات المتاحة أثناء التدريب.

</Tip>

قم بتقسيم مجموعة البيانات التدريبية إلى مجموعة تدريب واختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:


```python
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
```

دعونا نعرض بعض العينات من مجموعة التدريب.


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
        pltMultiplier = plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")


sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)
```
    
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_training_images_image_cap.png" alt="Sample training images"/>
</div>

## معالجة مجموعة البيانات مسبقًا

نظرًا لأن مجموعة البيانات لها وسيلتان (الصورة والنص)، فإن خط أنابيب المعالجة المسبقة ستقوم بمعالجة الصور والتعليقات التوضيحية.

للقيام بذلك، قم بتحميل فئة المعالج المرتبطة بالنموذج الذي ستقوم بضبطه بدقة.

```python
from transformers import AutoProcessor

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
```
```python
from transformers import AutoProcessor

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
```

سيقوم المعالج داخليًا بمعالجة الصورة (التي تشمل تغيير الحجم، وتدرج البكسل) ورموز التعليق التوضيحي.

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

مع تحضير مجموعة البيانات، يمكنك الآن إعداد النموذج للضبط الدقيق.

## تحميل نموذج أساسي

قم بتحميل ["microsoft/git-base"](https://huggingface.co/microsoft/git-base) في كائن [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM).


```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

## التقييم

تُقيَّم نماذج تعليقات الصور التوضيحية عادةً باستخدام [Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge) أو [Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer). وفي هذا الدليل، ستستخدم معدل خطأ الكلمات (WER).

نستخدم مكتبة 🤗 Evaluate للقيام بذلك. وللاطلاع على القيود المحتملة والمشكلات الأخرى لمعدل WER، راجع [هذا الدليل](https://huggingface.co/spaces/evaluate-metric/wer).


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

الآن، أنت مستعد لبدء الضبط الدقيق للنموذج. ستستخدم 🤗 [`Trainer`] لهذا الغرض.

أولاً، حدد وسائط التدريب باستخدام [`TrainingArguments`].


```python
from transformers import TrainingArguments, Trainer

model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon"،
    learning_rate=5e-5،
    num_train_epochs=50،
    fp16=True،
    per_device_train_batch_size=32،
    per_device_eval_batch_size=32،
    gradient_accumulation_steps=2،
    save_total_limit=3،
    eval_strategy="steps"،
    eval_steps=50،
    save_strategy="steps"،
    save_steps=50،
    logging_steps=50،
    remove_unused_columns=False،
    push_to_hub=True،
    label_names=["labels"]،
    load_best_model_at_end=True،
)
```

ثم مررها إلى جانب مجموعات البيانات والنموذج إلى 🤗 Trainer.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
```

لبدء التدريب، ما عليك سوى استدعاء [`~Trainer.train`] على كائن [`Trainer`].

```python 
trainer.train()
```

يجب أن تشاهد انخفاض خسارة التدريب بسلاسة مع تقدم التدريب.

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:


```python
trainer.push_to_hub()
```

## الاستنتاج

خذ عينة صورة من `test_ds` لاختبار النموذج.


```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/test_image_image_cap.png" alt="Test image"/>
</div>
    
قم بإعداد الصورة للنموذج.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values
```
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values
```

استدعِ [`generate`] وفك تشفير التنبؤات.

```python
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```
```bash
a drawing of a pink and blue pokemon
```

يبدو أن النموذج المضبوط بدقة قد ولَّد تعليقًا توضيحيًا جيدًا!