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

# تصنيف الصور بدون تدريب مسبق (Zero-shot image classification)

[[open-in-colab]]

تصنيف الصور بدون تدريب مسبق هو مهمة تتضمن تصنيف الصور إلى فئات مختلفة باستخدام نموذج لم يُدرَّب صراحةً على بيانات تحتوي أمثلة معنونة من تلك الفئات المحددة.

تقليديًا، يتطلب تصنيف الصور تدريب نموذج على مجموعة محددة من الصور المعلَّمة، ويتعلم هذا النموذج "مواءمة" سمات معينة في الصورة مع الوسوم. عند الحاجة لاستخدام هذا النموذج لمهمة تصنيف تُقدّم مجموعة جديدة من الوسوم، يلزم الضبط الدقيق لإعادة "معايرة" النموذج.

على النقيض، تكون نماذج تصنيف الصور بدون تدريب مسبق أو ذات المفردات المفتوحة متعددة الوسائط عادةً، وقد دُرّبت على مجموعة بيانات كبيرة من الصور والأوصاف المرتبطة بها. تتعلم هذه النماذج تمثيلاتٍ مُتّسقة بين الرؤية واللغة يمكن استخدامها في العديد من المهام اللاحقة، بما في ذلك تصنيف الصور بدون تدريب مسبق.

يُعدّ هذا نهجًا أكثر مرونة لتصنيف الصور لأنه يسمح للنماذج بالتعميم على الفئات الجديدة وغير المرئية من دون الحاجة لبيانات تدريب إضافية، كما يتيح للمستخدمين الاستعلام عن الصور بأوصاف نصية حرة لأهدافهم.

في هذا الدليل ستتعلم كيف:

- تنشئ أنبوب تصنيف صور بدون تدريب مسبق.
- تُشغّل استدلال تصنيف الصور بدون تدريب مسبق يدويًا.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات اللازمة:

```bash
pip install -q "transformers[torch]" pillow
```

## أنبوب تصنيف الصور بدون تدريب مسبق

أبسط طريقة لتجربة الاستدلال باستخدام نموذج يدعم تصنيف الصور بدون تدريب مسبق هي استخدام [`pipeline`] الموافق. أنشئ الأنبوب من [نقطة تفتيش على Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads):

```python
>>> from transformers import pipeline

>>> checkpoint = "openai/clip-vit-large-patch14"
>>> detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
```

بعد ذلك، اختر صورة ترغب في تصنيفها.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/owl.jpg" alt="Photo of an owl"/>
</div>

مرّر الصورة والوسوم المرشحة إلى الأنبوب. هنا نمرّر الصورة مباشرة؛ من الخيارات المناسبة الأخرى مسارًا محليًا لصورة أو رابط صورة. يمكن أن تكون الوسوم كلمات بسيطة كما في هذا المثال، أو أكثر وصفًا.

```py
>>> predictions = detector(image, candidate_labels=["fox", "bear", "seagull", "owl"])
>>> predictions
[{'score': 0.9996670484542847, 'label': 'owl'},
 {'score': 0.000199399160919711, 'label': 'seagull'},
 {'score': 7.392891711788252e-05, 'label': 'fox'},
 {'score': 5.96074532950297e-05, 'label': 'bear'}]
```

## تصنيف الصور بدون تدريب مسبق يدويًا

بعد أن رأيت كيفية استخدام أنبوب تصنيف الصور بدون تدريب مسبق، لنرَ كيف يمكنك تشغيل التصنيف يدويًا.

ابدأ بتحميل النموذج والمُعالج المرتبط من [نقطة تفتيش على Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads). سنستخدم نفس نقطة التفتيش كما سبق:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

>>> model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

لنأخذ صورةً مختلفة للتغيير.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" alt="Photo of a car"/>
</div>

استخدم المُعالج لتحضير المدخلات للنموذج. يجمع المُعالج بين معالج صور يُحضّر الصورة للنموذج عبر تغيير الحجم والتطبيع، ومُقسّمٍ يُعنى بالمدخلات النصية.

```py
>>> candidate_labels = ["tree", "car", "bike", "cat"]
# follows the pipeline prompt template to get same results
>>> candidate_labels = [f'This is a photo of {label}.' for label in candidate_labels]
>>> inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
```

مرّر المدخلات عبر النموذج، ثم عالِج النتائج بعديًا:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits_per_image[0]
>>> probs = logits.softmax(dim=-1).numpy()
>>> scores = probs.tolist()

>>> result = [
...     {"score": score, "label": candidate_label}
...     for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
... ]

>>> result
[{'score': 0.998572, 'label': 'car'},
 {'score': 0.0010570387, 'label': 'bike'},
 {'score': 0.0003393686, 'label': 'tree'},
 {'score': 3.1572064e-05, 'label': 'cat'}]
```
