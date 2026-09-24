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

# كشف الأجسام بدون تدريب مسبق (Zero-shot object detection)

[[open-in-colab]]

تقليديًا، تتطلب النماذج المستخدمة في [كشف الأجسام](object_detection) مجموعات بيانات صور مُعنونة للتدريب،
وتكون محدودة بالكشف عن مجموعة الفئات الموجودة في بيانات التدريب فقط.

كشف الأجسام بدون تدريب مسبق هو مهمة في الرؤية الحاسوبية لاكتشاف الأجسام وفئاتها في الصور دون أي تدريبٍ مسبق
أو معرفةٍ مسبقة بالفئات. تتلقى نماذج الكشف بدون تدريب مسبق صورة كمدخل، بالإضافة إلى قائمة بالفئات المرشحة،
وتُخرج صناديق الإحاطة والوسوم حيث تم اكتشاف الأجسام.

> [!NOTE]
> يستضيف Hugging Face العديد من [كواشف الأجسام ذات المفردات المفتوحة بدون تدريب مسبق](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection).

في هذا الدليل، ستتعلم كيفية استخدام مثل هذه النماذج:
- للكشف عن الأجسام بناءً على مطالبات نصية
- للمعالجة الدفعية للكشف عن الأجسام
- للكشف عن الأجسام الموجّه بالصور

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات اللازمة:

```bash
pip install -q transformers
```

## أنبوب الكشف بدون تدريب مسبق

أبسط طريقة لتجربة الاستدلال مع النماذج هي استخدام [`pipeline`]. أنشئ أنبوبًا للكشف بدون تدريب مسبق انطلاقًا من
[نقطة تفتيش على Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection):

```python
>>> from transformers import pipeline

>>> # استخدم أي نقطة تفتيش من hf.co/models?pipeline_tag=zero-shot-object-detection
>>> checkpoint = "iSEE-Laboratory/llmdet_large"
>>> detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
```

بعد ذلك، اختر صورة ترغب في الكشف عن الأجسام فيها. سنستخدم هنا صورة رائدة الفضاء Eileen Collins وهي جزء من
مجموعة صور [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html) Great Images.

```py
>>> from transformers.image_utils import load_image

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png"
>>> image = load_image(url)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png" alt="Astronaut Eileen Collins"/>
</div>

مرّر الصورة والوسوم المرشحة للأجسام التي نبحث عنها إلى الأنبوب. هنا نمرّر الصورة مباشرة؛ من الخيارات المناسبة الأخرى
مسارًا محليًا لصورة أو رابط صورة. نمرّر أيضًا أوصافًا نصية لكل عنصر نريد الاستعلام عنه في الصورة.

```py
>>> predictions = detector(
...     image,
...     candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner"],
...     threshold=0.45,
... )
>>> predictions
[{'score': 0.8409242033958435,
  'label': 'human face',
  'box': {'xmin': 179, 'ymin': 74, 'xmax': 272, 'ymax': 179}},
 {'score': 0.7380027770996094,
  'label': 'rocket',
  'box': {'xmin': 353, 'ymin': 0, 'xmax': 466, 'ymax': 284}},
 {'score': 0.5850900411605835,
  'label': 'star-spangled banner',
  'box': {'xmin': 0, 'ymin': 0, 'xmax': 96, 'ymax': 511}},
 {'score': 0.5697067975997925,
  'label': 'human face',
  'box': {'xmin': 18, 'ymin': 15, 'xmax': 366, 'ymax': 511}},
 {'score': 0.47813931107521057,
  'label': 'star-spangled banner',
  'box': {'xmin': 353, 'ymin': 0, 'xmax': 459, 'ymax': 274}},
 {'score': 0.46597740054130554,
  'label': 'nasa badge',
  'box': {'xmin': 353, 'ymin': 0, 'xmax': 462, 'ymax': 279}},
 {'score': 0.4585932493209839,
  'label': 'nasa badge',
  'box': {'xmin': 132, 'ymin': 348, 'xmax': 208, 'ymax': 423}}]
```

لنُصوّر التنبؤات:

```py
>>> from PIL import ImageDraw

>>> draw = ImageDraw.Draw(image)

>>> for prediction in predictions:
...     box = prediction["box"]
...     label = prediction["label"]
...     score = prediction["score"]

...     xmin, ymin, xmax, ymax = box.values()
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_2.png" alt="Visualized predictions on NASA image"/>
</div>

## كشف الأجسام بالمطالبة النصية يدويًا

الآن بعد أن رأيت كيفية استخدام أنبوب الكشف بدون تدريب مسبق، لنُكرر نفس النتيجة يدويًا.

ابدأ بتحميل النموذج والمُعالج المرتبط من [نقطة تفتيش على Hugging Face Hub](hf.co/iSEE-Laboratory/llmdet_large).
سنستخدم هنا نفس نقطة التفتيش كما قبل:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, device_map="auto")
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

لنأخذ صورة مختلفة للتغيير:

```py
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png"
>>> image = load_image(url)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png" alt="Beach photo"/>
</div>

استخدم المُعالج لتحضير المدخلات للنموذج.

```py
>>> text_labels = ["hat", "book", "sunglasses", "camera"]
>>> inputs = processor(text=text_labels, images=image, return_tensors="pt")to(model.device)
```

مرّر المدخلات عبر النموذج، ثم عالِج النتائج بعديًا وصوّرها. بما أن معالج الصور قد غيّر حجم الصور قبل تمريرها إلى
النموذج، تحتاج لاستخدام التابع `post_process_object_detection` للتأكد من أن صناديق الإحاطة المتنبأ بها لها إحداثيات صحيحة
بالنسبة للصورة الأصلية:

```py
>>> import torch

>>> with torch.inference_mode():
...     outputs = model(**inputs)

>>> results = processor.post_process_grounded_object_detection(
...    outputs, threshold=0.50, target_sizes=[(image.height, image.width)], text_labels=text_labels,
...)[0]

>>> draw = ImageDraw.Draw(image)

>>> scores = results["scores"]
>>> text_labels = results["text_labels"]
>>> boxes = results["boxes"]

>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_label}: {round(score.item(),2)}", fill="white")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="Beach photo with detected objects"/>
</div>

## المعالجة الدفعية

يمكنك تمرير مجموعات متعددة من الصور والاستعلامات النصية للبحث عن أجسام مختلفة (أو متشابهة) في عدة صور.
لنستخدم كلًا من صورة رائد الفضاء وصورة الشاطئ معًا. في المعالجة الدفعية، ينبغي تمرير الاستعلامات النصية كقائمة متداخلة إلى المُعالج،
والصور كقوائم من صور PIL أو موترات PyTorch أو مصفوفات NumPy.

```py
>>> url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png"
>>> url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png"
>>> images = [load_image(url1), load_image(url2)]
>>> text_queries = [
...     ["human face", "rocket", "nasa badge", "star-spangled banner"],
...     ["hat", "book", "sunglasses", "camera", "can"],
... ]
>>> inputs = processor(text=text_queries, images=images, return_tensors="pt", padding=True)
```

سابقًا، لمرحلة ما بعد المعالجة مرّرت حجم صورة مفردة كموتر، لكن يمكنك أيضًا تمرير زوج مرتب (tuple)، أو في حالة عدة صور، قائمة من الأزواج.
لننشئ تنبؤات للمثالين ونُصوّر المثال الثاني (`image_idx = 1`).

```py
>>> with torch.no_grad():
>>>     outputs = model(**inputs)

>>> target_sizes = [(image.height, image.width) for image in images]
>>> results = processor.post_process_grounded_object_detection(
...     outputs, threshold=0.3, target_sizes=target_sizes, text_labels=text_labels,
... )
```

لنصوّر النتائج:

```py
>>> image_idx = 1
>>> draw = ImageDraw.Draw(images[image_idx])

>>> scores = results[image_idx]["scores"].tolist()
>>> text_labels = results[image_idx]["text_labels"]
>>> boxes = results[image_idx]["boxes"].tolist()

>>> for box, score, text_label in zip(boxes, scores, text_labels):
>>>     xmin, ymin, xmax, ymax = box
>>>     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
>>>     draw.text((xmin, ymin), f"{text_label}: {round(score,2)}", fill="white")

>>> images[image_idx]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="Beach photo with detected objects"/>
</div>

## الكشف الموجّه بالصور

بالإضافة إلى الكشف بدون تدريب مسبق باستخدام الاستعلامات النصية، تقدم نماذج مثل [OWL-ViT](https://huggingface.co/collections/ariG23498/owlvit-689b0d0872a7634a6ea17ae7) و[OWLv2](https://huggingface.co/collections/ariG23498/owlv2-689b0d27bd7d96ba3c7f7530) كشفًا موجّهًا بالصور. هذا يعني أنه يمكنك استخدام صورة استعلام للعثور على أجسام مشابهة في الصورة الهدف.

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> checkpoint = "google/owlv2-base-patch16-ensemble"
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, device_map="auto")
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

على عكس الاستعلامات النصية، يُسمح بصورة استعلام واحدة فقط.

لنأخذ صورة هدف تحتوي على قطتين على أريكة، وصورة استعلام تحتوي على قط واحد:

```py
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image_target = Image.open(requests.get(url, stream=True).raw)

>>> query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
```

لنلقِ نظرة سريعة على الصورتين:

```py
>>> import matplotlib.pyplot as plt

>>> fig, ax = plt.subplots(1, 2)
>>> ax[0].imshow(image_target)
>>> ax[1].imshow(query_image)
>>> fig.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_5.png" alt="Cats"/>
</div>

في خطوة المعالجة المسبقة، بدلًا من الاستعلامات النصية، تحتاج الآن لاستخدام `query_images`:

```py
>>> inputs = processor(images=image_target, query_images=query_image, return_tensors="pt")
```

للتنبؤات، بدلًا من تمرير المدخلات إلى النموذج، مرّرها إلى [`~OwlViTForObjectDetection.image_guided_detection`]. ارسم التنبؤات كما سبق، باستثناء أنه لا توجد وسوم الآن.

```py
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
...     target_sizes = torch.tensor([image_target.size[::-1]])
...     results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(image_target)

>>> scores = results["scores"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score in zip(boxes, scores):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)

>>> image_target
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_6.png" alt="Cats with bounding boxes"/>
</div>
