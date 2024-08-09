# الكشف عن الأشياء باستخدام الصفرية

[[open-in-colab]]

تتطلب النماذج المستخدمة تقليديًا في [الكشف عن الأشياء](object_detection) مجموعات بيانات للصور الموسومة للتدريب، وهي مقيدة بالكشف عن مجموعة الفئات من بيانات التدريب.

يتم دعم الكشف عن الأشياء باستخدام الصفرية بواسطة نموذج [OWL-ViT](../model_doc/owlvit) الذي يستخدم نهجًا مختلفًا. OWL-ViT
هو كاشف كائنات مفتوح المفردات. وهذا يعني أنه يمكنه اكتشاف الأشياء في الصور بناءً على استعلامات نصية حرة دون
الحاجة إلى ضبط نموذج على مجموعات بيانات موسومة.

يستفيد OWL-ViT من التمثيلات متعددة الوسائط لأداء الكشف عن المفردات المفتوحة. فهو يجمع بين [CLIP](../model_doc/clip) مع
رؤوس تصنيف وتحديد موقع الكائنات خفيفة الوزن. يتم تحقيق الكشف عن المفردات المفتوحة عن طريق تضمين استعلامات نصية حرة مع مشفر النص CLIP واستخدامها كإدخال لرؤوس تصنيف الكائنات وتحديد الموقع.
تربط الصور ووصفاتها النصية المقابلة، وتعالج ViT رقع الصور كإدخالات. قام مؤلفو OWL-ViT بتدريب CLIP من البداية ثم ضبط نموذج OWL-ViT من النهاية إلى النهاية على مجموعات بيانات الكشف عن الكائنات القياسية باستخدام
خسارة المطابقة الثنائية.

مع هذا النهج، يمكن للنموذج اكتشاف الأشياء بناءً على الأوصاف النصية دون تدريب مسبق على مجموعات بيانات موسومة.

في هذا الدليل، ستتعلم كيفية استخدام OWL-ViT:
- للكشف عن الأشياء بناءً على موجهات النص
- للكشف عن الكائنات الدفعية
- للكشف عن الأشياء الموجهة بالصور

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install -q transformers
```

## خط أنابيب الكشف عن الأشياء باستخدام الصفرية

أبسط طريقة لتجربة الاستدلال مع OWL-ViT هي استخدامه في [`pipeline`]. قم بتنفيذ خط أنابيب
للكشف عن الأشياء باستخدام الصفرية من [نقطة تفتيش على مركز Hub](https://huggingface.co/models?other=owlvit):

```python
>>> from transformers import pipeline

>>> checkpoint = "google/owlv2-base-patch16-ensemble"
>>> detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
```

بعد ذلك، اختر صورة تريد اكتشاف الأشياء فيها. هنا سنستخدم صورة رائدة الفضاء إيلين كولينز التي تعد
جزءًا من مجموعة صور [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html) الرائعة.

```py
>>> import skimage
>>> import numpy as np
>>> from PIL import Image

>>> image = skimage.data.astronaut()
>>> image = Image.fromarray(np.uint8(image)).convert("RGB")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png" alt="رائد الفضاء إيلين كولينز"/>
</div>

مرر الصورة وعلامات تصنيف الكائنات المرشحة التي تريد البحث عنها إلى خط الأنابيب.
هنا نمرر الصورة مباشرة؛ تشمل الخيارات الأخرى مسارًا محليًا إلى صورة أو عنوان URL للصورة. كما نمرر الأوصاف النصية لجميع العناصر التي نريد الاستعلام عنها في الصورة.
```py
>>> predictions = detector(
...     image,
...     candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner"],
... )
>>> predictions
[{'score': 0.3571370542049408,
  'label': 'human face',
  'box': {'xmin': 180, 'ymin': 71, 'xmax': 271, 'ymax': 178}},
 {'score': 0.28099656105041504,
  'label': 'nasa badge',
  'box': {'xmin': 129, 'ymin': 348, 'xmax': 206, 'ymax': 427}},
 {'score': 0.2110239565372467,
  'label': 'rocket',
  'box': {'xmin': 350, 'ymin': -1, 'xmax': 468, 'ymax': 288}},
 {'score': 0.13790413737297058,
  'label': 'star-spangled banner',
  'box': {'xmin': 1, 'ymin': 1, 'xmax': 105, 'ymax': 509}},
 {'score': 0.11950037628412247,
  'label': 'nasa badge',
  'box': {'xmin': 277, 'ymin': 338, 'xmax': 327, 'ymax': 380}},
 {'score': 0.10649408400058746,
  'label': 'rocket',
  'box': {'xmin': 358, 'ymin': 64, 'xmax': 424, 'ymax': 280}}]
```

دعونا نصور التوقعات:

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
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_2.png" alt="التوقعات الموضحة على صورة وكالة ناسا"/>
</div>

## الكشف عن الأشياء باستخدام الصفرية الموجهة بالنص يدويًا

الآن بعد أن رأيت كيفية استخدام خط أنابيب الكشف عن الأشياء باستخدام الصفرية، دعنا نكرر نفس
النتيجة يدويًا.

ابدأ بتحميل النموذج والمعالج المرتبط من [نقطة تفتيش على مركز Hub](https://huggingface.co/models?other=owlvit).
هنا سنستخدم نفس نقطة التفتيش كما هو الحال من قبل:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

دعونا نأخذ صورة مختلفة لتغيير الأشياء.

```py
>>> import requests

>>> url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
>>> im = Image.open(requests.get(url, stream=True).raw)
>>> im
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png" alt="صورة شاطئ"/>
</div>

استخدم المعالج لإعداد الإدخالات للنموذج. يجمع المعالج بين معالج الصور الذي يعد
الصورة للنموذج عن طريق تغيير حجمها وتطبيعها، و [`CLIPTokenizer`] الذي يعتني بإدخالات النص.

```py
>>> text_queries = ["hat", "book", "sunglasses", "camera"]
>>> inputs = processor(text=text_queries, images=im, return_tensors="pt")
```

مرر الإدخالات عبر النموذج، وقم بمعالجة النتائج، وعرض النتائج. نظرًا لأن معالج الصور قام بتغيير حجم الصور قبل
إطعامها للنموذج، فأنت بحاجة إلى استخدام طريقة [`~OwlViTImageProcessor.post_process_object_detection`] للتأكد من أن صناديق الإحداثيات المتوقعة
صحيحة بالنسبة إلى الصورة الأصلية:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)
...     target_sizes = torch.tensor([im.size[::-1]])
...     results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(im)

>>> scores = results["scores"].tolist()
>>> labels = results["labels"].tolist()
>>> boxes = results["boxes"].tolist()
>>> draw = ImageDraw.Draw(im)

>>> scores = results["scores"].tolist()
>>> labels = results["labels"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

>>> im
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="صورة الشاطئ مع الأشياء المكتشفة"/>
</div>

## معالجة الدفعات

يمكنك تمرير مجموعات متعددة من الصور والاستعلامات النصية للبحث عن كائنات مختلفة (أو متطابقة) في عدة صور.
دعونا نستخدم صورة رائد فضاء وصورة شاطئ معًا.
بالنسبة للمعالجة الدفعية، يجب تمرير استعلامات النص كقائمة متداخلة إلى المعالج والصور كقوائم من صور PIL،
تنسورات PyTorch، أو صفائف NumPy.

```py
>>> images = [image, im]
>>> text_queries = [
...     ["human face", "rocket", "nasa badge", "star-spangled banner"],
...     ["hat", "book", "sunglasses", "camera"],
... ]
>>> inputs = processor(text=text_queries, images=images, return_tensors="pt")
```

سبق أن مررت بحجم الصورة الفردية كموتر، ولكن يمكنك أيضًا تمرير زوج، أو في حالة
العديد من الصور، قائمة من الأزواج. دعونا ننشئ تنبؤات للمثالين، ونصور الثاني (`image_idx = 1`).

```py
>>> with torch.no_grad():
...     outputs = model(**inputs)
...     target_sizes = [x.size[::-1] for x in images]
...     results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)

>>> image_idx = 1
>>> draw = ImageDraw.Draw(images[image_idx])

>>> scores = results[image_idx]["scores"].tolist()
>>> labels = results[image_idx]["labels"].tolist()
>>> boxes = results[image_idx]["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_queries[image_idx][label]}: {round(score,2)}", fill="white")

>>> images[image_idx]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="صورة الشاطئ مع الأشياء المكتشفة"/>
</div>

## الكشف عن الأشياء الموجهة بالصور

بالإضافة إلى الكشف عن الأشياء باستخدام الصفرية باستخدام الاستعلامات النصية، يوفر OWL-ViT الكشف عن الأشياء الموجهة بالصور. وهذا يعني
يمكنك استخدام استعلام الصورة للعثور على كائنات مماثلة في صورة الهدف.
على عكس الاستعلامات النصية، لا يُسمح إلا بصورة مثال واحدة.

دعونا نأخذ صورة بها قطتان على الأريكة كصورة مستهدفة، وصورة لقطة واحدة
كاستعلام:

```py
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image_target = Image.open(requests.get(url, stream=True).raw)

>>> query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
```

دعونا نلقي نظرة سريعة على الصور:

```py
>>> import matplotlib.pyplot as plt

>>> fig, ax = plt.subplots(1, 2)
>>> ax[0].imshow(image_target)
>>> ax[1].imshow(query_image)
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_5.png" alt="قطط"/>
</div>

في خطوة ما قبل المعالجة، بدلاً من الاستعلامات النصية، فأنت الآن بحاجة إلى استخدام `query_images`:

```py
>>> inputs = processor(images=image_target, query_images=query_image, return_tensors="pt")
```

بالنسبة للتنبؤات، بدلاً من تمرير الإدخالات إلى النموذج، قم بتمريرها إلى [`~OwlViTForObjectDetection.image_guided_detection`]. ارسم التوقعات
كما هو الحال من قبل باستثناء أنه لا توجد الآن تسميات.
بالنسبة للتنبؤات، بدلاً من تمرير الإدخالات إلى النموذج، قم بتمريرها إلى [`~OwlViTForObjectDetection.image_guided_detection`]. ارسم التوقعات
كما هو الحال من قبل باستثناء أنه لا توجد الآن تسميات.

```py
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
...     target_sizes = torch.tensor([image_target.size[::-1]])
...     results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(image_target)

>>> scores = results["scores"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)

>>> image_target
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_6.png" alt="قطط مع صناديق الإحداثيات"/>
</div>