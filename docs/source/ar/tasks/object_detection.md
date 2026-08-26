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

# اكتشاف الكائنات (Object detection)

[[open-in-colab]]

اكتشاف الكائنات هو مهمة من مهام الرؤية الحاسوبية تهدف إلى اكتشاف الكيانات (مثل البشر أو المباني أو السيارات) داخل الصورة. تستقبل نماذج اكتشاف الكائنات صورةً كمدخل، وتُخرج إحداثيات المربعات المحيطة (Bounding boxes) والتسميات المرتبطة بالكائنات المكتشفة. يمكن أن تحتوي الصورة على عدة كائنات، لكلٍ منها مربع محيط وتسميته الخاصة (على سبيل المثال، قد تحتوي الصورة على سيارة ومبنى)، ويمكن أن تتواجد الكائنات في مواضع مختلفة من الصورة (مثل وجود عدة سيارات).
هذا النوع من المهام شائع في القيادة الذاتية لاكتشاف المشاة وعلامات الطرق وإشارات المرور. تتضمن تطبيقات أخرى العدّ في الصور، والبحث بالصور، وغيرها.

في هذا الدليل، ستتعلم كيفية:

 1. ضبط نموذج [DETR](https://huggingface.co/docs/transformers/model_doc/detr) بدقة (Fine-tuning) — وهو نموذج يجمع بين عمود فقري التفافيي (Convolutional backbone) ومحوّل Encoder-Decoder — على مجموعة بيانات [CPPE-5](https://huggingface.co/datasets/cppe-5).
 2. استخدام النموذج المضبوط للاستدلال (Inference).

<Tip>

للاطلاع على جميع المعماريات ونقاط التفتيش (Checkpoints) المتوافقة مع هذه المهمة، نوصي بمراجعة [صفحة المهمة](https://huggingface.co/tasks/object-detection)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات اللازمة:

```bash
pip install -q datasets transformers accelerate timm
pip install -q -U albumentations>=1.4.5 torchmetrics pycocotools
```

سنستخدم 🤗 Datasets لتحميل مجموعة بيانات من Hugging Face Hub، و🤗 Transformers لتدريب النموذج، ومكتبة `albumentations` لزيادة البيانات (Data augmentation).

نشجعك على مشاركة نموذجك مع المجتمع. سجّل الدخول إلى حسابك في Hugging Face لرفعه إلى Hub. عند المطالبة، أدخل رمز الوصول لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

للبدء، سنعرّف ثوابت عامة، وهي اسم النموذج وحجم الصورة. سنستخدم في هذا الدليل نموذج DETR الشرطي (Conditional DETR) نظرًا لتقاربه الأسرع. لا تتردد في اختيار أي نموذج لاكتشاف الكائنات متاح في مكتبة `transformers`.

```py
>>> MODEL_NAME = "microsoft/conditional-detr-resnet-50"  # أو "facebook/detr-resnet-50"
>>> IMAGE_SIZE = 480
```

## تحميل مجموعة بيانات CPPE-5

تحتوي [مجموعة بيانات CPPE-5](https://huggingface.co/datasets/cppe-5) على صور مع تعليقات (Annotations) تُحدِّد معدات الوقاية الشخصية الطبية (PPE) في سياق جائحة كوفيد-19.

ابدأ بتحميل مجموعة البيانات وإنشاء قسم `validation` من `train`:

```py
>>> from datasets import load_dataset

>>> cppe5 = load_dataset("cppe-5")

>>> if "validation" not in cppe5:
...     split = cppe5["train"].train_test_split(0.15, seed=1337)
...     cppe5["train"] = split["train"]
...     cppe5["validation"] = split["test"]

>>> cppe5
DatasetDict({
    train: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 850
    })
    test: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 29
    })
    validation: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 150
    })
})
```

ستلاحظ أن هذه المجموعة تحتوي على 1000 صورة لمجموعتي التدريب والتحقق، ومجموعة اختبار تضم 29 صورة.

للتعرف أكثر على البيانات، استكشف شكل الأمثلة:

```py
>>> cppe5["train"][0]
{
  'image_id': 366,
  'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=500x290>,
  'width': 500,
  'height': 500,
  'objects': {
    'id': [1932, 1933, 1934],
    'area': [27063, 34200, 32431],
    'bbox': [[29.0, 11.0, 97.0, 279.0],
      [201.0, 1.0, 120.0, 285.0],
      [382.0, 0.0, 113.0, 287.0]],
    'category': [0, 0, 0]
  }
}
```

تتضمن أمثلة المجموعة الحقول التالية:
- `image_id`: معرّف صورة المثال
- `image`: كائن `PIL.Image.Image` يحتوي على الصورة
- `width`: عرض الصورة
- `height`: ارتفاع الصورة
- `objects`: قاموس يحتوي بيانات المربعات المحيطة للكائنات في الصورة:
  - `id`: معرّف التعليق (annotation)
  - `area`: مساحة المربع المحيط
  - `bbox`: المربع المحيط للكائن (بصيغة [COCO](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco))
  - `category`: فئة الكائن، بالقيم المحتملة: `Coverall (0)`, `Face_Shield (1)`, `Gloves (2)`, `Goggles (3)`, `Mask (4)`

قد تلاحظ أن الحقل `bbox` يتبع صيغة COCO، وهي الصيغة التي يتوقعها نموذج DETR. ومع ذلك، فإن تجميع الحقول داخل `objects` يختلف عن تنسيق التعليقات الذي يتطلبه DETR. ستحتاج لتطبيق بعض تحويلات المعالجة المسبقة قبل استخدام هذه البيانات في التدريب.

لزيادة الفهم، اعرض مثالًا من المجموعة:

```py
>>> import numpy as np
>>> import os
>>> from PIL import Image, ImageDraw

>>> image = cppe5["train"][2]["image"]
>>> annotations = cppe5["train"][2]["objects"]
>>> draw = ImageDraw.Draw(image)

>>> categories = cppe5["train"].features["objects"]["category"].feature.names

>>> id2label = {index: x for index, x in enumerate(categories, start=0)}
>>> label2id = {v: k for k, v in id2label.items()}

>>> for i in range(len(annotations["id"])):
...     box = annotations["bbox"][i]
...     class_idx = annotations["category"][i]
...     x, y, w, h = tuple(box)
...     # تحقّق مما إذا كانت الإحداثيات مُطبَّعة (Normalized) أم لا
...     if max(box) > 1.0:
...         # الإحداثيات غير مُطبَّعة، لا حاجة لإعادة التحجيم
...         x1, y1 = int(x), int(y)
...         x2, y2 = int(x + w), int(y + h)
...     else:
...         # الإحداثيات مُطبَّعة، أعد تحجيمها إلى إحداثيات مطلقة
...         x1 = int(x * width)
...         y1 = int(y * height)
...         x2 = int((x + w) * width)
...         y2 = int((y + h) * height)
...     draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
...     draw.text((x, y), id2label[class_idx], fill="white")

>>> image
```
<div class="flex justify-center">
    <img src="https://i.imgur.com/oVQb9SF.png" alt="CPPE-5 Image Example"/>
</div>

لعرض المربعات المحيطة مع التسميات المرتبطة بها، يمكنك الحصول على التسميات من بيانات تعريف (Metadata) المجموعة، وتحديدًا حقل `category`.
ستحتاج أيضًا لإنشاء قاموسين يُحوِّلان بين معرّف التسيمة والفئة (`id2label`) وبالعكس (`label2id`). ستستخدمهما لاحقًا عند تهيئة النموذج. لاحظ أن جزء الرسم أعلاه يفترض أن الصيغة هي `COCO` أي `(x_min, y_min, width, height)`. يجب تعديل ذلك إذا كنت تعمل بصيغ أخرى مثل `(x_min, y_min, x_max, y_max)`.

كخطوة أخيرة للتعرّف على البيانات، افحصها بحثًا عن مشاكل محتملة. من المشاكل الشائعة في مجموعات بيانات اكتشاف الكائنات وجود مربعات محيطة "تتجاوز" حافة الصورة. مثل هذه المربعات قد تُسبب أخطاء أثناء التدريب ويجب معالجتها. توجد بعض الأمثلة على ذلك في هذه المجموعة. لتبسيط الأمور في هذا الدليل، سنضبط `clip=True` في `BboxParams` ضمن التحويلات أدناه.

## المعالجة المسبقة للبيانات (Preprocess)

لإجراء الضبط الدقيق (Fine-tuning) لنموذج، يجب أن تُجهِّز البيانات لتطابق بدقة الطريقة المستخدمة أثناء ما قبل التدريب (Pretraining) للنموذج.
يقوم [`AutoImageProcessor`] بمعالجة بيانات الصور لإنتاج `pixel_values` و`pixel_mask` و`labels` التي يمكن لنموذج DETR التدريب عليها. يحتوي المعالج الصوري (Image processor) على بعض الخصائص الجاهزة التي لست مضطرًا للقلق بشأنها:

- `image_mean = [0.485, 0.456, 0.406 ]`
- `image_std = [0.229, 0.224, 0.225]`

هذه هي قيم المتوسط والانحراف المعياري المستخدمة لتطبيع الصور أثناء ما قبل التدريب. من الضروري إعادة استخدامها أثناء الاستدلال أو الضبط الدقيق لنموذج الصور.

أنشئ كائن المعالجة الصورية من نفس نقطة التفتيش (Checkpoint) الخاصة بالنموذج الذي ترغب بضبطه:

```py
>>> from transformers import AutoImageProcessor

>>> MAX_SIZE = IMAGE_SIZE

>>> image_processor = AutoImageProcessor.from_pretrained(
...     MODEL_NAME,
...     do_resize=True,
...     size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
...     do_pad=True,
...     pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
... )
```

قبل تمرير الصور إلى `image_processor`، طبّق تحويلين مسبقين على المجموعة:
- زيادة البيانات (Augmentation) للصور
- إعادة تنسيق التعليقات لتتوافق مع توقعات DETR

أولًا، لتقليل فرط التكيّف (Overfitting) على بيانات التدريب، يمكنك تطبيق زيادة للبيانات باستخدام أي مكتبة مناسبة. هنا نستخدم [Albumentations](https://albumentations.ai/docs/).
تضمن هذه المكتبة أن التحويلات تؤثر على الصورة وتحدّث المربعات المحيطة وفقًا لذلك.
تحتوي توثيقات مكتبة 🤗 Datasets على [دليل مفصّل لزيادة الصور لاكتشاف الكائنات](https://huggingface.co/docs/datasets/object_detection) ويستخدم نفس مجموعة البيانات كمثال. طبّق بعض التحويلات الهندسية واللونية على الصورة. لاستكشاف المزيد من خيارات الزيادة، راجع [Albumentations Demo Space](https://huggingface.co/spaces/qubvel-hf/albumentations-demo).

```py
>>> import albumentations as A

>>> train_augment_and_transform = A.Compose(
...     [
...         A.Perspective(p=0.1),
...         A.HorizontalFlip(p=0.5),
...         A.RandomBrightnessContrast(p=0.5),
...         A.HueSaturationValue(p=0.1),
...     ],
...     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
... )

>>> validation_transform = A.Compose(
...     [A.NoOp()],
...     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
... )
```

يتوقع `image_processor` أن تكون التعليقات في الصيغة التالية: `{'image_id': int, 'annotations': list[Dict]}` حيث يمثل كل قاموس تعليق كائن بصيغة COCO. لِنضِف دالة لإعادة تنسيق تعليقات مثال واحد:

```py
>>> def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
...     """تنسيق مجموعة تعليقات لصورة واحدة إلى صيغة COCO

...     الوسائط (Args):
...         image_id (str): معرّف الصورة. مثلًا: "0001"
...         categories (list[int]): قائمة الفئات/التسميات الموافقة للمربعات المحيطة
...         areas (list[float]): قائمة المساحات الموافقة للمربعات المحيطة
...         bboxes (list[tuple[float]]): قائمة المربعات المحيطة بصيغة COCO
...             ([center_x, center_y, width, height] بإحداثيات مطلقة)

...     القيمة المعادة (Returns):
...         dict: {
...             "image_id": معرّف الصورة,
...             "annotations": قائمة التعليقات المنسّقة
...         }
...     """
...     annotations = []
...     for category, area, bbox in zip(categories, areas, bboxes):
...         formatted_annotation = {
...             "image_id": image_id,
...             "category_id": category,
...             "iscrowd": 0,
...             "area": area,
...             "bbox": list(bbox),
...         }
...         annotations.append(formatted_annotation)

...     return {
...         "image_id": image_id,
...         "annotations": annotations,
...     }

```

يمكنك الآن دمج تحويلات الصورة والتعليقات للاستخدام على دفعة (Batch) من الأمثلة:

```py
>>> def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
...     """تطبيق الزيادات وإخراج التعليقات بصيغة COCO لمهمة اكتشاف الكائنات"""

...     images = []
...     annotations = []
...     for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
...         image = np.array(image.convert("RGB"))

...         # تطبيق زيادات البيانات
...         output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
...         images.append(output["image"])

...         # تنسيق التعليقات بصيغة COCO
...         formatted_annotations = format_image_annotations_as_coco(
...             image_id, output["category"], objects["area"], output["bboxes"]
...         )
...         annotations.append(formatted_annotations)

...     # تطبيق تحويلات المعالج الصوري: تغيير الحجم، إعادة القياس، التطبيع
...     result = image_processor(images=images, annotations=annotations, return_tensors="pt")

...     if not return_pixel_mask:
...         result.pop("pixel_mask", None)

...     return result
```

طبّق دالة المعالجة المسبقة هذه على مجموعة البيانات كاملةً باستخدام أسلوب 🤗 Datasets [`~datasets.Dataset.with_transform`]. يطبق هذا الأسلوب التحويلات أثناء تحميل عناصر المجموعة عند الطلب.

في هذه المرحلة، يمكنك فحص كيف يبدو المثال بعد التحويلات. ينبغي أن ترى موترًا (Tensor) لـ`pixel_values`، وموترًا لـ`pixel_mask`، وحقل `labels`.

```py
>>> from functools import partial

>>> # إنشاء دوال التحويل على دفعات وتطبيقها على الأقسام
>>> train_transform_batch = partial(
...     augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
... )
>>> validation_transform_batch = partial(
...     augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
... )

>>> cppe5["train"] = cppe5["train"].with_transform(train_transform_batch)
>>> cppe5["validation"] = cppe5["validation"].with_transform(validation_transform_batch)
>>> cppe5["test"] = cppe5["test"].with_transform(validation_transform_batch)

>>> cppe5["train"][15]
{'pixel_values': tensor([[[ 1.9235,  1.9407,  1.9749,  ..., -0.7822, -0.7479, -0.6965],
          [ 1.9578,  1.9749,  1.9920,  ..., -0.7993, -0.7650, -0.7308],
          [ 2.0092,  2.0092,  2.0263,  ..., -0.8507, -0.8164, -0.7822],
          ...,
          [ 0.0741,  0.0741,  0.0741,  ...,  0.0741,  0.0741,  0.0741],
          [ 0.0741,  0.0741,  0.0741,  ...,  0.0741,  0.0741,  0.0741],
          [ 0.0741,  0.0741,  0.0741,  ...,  0.0741,  0.0741,  0.0741]],

          [[ 1.6232,  1.6408,  1.6583,  ...,  0.8704,  1.0105,  1.1331],
          [ 1.6408,  1.6583,  1.6758,  ...,  0.8529,  0.9930,  1.0980],
          [ 1.6933,  1.6933,  1.7108,  ...,  0.8179,  0.9580,  1.0630],
          ...,
          [ 0.2052,  0.2052,  0.2052,  ...,  0.2052,  0.2052,  0.2052],
          [ 0.2052,  0.2052,  0.2052,  ...,  0.2052,  0.2052,  0.2052],
          [ 0.2052,  0.2052,  0.2052,  ...,  0.2052,  0.2052,  0.2052]],

          [[ 1.8905,  1.9080,  1.9428,  ..., -0.1487, -0.0964, -0.0615],
          [ 1.9254,  1.9428,  1.9603,  ..., -0.1661, -0.1138, -0.0790],
          [ 1.9777,  1.9777,  1.9951,  ..., -0.2010, -0.1138, -0.0790],
          ...,
          [ 0.4265,  0.4265,  0.4265,  ...,  0.4265,  0.4265,  0.4265],
          [ 0.4265,  0.4265,  0.4265,  ...,  0.4265,  0.4265,  0.4265],
          [ 0.4265,  0.4265,  0.4265,  ...,  0.4265,  0.4265,  0.4265]]]),
  'labels': {'image_id': tensor([688]), 'class_labels': tensor([3, 4, 2, 0, 0]), 'boxes': tensor([[0.4700, 0.1933, 0.1467, 0.0767],
          [0.4858, 0.2600, 0.1150, 0.1000],
          [0.4042, 0.4517, 0.1217, 0.1300],
          [0.4242, 0.3217, 0.3617, 0.5567],
          [0.6617, 0.4033, 0.5400, 0.4533]]), 'area': tensor([ 4048.,  4140.,  5694., 72478., 88128.]), 'iscrowd': tensor([0, 0, 0, 0, 0]), 'orig_size': tensor([480, 480])}}
```

لقد قمتَ بزيادة الصور الفردية وإعداد تعليقاتها. لكن المعالجة المسبقة لم تكتمل بعد. في الخطوة الأخيرة، أنشئ دالة `collate_fn` مخصّصة لتجميع الصور في دفعات.
قم بملء الصور (التي أصبحت الآن `pixel_values`) إلى أكبر حجم في الدفعة، وأنشئ `pixel_mask` مطابقًا يحدّد أي البكسلات حقيقية (1) وأيها تعبئة (0).

```py
>>> import torch

>>> def collate_fn(batch):
...     data = {}
...     data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
...     data["labels"] = [x["labels"] for x in batch]
...     if "pixel_mask" in batch[0]:
...         data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
...     return data

```

<!-- INSERT_MAP_SECTION -->

<!-- INSERT_TRAINING_SECTION -->

<!-- INSERT_EVAL_INFER_SECTION -->
