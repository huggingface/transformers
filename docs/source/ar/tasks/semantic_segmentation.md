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

# تقسيم الصور (Image Segmentation)

[[open-in-colab]]

<Youtube id="dKE8SIt9C-w"/>

نماذج تقسيم الصور تفصل المناطق المقابلة لمناطق الاهتمام المختلفة في الصورة. تعمل هذه النماذج عن طريق إسناد تصنيف لكل بكسل. توجد عدة أنواع من التقسيم: التجزئة الدلالية (Semantic Segmentation)، وتجزئة المثيل (Instance Segmentation)، والتجزئة الشاملة (Panoptic Segmentation).

في هذا الدليل سنقوم بما يلي:
1. إلقاء نظرة على الأنواع المختلفة للتقسيم.
2. تقديم مثال كامل لضبط دقيق (Fine-tuning) للتجزئة الدلالية.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات اللازمة:

```py
# uncomment to install the necessary libraries
!pip install -q datasets transformers evaluate accelerate
```

نوصيك بتسجيل الدخول إلى حسابك على Hugging Face حتى تتمكن من رفع ومشاركة نموذجك مع المجتمع. عند المطالبة، أدخل الرمز (token) لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## أنواع التقسيم (Types of Segmentation)

تُسند التجزئة الدلالية تصنيفًا أو فئة لكل بكسل في الصورة. دعنا نلقي نظرة على مخرجات نموذج للتجزئة الدلالية. سيُسند النموذج نفس الفئة لكل مثيل من الكائن الذي يصادفه في الصورة؛ على سبيل المثال، سيتم تصنيف كل القطط كـ "cat" بدلًا من "cat-1"، "cat-2".
يمكننا استخدام أنبوب `image-segmentation` في مكتبة Transformers للاستدلال سريعًا بنموذج تجزئة دلالية. لنتفحص الصورة المثال.

```python
from transformers import pipeline
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg" alt="Segmentation Input"/>
</div>

سنستخدم النموذج [nvidia/segformer-b1-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024).

```python
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)
results
```

يتضمن خرج أنبوب التجزئة قناعًا (mask) لكل فئة متنبأ بها.
```bash
[{'score': None,
  'label': 'road',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'sidewalk',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'building',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'pole',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'traffic sign',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'vegetation',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'terrain',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

عند النظر إلى قناع فئة السيارة، يمكننا ملاحظة أن كل السيارات مُصنّفة بنفس القناع.

```python
results[-1]["mask"]
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/semantic_segmentation_output.png" alt="Semantic Segmentation Output"/>
</div>

في تجزئة المثيل (Instance Segmentation)، الهدف ليس تصنيف كل بكسل، وإنما توقع قناع لكل مثيل من كائن معين في الصورة. يشبه ذلك كثيرًا اكتشاف الكائنات (Object Detection)، حيث يوجد صندوق إحاطة لكل مثيل، لكن بدلًا من الصندوق يوجد قناع تجزئة. سنستخدم [facebook/mask2former-swin-large-cityscapes-instance](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-instance) لهذا الغرض.

```python
instance_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-instance")
results = instance_segmentation(image)
results
```

كما ترى أدناه، هناك عدة سيارات مُصنّفة، ولا يوجد تصنيف للبكسلات الأخرى بخلاف تلك التي تنتمي إلى مثيلات "car" و"person".

```bash
[{'score': 0.999944,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999945,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999652,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.903529,
  'label': 'person',
  'mask': <PIL.Image.Image image mode=L size=612x415>}]
```
تفحّص أحد أقنعة السيارة أدناه.

```python
results[2]["mask"]
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/instance_segmentation_output.png" alt="Semantic Segmentation Output"/>
</div>

تجمع التجزئة الشاملة (Panoptic Segmentation) بين التجزئة الدلالية وتجزئة المثيل، حيث يُصنّف كل بكسل ضمن فئة ومثيل لتلك الفئة، وتوجد عدة أقنعة لكل مثيل من الفئة. يمكننا استخدام [facebook/mask2former-swin-large-cityscapes-panoptic](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) لهذا.

```python
panoptic_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-panoptic")
results = panoptic_segmentation(image)
results
```
كما ترى أدناه، لدينا فئات أكثر. سنوضّح لاحقًا أن كل بكسل مُصنّف ضمن إحدى هذه الفئات.

```bash
[{'score': 0.999981,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999958,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.99997,
  'label': 'vegetation',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999575,
  'label': 'pole',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999958,
  'label': 'building',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999634,
  'label': 'road',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.996092,
  'label': 'sidewalk',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999221,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.99987,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

دعنا نجري مقارنة جنبًا إلى جنب لجميع أنواع التقسيم.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation-comparison.png" alt="Segmentation Maps Compared"/>
</div>

بعد رؤية كل أنواع التقسيم، سنغوص في تفاصيل ضبط نموذج للتجزئة الدلالية.

تشمل التطبيقات الواقعية الشائعة للتجزئة الدلالية تدريب السيارات ذاتية القيادة على تحديد المشاة والمعلومات المرورية المهمة، وتحديد الخلايا والشذوذات في الصور الطبية، ومراقبة التغيّرات البيئية من صور الأقمار الصناعية.

## ضبط نموذج للتقسيم (Fine-tuning a Model for Segmentation)

سنقوم الآن بما يلي:

1. إجراء ضبط دقيق لـ [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) على مجموعة البيانات [SceneParse150](https://huggingface.co/datasets/scene_parse_150).
2. استخدام النموذج المُضبط للاستدلال.

<Tip>

لعرض جميع البُنى ونقاط التحقق (checkpoints) المتوافقة مع هذه المهمة، نوصي بالاطلاع على [صفحة المهمة](https://huggingface.co/tasks/image-segmentation)

</Tip>


### تحميل مجموعة بيانات SceneParse150

ابدأ بتحميل مجموعة فرعية أصغر من SceneParse150 من مكتبة 🤗 Datasets. سيسمح لك هذا بالتجربة والتأكد من أن كل شيء يعمل قبل قضاء وقت أطول في التدريب على المجموعة الكاملة.

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

قسّم قسم `train` في مجموعة البيانات إلى مجموعتي تدريب واختبار باستخدام الدالة [`~datasets.Dataset.train_test_split`]:

```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

ثم ألقِ نظرة على مثال:

```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}

# view the image
>>> train_ds[0]["image"]
```

- `image`: صورة PIL للمشهد.
- `annotation`: صورة PIL لخريطة التجزئة، وهي الهدف للنموذج.
- `scene_category`: معرّف فئة يصف مشهد الصورة مثل "kitchen" أو "office". في هذا الدليل ستحتاج فقط إلى `image` و`annotation`، وكلاهما صور PIL.

ستحتاج أيضًا إلى إنشاء قاموس يربط معرّف الفئة باسم الفئة، وهو ما سيكون مفيدًا عند إعداد النموذج لاحقًا. نزّل الخرائط من Hub وأنشئ قاموسي `id2label` و`label2id`:

```py
>>> import json
>>> from pathlib import Path
>>> from huggingface_hub import hf_hub_download

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

#### مجموعة بيانات مخصّصة (Custom dataset)

يمكنك أيضًا إنشاء واستخدام مجموعة البيانات الخاصة بك إذا كنت تفضّل التدريب باستخدام السكربت [run_semantic_segmentation.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_semantic_segmentation.py) بدلًا من مفكرة (notebook). يتطلّب السكربت ما يلي:

1. كائن [`~datasets.DatasetDict`] يحتوي عمودين من نوع [`~datasets.Image`] هما "image" و"label"

     ```py
     from datasets import Dataset, DatasetDict, Image

     image_paths_train = ["path/to/image_1.jpg/jpg", "path/to/image_2.jpg/jpg", ..., "path/to/image_n.jpg/jpg"]
     label_paths_train = ["path/to/annotation_1.png", "path/to/annotation_2.png", ..., "path/to/annotation_n.png"]

     image_paths_validation = [...]
     label_paths_validation = [...]

     def create_dataset(image_paths, label_paths):
         dataset = Dataset.from_dict({"image": sorted(image_paths),
                                     "label": sorted(label_paths)})
         dataset = dataset.cast_column("image", Image())
         dataset = dataset.cast_column("label", Image())
         return dataset

     # step 1: create Dataset objects
     train_dataset = create_dataset(image_paths_train, label_paths_train)
     validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

     # step 2: create DatasetDict
     dataset = DatasetDict({
          "train": train_dataset,
          "validation": validation_dataset,
          }
     )

     # step 3: push to Hub (assumes you have ran the hf auth login command in a terminal/notebook)
     dataset.push_to_hub("your-name/dataset-repo")

     # optionally, you can push to a private repo on the Hub
     # dataset.push_to_hub("name of repo on the hub", private=True)
     ```

2. قاموس `id2label` يربط الأعداد الصحيحة للفئات بأسمائها

     ```py
     import json
     # simple example
     id2label = {0: 'cat', 1: 'dog'}
     with open('id2label.json', 'w') as fp:
     json.dump(id2label, fp)
     ```

كمثال، طالع [مجموعة البيانات هذه](https://huggingface.co/datasets/nielsr/ade20k-demo) التي أُنشئت وفق الخطوات الموضّحة أعلاه.

### المعالجة المسبقة (Preprocess)

الخطوة التالية هي تحميل معالج صور SegFormer لإعداد الصور والوسوم (annotations) للنموذج. بعض مجموعات البيانات، مثل هذه، تستخدم الفهرس الصفري كفئة الخلفية. ولكن فئة الخلفية ليست مُضمّنة فعليًا ضمن 150 فئة، لذا ستحتاج إلى ضبط `do_reduce_labels=True` لطرح واحد من جميع الوسوم. يُستبدل الفهرس الصفري بالقيمة `255` حتى يتجاهله دالّة الخسارة في SegFormer:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)
```

<frameworkcontent>
<pt>

من الشائع تطبيق بعض تعظيمات البيانات (Data Augmentations) على مجموعات الصور لجعل النموذج أكثر متانة ضد فرط التكيّف (overfitting). في هذا الدليل، ستستخدم الدالة [`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) من مكتبة [torchvision](https://pytorch.org/vision/stable/index.html) لتغيير خصائص الألوان عشوائيًا، ويمكنك استخدام أي مكتبة صور تُفضّلها.

```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

أنشئ الآن دالتين للمعالجة المسبقة لإعداد الصور والوسوم للنموذج. تُحوّل هذه الدوال الصور إلى `pixel_values` والوسوم إلى `labels`. لبيانات التدريب، يُطبّق `jitter` قبل تمرير الصور إلى معالج الصور. لبيانات الاختبار، يقوم معالج الصور بالقص والتطبيع للصور (`images`)، بينما يقوم بقص `labels` فقط لأننا لا نطبّق تعظيم بيانات أثناء الاختبار.

```py
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

لتطبيق `jitter` على كامل مجموعة البيانات، استخدم الدالة [`~datasets.Dataset.set_transform`] من 🤗 Datasets. يجري تطبيق التحويل عند الطلب مما يجعله أسرع ويستهلك مساحة قرص أقل:

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
من الشائع تطبيق بعض تعظيمات البيانات على مجموعات الصور لجعل النموذج أكثر متانة ضد فرط التكيّف.
في هذا الدليل، ستستخدم [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) لتغيير خصائص الألوان عشوائيًا، ويمكنك استخدام أي مكتبة صور تُفضّلها.
عرّف دالتي تحويل منفصلتين:
- تحويلات بيانات التدريب التي تتضمن تعظيم الصور
- تحويلات بيانات التحقق (validation) التي تقوم فقط بتبديل ترتيب الأبعاد، نظرًا لأن نماذج الرؤية الحاسوبية في 🤗 Transformers تتوقع ترتيب القنوات أولًا (channels-first)

```py
>>> import tensorflow as tf


>>> def aug_transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.image.random_brightness(image, 0.25)
...     image = tf.image.random_contrast(image, 0.5, 2.0)
...     image = tf.image.random_saturation(image, 0.75, 1.25)
...     image = tf.image.random_hue(image, 0.1)
...     image = tf.transpose(image, (2, 0, 1))
...     return image


>>> def transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.transpose(image, (2, 0, 1))
...     return image
```

بعد ذلك، أنشئ دالتي معالجة مسبقة لإعداد دفعات من الصور والوسوم للنموذج. تطبق هذه الدوال تحويلات الصور وتستعمل `image_processor` المُحمّل سابقًا لتحويل الصور إلى `pixel_values` والوسوم إلى `labels`. يتكفل `ImageProcessor` أيضًا بتغيير الحجم (resize) وتطبيع الصور.

```py
>>> def train_transforms(example_batch):
...     images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs

>>> def val_transforms(example_batch):
...     images = [transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

لتطبيق تحويلات المعالجة المسبقة على كامل مجموعة البيانات، استخدم الدالة [`~datasets.Dataset.set_transform`] من 🤗 Datasets. يجري تطبيق التحويل عند الطلب مما يجعله أسرع ويستهلك مساحة قرص أقل:

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```
</tf>
</frameworkcontent>

### التقييم (Evaluate)

يكون تضمين مقياس أثناء التدريب مفيدًا غالبًا لتقييم أداء النموذج. يمكنك تحميل أسلوب تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، حمّل مقياس [متوسط تقاطع الاتحاد](https://huggingface.co/spaces/evaluate-metric/accuracy) (mean Intersection over Union - IoU) (راجع [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لتعرف المزيد حول كيفية التحميل والحساب):

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

بعد ذلك، أنشئ دالة لحساب المقاييس باستخدام [`~evaluate.EvaluationModule.compute`]. يجب تحويل تنبؤاتك إلى `logits` أولًا، ثم إعادة تشكيلها لتطابق حجم الوسوم قبل أن تتمكن من استدعاء [`~evaluate.EvaluationModule.compute`]:

<frameworkcontent>
<pt>

```py
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if isinstance(value, np.ndarray):
...                 metrics[key] = value.tolist()
...         return metrics
```

</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     logits = tf.transpose(logits, perm=[0, 2, 3, 1])
...     logits_resized = tf.image.resize(
...         logits,
...         size=tf.shape(labels)[1:],
...         method="bilinear",
...     )

...     pred_labels = tf.argmax(logits_resized, axis=-1)
...     metrics = metric.compute(
...         predictions=pred_labels,
...         references=labels,
...         num_labels=num_labels,
...         ignore_index=-1,
...         reduce_labels=image_processor.do_reduce_labels,
...     )

...     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
...     per_category_iou = metrics.pop("per_category_iou").tolist()

...     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
...     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
...     return {"val_" + k: v for k, v in metrics.items()}
```

</tf>
</frameworkcontent>

الآن أصبحت دالة `compute_metrics` جاهزة، وسنعود إليها عند إعداد التدريب.

### التدريب (Train)
<frameworkcontent>
<pt>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`]، فألقِ نظرة على الدليل الأساسي [هنا](../training#finetune-with-trainer)!

</Tip>

أصبحت جاهزًا لبدء تدريب النموذج! حمّل SegFormer باستخدام [`AutoModelForSemanticSegmentation`]، ومرّر إلى النموذج خريطة الربط بين معرفات الفئات وأسمائها:

```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

في هذه المرحلة، تبقّت ثلاث خطوات فقط:

1. حدد معاملات التدريب الفائقة في [`TrainingArguments`]. من المهم ألا تزيل الأعمدة غير المستخدمة لأن ذلك سيحذف عمود `image`. بدون عمود `image`، لن تتمكن من إنشاء `pixel_values`. اضبط `remove_unused_columns=False` لمنع هذا السلوك! المعامل الإلزامي الآخر هو `output_dir` الذي يحدد مكان حفظ النموذج. ستدفع هذا النموذج إلى Hub عبر ضبط `push_to_hub=True` (تحتاج لتسجيل الدخول إلى Hugging Face لرفع نموذجك). في نهاية كل عهدة (epoch)، سيقيّم [`Trainer`] مقياس IoU ويحفظ نقطة تحقق التدريب.
2. مرّر معاملات التدريب إلى [`Trainer`] مع النموذج ومجموعة البيانات والمُرمّز (tokenizer) والمجمّع (data collator) ودالة `compute_metrics`.
3. استدعِ [`~Trainer.train`] لإجراء الضبط الدقيق للنموذج.

```py
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     eval_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بعد اكتمال التدريب، شارك نموذجك على Hub باستخدام الدالة [`~transformers.Trainer.push_to_hub`] ليصبح متاحًا للجميع:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فاطّلع أولًا على [الدليل الأساسي](./training#train-a-tensorflow-model-with-keras)!

</Tip>

لضبط نموذج في TensorFlow، اتبع الخطوات التالية:
1. حدد معاملات التدريب الفائقة، واضبط المُحسِّن (optimizer) وجدول معدل التعلّم.
2. أنشئ نموذجًا مُدرّبًا مسبقًا.
3. حوّل مجموعة 🤗 Dataset إلى `tf.data.Dataset`.
4. قم بتجميع (compile) النموذج.
5. أضف الاستدعاءات التراجعية (callbacks) لحساب المقاييس ورفع النموذج إلى 🤗 Hub.
6. استخدم `fit()` لتشغيل التدريب.

ابدأ بتحديد المعاملات الفائقة والمُحسِّن وجدول معدل التعلّم:

```py
>>> from transformers import create_optimizer

>>> batch_size = 2
>>> num_epochs = 50
>>> num_train_steps = len(train_ds) * num_epochs
>>> learning_rate = 6e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

ثم حمّل SegFormer باستخدام [`TFAutoModelForSemanticSegmentation`] مع خرائط الفئات، وقم بتجميعه مع المُحسِّن. لاحظ أن نماذج Transformers تمتلك دالة خسارة افتراضية مناسبة للمهمة، لذا لا تحتاج لتحديد واحدة إلا إذا رغبت بذلك:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
>>> model.compile(optimizer=optimizer)  # No loss argument!
```

حوّل مجموعات البيانات إلى صيغة `tf.data.Dataset` باستخدام [`~datasets.Dataset.to_tf_dataset`] و[`DefaultDataCollator`]:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")

>>> tf_train_dataset = train_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_eval_dataset = test_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

لحساب الدقّة من التنبؤات ودفع النموذج إلى 🤗 Hub، استخدم [استدعاءات Keras](../main_classes/keras_callbacks).
مرّر دالة `compute_metrics` إلى [`KerasMetricCallback`],
واستخدم [`PushToHubCallback`] لرفع النموذج:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(
...     metric_fn=compute_metrics, eval_dataset=tf_eval_dataset, batch_size=batch_size, label_cols=["labels"]
... )

>>> push_to_hub_callback = PushToHubCallback(output_dir="scene_segmentation", tokenizer=image_processor)

>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أصبحت جاهزًا لتدريب النموذج! استدعِ `fit()` مع بيانات التدريب والتحقق، وعدد العُهد (epochs)، والاستدعاءات التراجعية لضبط النموذج:

```py
>>> model.fit(
...     tf_train_dataset,
...     validation_data=tf_eval_dataset,
...     callbacks=callbacks,
...     epochs=num_epochs,
... )
```

تهانينا! لقد أجريت الضبط الدقيق لنموذجك وشاركته على 🤗 Hub. يمكنك الآن استخدامه للاستدلال!
</tf>
</frameworkcontent>

### الاستدلال (Inference)

رائع، الآن بعد أن أجريت الضبط الدقيق لنموذج، يمكنك استخدامه في الاستدلال!

أعد تحميل مجموعة البيانات وحمّل صورة للاستدلال.

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
>>> ds = ds.train_test_split(test_size=0.2)
>>> test_ds = ds["test"]
>>> image = ds["test"][0]["image"]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="Image of bedroom"/>
</div>

<frameworkcontent>
<pt>

سنرى الآن كيفية الاستدلال بدون استخدام الأنبوب (pipeline). عالج الصورة باستخدام معالج الصور وضع `pixel_values` على وحدة المعالجة المناسبة (GPU/CPU):

```py
>>> from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
>>> device, _, _ = get_backend()
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

مرّر الإدخال إلى النموذج وأرجِع `logits`:

```py
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

بعد ذلك، قم بإعادة تحجيم (rescale) `logits` إلى حجم الصورة الأصلي:

```py
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
حمّل معالج صور لمعالجة الصورة وأرجِع الإدخال كموترات TensorFlow:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
>>> inputs = image_processor(image, return_tensors="tf")
```

مرّر الإدخال إلى النموذج وأرجِع `logits`:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
>>> logits = model(**inputs).logits
```

بعد ذلك، أعد تحجيم `logits` إلى حجم الصورة الأصلي وطبّق `argmax` على بُعد الفئات:
```py
>>> logits = tf.transpose(logits, [0, 2, 3, 1])

>>> upsampled_logits = tf.image.resize(
...     logits,
...     # We reverse the shape of `image` because `image.size` returns width and height.
...     image.size[::-1],
... )

>>> pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

</tf>
</frameworkcontent>

لعرض النتائج بصريًا، حمّل [لوحة ألوان مجموعة البيانات](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) كدالة `ade_palette()` التي تُطابق كل فئة بقيم RGB:

```py
def ade_palette():
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])
```

بعد ذلك يمكنك دمج صورتك وخريطة التجزئة المتوقعة وعرضهما:

```py
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1]  # convert to BGR

>>> img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png" alt="Image of bedroom overlaid with segmentation map"/>
</div>
