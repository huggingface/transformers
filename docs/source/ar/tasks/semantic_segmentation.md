# تجزئة الصور

[[open-in-colab]]

<Youtube id="dKE8SIt9C-w"/>

تفصل نماذج تجزئة الصور المناطق التي تتوافق مع مناطق مختلفة ذات أهمية في صورة. تعمل هذه النماذج عن طريق تعيين تسمية لكل بكسل. هناك عدة أنواع من التجزئة: التجزئة الدلالية، وتجزئة المثيل، وتجزئة المشهد الكلي.

في هذا الدليل، سوف:

1. [إلقاء نظرة على الأنواع المختلفة من التجزئة](#أنواع-التجزئة).
2. [لديك مثال شامل لضبط دقيق لنموذج التجزئة الدلالية](#ضبط-نموذج-بشكل-دقيق-للتجزئة).

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية
!pip install -q datasets transformers evaluate accelerate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## أنواع التجزئة

تعين التجزئة الدلالية تسمية أو فئة لكل بكسل في الصورة. دعونا نلقي نظرة على إخراج نموذج التجزئة الدلالية. سوف يعين نفس الفئة لكل مثيل من كائن يصادفه في صورة، على سبيل المثال، سيتم تصنيف جميع القطط على أنها "قطة" بدلاً من "cat-1"، "cat-2".
يمكننا استخدام خط أنابيب تجزئة الصور في المحولات للتنبؤ بسرعة بنموذج التجزئة الدلالية. دعونا نلقي نظرة على صورة المثال.

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

سنستخدم [nvidia/segformer-b1-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024).

```python
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)
results
```
```python
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)
results
```

يشمل إخراج خط أنابيب التجزئة قناعًا لكل فئة متوقعة.
```bash
[{'score': None,
  'label': 'road',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'sidewalk',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'building',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'pole',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'traffic sign',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'vegetation',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'terrain',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=612x414>},
 {'score': None,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x414>}
]
```

عند النظر إلى القناع لفئة السيارات، يمكننا أن نرى أن كل سيارة مصنفة بنفس القناع.

```python
results[-1]["mask"]
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/semantic_segmentation_output.png" alt="Semantic Segmentation Output"/>
</div>

في تجزئة المثيل، الهدف ليس تصنيف كل بكسل، ولكن التنبؤ بقناع لكل **مثيل كائن** في صورة معينة. إنه يعمل بشكل مشابه جدًا للكشف عن الأشياء، حيث يوجد مربع محدد لكل مثيل، وهناك قناع تجزئة بدلاً من ذلك. سنستخدم [facebook/mask2former-swin-large-cityscapes-instance](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-instance) لهذا.

```python
instance_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-instance")
results = instance_segmentation(image)
results
```

كما ترون أدناه، هناك العديد من السيارات المصنفة، ولا توجد تصنيفات للبكسلات بخلاف البكسلات التي تنتمي إلى سيارات وأشخاص.

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
  'mask': <PIL.Image.Image image mode=L size=612x415>}
]
```
التحقق من إحدى أقنعة السيارات أدناه.

```python
results[2]["mask"]
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/instance_segmentation_output.png" alt="Semantic Segmentation Output"/>
</div>

تدمج التجزئة المشهدية التجزئة الدلالية وتجزئة المثيل، حيث يتم تصنيف كل بكسل إلى فئة ومثيل من تلك الفئة، وهناك أقنعة متعددة لكل مثيل من فئة. يمكننا استخدام [facebook/mask2former-swin-large-cityscapes-panoptic](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) لهذا.

```python
panoptic_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-panoptic")
results = panoptic_segmentation(image)
results
```
كما ترون أدناه، لدينا المزيد من الفئات. سنقوم لاحقًا بتوضيح أن كل بكسل مصنف إلى واحدة من الفئات.
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
 
...
```

دعونا نقارن جنبا إلى جنب لجميع أنواع التجزئة.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation-comparison.png" alt="Segmentation Maps Compared"/>
</div>

بعد رؤية جميع أنواع التجزئة، دعونا نتعمق في ضبط نموذج التجزئة الدلالية.

تشمل التطبيقات الواقعية الشائعة للتجزئة الدلالية تدريب السيارات ذاتية القيادة على التعرف على المشاة ومعلومات المرور المهمة، وتحديد الخلايا والتشوهات في الصور الطبية، ومراقبة التغيرات البيئية من صور الأقمار الصناعية.

## ضبط نموذج بشكل دقيق لتجزئة

سنقوم الآن بما يلي:

1. ضبط دقيق [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) على مجموعة البيانات [SceneParse150](https://huggingface.co/datasets/scene_parse_150).
2. استخدام نموذجك المضبوط بدقة للاستنتاج.

<Tip>

لمعرفة جميع التصميمات ونقاط التفتيش المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/image-segmentation)

</Tip>


### تحميل مجموعة بيانات SceneParse150

ابدأ بتحميل جزء فرعي أصغر من مجموعة بيانات SceneParse150 من مكتبة Datasets. سيعطيك هذا فرصة لتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

قم بتقسيم مجموعة البيانات "train" إلى مجموعة تدريب واختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

ثم الق نظرة على مثال:

```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}

# عرض الصورة
>>> train_ds[0]["image"]
```

- `image`: صورة PIL للمشهد.
- `annotation`: صورة PIL لخريطة التجزئة، والتي تعد أيضًا هدف النموذج.
- `scene_category`: معرف فئة يصف فئة المشهد مثل "المطبخ" أو "المكتب". في هذا الدليل، ستحتاج فقط إلى `image` و`annotation`، وكلاهما عبارة عن صور PIL.

ستحتاج أيضًا إلى إنشاء قاموس يقوم بتعيين معرف التسمية إلى فئة التسمية والتي ستكون مفيدة عند إعداد النموذج لاحقًا. قم بتنزيل التعيينات من Hub وإنشاء القواميس `id2label` و`label2id`:

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

#### مجموعة بيانات مخصصة
#### مجموعة بيانات مخصصة

يمكنك أيضًا إنشاء واستخدام مجموعة البيانات الخاصة بك إذا كنت تفضل التدريب باستخدام النص البرمجي [run_semantic_segmentation.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_semantic_segmentation.py) بدلاً من مثيل دفتر الملاحظات. يتطلب النص البرمجي ما يلي:

1. [`~datasets.DatasetDict`] مع عمودين [`~datasets.Image`]، "image" و"label"

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

     # الخطوة 1: إنشاء كائنات Dataset
     train_dataset = create_dataset(image_paths_train, label_paths_train)
     validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

     # الخطوة 2: إنشاء DatasetDict
     dataset = DatasetDict({
          "train": train_dataset,
          "validation": validation_dataset,
          }
     )

     # الخطوة 3: الدفع إلى Hub (يفترض أنك قمت بتشغيل الأمر huggingface-cli login في terminal/notebook)
     dataset.push_to_hub("your-name/dataset-repo")

     # بشكل اختياري، يمكنك الدفع إلى مستودع خاص على Hub
     # dataset.push_to_hub("name of repo on the hub", private=True)
     ```

2. قاموس id2label الذي يقوم بتعيين أعداد صحيحة للفئة إلى أسماء فئاتها

     ```py
     import json
     # مثال بسيط
     id2label = {0: 'cat', 1: 'dog'}
     with open('id2label.json', 'w') as fp:
     json.dump(id2label, fp)
     ```

كمثال، الق نظرة على [مجموعة البيانات هذه](https://huggingface.co/datasets/nielsr/ade20k-demo) التي تم إنشاؤها بالخطوات الموضحة أعلاه.

### معالجة مسبقة

الخطوة التالية هي تحميل معالج صور SegFormer لإعداد الصور والتعليقات التوضيحية للنموذج. تستخدم بعض مجموعات البيانات، مثل هذه، الفهرس الصفري كفئة خلفية. ومع ذلك، فإن فئة الخلفية ليست مدرجة بالفعل في 150 فئة، لذلك ستحتاج إلى تعيين `do_reduce_labels=True` لطرح واحد من جميع التسميات. يتم استبدال الفهرس الصفري بـ `255` حتى يتم تجاهله بواسطة دالة الخسارة SegFormer:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)
```

<frameworkcontent>
<pt>

من الشائع تطبيق بعض عمليات زيادة البيانات على مجموعة بيانات الصور لجعل النموذج أكثر قوة ضد الإفراط في التخصيص. في هذا الدليل، ستستخدم دالة [`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) من [torchvision](https://pytorch.org/vision/stable/index.html) لتغيير الخصائص اللونية للصورة بشكل عشوائي، ولكن يمكنك أيضًا استخدام أي مكتبة صور تفضلها.

```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

الآن قم بإنشاء دالتين لمعالجة مسبقة
بالتأكيد، سأقوم بترجمة النص مع اتباع التعليمات التي قدمتها.

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

بعد الانتهاء من التدريب، شارك نموذجك على Hub باستخدام طريقة [~transformers.Trainer.push_to_hub] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع [البرنامج التعليمي الأساسي](./training#train-a-tensorflow-model-with-keras) أولاً!

</Tip>

لضبط نموذج في TensorFlow، اتبع الخطوات التالية:

1. حدد فرط المعلمات التدريبية، وقم بإعداد محسن وجدول معدل التعلم.
2. قم بتنفيذ مثيل لنموذج مسبق التدريب.
3. قم بتحويل مجموعة بيانات 🤗 إلى `tf.data.Dataset`.
4. قم بتجميع نموذجك.
5. أضف استدعاءات رجوع لحساب المقاييس وتحميل نموذجك إلى 🤗 Hub
6. استخدم طريقة `fit()` لتشغيل التدريب.

ابدأ بتحديد فرط المعلمات والمحسن وجدول معدل التعلم:

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

بعد ذلك، قم بتحميل SegFormer مع [TFAutoModelForSemanticSegmentation] جنبًا إلى جنب مع تعيينات التسميات، وقم بتجميعها مع المحسن. لاحظ أن جميع نماذج Transformers تحتوي على دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
>>> model.compile(optimizer=optimizer)  # لا توجد حجة الخسارة!
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [~datasets.Dataset.to_tf_dataset] و [DefaultDataCollator]:

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

لحساب الدقة من التوقعات وتحميل نموذجك إلى 🤗 Hub، استخدم [Keras callbacks](../main_classes/keras_callbacks).
مرر دالة `compute_metrics` إلى [KerasMetricCallback]،
واستخدم [PushToHubCallback] لتحميل النموذج:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(
...     metric_fn=compute_metrics, eval_dataset=tf_eval_dataset, batch_size=batch_size, label_cols=["labels"]
... )

>>> push_to_hub_callback = PushToHubCallback(output_dir="scene_segmentation", tokenizer=image_processor)

>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لتدريب نموذجك! اتصل بـ `fit()` باستخدام مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور،
واستدعاءات الرجوع الخاصة بك لضبط النموذج:
أخيرًا، أنت مستعد لتدريب نموذجك! اتصل بـ `fit()` باستخدام مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور،
واستدعاءات الرجوع الخاصة بك لضبط النموذج:

```py
>>> model.fit(
...     tf_train_dataset,
...     validation_data=tf_eval_dataset,
...     callbacks=callbacks,
...     epochs=num_epochs,
... )
```

تهانينا! لقد ضبطت نموذجك وشاركته على 🤗 Hub. يمكنك الآن استخدامه للاستنتاج!
</tf>
</frameworkcontent>

### الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

أعد تحميل مجموعة البيانات وتحميل صورة للاستدلال.

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
>>> ds = ds.train_test_split(test_size=0.2)
>>> test_ds = ds["test"]
>>> image = ds["test"][0]["image"]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="صورة لغرفة نوم"/>
</div>

<frameworkcontent>
<pt>

سنرى الآن كيفية الاستدلال بدون خط أنابيب. قم بمعالجة الصورة بمعالج الصور وقم بوضع `pixel_values` على GPU:

```py
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # استخدم GPU إذا كان متاحًا، وإلا استخدم CPU
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

مرر المدخلات إلى النموذج وأعد الخرج `logits`:

```py
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

بعد ذلك، قم بإعادة تحجيم logits إلى حجم الصورة الأصلي:

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
قم بتحميل معالج الصور لمعالجة الصورة وإعادة الإدخال كتناسق TensorFlow:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
>>> inputs = image_processor(image, return_tensors="tf")
```

مرر المدخلات إلى النموذج وأعد الخرج `logits`:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
>>> logits = model(**inputs).logits
```

بعد ذلك، قم بإعادة تحجيم logits إلى حجم الصورة الأصلي وقم بتطبيق argmax على البعد class:
```py
>>> logits = tf.transpose(logits, [0, 2, 3, 1])

>>> upsampled_logits = tf.image.resize(
...     logits,
...     # نعكس شكل `image` لأن `image.size` يعيد العرض والارتفاع.
...     image.size[::-1],
... )

>>> pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

</tf>
</frameworkcontent>

لعرض النتائج، قم بتحميل لوحة الألوان الخاصة بمجموعة البيانات [dataset color palette](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) كـ `ade_palette()` التي تقوم بتعيين كل فئة إلى قيم RGB الخاصة بها.

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

Then you can combine and plot your image and the predicted segmentation map:

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