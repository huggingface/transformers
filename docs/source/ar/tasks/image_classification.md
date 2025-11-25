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

# تصنيف الصور (Image classification)

[[open-in-colab]]

<Youtube id="tjAIM7BOYhw"/>

يُسنِد تصنيف الصور وسمًا أو فئة للصورة. على عكس تصنيف النص أو الصوت، تكون المدخلات هي قيم البكسلات التي تُكوِّن الصورة. هناك العديد من الاستخدامات لتصنيف الصور، مثل اكتشاف الأضرار بعد الكوارث الطبيعية، ومراقبة صحة المحاصيل، أو المساعدة في فحص الصور الطبية بحثًا عن مؤشرات المرض.

يوضح هذا الدليل كيفية:

1. ضبط نموذج [ViT](../model_doc/vit) على مجموعة بيانات [Food-101](https://huggingface.co/datasets/food101) لتصنيف عنصر غذائي في صورة.
2. استخدام النموذج المضبوط للاستدلال.

<Tip>

للاطلاع على جميع البنى ونقاط التفتيش المتوافقة مع هذه المهمة، نوصي بزيارة [صفحة المهمة](https://huggingface.co/tasks/image-classification)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات اللازمة:

```bash
pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn
```

نوصيك بتسجيل الدخول إلى حسابك على Hugging Face لرفع نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات Food-101

ابدأ بتحميل جزء أصغر من مجموعة بيانات Food-101 من مكتبة 🤗 Datasets. سيمنحك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء وقت أطول في التدريب على المجموعة الكاملة.

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

قسّم جزء `train` إلى مجموعتي تدريب واختبار باستخدام التابع [`~datasets.Dataset.train_test_split`]:

```py
>>> food = food.train_test_split(test_size=0.2)
```

ثم ألقِ نظرة على مثال:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

كل مثال في المجموعة يحتوي على حقلين:

- `image`: صورة PIL للعنصر الغذائي
- `label`: الفئة التصنيفية للعنصر الغذائي

لتسهيل حصول النموذج على اسم الفئة من معرّف الفئة، أنشئ قاموسًا يُحوِّل اسم الفئة إلى عدد صحيح والعكس:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

الآن يمكنك تحويل معرّف الفئة إلى اسم الفئة:

```py
>>> id2label[str(79)]
'prime_rib'
```

## المعالجة المسبقة

الخطوة التالية هي تحميل معالج صور ViT لمعالجة الصورة إلى موتر:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

<frameworkcontent>
<pt>
طبّق بعض تحويلات الصور لجعل النموذج أكثر قوة ضد فرط التكيّف. هنا سنستخدم وحدة [`transforms`](https://pytorch.org/vision/stable/transforms.html) من torchvision، ولكن يمكنك أيضًا استخدام أي مكتبة صور تفضلها.

قص جزءًا عشوائيًا من الصورة، ثم غيّر الحجم وطبّعها بمتوسط الصورة وانحرافها المعياري:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

ثم أنشئ دالة معالجة مسبقة لتطبيق التحويلات وإرجاع `pixel_values` — وهي مدخلات النموذج — للصورة:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

لتطبيق دالة المعالجة على كامل المجموعة، استخدم تابع 🤗 Datasets [`~datasets.Dataset.with_transform`]. تُطبّق التحويلات عند الطلب عند تحميل عنصر من المجموعة:

```py
>>> food = food.with_transform(transforms)
```

الآن أنشئ دفعة أمثلة باستخدام [`DefaultDataCollator`]. على عكس مجمّعات البيانات الأخرى في 🤗 Transformers، لا يقوم `DefaultDataCollator` بتطبيق معالجة إضافية مثل الحشو.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

لتجنب فرط التكيّف وجعل النموذج أكثر قوة، أضِف بعض تعزيزات البيانات إلى جزء التدريب من المجموعة. هنا نستخدم طبقات المعالجة المسبقة في Keras لتعريف التحويلات لبيانات التدريب (بما يشمل التعزيز)، والتحويلات لبيانات التحقق (اقتصاص مركزي وتغيير حجم وتطبيع فقط). يمكنك استخدام `tf.image` أو أي مكتبة أخرى تفضّلها.

```py
>>> from tensorflow import keras
>>> from tensorflow.keras import layers

>>> size = (image_processor.size["height"], image_processor.size["width"])

>>> train_data_augmentation = keras.Sequential(
...     [
...         layers.RandomCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...         layers.RandomFlip("horizontal"),
...         layers.RandomRotation(factor=0.02),
...         layers.RandomZoom(height_factor=0.2, width_factor=0.2),
...     ],
...     name="train_data_augmentation",
... )

>>> val_data_augmentation = keras.Sequential(
...     [
...         layers.CenterCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...     ],
...     name="val_data_augmentation",
... )
```

بعد ذلك، أنشئ دوالًا لتطبيق التحويلات المناسبة على دفعة من الصور، بدلًا من صورة واحدة في كل مرة.

```py
>>> import numpy as np
>>> import tensorflow as tf
>>> from PIL import Image


>>> def convert_to_tf_tensor(image: Image):
...     np_image = np.array(image)
...     tf_image = tf.convert_to_tensor(np_image)
...     # تُستخدم `expand_dims()` لإضافة بُعد الدفعة لأن
...     # طبقات التعزيز في TF تعمل على مدخلات مُجمّعة.
...     return tf.expand_dims(tf_image, 0)


>>> def preprocess_train(example_batch):
...     """تطبيق تحويلات التدريب على دفعة."""
...     images = [
...         train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch


... def preprocess_val(example_batch):
...     """تطبيق تحويلات التحقق على دفعة."""
...     images = [
...         val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch
```

استخدم [`~datasets.Dataset.set_transform`] من 🤗 Datasets لتطبيق التحويلات عند الطلب:

```py
>>> food["train"].set_transform(preprocess_train)
>>> food["test"].set_transform(preprocess_val)
```

كخطوة معالجة أخيرة، أنشئ دفعة أمثلة باستخدام `DefaultDataCollator`. على عكس مجمّعات البيانات الأخرى في 🤗 Transformers، لا يطبّق `DefaultDataCollator` أي معالجة إضافية مثل الحشو.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## التقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء النموذج. يمكنك بسرعة تحميل طريقة تقييم باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، حمّل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (اطلع على [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية التحميل والحساب):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم أنشئ دالة تمُرِّر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

دالة `compute_metrics` جاهزة الآن، وسنعود إليها عند إعداد التدريب.

## التدريب

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`]، ألقِ نظرة على الدليل الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أصبحت جاهزًا لبدء تدريب نموذجك الآن! حمّل ViT باستخدام [`AutoModelForImageClassification`]. حدّد عدد الملصقات مع عدد الفئات المتوقعة، وخرائط الفئات:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

في هذه المرحلة، تبقى ثلاث خطوات فقط:

1. عرّف فرطّيات التدريب في [`TrainingArguments`]. من المهم ألّا تزيل الأعمدة غير المستخدمة لأن ذلك سيحذف عمود `image`. من دون عمود `image` لا يمكنك إنشاء `pixel_values`. عيّن `remove_unused_columns=False` لتجنب هذا السلوك! المعامل الوحيد المطلوب الآخر هو `output_dir` الذي يحدد مكان حفظ نموذجك. سنرفع هذا النموذج إلى Hub عبر تعيين `push_to_hub=True` (يجب أن تكون مسجلًا في Hugging Face لرفع نموذجك). في نهاية كل عهدة، سيقيِّم [`Trainer`] الدقة ويحفظ نقطة التحقق.
2. مرّر معاملات التدريب إلى [`Trainer`] مع النموذج والمجموعة والمعالج ومجمّع البيانات ودالة `compute_metrics`.
3. استدعِ [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بعد اكتمال التدريب، شارك نموذجك على Hub باستخدام التابع [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدامه:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، اطلع أولًا على [الدليل الأساسي](./training#train-a-tensorflow-model-with-keras)!

</Tip>

لضبط نموذج في TensorFlow، اتبع الخطوات التالية:
1. عرّف فرطّيات التدريب، واضبط المُحسِّن وجدول معدل التعلم.
2. استدعِ نموذجًا مُدرَّبًا مسبقًا.
3. حوِّل مجموعة 🤗 Dataset إلى `tf.data.Dataset`.
4. اجمّع نموذجك.
5. أضِف الاستدعاءات (callbacks) واستخدم `fit()` لتشغيل التدريب.
6. ارفع نموذجك إلى 🤗 Hub لمشاركته مع المجتمع.

ابدأ بتعريف الفرطّيات والمُحسِّن وجدول معدل التعلم:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 5
>>> num_train_steps = len(food["train"]) * num_epochs
>>> learning_rate = 3e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

بعد ذلك، حمّل ViT باستخدام [`TFAutoModelForImageClassification`] مع خرائط الفئات:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
```

حوِّل مجموعتي البيانات إلى صيغة `tf.data.Dataset` باستخدام [`~datasets.Dataset.to_tf_dataset`] و`data_collator`:

```py
>>> # تحويل مجموعة التدريب إلى tf.data.Dataset
>>> tf_train_dataset = food["train"].to_tf_dataset(
...     columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
... )

>>> # تحويل مجموعة الاختبار إلى tf.data.Dataset
>>> tf_eval_dataset = food["test"].to_tf_dataset(
...     columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
... )
```

هيّئ النموذج للتدريب باستخدام `compile()`:

```py
>>> from tensorflow.keras.losses import SparseCategoricalCrossentropy

>>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
>>> model.compile(optimizer=optimizer, loss=loss)
```

لحساب الدقة من التنبؤات ورفع نموذجك إلى 🤗 Hub، استخدم [استدعاءات Keras](../main_classes/keras_callbacks). مرّر دالتك `compute_metrics` إلى [KerasMetricCallback](../main_classes/keras_callbacks#transformers.KerasMetricCallback)، واستخدم [PushToHubCallback](../main_classes/keras_callbacks#transformers.PushToHubCallback) لرفع النموذج:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="food_classifier",
...     tokenizer=image_processor,
...     save_strategy="no",
... )
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أصبحت جاهزًا لتدريب النموذج! استدعِ `fit()` مع مجموعتي التدريب والتحقق، وعدد العُهَد، والاستدعاءات الخاصة بك لضبط النموذج:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
Epoch 1/5
250/250 [==============================] - 313s 1s/step - loss: 2.5623 - val_loss: 1.4161 - accuracy: 0.9290
Epoch 2/5
250/250 [==============================] - 265s 1s/step - loss: 0.9181 - val_loss: 0.6808 - accuracy: 0.9690
Epoch 3/5
250/250 [==============================] - 252s 1s/step - loss: 0.3910 - val_loss: 0.4303 - accuracy: 0.9820
Epoch 4/5
250/250 [==============================] - 251s 1s/step - loss: 0.2028 - val_loss: 0.3191 - accuracy: 0.9900
Epoch 5/5
250/250 [==============================] - 238s 949ms/step - loss: 0.1232 - val_loss: 0.3259 - accuracy: 0.9890
```

تهانينا! لقد ضبطت نموذجك وشاركته على 🤗 Hub. يمكنك الآن استخدامه للاستدلال!
</tf>
</frameworkcontent>


<Tip>

للحصول على مثال أكثر تفصيلًا حول كيفية ضبط نموذج لتصنيف الصور، ألقِ نظرة على [دفتر بايثورش](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) المقابل.

</Tip>

## الاستدلال

رائع! الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستدلال.

حمّل صورة ترغب بتشغيل الاستدلال عليها:

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="صورة لبيجنِه"/>
</div>

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه ضمن [`pipeline`]. أنشئ `pipeline` لتصنيف الصور باستخدام نموذجك، ثم مرّر إليه صورتك:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا رغبت:

<frameworkcontent>
<pt>
حمّل معالج صور لمعالجة الصورة وأعد `input` كموترات PyTorch:

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

مرّر المدخلات إلى النموذج وأعد القيم اللوجيتية:

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على الفئة المتنبأ بها ذات الاحتمالية الأعلى، واستخدم خريطة `id2label` في النموذج لتحويلها إلى وسم:

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
حمّل معالج صور لمعالجة الصورة وأعد `input` كموترات TensorFlow:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
>>> inputs = image_processor(image, return_tensors="tf")
```

مرّر المدخلات إلى النموذج وأعد القيم اللوجيتية:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
>>> logits = model(**inputs).logits
```

احصل على الفئة المتنبأ بها ذات الاحتمالية الأعلى، واستخدم خريطة `id2label` في النموذج لتحويلها إلى وسم:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'beignets'
```

</tf>
</frameworkcontent>
