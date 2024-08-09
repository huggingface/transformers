# تصنيف الصور

[[open-in-colab]]

<Youtube id="tjAIM7BOYhw"/>

يقوم تصنيف الصور بتعيين تسمية أو فئة لصورة. على عكس تصنيف النص أو الصوت، تكون المدخلات هي قيم البكسل التي تتكون منها الصورة. هناك العديد من التطبيقات لتصنيف الصور، مثل الكشف عن الأضرار بعد كارثة طبيعية، أو مراقبة صحة المحاصيل، أو المساعدة في فحص الصور الطبية للبحث عن علامات المرض.

يوضح هذا الدليل كيفية:

1. ضبط نموذج [ViT](model_doc/vit) الدقيق على مجموعة بيانات [Food-101](https://huggingface.co/datasets/food101) لتصنيف عنصر غذائي في صورة.
2. استخدام نموذجك الدقيق للاستنتاج.

<Tip>

لرؤية جميع البنى ونقاط التفتيش المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/image-classification)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات Food-101

ابدأ بتحميل جزء فرعي أصغر من مجموعة بيانات Food-101 من مكتبة Datasets 🤗. سيعطيك هذا فرصة لتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

قسِّم مجموعة البيانات إلى مجموعتين فرعيتين للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> food = food.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

يحتوي كل مثال في مجموعة البيانات على حقلين:

- `image`: صورة PIL لصنف الطعام
- `label`: فئة التسمية لصنف الطعام

ولتسهيل الأمر على النموذج للحصول على اسم التسمية من معرف التسمية، قم بإنشاء قاموس يقوم بتعيين اسم التسمية إلى رقم صحيح والعكس صحيح:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

الآن يمكنك تحويل معرف التسمية إلى اسم التسمية:

```py
>>> id2label[str(79)]
'prime_rib'
```

## معالجة مسبقة

الخطوة التالية هي تحميل معالج صور ViT لمعالجة الصورة إلى Tensor:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

<frameworkcontent>
<pt>
تطبيق بعض التحولات على الصور لجعل النموذج أكثر قوة ضد الإفراط في التكيّف. هنا ستستخدم وحدة [`transforms`](https://pytorch.org/vision/stable/transforms.html) من torchvision، ولكن يمكنك أيضًا استخدام أي مكتبة صور تفضلها.

اقتص جزءًا عشوائيًا من الصورة، وقم بتغيير حجمها، وقم بتطبيعها باستخدام متوسط الصورة والانحراف المعياري:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
Crop a random part of the image, resize it, and normalize it with the image mean and standard deviation:

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

ثم قم بإنشاء دالة معالجة مسبقة لتطبيق التحولات وإرجاع `pixel_values` - المدخلات إلى النموذج - للصورة:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.with_transform`] من مكتبة 🤗 Datasets. يتم تطبيق التحولات أثناء التنقل عند تحميل عنصر من مجموعة البيانات:

```py
>>> food = food.with_transform(transforms)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DefaultDataCollator`]. على عكس برامج الجمع الأخرى للبيانات في مكتبة 🤗 Transformers، لا يطبق `DefaultDataCollator` معالجة مسبقة إضافية مثل التوسيد.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

لتجنب الإفراط في التكيّف وجعل النموذج أكثر قوة، أضف بعض التعزيزات للبيانات إلى الجزء التدريبي من مجموعة البيانات.
هنا نستخدم طبقات المعالجة المسبقة من Keras لتحديد التحولات لبيانات التدريب (بما في ذلك تعزيز البيانات)،
والتحولات لبيانات التحقق (الاقتصاص المركزي فقط، تغيير الحجم والتطبيع). يمكنك استخدام `tf.image` أو
أي مكتبة أخرى تفضلها.

```py
>>> from tensorflow import keras
>>> from tensorflow.keras import layers

>>> size = (image_processor.size["height"], image_processor.size["width"])

>>> train_data_augmentation = keras.Sequential(
...     [
...         layers.RandomCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...         layers
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

بعد ذلك، قم بإنشاء وظائف لتطبيق التحولات المناسبة على دفعة من الصور، بدلاً من صورة واحدة في كل مرة.

```py
>>> import numpy as np
>>> import tensorflow as tf
>>> from PIL import Image


>>> def convert_to_tf_tensor(image: Image):
...     np_image = np.array(image)
...     tf_image = tf.convert_to_tensor(np_image)
...     # `expand_dims()` is used to add a batch dimension since
...     # the TF augmentation layers operates on batched inputs.
...     return tf.expand_dims(tf_image, 0)


>>> def preprocess_train(example_batch):
...     """Apply train_transforms across a batch."""
...     images = [
...         train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch


... def preprocess_val(example_batch):
...     """Apply val_transforms across a batch."""
...     images = [
...         val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch
```

استخدم طريقة [`~datasets.Dataset.set_transform`] من مكتبة 🤗 Datasets لتطبيق التحولات أثناء التنقل:
Use 🤗 Datasets [`~datasets.Dataset.set_transform`] to apply the transformations on the fly:

```py
food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)
```

كخطوة معالجة مسبقة نهائية، قم بإنشاء دفعة من الأمثلة باستخدام `DefaultDataCollator`. على عكس برامج الجمع الأخرى للبيانات في مكتبة 🤗 Transformers، فإن
`DefaultDataCollator` لا يطبق معالجة مسبقة إضافية، مثل التوسيد.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل
مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

وظيفتك `compute_metrics` جاهزة الآن، وستعود إليها عندما تقوم بإعداد تدريبك.

## تدريب

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام [`Trainer`، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل ViT باستخدام [`AutoModelForImageClassification`]. حدد عدد التسميات إلى جانب عدد التسميات المتوقعة، وخرائط التسميات:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

في هذه المرحلة، هناك ثلاث خطوات فقط:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. من المهم ألا تقوم بإزالة الأعمدة غير المستخدمة لأن ذلك سيؤدي إلى إسقاط عمود "الصورة". بدون عمود "الصورة"، لا يمكنك إنشاء "pixel_values". قم بتعيين `remove_unused_columns=False` لمنع هذا السلوك! المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد أين يتم حفظ نموذجك. ستقوم بالدفع بهذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم الدقة وحفظ نقطة تفتيش التدريب.
2. قم بتمرير فرط معلمات التدريب إلى [`Trainer`] إلى جانب النموذج ومجموعة البيانات ومعالج الرموز ومبرمج البيانات ووظيفة `compute_metrics`.
3. استدعاء [`~Trainer.train`] لضبط نموذجك بشكل دقيق.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     eval_strategy="epoch"،
...     save_strategy="epoch"،
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy"،
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"]
...     eval_dataset=food["test"]
...     tokenizer=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`]:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

<Tip>

إذا كنت غير معتاد على ضبط نموذج باستخدام Keras، فراجع [البرنامج التعليمي الأساسي](./training#train-a-tensorflow-model-with-keras) أولاً!

</Tip>

لضبط نموذج في TensorFlow، اتبع الخطوات التالية:
1. حدد فرط معلمات التدريب، وقم بإعداد مثاليًا ومعدل تعلم جدول.
2. قم بتحميل نموذج مسبق التدريب.
3. قم بتحويل مجموعة بيانات 🤗 إلى تنسيق `tf.data.Dataset`.
4. قم بتجميع نموذجك.
5. أضف استدعاءات الإرجاع واستخدم طريقة `fit()` لتشغيل التدريب.
6. قم بتحميل نموذجك إلى 🤗 Hub لمشاركته مع المجتمع.

ابدأ بتحديد فرط معلماتك، ومثاليًا ومعدل التعلم:

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

بعد ذلك، قم بتحميل ViT باستخدام [`TFAutoModelForImageClassification`] إلى جانب خرائط التسميات:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~datasets.Dataset.to_tf_dataset`] وبرنامج التجميع الخاص بك `data_collator`:

```py
>>> # converting our train dataset to tf.data.Dataset
>>> tf_train_dataset = food["train"].to_tf_dataset(
...     columns="pixel_values"، label_cols="label"، shuffle=True، batch_size=batch_size، collate_fn=data_collator
... )

>>> # converting our test dataset to tf.data.Dataset
>>> tf_eval_dataset = food["test"].to_tf_dataset(
...     columns="pixel_values"، label_cols="label"، shuffle=True، batch_size=batch_size، collate_fn=data_collator
... )
```

قم بتكوين النموذج للتدريب باستخدام `compile()`


```py
>>> from tensorflow.keras.losses import SparseCategoricalCrossentropy

>>> loss = tf.keras.lossetrops.SparseCategoricalCrosseny(from_logits=True)
>>> model.compile(optimizer=optimizer, loss=loss)
```

لحساب الدقة من التوقعات ودفع نموذجك إلى 🤗 Hub، استخدم [Keras callbacks](../main_classes/keras_callbacks).
مرر دالتك `compute_metrics` إلى [KerasMetricCallback](../main_classes/keras_callbacks#transformers.KerasMetricCallback)،
واستخدم [PushToHubCallback](../main_classes/keras_callbacks#transformers.PushToHubCallback) لتحميل النموذج:

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

أخيرًا، أنت مستعد لتدريب نموذجك! استدع `fit()` باستخدام مجموعات البيانات التدريبية والتحقق من الصحة الخاصة بك، وعدد العصور،
والاستدعاءات الخاصة بك لضبط نموذجك بدقة:

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

تهانينا! لقد قمت بضبط نموذجك بدقة ومشاركته على 🤗 Hub. يمكنك الآن استخدامه للاستنتاج!
</tf>
</frameworkcontent>


<Tip>

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج لتصنيف الصور، راجع دفتر ملاحظات PyTorch المقابل [هنا](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

</Tip>

## الاستنتاج

رائع، الآن بعد أن قمت بضبط نموذج بدقة، يمكنك استخدامه للاستنتاج!

قم بتحميل صورة تريد تشغيل الاستنتاج عليها:

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="صورة beignets"/>
</div>

أبسط طريقة لتجربة نموذجك المضبوط الدقيق للاستنتاج هي استخدامه في [`pipeline`]. قم بتنفيذ مثيل `pipeline` لتصنيف الصور باستخدام نموذجك، ومرر صورتك إليه:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224، 'label': 'beignets'}،
 {'score': 0.015232225880026817، 'label': 'bruschetta'}،
 {'score': 0.01519392803311348، 'label': 'chicken_wings'}،
 {'score': 0.013022331520915031، 'label': 'pork_chop'}،
 {'score': 0.012728818692266941، 'label': 'prime_rib'}]
```

يمكنك أيضًا محاكاة نتائج `pipeline` يدويًا إذا أردت:

<frameworkcontent>
<pt>
قم بتحميل معالج الصور لمعالجة الصورة وإرجاع `input` كرموز تعبيرية PyTorch:

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

مرر المدخلات إلى النموذج وأعد الخرجات:

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على التسمية المتوقعة مع أعلى احتمال، واستخدم تعيين `id2label` للنموذج لتحويله إلى تسمية:

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
قم بتحميل معالج الصور لمعالجة الصورة وإرجاع `input` كرموز تعبيرية TensorFlow:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
>>> inputs = image_processor(image, return_tensors="tf")
```

مرر المدخلات إلى النموذج وأعد الخرجات:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
>>> logits = model(**inputs).logits
```

احصل على التسمية المتوقعة مع أعلى احتمال، واستخدم تعيين `id2label` للنموذج لتحويله إلى تسمية:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'beignets'
```

</tf>
</frameworkcontent>