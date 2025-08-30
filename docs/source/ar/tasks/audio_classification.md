<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# تصنيف الصوت (Audio Classification)

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

تصنيف الصوت — تمامًا مثل النص — يُسند تسمية فئوية (class label) كمُخرج انطلاقًا من بيانات الإدخال. الفارق الوحيد هو أنه بدلًا من مُدخلات نصية، لديك أشكال موجية صوتية خام. من التطبيقات العملية لتصنيف الصوت: تحديد نية المتحدث، وتصنيف اللغة، وحتى تمييز أنواع الحيوانات من خلال أصواتها.

سيُرشدك هذا الدليل إلى كيفية:

1. إجراء ضبط دقيق (fine-tuning) لـ [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) على مجموعة البيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) لتصنيف نية المتحدث.
2. استخدام نموذجك المضبوط للاستدلال (inference).

<Tip>

لمعرفة جميع البُنى ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالاطلاع على [صفحة المهمة](https://huggingface.co/tasks/audio-classification).

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نوصيك بتسجيل الدخول إلى حسابك على Hugging Face حتى تتمكن من رفع نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل الرمز المميز لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة البيانات MInDS-14

ابدأ بتحميل مجموعة البيانات MInDS-14 من مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

قسّم جزء `train` من مجموعة البيانات إلى مجموعة تدريب واختبار أصغر باستخدام الطريقة [`~datasets.Dataset.train_test_split`]. سيمنحك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت على المجموعة الكاملة.

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

ثم ألقِ نظرة على مجموعة البيانات:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

بينما تحتوي مجموعة البيانات على الكثير من المعلومات المفيدة، مثل `lang_id` و`english_transcription`، ستركّز في هذا الدليل على `audio` و`intent_class`. أزِل الأعمدة الأخرى باستخدام الطريقة [`~datasets.Dataset.remove_columns`]:

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

إليك مثالًا:

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

هناك حقلان:

- `audio`: مصفوفة أحادية البُعد `array` للإشارة الصوتية يجب استدعاؤها لتحميل ملف الصوت وإعادة تشكيل معدل العيّنة.
- `intent_class`: يمثّل معرّف الفئة (class id) الخاص بنيّة المتحدث.

لتسهيل حصول النموذج على اسم التسمية (label name) من معرّف التسمية (label id)، أنشئ قاموسًا يربط اسم التسمية بعدد صحيح والعكس صحيح:

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

يمكنك الآن تحويل معرّف التسمية إلى اسم التسمية:

```py
>>> id2label[str(2)]
'app_error'
```

## المعالجة المسبقة (Preprocess)

الخطوة التالية هي تحميل مستخرج الخصائص (feature extractor) الخاص بـ Wav2Vec2 لمعالجة الإشارة الصوتية:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

تملك مجموعة بيانات MInDS-14 معدل عيّنة 8kHz (يمكنك العثور على هذه المعلومة في [بطاقة مجموعة البيانات](https://huggingface.co/datasets/PolyAI/minds14))، ما يعني أنك ستحتاج إلى إعادة تشكيلها إلى 16kHz لاستخدام نموذج Wav2Vec2 المُدرّب مسبقًا:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

أنشئ الآن دالة للمعالجة المسبقة تقوم بما يلي:

1. تستدعي عمود `audio` لتحميل ملف الصوت، وإذا لزم الأمر، إعادة تشكيل معدل العيّنة.
2. تتحقق مما إذا كان معدل عيّنة ملف الصوت يطابق معدل عيّنة بيانات الصوت التي تم تدريب النموذج عليها مسبقًا. يمكنك العثور على هذه المعلومة في [بطاقة النموذج](https://huggingface.co/facebook/wav2vec2-base) الخاصة بـ Wav2Vec2.
3. تعيين طول إدخال أقصى (maximum input length) لتجميع مُدخلات أطول دون اقتطاعها.

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات كاملة، استخدم دالة [`~datasets.Dataset.map`] في 🤗 Datasets. يمكنك تسريع `map` بتعيين `batched=True` لمعالجة عدة عناصر دفعة واحدة. أزِل الأعمدة غير الضرورية وأعد تسمية `intent_class` إلى `label` كما يتطلب النموذج:

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## التقييم (Evaluate)

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك بسرعة تحميل طريقة تقييم عبر مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، حمّل مقياس [الدقّة (Accuracy)](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) في 🤗 Evaluate لمعرفة المزيد حول التحميل والحساب):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم أنشئ دالة تمُرّر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقّة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

أصبحت دالة `compute_metrics` جاهزة الآن، وسنعود إليها عند إعداد التدريب.

## التدريب (Train)

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن معتادًا على إجراء الضبط الدقيق لنموذج باستخدام [`Trainer`]، ألقِ نظرة على الدليل الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت الآن جاهز لبدء تدريب نموذجك! قم بتحميل Wav2Vec2 باستخدام [`AutoModelForAudioClassification`] مع عدد التسميات المتوقعة (labels) وربط التسميات (label mappings):

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

في هذه المرحلة، تبقّت ثلاث خطوات فقط:

1. عرّف فرط-معاملات التدريب (hyperparameters) في [`TrainingArguments`]. المعامل الوحيد المطلوب هو `output_dir`، والذي يحدد مكان حفظ نموذجك. ستدفع هذا النموذج إلى Hub بتعيين `push_to_hub=True` (تحتاج إلى تسجيل الدخول إلى Hugging Face لرفع نموذجك). في نهاية كل حقبة (epoch)، سيُقيّم [`Trainer`] قيمة الدقّة ويحفظ نقطة التحقق التدريبية.
2. مرّر معاملات التدريب إلى [`Trainer`] مع النموذج ومجموعة البيانات والمُعالج/الم ميّز (tokenizer/feature extractor) ودالة `compute_metrics`.
3. استدعِ [`~Trainer.train`] لإجراء الضبط الدقيق لنموذجك.


```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام الطريقة [`~transformers.Trainer.push_to_hub`] ليتمكن الجميع من استخدامه:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر تفصيلًا حول كيفية إجراء الضبط الدقيق لنموذج لتصنيف الصوت، اطّلع على [دفتر Jupyter الخاص بـ PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

</Tip>

## الاستدلال (Inference)

رائع! بعد أن أجريت الضبط الدقيق لنموذجك، يمكنك استخدامه الآن للاستدلال.

حمّل ملفًا صوتيًا لإجراء الاستدلال. تذكّر إعادة تشكيل معدل العيّنة لملف الصوت ليتوافق مع معدل العيّنة الخاص بالنموذج إذا لزم الأمر.

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

أسهل طريقة لتجربة نموذجك المضبوط من أجل الاستدلال هي استخدامه ضمن [`pipeline`]. قم بإنشاء `pipeline` خاص بتصنيف الصوت باستخدام نموذجك، ثم مرّر له ملف الصوت:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

يمكنك أيضًا إعادة تنفيذ نتائج `pipeline` يدويًا إذا رغبت بذلك:

<frameworkcontent>
<pt>
حمّل مستخرج خصائص (feature extractor) لتهيئة ملف الصوت وإرجاع `input` على شكل موترات PyTorch:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

مرّر المُدخلات إلى النموذج واسترجع القيم اللوغارية (logits):

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على الفئة ذات الاحتمالية الأعلى، ثم استخدم ربط `id2label` الخاص بالنموذج لتحويلها إلى تسمية (label):

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
</pt>
</frameworkcontent>
