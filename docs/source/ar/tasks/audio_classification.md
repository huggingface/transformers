# تصنيف الصوت

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

يصنف تصنيف الصوت - تمامًا مثل النص - تسمية فئة الإخراج من بيانات الإدخال. والفرق الوحيد هو بدلاً من إدخالات النص، لديك أشكال موجية صوتية خام. تشمل التطبيقات العملية لتصنيف الصوت التعرف على نية المتحدث، وتصنيف اللغة، وحتى الأنواع الحيوانية من أصواتها.

سيوضح لك هذا الدليل كيفية:

1. ضبط نموذج [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) الدقيق على مجموعة بيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) لتصنيف نية المتحدث.
2. استخدام نموذجك الدقيق للاستنتاج.

<Tip>

لمشاهدة جميع البنيات ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/audio-classification)

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات MInDS-14

ابدأ بتحميل مجموعة بيانات MInDS-14 من مكتبة Datasets 🤗:

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

قسِّم تقسيم "التدريب" في مجموعة البيانات إلى مجموعة تدريب واختبار أصغر باستخدام طريقة [`~datasets.Dataset.train_test_split`]. سيعطيك هذا فرصة لتجربة التأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في مجموعة البيانات الكاملة.

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

ثم الق نظرة على مجموعة البيانات:

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

في حين أن مجموعة البيانات تحتوي على الكثير من المعلومات المفيدة، مثل `lang_id` و`english_transcription`، فسوف تركز على `audio` و`intent_class` في هذا الدليل. أزل الأعمدة الأخرى باستخدام طريقة [`~datasets.Dataset.remove_columns`]:

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

الق نظرة على مثال الآن:

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

هناك حقلان:

- `audio`: مصفوفة أحادية البعد للإشارة الصوتية التي يجب استدعاؤها لتحميل وإعادة أخذ عينات ملف الصوت.
- `intent_class`: يمثل معرف فئة نية المتحدث.

ولتسهيل الأمر على النموذج للحصول على اسم التسمية من معرف التسمية، قم بإنشاء قاموس يقوم بتعيين اسم التسمية إلى عدد صحيح والعكس صحيح:

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

الآن يمكنك تحويل معرف التسمية إلى اسم التسمية:

```py
>>> id2label[str(2)]
'app_error'
```

## معالجة مسبقة

الخطوة التالية هي تحميل مستخرج ميزات Wav2Vec2 لمعالجة الإشارة الصوتية:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

تحتوي مجموعة بيانات MInDS-14 على معدل أخذ عينات يبلغ 8000 كيلو هرتز (يمكنك العثور على هذه المعلومات في [بطاقة مجموعة البيانات](https://huggingface.co/datasets/PolyAI/minds14))، مما يعني أنه سيتعين عليك إعادة أخذ عينات من مجموعة البيانات إلى 16000 كيلو هرتز لاستخدام نموذج Wav2Vec2 المدرب مسبقًا:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 1600Multiplier: 1000
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

الآن قم بإنشاء دالة معالجة مسبقة تقوم بما يلي:

1. استدعاء عمود "الصوت" لتحميل، وإذا لزم الأمر، إعادة أخذ عينات ملف الصوت.
2. التحقق مما إذا كان معدل أخذ العينات لملف الصوت يتطابق مع معدل أخذ العينات لبيانات الصوت التي تم تدريب النموذج عليها. يمكنك العثور على هذه المعلومات في [بطاقة نموذج](https://huggingface.co/facebook/wav2vec2-base) Wav2Vec2.
3. قم بتعيين طول إدخال أقصى لدفعات المدخلات الأطول دون اقتطاعها.

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`] في Datasets 🤗. يمكنك تسريع `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد. أزل الأعمدة التي لا تحتاجها، وأعد تسمية `intent_class` إلى `label` لأن هذا هو الاسم الذي يتوقعه النموذج:

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

دالة `compute_metrics` الخاصة بك جاهزة الآن، وستعود إليها عند إعداد التدريب الخاص بك.

## تدريب

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام [`Trainer`]]، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>
<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام [`Trainer`]]، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل Wav2Vec2 باستخدام [`AutoModelForAudioClassification`] جنبًا إلى جنب مع عدد التسميات المتوقعة، وخرائط التسميات:

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

في هذه المرحلة، هناك ثلاث خطوات فقط:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. ستقوم بالدفع بهذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجل الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم الدقة وحفظ نقطة التحقق التدريبية.
2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج، ومجموعة البيانات، ومعالج الرموز، وملف تجميع البيانات، ودالة `compute_metrics`.
3. استدعاء [`~Trainer.train`] لضبط نموذجك بشكل دقيق.


```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_Multiplier: 1000
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
...     tokenizer=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب شارك نموذجك في Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر شمولاً حول كيفية ضبط نموذج دقيق لتصنيف الصوت، راجع الدفتر [المقابل لـ PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

</Tip>

## الاستدلال

رائع، الآن بعد أن ضبطت نموذجًا دقيقًا، يمكنك استخدامه للاستدلال!

قم بتحميل ملف صوتي تريد تشغيل الاستدلال عليه. تذكر إعادة أخذ عينات معدل أخذ العينات لملف الصوت لمطابقة معدل أخذ العينات للنموذج إذا لزم الأمر!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

أبسط طريقة لتجربة نموذجك الدقيق للاستدلال هي استخدامه في [`pipeline`]. قم بتنفيذ مثيل `pipeline` لتصنيف الصوت باستخدام نموذجك، ومرر ملف الصوت الخاص بك إليه:

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

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا كنت تريد ذلك:

<frameworkcontent>
<pt>
قم بتحميل مستخرج ميزات لمعالجة ملف الصوت وإرجاع `input` كرموز تعبيرية PyTorch:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

Pass your inputs to the model and return the logits:

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Get the class with the highest probability, and use the model's `id2label` mapping to convert it to a label:

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
</pt>
</frameworkcontent>