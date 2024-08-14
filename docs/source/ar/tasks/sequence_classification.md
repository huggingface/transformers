# تصنيف النص

[[open-in-colab]]

<Youtube id="leNG9fN9FQU"/>

تصنيف النص هي مهمة شائعة في معالجة اللغات الطبيعية (NLP) تقوم بتعيين تسمية أو فئة للنص. وتستخدم بعض أكبر الشركات تصنيف النص في الإنتاج لمجموعة واسعة من التطبيقات العملية. أحد أكثر أشكال تصنيف النص شيوعًا هو تحليل المشاعر، والذي يعين تسمية مثل 🙂 إيجابية، 🙁 سلبية، أو 😐 محايدة لتسلسل نصي.

سيوضح هذا الدليل لك كيفية:

1. ضبط دقة نموذج [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) باستخدام مجموعة بيانات [IMDb](https://huggingface.co/datasets/imdb) لتحديد ما إذا كان تقييم الفيلم إيجابيًا أم سلبيًا.
2. استخدام نموذجك المضبوط دقة للتنبؤ.

<Tip>

لمعرفة جميع البنى ونقاط المراقبة المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/text-classification).

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate accelerate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات IMDb

ابدأ بتحميل مجموعة بيانات IMDb من مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> imdb = load_dataset("imdb")
```

ثم الق نظرة على مثال:

```py
>>> imdb["test"][0]
{
    "label": 0,
    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
}
```

هناك حقلاًن في هذه المجموعة من البيانات:

- `text`: نص مراجعة الفيلم.
- `label`: قيمة إما `0` لمراجعة سلبية أو `1` لمراجعة إيجابية.

## معالجة مسبقة

الخطوة التالية هي تحميل معالج نصوص DistilBERT لمعالجة حقل `النص`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

قم بإنشاء دالة معالجة مسبقة لترميز `النص` وتقليص التسلسلات بحيث لا تكون أطول من طول الإدخال الأقصى لـ DistilBERT:

```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`] من مكتبة 🤗 Datasets. يمكنك تسريع `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorWithPadding`]. من الأكثر كفاءة *توسيد* الديناميكي للجمل إلى أطول طول في دفعة أثناء التجميع، بدلاً من توسيد مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## تقييم

من المفيد غالبًا تضمين مقياس أثناء التدريب لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

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

دالتك `compute_metrics` جاهزة الآن، وستعود إليها عندما تقوم بإعداد تدريبك.

## تدريب

قبل البدء في تدريب نموذجك، قم بإنشاء خريطة من معرفات التسميات المتوقعة إلى تسمياتها باستخدام `id2label` و`label2id`:

```py
>>> id2label = {0: "NEGATIVE", 1: "POSITIVE"}
>>> label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط دقة نموذج باستخدام [`Trainer`], فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل DistilBERT باستخدام [`AutoModelForSequenceClassification`] إلى جانب عدد التسميات المتوقعة، وخرائط التسميات:

```py
>>> from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

>>> model = AutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

في هذه المرحلة، هناك ثلاث خطوات فقط متبقية:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. ستقوم بالدفع إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقيم [`Trainer`] الدقة ويحفظ نقطة المراقبة التدريبية.
2. مرر فرط معلمات التدريب إلى [`Trainer`] إلى جانب النموذج ومجموعة البيانات والمعالج النصي ومجمع البيانات و`compute_metrics` الدالة.
3. استدعاء [`~Trainer.train`] لضبط دقة نموذجك.
```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_Multiplier=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_imdb["train"],
...     eval_dataset=tokenized_imdb["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

<Tip>

يطبق [`Trainer`] التوسيد الديناميكي بشكل افتراضي عند تمرير `tokenizer` إليه. في هذه الحالة، لا تحتاج إلى تحديد مجمع بيانات بشكل صريح.

</Tip>

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن، وجدول معدل التعلم، وبعض فرط معلمات التدريب:

```py
>>> from transformers import create_optimizer
>>> import tensorflow as tf

>>> batch_size = 16
>>> num_epochs = 5
>>> batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
>>> total_train_steps = int(batches_per_epoch * num_epochs)
>>> optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

بعد ذلك، يمكنك تحميل DistilBERT باستخدام [`TFAutoModelForSequenceClassification`] جنبًا إلى جنب مع عدد التصنيفات المتوقعة، وخرائط التسميات:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased"، num_labels=2، id2label=id2label، label2id=label2id
... )
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_imdb["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_imdb["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة ذات صلة بالمهمة الافتراضية، لذلك لا تحتاج إلى تحديدها ما لم تكن تريد ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer) # لا توجد وسيطة خسارة!
```

آخر شيئين يجب إعدادهما قبل بدء التدريب هما حساب الدقة من التوقعات، وتوفير طريقة لدفع نموذجك إلى Hub. يتم ذلك باستخدام [استدعاءات Keras](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد المكان الذي ستدفع فيه نموذجك ومصنف الرموز في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model"،
...     tokenizer=tokenizer،
... )
```

بعد ذلك، قم بتجميع استدعاءات الإرجاع الخاصة بك معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! استدعاء [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات الخاصة بالتدريب والتحقق من الصحة، وعدد العصور، واستدعاءات الإرجاع الخاصة بك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

لمثال أكثر عمقًا حول كيفية ضبط نموذج لتصنيف النصوص، راجع الدفتر الملاحظات المقابل
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

</Tip>

## الاستنتاج

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستنتاج!

احصل على بعض النصوص التي تود تشغيل الاستدلال عليها:

```py
>>> text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`]. قم بتنفيذ `pipeline` لتحليل المشاعر باستخدام نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
>>> classifier(text)
[{'label': 'POSITIVE', 'score': 0.9994940757751465}]
```

يمكنك أيضًا محاكاة نتائج `pipeline` يدويًا إذا كنت ترغب في ذلك:

<frameworkcontent>
<pt>
قم بتوكينز النص وإرجاع tensers PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

مرر المدخلات الخاصة بك إلى النموذج وإرجاع "logits":

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على الفئة ذات أعلى احتمال، واستخدم خريطة `id2label` للنموذج لتحويلها إلى تسمية نصية:

```py
>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</pt>
<tf>
قم بتوكينز النص وإرجاع tensers TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="tf")
```

مرر المدخلات الخاصة بك إلى النموذج وإرجاع "logits":

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> logits = model(**inputs).logits
```

احصل على الفئة ذات أعلى احتمال، واستخدم خريطة `id2label` للنموذج لتحويلها إلى تسمية نصية:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</tf>
</frameworkcontent>