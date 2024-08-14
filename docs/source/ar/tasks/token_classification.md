# تصنيف الرموز

[[open-in-colab]]

<Youtube id="wVHdVlPScxA"/>

يُخصص تصنيف الرموز تسمية لعلامات فردية في جملة. إحدى مهام تصنيف الرموز الشائعة هي التعرف على الكيانات المسماة (NER). تحاول NER إيجاد تسمية لكل كيان في جملة، مثل شخص أو موقع أو منظمة.

سيوضح هذا الدليل كيفية:

1. ضبط دقة [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) على مجموعة بيانات [WNUT 17](https://huggingface.co/datasets/wnut_17) للكشف عن كيانات جديدة.
2. استخدام نموذجك المضبوط الدقة للاستنتاج.

<Tip>

لمشاهدة جميع البنى ونقاط التفتيش المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/token-classification).

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate seqeval
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات WNUT 17

ابدأ بتحميل مجموعة بيانات WNUT 17 من مكتبة Datasets 🤗:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

ثم الق نظرة على مثال:

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

يمثل كل رقم في `ner_tags` كيانًا. قم بتحويل الأرقام إلى أسماء تسمياتها لمعرفة ما هي الكيانات:

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

يشير الحرف الذي يسبق كل `ner_tag` إلى موضع الرمز المميز للكيان:

- `B-` يشير إلى بداية الكيان.
- `I-` يشير إلى أن الرمز المميز موجود داخل نفس الكيان (على سبيل المثال، رمز مميز `State` هو جزء من كيان مثل `Empire State Building`).
- `0` يشير إلى أن الرمز المميز لا يقابل أي كيان.

## معالجة مسبقة

<Youtube id="iY2AZYdZAr0"/>

الخطوة التالية هي تحميل معالج رموز DistilBERT لمعالجة حقل `tokens`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

كما رأيت في حقل `tokens` المثال أعلاه، يبدو أن المدخلات قد تم تمييزها بالفعل. لكن المدخلات لم يتم تمييزها بعد، وستحتاج إلى تعيين `is_split_into_words=True` لتمييز الكلمات إلى كلمات فرعية. على سبيل المثال:
```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

ومع ذلك، يضيف هذا بعض الرموز المميزة الخاصة `[CLS]` و`[SEP]`، وتؤدي عملية تمييز الكلمات الفرعية إلى عدم تطابق بين المدخلات والتسميات. قد يتم الآن تقسيم كلمة واحدة تقابل تسمية واحدة إلى كلمتين فرعيتين. ستحتاج إلى إعادة محاذاة الرموز المميزة والتسميات عن طريق:

1. قم بتعيين جميع الرموز المميزة إلى كلماتها المقابلة باستخدام طريقة [`word_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.word_ids).
2. قم بتعيين التسمية `-100` إلى الرموز المميزة الخاصة `[CLS]` و`[SEP]` حتى يتم تجاهلها بواسطة دالة الخسارة PyTorch (راجع [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
3. قم بتسمية الرمز المميز الأول للكلمة فقط. قم بتعيين `-100` إلى الرموز الفرعية الأخرى من نفس الكلمة.

هنا كيف يمكنك إنشاء وظيفة لإعادة محاذاة الرموز المميزة والتسميات، وقص التسلسلات بحيث لا تكون أطول من طول المدخلات الأقصى لـ DistilBERT:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # قم بتعيين الرموز المميزة إلى كلماتها المقابلة.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # قم بتعيين الرموز المميزة الخاصة إلى -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # قم بتسمية الرمز المميز الأول للكلمة فقط.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

لتطبيق وظيفة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`] في مكتبة Datasets 🤗. يمكنك تسريع وظيفة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorWithPadding`]. من الأكثر كفاءة *توسيد* الديناميكي للجمل إلى أطول طول في دفعة أثناء التجميع، بدلاً من توسيد مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل إطار [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس). في الواقع، ينتج Seqeval عدة درجات: الدقة والاستدعاء وF1 والدقة.

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```
```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

احصل على تسميات NER أولاً، ثم قم بإنشاء وظيفة تمرر تنبؤاتك الصحيحة وتسمياتك الصحيحة إلى [`~evaluate.EvaluationModule.compute`] لحساب الدرجات:

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

وظيفتك `compute_metrics` جاهزة الآن، وستعود إليها عند إعداد تدريبك.

## تدريب

قبل البدء في تدريب نموذجك، قم بإنشاء خريطة من معرفات التسميات المتوقعة إلى تسمياتها باستخدام `id2label` و`label2id`:

```py
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط دقة نموذج باستخدام [`Trainer`], فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل DistilBERT باستخدام [`AutoModelForTokenClassification`] جنبًا إلى جنب مع عدد التسميات المتوقعة، وخرائط التسميات:

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

في هذه المرحلة، هناك ثلاث خطوات فقط:

1. حدد معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد أين يتم حفظ نموذجك. ستقوم بالدفع بهذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقيم [`Trainer`] درجات seqeval ويحفظ نقطة تفتيش التدريب.
2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمعالج الرمزي ومجمع البيانات ووظيفة `compute_metrics`.
3. استدعاء [`~Trainer.train`] لضبط دقة نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_Multiplier: 16,
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
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن على دراية بضبط دقة نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط دقة نموذج في TensorFlow، ابدأ بإعداد دالة تحسين ومعدل تعلم وجدول، وبعض معلمات التدريب:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 3
>>> num_train_steps = (len(tokenized_wnut["train"]) // batch_size) * num_train_epochs
>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=2e-5,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=0.01,
...     num_warmup_steps=0,
... )
```

ثم يمكنك تحميل DistilBERT باستخدام [`TFAutoModelForTokenClassification`] جنبًا إلى جنب مع عدد التسميات المتوقعة، وخرائط التسميات:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

قم بتحويل مجموعات بياناتك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_wnut["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_wnut["validation"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

تهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة افتراضية ذات صلة بالمهمة، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # لا توجد حجة الخسارة!
```

الشيئان الأخيران اللذان يجب إعدادهما قبل بدء التدريب هما حساب درجات seqeval من التوقعات، وتوفير طريقة لدفع نموذجك إلى المنصة. يتم تنفيذ كلاهما باستخدام [Keras callbacks](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد أين تريد دفع نموذجك ومحولك في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_wnut_model"،
...     tokenizer=tokenizer,
... )
```

ثم قم بتجميع مكالمات الإرجاع الخاصة بك معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! اتصل بـ [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) باستخدام مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، ومكالمات الإرجاع الخاصة بك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set، validation_data=tf_validation_set، epochs=3، callbacks=callbacks)
```

بمجرد الانتهاء من التدريب، يتم تحميل نموذجك تلقائيًا إلى المنصة حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

لمثال أكثر تفصيلاً حول كيفية ضبط نموذج لتصنيف الرموز المميزة، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

</Tip>

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

احصل على بعض النصوص التي تريد تشغيل الاستدلال عليها:

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`]. قم بتنفيذ `pipeline` لـ NER باستخدام نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner"، model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location',
  'score': 0.42658573,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856336,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.3064001,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.65523505,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668663,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا كنت تريد ذلك:

<frameworkcontent>
<pt>
قم برمز النص وإرجاع الرموز المتوترة PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text، return_tensors="pt")
```

مرر المدخلات الخاصة بك إلى النموذج وإرجاع `logits`:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على الفئة ذات الاحتمالية الأعلى، واستخدم خريطة "id2label" للنموذج لتحويلها إلى تسمية نصية:

```py
>>> predictions = torch.argmax(logits، dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O'،
 'O'،
 'B-location'،
 'I-location'،
 'B-group'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'B-location'،
 'B-location'،
 'O'،
 'O']
```
</pt>
<tf>
قم برمز النص وإرجاع رموز TensorFlow المتوترة:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text، return_tensors="tf")
```

مرر المدخلات الخاصة بك إلى النموذج وإرجاع `logits`:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> logits = model(**inputs).logits
```

احصل على الفئة ذات الاحتمالية الأعلى، واستخدم خريطة "id2label" للنموذج لتحويلها إلى تسمية نصية:

```py
>>> predicted_token_class_ids = tf.math.argmax(logits، axis=-1)
>>> predicted_token_class = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
>>> predicted_token_class
['O'،
 'O'،
 'B-location'،
 'I-location'،
 'B-group'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'O'،
 'B-location'،
 'B-location'،
 'O'،
 'O']
```
</tf>
</frameworkcontent>