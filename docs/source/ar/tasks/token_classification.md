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

# تصنيف الرموز(Token classification)

[[open-in-colab]]

<Youtube id="wVHdVlPScxA"/>

يهدف تصنيف الرموز إلى إعطاء تسمية لكل رمز على حدة في الجملة. من أكثر مهام تصنيف الرموز شيوعًا هو التعرف على الكيانات المسماة (NER). يحاول NER تحديد تسمية لكل كيان في الجملة، مثل شخص، أو مكان، أو منظمة.

سيوضح لك هذا الدليل كيفية:

1. ضبط [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) على مجموعة بيانات [WNUT 17](https://huggingface.co/datasets/wnut_17) للكشف عن كيانات جديدة.
2.  استخدام نموذجك المضبوط بدقة للاستدلال.

<Tip>

للاطلاع جميع البنى والنقاط المتوافقة مع هذه المهمة، نوصي بالرجوع من [صفحة المهمة](https://huggingface.co/tasks/token-classification).

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate seqeval
```

نحن نشجعك على تسجيل الدخول إلى حساب HuggingFace الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات WNUT 17

ابدأ بتحميل مجموعة بيانات WNUT 17 من مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

ثم ألق نظرة على مثال:

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

يمثل كل رقم في `ner_tags` كياناً. حوّل الأرقام إلى أسماء التصنيفات لمعرفة ماهية الكيانات:

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

يشير الحرف الذي يسبق كل `ner_tag` إلى موضع الرمز للكيان:

- `B-` يشير إلى بداية الكيان.
- `I-` يشير إلى أن الرمز يقع ضمن نفس الكيان (على سبيل المثال، الرمز `State` هو جزء من كيان مثل `Empire State Building`).
- `0` يشير إلى أن الرمز لا يمثل أي كيان.

## المعالجة المسبقة(Preprocess)

<Youtube id="iY2AZYdZAr0"/>

الخطوة التالية هي تحميل مُجزِّئ النصوص DistilBERT للمعالجة المسبقة لحقل `tokens`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

كما رأيت في حقل `tokens` المثال أعلاه، يبدو أن المدخل قد تم تحليله بالفعل. لكن المدخل  لم يُجزأ بعد ويتعيّن عليك ضبط `is_split_into_words=True` لتقسيم الكلمات إلى كلمات فرعية. على سبيل المثال:

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

ومع ذلك، يضيف هذا بعض الرموز الخاصة `[CLS]` و`[SEP]` وتقسيم الكلمات إلى أجزاء يُنشئ عدم تطابق بين المُدخلات والتسميات. قد يتم تقسيم كلمة واحدة تقابل تسمية واحدة الآن إلى كلمتين فرعيتين. ستحتاج إلى إعادة محاذاة الرموز والتسميات عن طريق:

1. ربط كل رمز بالكلمة الأصلية باستخدام الخاصية [`word_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.word_ids).
2. تعيين التسمية `-100` للرموز الخاصة `[CLS]` و`[SEP]` بحيث يتم تجاهلها بواسطة دالة الخسارة PyTorch (انظر [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
3. تسمية الرمز الأول فقط لكلمة معينة. قم بتعيين `-100` لأجزاء الكلمة الأخرى.

هنا كيف يمكنك إنشاء وظيفة لإعادة محاذاة الرموز والتسميات، وقص الجمل لتتجاوز الحد الأقصى لطول مُدخلات DistilBERT:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # تعيين الرموز إلى كلماتهم المقابلة.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # تعيين الرموز الخاصة إلى -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # تسمية الرمز الأول فقط لكلمة معينة.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

لتطبيق هذه العملية على كامل مجموعة البيانات، استخدم الدالة [`~datasets.Dataset.map`] لمجموعة بيانات 🤗. يمكنك تسريع الدالة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorWithPadding`].من الأفضل استخدام *الحشو الديناميكي* للجمل إلى أطول طول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بالكامل إلى الطول الأقصى.

```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

## التقييم(Evaluate)

يُعدّ تضمين مقياس أثناء التدريب مفيدًا في تقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة مع مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، قم بتحميل إطار [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) (انظر جولة 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس). يُخرج seqeval عدة نتائج: الدقة، والاستذكار، ومقياس F1، والدقة.

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

احصل على تسميات الكيانات المسماة (NER) أولاً،ثم أنشئ دالة تُمرر تنبؤاتك وتسمياتك الصحيحة إلى [`~evaluate.EvaluationModule.compute`] لحساب النتائج:

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

دالة `compute_metrics` جاهزة للاستخدام، وستحتاج إليها عند إعداد التدريب.

## التدريب(Train)

قبل تدريب النموذج، جهّز خريطة تربط بين المعرّفات المتوقعة وتسمياتها باستخدام `id2label` و `label2id`:

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

<Tip>

إذا لم تكن على دراية بتعديل نموذج باستخدام [`Trainer`], ألق نظرة على الدليل التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل DistilBERT مع [`AutoModelForTokenClassification`] إلى جانب عدد التصنيفات المتوقعة، وخريطة التسميات:

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

في هذه المرحلة، هناك ثلاث خطوات فقط متبقية:

1. حدد معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعامل الوحيد المطلوب هو `output_dir` الذي يحدد مكان حفظ نموذجك. ستقوم بدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم درجات seqeval وحفظ تسخة التدريب.
2. قم بتمرير معاملات التدريب إلى [`Trainer`] إلى جانب النموذج، ومجموعة البيانات، والمُجزِّئ اللغوي، و`data collator`، ودالة `compute_metrics`.
3.استدعِ [`~Trainer.train`] لتدريب نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
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
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```

<Tip>

للحصول على مثال أكثر تفصيلاً حول كيفية تعديل نموذج لتصنيف الرموز، ألق نظرة على الدفتر المقابل
[دفتر PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
أو [دفتر TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

</Tip>

## الاستدلال(Inference)

رائع، الآن بعد أن قمت بتعديل نموذج، يمكنك استخدامه للاستدلال!

احصل على بعض النصوص التي تريد تشغيل الاستدلال عليها:

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

أبسط طريقة لتجربة نموذجك المُدرب مسبقًا للاستدلال هي استخدامه في [`pipeline`]. قم بتنفيذ `pipeline` لتصنيف الكيانات المسماة مع نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
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

يمكنك أيضًا تكرار نتائج `pipeline` يدويًا إذا أردت:

قسّم النص إلى رموز وأرجع المُوتّرات بلغة PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

مرر مدخلاتك إلى النموذج واحصل على `logits`:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

استخرج الفئة ذات الاحتمالية الأعلى، واستخدم جدول `id2label` الخاصة بالنموذج لتحويلها إلى تسمية نصية:

```py
>>> predictions = torch.argmax(logits, dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
