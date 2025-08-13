<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# الإجابة عن الأسئلة البصرية (Visual Question Answering)

[[open-in-colab]]

الإجابة عن الأسئلة البصرية (VQA) هي مهمة الإجابة عن أسئلة مفتوحة استنادًا إلى صورة.
عادةً ما يكون مُدخل النماذج الداعمة لهذه المهمة عبارة عن مزيج من صورة وسؤال، ويكون المخرج إجابة مُعبَّرًا عنها بلغة طبيعية.

أمثلة جديرة بالذكر لاستخدامات VQA:
- الإتاحة: تطبيقات لمساعدة ذوي الإعاقة البصرية.
- التعليم: طرح أسئلة حول المواد المرئية المقدَّمة في المحاضرات أو الكتب. يمكن أيضًا استخدام VQA في المعارض المتحفية التفاعلية أو المواقع التاريخية.
- خدمة العملاء والتجارة الإلكترونية: يمكن أن يعزِّز VQA تجربة المستخدم بالسماح لهم بطرح أسئلة حول المنتجات.
- استرجاع الصور: يمكن استخدام نماذج VQA لاسترجاع الصور ذات الخصائص المحددة. على سبيل المثال، يستطيع المستخدم أن يسأل "هل يوجد كلب؟" للعثور على جميع الصور التي تحتوي على كلاب من مجموعة صور.

في هذا الدليل ستتعلم كيف:

- تضبط نموذج VQA تصنيفيًا، تحديدًا [ViLT](../model_doc/vilt)، على [مجموعة بيانات `Graphcore/vqa`](https://huggingface.co/datasets/Graphcore/vqa).
- تستخدم نموذج ViLT المضبوط للاستدلال.
- تشغّل استدلال VQA بدون تدريب مسبق باستخدام نموذج توليدي مثل BLIP-2.

## ضبط ViLT

يُضمِّن نموذج ViLT تضمينات نصية داخل محوّل رؤية (ViT)، مما يتيح تصميمًا بسيطًا لمرحلة التدريب المسبق للرؤية واللغة (VLP).
يمكن استخدام هذا النموذج لعدة مهام لاحقة. لمهمة VQA، يُوضَع رأس تصنيفي في الأعلى (طبقة خطية فوق الحالة المخفية النهائية لرمز `[CLS]`) ويُهيّأ عشوائيًا.
وهكذا تُعامَل مهمة VQA على أنها "مشكلة تصنيف".

تتعامل النماذج الأحدث، مثل BLIP وBLIP-2 وInstructBLIP، مع VQA بوصفها مهمة توليدية. سنوضّح لاحقًا في هذا الدليل كيفية استخدامها للاستدلال بدون تدريب مسبق.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات اللازمة.

```bash
pip install -q transformers datasets
```

نوصيك بمشاركة نموذجك مع المجتمع. سجّل الدخول إلى حسابك على Hugging Face لرفعه إلى 🤗 Hub.
عند المطالبة، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

لنعرّف نقطة تفتيش النموذج كمتغير عام:

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## تحميل البيانات

لأغراض الشرح، نستخدم في هذا الدليل عينة صغيرة جدًا (200 مثال) من جزء التحقق في مجموعة بيانات `Graphcore/vqa` الخاصة بالإجابة عن الأسئلة البصرية.
يمكنك العثور على المجموعة الكاملة على [🤗 Hub](https://huggingface.co/datasets/Graphcore/vqa).

كبديل عن [مجموعة بيانات `Graphcore/vqa`](https://huggingface.co/datasets/Graphcore/vqa)، يمكنك تنزيل نفس البيانات يدويًا من [صفحة مجموعة بيانات VQA الرسمية](https://visualqa.org/download.html).
إذا فضّلت متابعة الدليل باستخدام بياناتك المخصّصة، راجع كيفية [إنشاء مجموعة بيانات صور](https://huggingface.co/docs/datasets/image_dataset#loading-script) في توثيق 🤗 Datasets.

لنحمّل أول 200 مثال من جزء التحقق ونستكشف ميزات المجموعة:

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 200
})
```

لنلقِ نظرة على مثال لفهم ميزات المجموعة:

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

الميزات ذات الصلة بالمهمة تتضمن:
- `question`: السؤال المُراد الإجابة عنه من الصورة
- `image_id`: مسار الصورة التي يُشير إليها السؤال
- `label`: التوسيمات/الترميزات

يمكننا إزالة بقية الميزات لأنها لن تكون ضرورية:

```py
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
```

كما ترى، تحتوي ميزة `label` على عدة إجابات للسؤال نفسه (تسمى هنا `ids`) جُمِعت من مُعنِّفين بشريين مختلفين.
ذلك لأن الإجابة عن سؤال ما قد تكون ذاتية. في هذا المثال، السؤال هو "أين ينظر؟". بعض الأشخاص وسموه بـ"down"، وآخرون بـ"at table"، وآخر بـ"skateboard"، إلخ.

انظر إلى الصورة وفكّر أي إجابة ستعطي:

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div>

نظرًا لغموض الأسئلة والأجوبة، تُعامَل مجموعات كهذه كمشكلة تصنيف متعدد الملصقات (إذ قد تكون عدة إجابات صحيحة).
وعوضًا عن إنشاء متجه ترميز أحادي (one-hot) فقط، نُنشئ ترميزًا لينًا (soft) اعتمادًا على عدد مرات ظهور إجابة معيّنة في التعليقات.

على سبيل المثال أعلاه، لأن الإجابة "down" مُختارة أكثر بكثير من الإجابات الأخرى، تحصل على درجة (تُسمى `weight` في المجموعة) قدرها 1.0، بينما بقية الإجابات أقل من 1.0.

لاحقًا لتهيئة رأس تصنيفي مناسب في النموذج، لننشئ قاموسين: أحدهما يحوّل اسم الوسم إلى عدد صحيح، والآخر يعكسه:

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()}
```

الآن بعد أن أصبح لدينا الخرائط، يمكننا استبدال الإجابات النصية بمعرّفاتها، وتسطيح المجموعة لتهيئة المعالجة اللاحقة بشكل أسهل:

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'image_id': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## المعالجة المسبقة للبيانات

الخطوة التالية هي تحميل مُعالج ViLT لتحضير بيانات الصورة والنص للنموذج.
يلفّ [`ViltProcessor`] مُقسِّم BERT ومُعالج صور ViLT ضمن مُعالج واحد مريح:

```py
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

لمعالجة البيانات مسبقًا نحتاج إلى ترميز الصور والأسئلة باستخدام [`ViltProcessor`]. سيستخدم المُعالج [`BertTokenizerFast`] لتقسيم النص وإنشاء
`input_ids` و`attention_mask` و`token_type_ids` لبيانات النص. وبالنسبة للصور، سيستفيد المُعالج من [`ViltImageProcessor`] لتغيير الحجم وتطبيع الصورة، وإنشاء `pixel_values` و`pixel_mask`.

تتم جميع خطوات المعالجة هذه خلف الكواليس، وكل ما نحتاج إليه هو استدعاء `processor`. لكننا ما زلنا بحاجة لتحضير الملصقات الهدف. في هذا التمثيل، يقابل كل عنصر إجابةً محتملة (وسمًا). للإجابات الصحيحة، يحمل العنصر درجتها المقابلة (الوزن)، بينما تُضبط العناصر المتبقية على الصفر.

تُطبّق الدالة التالية `processor` على الصور والأسئلة وتنسّق الملصقات كما هو موصوف أعلاه:

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()

...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score

...         targets.append(target)

...     encoding["labels"] = targets

...     return encoding
```

لتطبيق دالة المعالجة على كامل المجموعة، استخدم دالة 🤗 Datasets [`~datasets.map`]. يمكنك تسريع `map` عبر ضبط `batched=True` لمعالجة عدة عناصر دفعةً واحدة. في هذه المرحلة، لا تتردد في إزالة الأعمدة غير اللازمة.

```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 200
})
```

كخطوة أخيرة، أنشئ دفعة أمثلة باستخدام [`DefaultDataCollator`]:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## تدريب النموذج

أصبحت جاهزًا لبدء تدريب نموذجك الآن! حمّل ViLT باستخدام [`ViltForQuestionAnswering`]. حدّد عدد الوسوم مع خرائط الوسوم:

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

في هذه المرحلة، تبقى ثلاث خطوات فقط:

1. عرّف فرطّيات التدريب في [`TrainingArguments`]:

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_200"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

2. مرّر معاملات التدريب إلى [`Trainer`] مع النموذج والمجموعة والمُعالج ومجمّع البيانات.

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     processing_class=processor,
... )
```

3. استدعِ [`~Trainer.train`] لضبط نموذجك.

```py
>>> trainer.train()
```

بعد اكتمال التدريب، شارك نموذجك على Hub باستخدام التابع [`~Trainer.push_to_hub`] لمشاركة نموذجك النهائي على 🤗 Hub:

```py
>>> trainer.push_to_hub()
```

## الاستدلال

الآن بعد أن ضبطت نموذج ViLT ورفعته إلى 🤗 Hub، يمكنك استخدامه للاستدلال. أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه ضمن [`Pipeline`].

```py
>>> from transformers import pipeline

>>> pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")
```

لم يُدرَّب النموذج في هذا الدليل إلا على 200 مثال، لذا لا تتوقع الكثير منه. لنرَ إن كان قد تعلّم شيئًا ما على الأقل، ولنأخذ المثال الأول من المجموعة لشرح الاستدلال:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
>>> print(question)
>>> pipe(image, question, top_k=1)
"Where is he looking?"
[{'score': 0.5498199462890625, 'answer': 'down'}]
```

على الرغم من أن الثقة ليست عالية، يبدو أن النموذج قد تعلّم شيئًا بالفعل. مع مزيد من الأمثلة وتدريب أطول، ستحصل على نتائج أفضل بكثير!

يمكنك أيضًا إعادة إنتاج نتائج الأنبوب يدويًا إذا رغبت:
1. خُذ صورةً وسؤالًا، وحضّرْهما للنموذج باستخدام المُعالج من نموذجك.
2. مرّر نتيجة المعالجة عبر النموذج.
3. من اللوجيت، احصل على معرّف الإجابة الأكثر احتمالًا، واعثر على الإجابة الفعلية في `id2label`.

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # prepare inputs
>>> inputs = processor(image, question, return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: down
```

## VQA بدون تدريب مسبق

عالج النموذج السابق VQA كمهمة تصنيف. بعض النماذج الحديثة مثل BLIP وBLIP-2 وInstructBLIP تتعامل مع VQA كمهمة توليدية.
لنأخذ [BLIP-2](../model_doc/blip-2) مثالًا. قدّم BLIP-2 نهج تدريب مسبق للرؤية واللغة يسمح باستخدام أي توليفة من مُرمّز رؤية ونموذج لغة كبير (تعرف أكثر في [مقال BLIP-2](https://huggingface.co/blog/blip-2)).
يُمكّن هذا من تحقيق أحدث النتائج الفنية في مهام الرؤية واللغة، بما في ذلك الإجابة عن الأسئلة البصرية.

لنوضّح كيف يمكنك استخدام هذا النموذج لـ VQA. أولًا، لنحمّل النموذج. سنرسل النموذج صراحةً إلى معالج رسومي GPU إن كان متاحًا، وهو ما لم نحتجه سابقًا أثناء التدريب؛ إذ يتكفّل [`Trainer`] بذلك تلقائيًا:

```py
>>> from transformers import AutoProcessor, Blip2ForConditionalGeneration
>>> import torch
>>> from accelerate.test_utils.testing import get_backend

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
>>> device, _, _ = get_backend() # يكتشف تلقائيًا نوع الجهاز الأساسي (CUDA أو CPU أو XPU أو MPS ...)
>>> model.to(device)
```

يتلقى النموذج صورة ونصًا كمدخل، لذا سنستخدم نفس زوج الصورة/السؤال من المثال الأول في مجموعة VQA:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

لاستخدام BLIP-2 في مهمة الإجابة عن الأسئلة البصرية، يجب أن يتبع المُوجِّه النصي تنسيقًا محددًا: `Question: {} Answer:`.

```py
>>> prompt = f"Question: {question} Answer:"
```

الآن نحتاج لمعالجة الصورة/الموجِّه بمُعالج النموذج، ثم تمرير المدخلات المعالجة عبر النموذج، وفك ترميز المخرجات:

```py
>>> inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs, max_new_tokens=10)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"He is looking at the crowd"
```

كما ترى، تعرّف النموذج على الحشد واتجاه الوجه (النظر للأسفل)، لكنه يبدو أنه لم يلتقط حقيقة أن الحشد خلف المتزلّج. ومع ذلك، في الحالات التي يصعب فيها الحصول على مجموعات بيانات مُعنّنة بشريًا، يمكن أن ينتج هذا النهج نتائج مفيدة بسرعة.
