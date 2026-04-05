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

# الإجابة على الأسئلة من المستندات (Document Question Answering)

[[open-in-colab]]

الإجابة على أسئلة المستندات، وتُعرف أيضًا باسم "الإجابة البصرية على أسئلة المستندات"، هي مهمة تتضمن تقديم إجابات عن أسئلة تُطرح حول صور مستندية. عادةً ما يكون إدخال النماذج الداعمة لهذه المهمة مزيجًا من صورة وسؤال، بينما يكون الإخراج إجابة باللغة الطبيعية. تستفيد هذه النماذج من عدة أنماط (modalities)، بما في ذلك النص، ومواقع الكلمات (الصناديق المُحدِّدة bounding boxes)، والصورة نفسها.

يوضّح هذا الدليل كيفية:

- ضبط [LayoutLMv2](../model_doc/layoutlmv2) على بيانات [DocVQA](https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut).
- استخدام نموذجك المضبوط للاستدلال.

<Tip>

لاطّلاع على جميع البُنى ونقاط التحقق المتوافقة مع هذه المهمة، ننصح بزيارة [صفحة المهمة](https://huggingface.co/tasks/image-to-text)

</Tip>

يعالج LayoutLMv2 مهمة الإجابة على أسئلة المستندات بإضافة رأس (head) خاص بالإجابة على الأسئلة أعلى الحالات المخفية النهائية للرموز (tokens)، للتنبؤ بمواقع رموز البداية والنهاية للإجابة. وبمعنى آخر، تُعامل المشكلة على أنها إجابة استخراجية: بالنظر إلى السياق، استخرج المعلومة التي تجيب عن السؤال. يأتي السياق من مخرجات محرك OCR، وهنا نستخدم Tesseract من Google.

قبل البدء، تأكد من تثبيت جميع المكتبات اللازمة. يعتمد LayoutLMv2 على detectron2 وtorchvision وtesseract.

```bash
pip install -q transformers datasets
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchvision
```

```bash
sudo apt install tesseract-ocr
pip install -q pytesseract
```

بعد تثبيت جميع التبعيات، أعد تشغيل بيئتك التنفيذية.

نشجّعك على مشاركة نموذجك مع المجتمع. سجّل الدخول إلى حسابك على Hugging Face لرفعه إلى 🤗 Hub. عند المطالبة، أدخل رمز الوصول الخاص بك:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

لنعرّف بعض المتغيرات العامة.

```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## تحميل البيانات

يستخدم هذا الدليل عينة صغيرة من DocVQA المُعالجة مسبقًا والمتوفرة على 🤗 Hub. إذا رغبت في استخدام مجموعة DocVQA كاملة، يمكنك التسجيل وتنزيلها من [الصفحة الرئيسية لـ DocVQA](https://rrc.cvc.uab.es/?ch=17). إذا فعلت ذلك، لمتابعة هذا الدليل اطلع على [كيفية تحميل الملفات إلى مجموعة بيانات 🤗](https://huggingface.co/docs/datasets/loading#local-and-remote-files).

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("nielsr/docvqa_1200_examples")
>>> dataset
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 200
    })
})
```

كما ترى، تنقسم مجموعة البيانات بالفعل إلى تدريبي (train) واختباري (test). ألقِ نظرة على مثال عشوائي للتعرّف على الميزات.

```py
>>> dataset["train"].features
```

يمثل كل حقل مما يلي:
* `id`: معرّف المثال
* `image`: كائن PIL.Image.Image يحتوي على صورة المستند
* `query`: السؤال (قد يحتوي على ترجمات متعددة، مثل `en`)
* `answers`: قائمة بالإجابات الصحيحة التي قدّمها المقيِّمون البشريون
* `words` و`bounding_boxes`: نتائج OCR، والتي لن نستخدمها هنا
* `answer`: إجابة طابقتْها أداة مختلفة ولن نستخدمها هنا

لنُبقِ على الأسئلة الإنجليزية فقط، ونُسقط الحقل `answer` الذي يبدو أنه يحتوي على تنبؤات نموذج آخر. سنأخذ أيضًا أول إجابة من مجموعة الإجابات التي وفّرها المقيّمون. بديلًا عن ذلك، يمكنك أخذ عينة عشوائية.

```py
>>> def keep_english_drop_prediction(example):
...     new_example = {}
...     new_example["id"] = example["id"]
...     new_example["image"] = example["image"]
...     new_example["question"] = example["query"]["en"]
...     new_example["answers"] = example["answers"]["en"]
...     new_example["words"] = example["words"]
...     new_example["bounding_boxes"] = example["bounding_boxes"]
...     return new_example

>>> updated_dataset = dataset.map(keep_english_drop_prediction)
```

قد يرغب البعض بتصفية الأمثلة التي تتجاوز حد طول الإدخال. يستخدم LayoutLMv2 حدًا أقصى بطول 512 رمزًا للنص مع مجموع الكلمات، لذا سنُسقِط الأمثلة الأطول.

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

في هذه المرحلة، لنُزِل ميزات OCR من هذه العيّنة. إذ إنها مخصصة لضبط نموذج آخر، وستتطلب معالجة إضافية لتلائم متطلبات إدخال نموذجنا. سنحسبها عبر تطبيق OCR بأنفسنا ضمن مسار المعالجة (preprocessing) الخاص بنا باستخدام Tesseract.

قبل تجهيز البيانات، نحتاج إلى تجهيز المعالِج (processor) المناسب للنموذج. تجمع [`LayoutLMv2Processor`] داخليًا بين معالج الصور (image processor) لمعالجة بيانات الصور ومُرمِّز النص (tokenizer) لترميز النص.

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### معالجة صور المستندات مسبقًا

سنكتب دوالًا مساعدة لتطبيق OCR على الصور واستخراج الكلمات والصناديق المحيطة (bounding boxes) باستخدام pytesseract.

```py
>>> import pytesseract
>>> from pytesseract import Output
>>> from PIL import Image

>>> def get_ocr_words_and_boxes(batch):
...     images = batch["image"]
...     words = []
...     boxes = []
...     for image in images:
...         data = pytesseract.image_to_data(image.convert("RGB"), output_type=Output.DICT)
...         words.append(data["text"])  # كلمات على مستوى الكلمة
...         # الصناديق: x, y, w, h -> حوّلها إلى نظام LayoutLMv2 [0, 1000]
...         image_width, image_height = image.size
...         normalized_boxes = []
...         for x, y, w, h in zip(data["left"], data["top"], data["width"], data["height"]):
...             x0, y0, x1, y1 = x, y, x + w, y + h
...             normalized_boxes.append([
...                 int(1000 * x0 / image_width),
...                 int(1000 * y0 / image_height),
...                 int(1000 * x1 / image_width),
...                 int(1000 * y1 / image_height),
...             ])
...         boxes.append(normalized_boxes)
...     return {"words": words, "boxes": boxes}

>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```

### معالجة البيانات النصية مسبقًا

نحوّل الآن الكلمات والصناديق التي حصلنا عليها في الخطوة السابقة إلى `input_ids` و`attention_mask` و`token_type_ids` و`bbox` على مستوى الرموز (token-level). سنحتاج إلى `tokenizer` من المعالج لمعالجة النص.

```py
>>> tokenizer = processor.tokenizer
```

سنكتب دالة مساعدة للعثور على موقع الإجابة (قائمة كلمات) ضمن قائمة كلمات المثال.

```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(words_list[i : i + len(answer_list)])
...             start_indices.append(i)
...             end_indices.append(i + len(answer_list) - 1)
...     if len(matches) > 0:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```

لنجرّبها على مثال واحد.

```py
>>> example = dataset_with_ocr["train"][0]
>>> words = [word.lower() for word in example["words"]]
>>> answer = [word.lower() for word in example["answers"][0].split()]
>>> answer, word_idx_start, word_idx_end = subfinder(words, answer)

>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
```

بعد ترميز الأمثلة ستبدو كالتالي:

```py
>>> encoding = tokenizer(example["question"], example["words"], example["boxes"])
>>> tokenizer.decode(encoding["input_ids"])
```

لنكتب دالة لترميز دفعة من الأمثلة وإنتاج مواضع البداية والنهاية للإجابة بكل مثال.

```py
>>> def encode_dataset(batch, max_length=512):
...     questions = batch["question"]
...     words = batch["words"]
...     boxes = batch["boxes"]
...     encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
...     start_positions = []
...     end_positions = []
...
...     # حلقة على الأمثلة في الدفعة
...     for i in range(len(questions)):
...         cls_index = encoding["input_ids"][i].index(tokenizer.cls_token_id)
...
...         # ابحث عن موقع الإجابة ضمن كلمات المثال
...         words_example = [word.lower() for word in words[i]]
...         answer = words_example  # placeholder إن لزم
...         # مبدئيًا: حاول مطابقة أول إجابة موفّرة
...         ans_tokens = batch["answers"][i][0].lower().split()
...         _, word_idx_start, word_idx_end = subfinder(words_example, ans_tokens)
...
...         # حوّل مواضع الكلمات إلى مواضع الرموز عبر `word_ids`
...         word_ids = encoding.word_ids(i)
...         token_start_index = 0
...         while token_start_index < len(word_ids) and word_ids[token_start_index] != word_idx_start:
...             token_start_index += 1
...         token_end_index = len(word_ids) - 1
...         while token_end_index >= 0 and word_ids[token_end_index] != word_idx_end:
...             token_end_index -= 1
...
...         # إن لم نجدها، اجعل الإجابة هي CLS
...         if token_start_index >= len(word_ids) or token_end_index < 0:
...             start_positions.append(cls_index)
...             end_positions.append(cls_index)
...         else:
...             start_positions.append(token_start_index)
...             end_positions.append(token_end_index)
...
...     encoding["start_positions"] = start_positions
...     encoding["end_positions"] = end_positions
...     return encoding
```

طبّق الترميز على قسمي التدريب والاختبار.

```py
>>> encoded_train_dataset = dataset_with_ocr["train"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["train"].column_names
... )
>>> encoded_test_dataset = dataset_with_ocr["test"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["test"].column_names
... )
```

لنطّلع على ميزات مجموعة البيانات بعد الترميز:

```py
>>> encoded_train_dataset.features
```

## التدريب

تهانينا! لقد تجاوزت أصعب جزء في هذا الدليل وأصبحت جاهزًا لتدريب نموذجك. يتضمن التدريب الخطوات التالية:

- تحميل النموذج المناسب للمهمة.
- تحديد معاملات التدريب.
- ترتيب مجمّع البيانات (data collator).
- استدعاء [`Trainer.train`].

أولًا، لنحمّل النموذج.

```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```

في [`TrainingArguments`] استخدم `output_dir` لتحديد مكان حفظ النموذج، واضبط الهايبر-بارامترز كما تراه مناسبًا. إذا رغبت في مشاركة نموذجك مع المجتمع، اضبط `push_to_hub` إلى `True` (يجب أن تكون مسجّل الدخول إلى Hugging Face). في هذه الحالة سيكون `output_dir` أيضًا اسم مستودع النموذج حيث ستُرفع نقاط التحقق.

```py
>>> from transformers import TrainingArguments

>>> # استبدل هذا بمعرّف المستودع الخاص بك
>>> repo_id = "MariaK/layoutlmv2-base-uncased_finetuned_docvqa"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     eval_strategy="steps",
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

عرّف مجمّع بيانات بسيطًا لجمع الأمثلة في دفعات.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

أخيرًا، لنجمع كل شيء ونستدعي [`~Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     processing_class=processor,
... )

>>> trainer.train()
```

لإضافة النموذج النهائي إلى 🤗 Hub، أنشئ بطاقة نموذج واستدعِ `push_to_hub`:

```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## الاستدلال

الآن بعد أن ضبطت نموذج LayoutLMv2 ورفعته إلى 🤗 Hub، يمكنك استخدامه للاستدلال. أبسط طريقة هي استخدامه ضمن [`Pipeline`].

لنأخذ مثالًا:
```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```

بعدها، أنشئ بايبلاين للإجابة على أسئلة المستندات باستخدام نموذجك، ومرّر الصورة + السؤال إليه.

```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
  'answer': 'Lee A. Waller',
  'start': 55,
  'end': 57}]
```

يمكنك أيضًا إعادة تنفيذ خطوات البايبلاين يدويًا إذا رغبت:
1. خذ صورة وسؤالًا، وجهّزهما للنموذج باستخدام المعالج.
2. مرّر المدخلات عبر النموذج.
3. يُرجع النموذج `start_logits` و`end_logits`، وتشير إلى رمز بداية الإجابة ورمز نهايتها، وكلاهما بالشكل (batch_size, sequence_length).
4. طبّق argmax على البُعد الأخير لكل من `start_logits` و`end_logits` للحصول على `start_idx` و`end_idx` المتنبأ بهما.
5. فك ترميز الإجابة باستخدام tokenizer.

```py
>>> import torch
>>> from transformers import AutoProcessor
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")

>>> with torch.no_grad():
...     encoding = processor(image.convert("RGB"), question, return_tensors="pt")
...     outputs = model(**encoding)
...     start_logits = outputs.start_logits
...     end_logits = outputs.end_logits
...     predicted_start_idx = start_logits.argmax(-1).item()
...     predicted_end_idx = end_logits.argmax(-1).item()

>>> processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1])
'lee a. waller'
```
