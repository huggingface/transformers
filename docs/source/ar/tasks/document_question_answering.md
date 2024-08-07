# الإجابة على الأسئلة الواردة في الوثيقة

[[open-in-colab]]

الإجابة على الأسئلة الواردة في الوثيقة، والتي يشار إليها أيضًا باسم الإجابة المرئية على الأسئلة الواردة في الوثيقة، هي مهمة تتضمن تقديم إجابات على الأسئلة المطروحة حول صور المستندات. المدخلات إلى النماذج التي تدعم هذه المهمة هي عادة مزيج من الصورة والسؤال، والناتج هو إجابة معبر عنها باللغة الطبيعية. تستخدم هذه النماذج أوضاعًا متعددة، بما في ذلك النص، ومواضع الكلمات (حدود الإحداثيات)، والصورة نفسها.

يوضح هذا الدليل كيفية:

- ضبط نموذج LayoutLMv2 الدقيق على مجموعة بيانات DocVQA.
- استخدام نموذجك المضبوط دقيقًا للاستنتاج.

<Tip>

لمعرفة جميع التصميمات ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/image-to-text)

</Tip>

يحل LayoutLMv2 مهمة الإجابة على الأسئلة الواردة في الوثيقة عن طريق إضافة رأس للإجابة على الأسئلة أعلى حالات الرموز النهائية للرموز، للتنبؤ بمواضع رموز البدء والنهاية للإجابة. وبعبارة أخرى، تتم معاملة المشكلة على أنها إجابة استخراجية: استخراج قطعة المعلومات التي تجيب على السؤال، بالنظر إلى السياق. يأتي السياق من إخراج محرك التعرف الضوئي على الحروف، وهو هنا Tesseract من Google.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية. يعتمد LayoutLMv2 على detectron2 و torchvision و tesseract.

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

بمجرد تثبيت جميع التبعيات، أعد تشغيل وقت التشغيل الخاص بك.

نحن نشجعك على مشاركة نموذجك مع المجتمع. قم بتسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميله إلى 🤗 Hub.
عند المطالبة، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

دعونا نحدد بعض المتغيرات العالمية.

```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## تحميل البيانات

في هذا الدليل، نستخدم عينة صغيرة من DocVQA المعالجة مسبقًا والتي يمكنك العثور عليها على 🤗 Hub. إذا كنت ترغب في استخدام مجموعة DocVQA الكاملة، فيمكنك التسجيل وتنزيلها من [الصفحة الرئيسية لـ DocVQA](https://rrc.cvc.uab.es/?ch=17). إذا قمت بذلك، للمتابعة مع هذا الدليل، تحقق من [كيفية تحميل الملفات إلى مجموعة بيانات 🤗](https://huggingface.co/docs/datasets/loading#local-and-remote-files).

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

كما ترى، تم تقسيم مجموعة البيانات إلى مجموعات تدريب واختبار بالفعل. الق نظرة على مثال عشوائي للتعرف على الميزات.

```py
>>> dataset["train"].features
```
كما ترى، تم تقسيم مجموعة البيانات إلى مجموعات تدريب واختبار بالفعل. الق نظرة على مثال عشوائي للتعرف على الميزات.

```py
>>> dataset["train"].features
```

هذا ما تمثله الحقول الفردية:
* `id`: معرف المثال
* `image`: كائن PIL.Image.Image يحتوي على صورة المستند
* `query`: سلسلة الاستعلام - سؤال اللغة الطبيعية المطروح، بعدة لغات
* `answers`: قائمة الإجابات الصحيحة التي قدمها المعلقون البشريون
* `words` و `bounding_boxes`: نتائج التعرف الضوئي على الحروف، والتي لن نستخدمها هنا
* `answer`: إجابة تمت مطابقتها بواسطة نموذج مختلف لن نستخدمه هنا

دعونا نترك فقط الأسئلة باللغة الإنجليزية، ونقوم بإسقاط ميزة "الإجابة" التي يبدو أنها تحتوي على تنبؤات بواسطة نموذج آخر.
سنقوم أيضًا بأخذ الإجابة الأولى من مجموعة الإجابات التي قدمها المعلقون. أو يمكنك أخذ عينة عشوائية منها.

```py
>>> updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
>>> updated_dataset = updated_dataset.map(
...     lambda example: {"answer": example["answers"][0]}, remove_columns=["answer", "answers"]
... )
```

لاحظ أن نقطة التحقق LayoutLMv2 التي نستخدمها في هذا الدليل تم تدريبها مع `max_position_embeddings = 512` (يمكنك
العثور على هذه المعلومات في ملف `config.json` الخاص بنقطة التحقق [هنا](https://huggingface.co/microsoft/layoutlmv2-base-uncased/blob/main/config.json#L18)).
يمكننا تقليص الأمثلة ولكن لتجنب الموقف الذي قد تكون فيه الإجابة في نهاية مستند طويل وتنتهي مقتطعة،
هنا سنقوم بإزالة الأمثلة القليلة التي من المحتمل أن ينتهي فيها تضمينها إلى أكثر من 512.
إذا كانت معظم المستندات في مجموعة البيانات الخاصة بك طويلة، فيمكنك تنفيذ إستراتيجية النافذة المنزلقة - تحقق من [هذا الدفتر](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb) للحصول على التفاصيل.

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

في هذه المرحلة، دعنا نزيل أيضًا ميزات التعرف الضوئي على الحروف من هذه المجموعة من البيانات. هذه هي نتيجة التعرف الضوئي على الحروف لضبط نموذج مختلف. لا يزالون بحاجة إلى بعض المعالجة إذا أردنا استخدامها، لأنها لا تتطابق مع متطلبات الإدخال
من النموذج الذي نستخدمه في هذا الدليل. بدلاً من ذلك، يمكننا استخدام [`LayoutLMv2Processor`] على البيانات الأصلية لكل من التعرف الضوئي على الحروف
والتنفيذ. بهذه الطريقة سنحصل على المدخلات التي تتطابق مع الإدخال المتوقع للنموذج. إذا كنت تريد معالجة الصور يدويًا،
تحقق من وثائق نموذج [`LayoutLMv2`](../model_doc/layoutlmv2) لمعرفة تنسيق الإدخال الذي يتوقعه النموذج.

```py
>>> updated_dataset = updated_dataset.remove_columns("words")
>>> updated_dataset = updated_dataset.remove_columns("bounding_boxes")
```

أخيرًا، لن يكون استكشاف البيانات مكتملاً إذا لم نلقي نظرة على مثال على الصورة.

```py
>>> updated_dataset["train"][11]["image"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/docvqa_example.jpg" alt="مثال DocVQA"/>
 </div>

## معالجة البيانات مسبقًا

مهمة الإجابة على الأسئلة الواردة في الوثيقة هي مهمة متعددة الوسائط، ويجب التأكد من معالجة المدخلات من كل وسيط
وفقًا لتوقعات النموذج. دعونا نبدأ بتحميل [`LayoutLMv2Processor`]، والذي يجمع داخليًا بين معالج الصور الذي يمكنه التعامل مع بيانات الصور ومعالج الرموز الذي يمكنه تشفير بيانات النص.

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### معالجة صور المستندات
```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### معالجة صور المستندات

أولاً، دعونا نقوم بإعداد صور المستندات للنموذج بمساعدة `image_processor` من المعالج.
بحكم التوصيف، يقوم معالج الصور بإعادة تحجيم الصور إلى 224x224، والتأكد من أن لديها الترتيب الصحيح لقنوات الألوان،
تطبيق التعرف الضوئي على الحروف باستخدام Tesseract للحصول على الكلمات وحدود الإحداثيات المعيارية. في هذا البرنامج التعليمي، هذه الافتراضيات هي بالضبط ما نحتاجه.
اكتب دالة تطبق المعالجة الافتراضية على دفعة من الصور وتعيد نتائج التعرف الضوئي على الحروف.

```py
>>> image_processor = processor.image_processor


>>> def get_ocr_words_and_boxes(examples):
...     images = [image.convert("RGB") for image in examples["image"]]
...     encoded_inputs = image_processor(images)

...     examples["image"] = encoded_inputs.pixel_values
...     examples["words"] = encoded_inputs.words
...     examples["boxes"] = encoded_inputs.boxes

...     return examples
```

لتطبيق هذه المعالجة المسبقة على مجموعة البيانات بأكملها بطريقة سريعة، استخدم [`~datasets.Dataset.map`].

```py
>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```

### معالجة بيانات النص

بمجرد تطبيق التعرف الضوئي على الصور، نحتاج إلى تشفير الجزء النصي من مجموعة البيانات لإعدادها للنموذج.
ينطوي هذا على تحويل الكلمات وحدود الإحداثيات التي حصلنا عليها في الخطوة السابقة إلى مستوى الرمز `input_ids` و `attention_mask`
، `token_type_ids` و `bbox`. لمعالجة النص، سنحتاج إلى `tokenizer` من المعالج.

```py
>>> tokenizer = processor.tokenizer
```

بالإضافة إلى المعالجة المذكورة أعلاه، نحتاج أيضًا إلى إضافة العلامات إلى النموذج. بالنسبة لنماذج `xxxForQuestionAnswering` في 🤗 Transformers،
تتكون العلامات من `start_positions` و `end_positions`، والتي تشير إلى الرمز الموجود في
بداية ونهاية الإجابة.

دعونا نبدأ بذلك. قم بتعريف دالة مساعدة يمكنها العثور على قائمة فرعية (الإجابة المقسمة إلى كلمات) في قائمة أكبر (قائمة الكلمات).

ستأخذ هذه الدالة كإدخال قائمتين، `words_list` و `answer_list`. ثم سيقوم بالتنقل عبر `words_list` والتحقق
إذا كان الكلمة الحالية في `words_list` (words_list [i]) متساوية مع الكلمة الأولى من answer_list (answer_list [0]) وإذا
كانت القائمة الفرعية لـ `words_list` بدءًا من الكلمة الحالية وطولها متساوٍ `مع answer_list`.
إذا كان هذا الشرط صحيحًا، فهذا يعني أنه تم العثور على تطابق، وستقوم الدالة بتسجيل المطابقة، وموضع البدء الخاص بها،
وموضع النهاية (idx + len (answer_list) - 1). إذا تم العثور على أكثر من تطابق واحد، فستعيد الدالة فقط الأول.
إذا لم يتم العثور على أي تطابق، فستعيد الدالة (None، 0، و 0).

```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(answer_list)
...             start_indices.append(idx)
...             end_indices.append(idx + len(answer_list) - 1)
...     if matches:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```

لتوضيح كيفية عثور هذه الدالة على موضع الإجابة، دعنا نستخدمها في مثال:
لتوضيح كيفية عثور هذه الدالة على موضع الإجابة، دعنا نستخدمها في مثال:

```py
>>> example = dataset_with_ocr["train"][1]
>>> words = [word.lower() for word in example["words"]]
>>> match, word_idx_start, word_idx_end = subfinder(words, example["answer"].lower().split())
>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
Question:  من هو في سي سي في هذه الرسالة؟
Words: ['wie', 'baw', 'brown', '&', 'williamson', 'tobacco', 'corporation', 'research', '&', 'development', 'internal', 'correspondence', 'to:', 'r.', 'h.', 'honeycutt', 'ce:', 't.f.', 'riehl', 'from:', '.', 'c.j.', 'cook', 'date:', 'may', '8,', '1995', 'subject:', 'review', 'of', 'existing', 'brainstorming', 'ideas/483', 'the', 'major', 'function', 'of', 'the', 'product', 'innovation', 'graup', 'is', 'to', 'develop', 'marketable', 'nove!', 'products', 'that', 'would', 'be', 'profitable', 'to', 'manufacture', 'and', 'sell.', 'novel', 'is', 'defined', 'as:', 'of', 'a', 'new', 'kind,', 'or', 'different', 'from', 'anything', 'seen', 'or', 'known', 'before.', 'innovation', 'is', 'defined', 'as:', 'something', 'new', 'or', 'different', 'introduced;', 'act', 'of', 'innovating;', 'introduction', 'of', 'new', 'things', 'or', 'methods.', 'the', 'products', 'may', 'incorporate', 'the', 'latest', 'technologies,', 'materials', 'and', 'know-how', 'available', 'to', 'give', 'then', 'a', 'unique', 'taste', 'or', 'look.', 'the', 'first', 'task', 'of', 'the', 'product', 'innovation', 'group', 'was', 'to', 'assemble,', 'review', 'and', 'categorize', 'a', 'list', 'of', 'existing', 'brainstorming', 'ideas.', 'ideas', 'were', 'grouped', 'into', 'two', 'major', 'categories', 'labeled', 'appearance', 'and', 'taste/aroma.', 'these', 'categories', 'are', 'used', 'for', 'novel', 'products', 'that', 'may', 'differ', 'from', 'a', 'visual', 'and/or', 'taste/aroma', 'point', 'of', 'view', 'compared', 'to', 'canventional', 'cigarettes.', 'other', 'categories', 'include', 'a', 'combination', 'of', 'the', 'above,', 'filters,', 'packaging', 'and', 'brand', 'extensions.', 'appearance', 'this', 'category', 'is', 'used', 'for', 'novel', 'cigarette', 'constructions', 'that', 'yield', 'visually', 'different', 'products', 'with', 'minimal', 'changes', 'in', 'smoke', 'chemistry', 'two', 'cigarettes', 'in', 'cne.', 'emulti-plug', 'te', 'build', 'yaur', 'awn', 'cigarette.', 'eswitchable', 'menthol', 'or', 'non', 'menthol', 'cigarette.', '*cigarettes', 'with', 'interspaced', 'perforations', 'to', 'enable', 'smoker', 'to', 'separate', 'unburned', 'section', 'for', 'future', 'smoking.', '«short', 'cigarette,', 'tobacco', 'section', '30', 'mm.', '«extremely', 'fast',
هذا هو النص المترجم مع اتباع التعليمات التي قدمتها: 

دعونا نتحقق من شكل خصائص مجموعة البيانات المشفرة:

```py
>>> encoded_train_dataset.features
{'image': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='uint8', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None),
 'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
 'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'bbox': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'start_positions': Value(dtype='int64', id=None),
 'end_positions': Value(dtype='int64', id=None)}
```

## التقييم

يتطلب تقييم الإجابة على الأسئلة المتعلقة بالوثائق قدرًا كبيرًا من المعالجة اللاحقة. ولتجنب استغراق الكثير من وقتك، يتخطى هذا الدليل خطوة التقييم. لا يزال [`Trainer`] يحسب خسارة التقييم أثناء التدريب حتى تظل على دراية بأداء نموذجك. عادةً ما يتم تقييم الإجابة الاستخراجية على الأسئلة باستخدام F1/exact match.

إذا كنت ترغب في تنفيذها بنفسك، فراجع فصل [Question Answering](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) في دورة Hugging Face للاستلهام.

## التدريب

تهانينا! لقد نجحت في تخطي أصعب جزء في هذا الدليل، والآن أنت مستعد لتدريب نموذجك الخاص. ينطوي التدريب على الخطوات التالية:

* قم بتحميل النموذج باستخدام [`AutoModelForDocumentQuestionAnswering`] باستخدام نفس نقطة التفتيش كما في مرحلة ما قبل المعالجة.
* حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`].
* حدد دالة لدمج الأمثلة معًا، حيث ستكون [`DefaultDataCollator`] مناسبة تمامًا
* قم بتمرير فرط معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات ودمج البيانات.
* استدعاء [`~Trainer.train`] لضبط نموذجك الدقيق.

```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```

في [`TrainingArguments`]، استخدم `output_dir` لتحديد مكان حفظ نموذجك، وقم بتكوين فرط المعلمات كما تراه مناسبًا.

إذا كنت ترغب في مشاركة نموذجك مع المجتمع، قم بتعيين `push_to_hub` إلى `True` (يجب أن تكون قد سجلت الدخول إلى Hugging Face لتحميل نموذجك). في هذه الحالة، سيكون `output_dir` أيضًا اسم المستودع حيث سيتم دفع نقطة تفتيش النموذج الخاص بك.

```py
>>> from transformers import TrainingArguments

>>> # استبدل هذا بمعرف مستودعك
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

قم بتعريف دالة بسيطة لدمج البيانات لدمج الأمثلة معًا.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

أخيرًا، قم بجمع كل شيء معًا، واستدعاء [`~Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     tokenizer=processor,
... )

>>> trainer.train()
```

لإضافة النموذج النهائي إلى 🤗 Hub، قم بإنشاء بطاقة نموذج واستدعاء `push_to_hub`:

```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## الاستنتاج

الآن بعد أن قمت بضبط نموذج LayoutLMv2 وتحميله إلى 🤗 Hub، يمكنك استخدامه للاستنتاج. أسهل طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`Pipeline`].

دعنا نأخذ مثالاً:

```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```

بعد ذلك، قم بتنفيذ خط أنابيب للإجابة على الأسئلة المتعلقة بالوثائق باستخدام نموذجك، ومرر مزيج الصورة + السؤال إليه.

```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
  'answer': 'Lee A. Waller',
  'start': 55,
  'end': 57}]
```

يمكنك أيضًا محاكاة نتائج خط الأنابيب يدويًا إذا كنت ترغب في ذلك:

1. خذ صورة وسؤال، وقم بإعدادهما للنموذج باستخدام المعالج من نموذجك.
2. قم بتمرير نتيجة المعالجة المسبقة عبر النموذج.
3. يعيد النموذج `start_logits` و`end_logits`، والتي تشير إلى الرمز الذي يكون في بداية الإجابة والرمز الذي يكون في نهاية الإجابة. كلاهما له شكل (batch_size، sequence_length).
4. قم بإجراء argmax على البعد الأخير لكل من `start_logits` و`end_logits` للحصول على `start_idx` المتوقع و`end_idx`.
5. فك تشفير الإجابة باستخدام المعالج.

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