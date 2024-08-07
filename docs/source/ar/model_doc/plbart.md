# PLBart

## نظرة عامة
تم اقتراح نموذج PLBART في [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) بواسطة Wasi Uddin Ahmad و Saikat Chakraborty و Baishakhi Ray و Kai-Wei Chang.
هذا النموذج مشابه لنموذج BART ويمكن استخدامه لأداء مهام توليد الملخصات البرمجية والترجمة البرمجية وتوليد الأكواد. تم تدريب النموذج المُدرب مسبقًا `plbart-base` باستخدام مهمة إزالة التشويش متعددة اللغات
على Java و Python واللغة الإنجليزية.

وفقًا للملخص:

> "تُمكّن عملية توليد الملخصات البرمجية والترجمة البرمجية التحويل بين لغة البرمجة (PL) واللغة الطبيعية (NL)، بينما تتيح ترجمة الأكواد إمكانية نقل الأكواد القديمة من لغة برمجة إلى أخرى. يقدم هذا البحث PLBART، وهو نموذج تسلسلي إلى تسلسلي قادر على أداء مجموعة واسعة من مهام فهم وتوليد الأكواد واللغات.
تم تدريب PLBART مسبقًا على مجموعة واسعة من وظائف Java و Python والنصوص المرتبطة باللغة الطبيعية من خلال الترميز التلقائي لإزالة التشويش.
تُظهر التجارب على توليد ملخصات الأكواد باللغة الإنجليزية، وتوليد الأكواد، وترجمة الأكواد في سبع لغات برمجة
أن PLBART يتفوق على النماذج الحالية أو ينافسها. علاوة على ذلك، تُظهر التجارب على المهام التمييزية، مثل إصلاح الأكواد، وكشف النسخ المماثلة، واكتشاف الأكواد المعرضة للخطر، فعالية PLBART في فهم الأكواد.
علاوة على ذلك، يكشف التحليل أن PLBART يتعلم بناء جملة الأكواد، والأسلوب (مثل اتفاقية تسمية المعرفات)، والتدفق المنطقي
(على سبيل المثال، كتلة if داخل كتلة else مكافئة لكتلة else if) والتي تعتبر حاسمة لدلالات الأكواد، وبالتالي يتفوق
حتى مع الملاحظات المحدودة."

تمت المساهمة بهذا النموذج من قبل [gchhablani](https://huggingface.co/gchhablani). يمكن العثور على كود المؤلفين [هنا](https://github.com/wasiahmad/PLBART).

## أمثلة الاستخدام
PLBart هو نموذج متعدد اللغات للتكويد وفك التكويد (التسلسلي إلى التسلسلي) المقصود في المقام الأول لمهام تحويل الكود إلى نص، والنص إلى كود، والتحويل بين الأكواد. نظرًا لأن
النموذج متعدد اللغات، فإنه يتوقع تسلسلًا بتنسيق مختلف. تتم إضافة رمز معرف اللغة الخاص في كل من
نص المصدر والهدف. تنسيق نص المصدر هو `X [eos, src_lang_code]` حيث `X` هو نص المصدر.
تنسيق النص المستهدف هو `[tgt_lang_code] X [eos]`. لا يتم استخدام `bos` مطلقًا.

ومع ذلك، في بعض الحالات، لا يتم توفير رمز اللغة أثناء الضبط الدقيق عند استخدام لغة واحدة فقط. يرجى الرجوع إلى [الورقة البحثية](https://arxiv.org/abs/2103.06333) لمعرفة المزيد حول هذا الموضوع.

في الحالات التي تكون فيها رموز اللغة مطلوبة، ستقوم الدالة العادية [`~PLBartTokenizer.__call__`] بتشفير تنسيق نص المصدر
عندما تقوم بتمرير النصوص كأول حجة أو باستخدام الحجة الكلمة `text`، وستقوم بتشفير تنسيق نص الهدف إذا
تم تمريره باستخدام الحجة الكلمة `text_target`.

### التدريب المُشرف

```python
>>> from transformers import PLBartForConditionalGeneration, PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base", src_lang="en_XX", tgt_lang="python")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> expected_translation_english = "Returns the maximum value of a b c."
>>> inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
>>> model(**inputs)
```

### التوليد

عند توليد نص الهدف، قم بتعيين `decoder_start_token_id` إلى معرف لغة الهدف. يوضح المثال التالي كيفية ترجمة Python إلى الإنجليزية باستخدام نموذج `uclanlp/plbart-python-en_XX`.

```python
>>> from transformers import PLBartForConditionalGeneration, PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> inputs = tokenizer(example_python_phrase, return_tensors="pt")
>>> model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-python-en_XX")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Returns the maximum value of a b c."
```

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة النمذجة اللغوية السببية](../tasks/language_modeling)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة توليد الملخصات](../tasks/summarization)

## PLBartConfig

[[autodoc]] PLBartConfig

## PLBartTokenizer

[[autodoc]] PLBartTokenizer

- build_inputs_with_special_tokens

## PLBartModel

[[autodoc]] PLBartModel

- forward

## PLBartForConditionalGeneration

[[autodoc]] PLBartForConditionalGeneration

- forward

## PLBartForSequenceClassification

[[autodoc]] PLBartForSequenceClassification

- forward

## PLBartForCausalLM

[[autodoc]] PLBartForCausalLM

- forward