# LUKE

## نظرة عامة
اقترح نموذج LUKE في ورقة البحثية بعنوان: "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention" من قبل Ikuya Yamada وAkari Asai وHiroyuki Shindo وHideaki Takeda وYuji Matsumoto.

يستند النموذج إلى RoBERTa ويضيف تضمين الكيانات، بالإضافة إلى آلية انتباه ذاتي على دراية بالكيانات، مما يساعد على تحسين الأداء في مهام مختلفة لأسفل تتضمن الاستدلال حول الكيانات مثل التعرف على الكيانات المسماة، والأسئلة والإجابات الاستخلاصية وأسلوب الإكمال، وتصنيف الكيانات والعلاقات.

الملخص من الورقة هو كما يلي:

*تمثل الكيانات مفيدة في المهام اللغوية الطبيعية التي تتضمن الكيانات. في هذه الورقة، نقترح تمثيلات سياقية جديدة للكلمات والكيانات بناءً على المحول ثنائي الاتجاه. ويعامل النموذج المقترح الكلمات والكيانات في نص معين كعلامات مستقلة، ويقوم بإخراج تمثيلات سياقية لها. يتم تدريب النموذج باستخدام مهمة تدريب مسبق جديدة بناءً على نموذج اللغة المقنعة من BERT. تتضمن المهمة التنبؤ بالكلمات والكيانات المقنعة بشكل عشوائي في مجموعة بيانات كبيرة موسومة بالكيانات تم استردادها من ويكيبيديا. نقترح أيضًا آلية انتباه ذاتي على دراية بالكيانات وهي امتداد لآلية الانتباه الذاتي لمحول، وتأخذ في الاعتبار أنواع العلامات (كلمات أو كيانات) عند حساب درجات الانتباه. يحقق النموذج المقترح أداءً تجريبيًا مثيرًا للإعجاب في مجموعة واسعة من المهام المتعلقة بالكيانات. وعلى وجه الخصوص، فإنه يحصل على نتائج متقدمة في خمس مجموعات بيانات شهيرة: Open Entity (تصنيف الكيانات)، TACRED (تصنيف العلاقات)، CoNLL-2003 (التعرف على الكيانات المسماة)، ReCoRD (الأسئلة والإجابات على طريقة الإكمال)، وSQuAD 1.1 (الأسئلة والإجابات الاستخلاصية).*

تمت المساهمة بهذا النموذج من قبل [ikuyamada](https://huggingface.co/ikuyamada) و [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/studio-ousia/luke).

## نصائح الاستخدام

- هذا التنفيذ هو نفسه [`RobertaModel`] مع إضافة تضمين الكيانات وكذلك آلية الانتباه الذاتي على دراية بالكيانات، والتي تحسن الأداء في المهام التي تتضمن الاستدلال حول الكيانات.

- يعامل LUKE الكيانات كعلامات إدخال؛ لذلك، فإنه يأخذ `entity_ids` و`entity_attention_mask` و`entity_token_type_ids` و`entity_position_ids` كإدخال إضافي. يمكنك الحصول على هذه المعلومات باستخدام [`LukeTokenizer`].

- يأخذ [`LukeTokenizer`] `entities` و`entity_spans` (مواضع البداية والنهاية المستندة إلى الأحرف للكيانات في نص الإدخال) كإدخال إضافي. تتكون `entities` عادةً من كيانات [MASK] أو كيانات Wikipedia. الوصف المختصر عند إدخال هذه الكيانات هو كما يلي:

- *إدخال كيانات [MASK] لحساب تمثيلات الكيانات*: تستخدم كيان [MASK] لإخفاء الكيانات التي سيتم التنبؤ بها أثناء التدريب المسبق. عندما يتلقى LUKE كيان [MASK]، فإنه يحاول التنبؤ بالكيان الأصلي عن طريق جمع المعلومات حول الكيان من نص الإدخال. وبالتالي، يمكن استخدام كيان [MASK] لمعالجة المهام لأسفل التي تتطلب معلومات الكيانات في النص مثل تصنيف الكيانات، وتصنيف العلاقات، والتعرف على الكيانات المسماة.

- *إدخال كيانات Wikipedia للحصول على تمثيلات الرموز المعززة بالمعرفة*: يتعلم LUKE معلومات غنية (أو معرفة) حول كيانات Wikipedia أثناء التدريب المسبق ويخزن المعلومات في تضمين الكيان الخاص به. من خلال استخدام كيانات Wikipedia كرموز إدخال، يقوم LUKE بإخراج تمثيلات الرموز المعززة بالمعلومات المخزنة في تضمين هذه الكيانات. وهذا فعال بشكل خاص للمهام التي تتطلب معرفة العالم الحقيقي، مثل الأسئلة والإجابات.

- هناك ثلاثة نماذج رأس لحالة الاستخدام السابقة:

- [`LukeForEntityClassification`]، للمهام التي تصنف كيانًا واحدًا في نص الإدخال مثل تصنيف الكيانات، على سبيل المثال مجموعة بيانات [Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html). يضع هذا النموذج رأسًا خطيًا أعلى تمثيل الكيان الإخراج.

- [`LukeForEntityPairClassification`]، للمهام التي تصنف العلاقة بين كيانين مثل تصنيف العلاقات، على سبيل المثال مجموعة بيانات [TACRED](https://nlp.stanford.edu/projects/tacred/). يضع هذا النموذج رأسًا خطيًا أعلى التمثيل الموحد لزوج الكيانات المعطاة.

- [`LukeForEntitySpanClassification`]، للمهام التي تصنف تسلسل نطاقات الكيانات، مثل التعرف على الكيانات المسماة (NER). يضع هذا النموذج رأسًا خطيًا أعلى تمثيلات الكيانات الإخراج. يمكنك معالجة NER باستخدام هذا النموذج عن طريق إدخال جميع نطاقات الكيانات المحتملة في النص إلى النموذج.

لدى [`LukeTokenizer`] حجة `task`، والتي تتيح لك إنشاء إدخال لهذه النماذج الرأس بسهولة عن طريق تحديد `task="entity_classification"` أو `task="entity_pair_classification"` أو `task="entity_span_classification"`. يرجى الرجوع إلى كود المثال لكل نموذج رأس.

مثال على الاستخدام:

```python
>>> from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification

>>> model = LukeModel.from_pretrained("studio-ousia/luke-base")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
# مثال 1: حساب التمثيل السياقي للكيان المقابل لذكر الكيان "Beyoncé"

>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [(0, 7)]  # entity span المستندة إلى الأحرف المقابلة لـ "Beyoncé"
>>> inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**inputs)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state
# مثال 2: إدخال كيانات Wikipedia للحصول على تمثيلات سياقية مثراة

>>> entities = [
...     "Beyoncé",
...     "Los Angeles",
... ]  # عناوين كيانات Wikipedia المقابلة لذكر الكيانات "Beyoncé" و "Los Angeles"
>>> entity_spans = [(0, 7), (17, 28)]  # entity spans المستندة إلى الأحرف المقابلة لـ "Beyoncé" و "Los Angeles"
>>> inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**inputs)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state
# مثال 3: تصنيف العلاقة بين كيانين باستخدام نموذج رأس LukeForEntityPairClassification

>>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> entity_spans = [(0, 7), (17, 28)]  # entity spans المستندة إلى الأحرف المقابلة لـ "Beyoncé" و "Los Angeles"
>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_idx = int(logits[0].argmax())
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## الموارد

- [مفكرة توضيحية حول كيفية ضبط نموذج [`LukeForEntityPairClassification`] لتصنيف العلاقات](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LUKE)

- [مفكرات توضح كيفية إعادة إنتاج النتائج كما هو موضح في الورقة باستخدام تنفيذ HuggingFace لـ LUKE](https://github.com/studio-ousia/luke/tree/master/notebooks)

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)

- [دليل مهمة تصنيف الرموز](../tasks/token_classification)

- [دليل مهمة الأسئلة والإجابات](../tasks/question_answering)

- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)

- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## LukeConfig

[[autodoc]] LukeConfig

## LukeTokenizer

[[autodoc]] LukeTokenizer

- __call__

- save_vocabulary

## LukeModel

[[autodoc]] LukeModel

- forward

## LukeForMaskedLM

[[autodoc]] LukeForMaskedLM

- forward

## LukeForEntityClassification

[[autodoc]] LukeForEntityClassification

- forward

## LukeForEntityPairClassification

[[autodoc]] LukeForEntityPairClassification

- forward

## LukeForEntitySpanClassification

[[autodoc]] LukeForEntitySpanClassification

- forward

## LukeForSequenceClassification

[[autodoc]] LukeForSequenceClassification

- forward

## LukeForMultipleChoice

[[autodoc]] LukeForMultipleChoice

- forward

## LukeForTokenClassification

[[autodoc]] LukeForTokenClassification

- forward

## LukeForQuestionAnswering

[[autodoc]] LukeForQuestionAnswering

- forward