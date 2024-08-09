# CANINE

## نظرة عامة

اقتُرح نموذج CANINE في الورقة البحثية المعنونة "CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation" بواسطة Jonathan H. Clark وDan Garrette وIulia Turc وJohn Wieting. وتعد هذه الورقة من أوائل الأوراق التي تدرب محولًا بدون استخدام خطوة تمييز صريح (مثل ترميز زوج البايت أو WordPiece أو SentencePiece). وبدلاً من ذلك، يتم تدريب النموذج مباشرة على مستوى أحرف Unicode. ويأتي التدريب على مستوى الأحرف حتمًا بطول تسلسل أطول، والذي يحله CANINE باستخدام استراتيجية تقليل فعالة، قبل تطبيق محول ترميز عميق.

وفيما يلي الملخص المستخرج من الورقة:

*على الرغم من أن أنظمة NLP المتسلسلة قد حلت محلها إلى حد كبير نمذجة عصبية من النهاية إلى النهاية، إلا أن معظم النماذج الشائعة الاستخدام لا تزال تتطلب خطوة تمييز صريحة. وفي حين أن أساليب التمييز الحديثة القائمة على معاجم الكلمات الفرعية المستمدة من البيانات أقل هشاشة من أدوات التمييز المصممة يدويًا، إلا أن هذه التقنيات لا تناسب جميع اللغات، وقد تحد أي مفردات ثابتة من قدرة النموذج على التكيف. وفي هذه الورقة، نقدم CANINE، وهو محول عصبي يعمل مباشرة على تسلسلات الأحرف، بدون تمييز أو مفردات صريحة، واستراتيجية تدريب مسبق تعمل إما مباشرة على الأحرف أو تستخدم الكلمات الفرعية كتحيز استقرائي ناعم. ولاستخدام مدخلاته الدقيقة بشكل فعال وكفء، يجمع CANINE بين تقليل العينات، الذي يقلل من طول تسلسل المدخلات، مع مكدس محول الترميز العميق، الذي يشفر السياق. ويتفوق CANINE على نموذج mBERT المماثل بمقدار 2.8 F1 على TyDi QA، وهو معيار مرجعي متعدد اللغات، على الرغم من أن لديه 28% معلمات نموذج أقل.*

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/language/tree/master/language/canine).

## نصائح الاستخدام

- يستخدم CANINE داخليًا ما لا يقل عن 3 محولات ترميز: محولان "ضحلان" (يتكونان من طبقة واحدة فقط) ومحول ترميز "عميق" (وهو محول ترميز BERT عادي). أولاً، يتم استخدام محول ترميز "ضحل" لسياق تضمين الأحرف، باستخدام الانتباه المحلي. بعد ذلك، بعد تقليل العينات، يتم تطبيق محول الترميز "العميق". وأخيراً، بعد زيادة العينات، يتم استخدام محول ترميز "ضحل" لإنشاء تضمين الأحرف النهائي. يمكن العثور على التفاصيل المتعلقة بزيادة وتقليل العينات في الورقة.

- يستخدم CANINE بشكل افتراضي طول تسلسل أقصاه 2048 حرفًا. يمكن استخدام [`CanineTokenizer`] لإعداد النص للنموذج.

- يمكن إجراء التصنيف عن طريق وضع طبقة خطية أعلى حالة الإخفاء النهائية للرمز الخاص [CLS] (الذي له نقطة رمز Unicode محددة مسبقًا). ومع ذلك، بالنسبة لمهام تصنيف الرموز، يجب زيادة تسلسل الرموز المخفضة مرة أخرى لمطابقة طول تسلسل الأحرف الأصلي (الذي يبلغ 2048). يمكن العثور على التفاصيل الخاصة بذلك في الورقة.

نقاط تفتيش النموذج:

- [google/canine-c](https://huggingface.co/google/canine-c): تم التدريب المسبق باستخدام خسارة الأحرف ذاتية الارتباط، 12 طبقة، 768 مخفية، 12 رأسًا، 121 مليون معلمة (الحجم ~500 ميجابايت).

- [google/canine-s](https://huggingface.co/google/canine-s): تم التدريب المسبق باستخدام خسارة الكلمات الفرعية، 12 طبقة، 768 مخفية، 12 رأسًا، 121 مليون معلمة (الحجم ~500 ميجابايت).

## مثال على الاستخدام

يعمل CANINE على الأحرف الخام، لذا يمكن استخدامه **بدون أداة تمييز**:

```python
>>> from transformers import CanineModel
>>> import torch

>>> model = CanineModel.from_pretrained("google/canine-c") # model pre-trained with autoregressive character loss

>>> text = "hello world"
>>> # use Python's built-in ord() function to turn each character into its unicode code point id
>>> input_ids = torch.tensor([[ord(char) for char in text]])

>>> outputs = model(input_ids) # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

بالنسبة للاستدلال بالدفعات والتدريب، يوصى مع ذلك باستخدام أداة التمييز (لضبط/اقتصاص جميع التسلسلات إلى نفس الطول):

```python
>>> from transformers import CanineTokenizer, CanineModel

>>> model = CanineModel.from_pretrained("google/canine-c")
>>> tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

>>> inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
>>> encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

>>> outputs = model(**encoding) # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)

- [دليل مهمة تصنيف الرموز](../tasks/token_classification)

- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)

- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## CanineConfig

[[autodoc]] CanineConfig

## CanineTokenizer

[[autodoc]] CanineTokenizer

- build_inputs_with_special_tokens

- get_special_tokens_mask

- create_token_type_ids_from_sequences

## المخرجات الخاصة بـ CANINE

[[autodoc]] models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineModel

[[autodoc]] CanineModel

- forward

## CanineForSequenceClassification

[[autodoc]] CanineForSequenceClassification

- forward

## CanineForMultipleChoice

[[autodoc]] CanineForMultipleChoice

- forward

## CanineForTokenClassification

[[autodoc]] CanineForTokenClassification

- forward

## CanineForQuestionAnswering

[[autodoc]] CanineForQuestionAnswering

- forward