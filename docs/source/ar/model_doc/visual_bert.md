# VisualBERT

## نظرة عامة

اقتُرح نموذج VisualBERT في ورقة بحثية بعنوان "VisualBERT: A Simple and Performant Baseline for Vision and Language" من قبل Liunian Harold Li و Mark Yatskar و Da Yin و Cho-Jui Hsieh و Kai-Wei Chang.

VisualBERT هو شبكة عصبية تم تدريبها على مجموعة متنوعة من أزواج الصور والنصوص.

الملخص المستخرج من الورقة البحثية هو كما يلي:

> *نقترح VisualBERT، وهو إطار عمل بسيط ومرن لنمذجة مجموعة واسعة من مهام الرؤية واللغة. يتكون VisualBERT من مجموعة من طبقات Transformer التي تقوم بمواءمة ضمنية لعناصر نص الإدخال والمناطق في صورة الإدخال المرتبطة باستخدام الاهتمام الذاتي. نقترح أيضًا هدفين لنموذج اللغة المدعوم بصريًا لمرحلة ما قبل التدريب على بيانات تعليق الصور. تُظهر التجارب على أربع مهام للرؤية واللغة بما في ذلك VQA و VCR و NLVR2 و Flickr30K أن VisualBERT يتفوق على النماذج الحالية أو ينافسها مع كونه أبسط بكثير. يُظهر التحليل الإضافي أن VisualBERT يمكنه ربط عناصر اللغة بمناطق الصورة دون أي إشراف صريح، وهو حساس حتى للعلاقات النحوية، ويتتبع على سبيل المثال الارتباطات بين الأفعال ومناطق الصورة المقابلة لحججها.*

تمت المساهمة بهذا النموذج من قبل [gchhablani](https://huggingface.co/gchhablani). يمكن العثور على الكود الأصلي [هنا](https://github.com/uclanlp/visualbert).

## نصائح الاستخدام

1. تعمل معظم نقاط التفتيش المقدمة مع تكوين [`VisualBertForPreTraining`] . نقاط التفتيش الأخرى المقدمة هي نقاط تفتيش ضبط دقيق لمهام المصب - VQA ('visualbert-vqa')، VCR ('visualbert-vcr')، NLVR2 ('visualbert-nlvr2'). لذلك، إذا كنت لا تعمل على مهام المصب هذه، يوصى باستخدام نقاط التفتيش المعاد تدريبها.

2. بالنسبة لمهمة VCR، يستخدم المؤلفون كاشفًا مضبوطًا دقيقًا لتوليد التضمينات المرئية، لجميع نقاط التفتيش. لا نوفر الكاشف وأوزانه كجزء من الحزمة، ولكنه سيكون متاحًا في مشاريع البحث، ويمكن تحميل حالاته مباشرة في الكاشف المقدم.

VisualBERT هو نموذج متعدد الوسائط للرؤية واللغة. يمكن استخدامه للإجابة على الأسئلة المرئية، والاختيار من متعدد، والاستدلال المرئي، ومهام المطابقة بين المنطقة والجملة. يستخدم VisualBERT محول BERT-like لإعداد التضمينات لأزواج الصور والنصوص. يتم بعد ذلك إسقاط كل من الميزات النصية والمرئية إلى مساحة كامنة ذات أبعاد متطابقة.

لتغذية الصور إلى النموذج، يتم تمرير كل صورة عبر كاشف كائنات مُدرب مسبقًا، واستخراج المناطق وحدود الصناديق. يستخدم المؤلفون الميزات التي تم إنشاؤها بعد تمرير هذه المناطق عبر شبكة عصبية CNN مُدربة مسبقًا مثل ResNet كتضمينات بصرية. كما أنهم يضيفون تضمينات الموضع المطلق، ويغذون تسلسل المتجهات الناتج إلى نموذج BERT القياسي. يتم دمج الإدخال النصي في مقدمة التضمينات المرئية في طبقة التضمين، ومن المتوقع أن يكون محاطًا برمزي [CLS] و [SEP]، كما هو الحال في BERT. يجب أيضًا تعيين معرفات القطاع بشكل مناسب للأجزاء النصية والمرئية.

يتم استخدام [`BertTokenizer`] لتشفير النص. يجب استخدام كاشف/معالج صور مخصص للحصول على التضمينات المرئية. توضح دفاتر الملاحظات التالية كيفية استخدام VisualBERT مع النماذج المشابهة لـ Detectron:

- [VisualBERT VQA demo notebook](https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert): يحتوي هذا الدفتر على مثال حول VisualBERT VQA.

- [Generate Embeddings for VisualBERT (Colab Notebook)](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing): يحتوي هذا الدفتر على مثال حول كيفية إنشاء التضمينات المرئية.

يوضح المثال التالي كيفية الحصول على حالة الإخفاء الأخيرة باستخدام [`VisualBertModel`]:

```python
>>> import torch
>>> from transformers import BertTokenizer, VisualBertModel

>>> model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("What is the man eating?", return_tensors="pt")
>>> # هذه دالة مخصصة تعيد التضمينات المرئية بالنظر إلى مسار الصورة
>>> visual_embeds = get_visual_embeddings(image_path)

>>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
>>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
>>> inputs.update(
...     {
...         "visual_embeds": visual_embeds,
...         "visual_token_type_ids": visual_token_type_ids,
...         "visual_attention_mask": visual_attention_mask,
...     }
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

## VisualBertConfig

[[autodoc]] VisualBertConfig

## VisualBertModel

[[autodoc]] VisualBertModel

- forward

## VisualBertForPreTraining

[[autodoc]] VisualBertForPreTraining

- forward

## VisualBertForQuestionAnswering

[[autodoc]] VisualBertForQuestionAnswering

- forward

## VisualBertForMultipleChoice

[[autodoc]] VisualBertForMultipleChoice

- forward

## VisualBertForVisualReasoning

[[autodoc]] VisualBertForVisualReasoning

- forward

## VisualBertForRegionToPhraseAlignment

[[autodoc]] VisualBertForRegionToPhraseAlignment

- forward