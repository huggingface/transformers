# UMT5

## نظرة عامة

اقترح نموذج UMT5 في [UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining](https://openreview.net/forum?id=kXwdL1cWOAi) بواسطة Hyung Won Chung و Xavier Garcia و Adam Roberts و Yi Tay و Orhan Firat و Sharan Narang و Noah Constant.

ملخص الورقة هو ما يلي:

* تستخدم النماذج اللغوية الكبيرة متعددة اللغات التي تم تدريبها مسبقًا عادةً عينات تعتمد على درجة حرارة الحدس لموازنة اللغات المختلفة. ومع ذلك، لم يقيم العمل السابق بشكل منهجي فعالية توزيعات اللغات المختلفة عبر نطاقات النماذج. في هذه الورقة، نقترح طريقة جديدة للعينات، تسمى UniMax، والتي توفر تغطية أكثر اتساقًا للغات الرأس مع تقليل الإفراط في تناسب لغات الذيل من خلال الحد صراحة من عدد التكرارات عبر مجموعة بيانات كل لغة. نقوم بسلسلة واسعة من عمليات الحجب التي تختبر مجموعة من استراتيجيات الاستخراج على مجموعة من المعايير المرجعية متعددة اللغات، مع تغيير نطاق النموذج. نجد أن UniMax يتفوق على العينات القياسية القائمة على درجة الحرارة، وتستمر الفوائد مع زيادة الحجم. كجزء من مساهمتنا، نقوم بإصدار: (1) مجموعة بيانات mC4 المحسنة والمحسنة متعددة اللغات والتي تتكون من 29 تريليون حرف عبر 107 لغات، و (2) مجموعة من نقاط التحقق من نموذج umT5 المسبق التدريب المدرب باستخدام عينات UniMax.

أصدرت Google المتغيرات التالية:

- [google/umt5-small](https://huggingface.co/google/umt5-small)
- [google/umt5-base](https://huggingface.co/google/umt5-base)
- [google/umt5-xl](https://huggingface.co/google/umt5-xl)
- [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).

تمت المساهمة بهذا النموذج بواسطة [agemagician](https://huggingface.co/agemagician) و [stefan-it](https://huggingface.co/stefan-it). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/t5x).

## نصائح الاستخدام

- تم تدريب UMT5 مسبقًا فقط على [mC4](https://huggingface.co/datasets/mc4) مع استبعاد أي تدريب خاضع للإشراف. لذلك، يجب ضبط دقة هذا النموذج قبل استخدامه في مهمة تدفق أسفل، على عكس نموذج T5 الأصلي.
- نظرًا لأن umT5 تم تدريبه مسبقًا بطريقة غير خاضعة للإشراف، فلا توجد ميزة حقيقية لاستخدام بادئة المهمة أثناء الضبط الدقيق لمهمة واحدة. إذا كنت تقوم بالضبط الدقيق متعدد المهام، فيجب عليك استخدام بادئة.

## ما الاختلافات مع mT5؟

`UmT5` مبني على mT5، مع انحياز موضعي نسبي غير مشترك يتم حسابه لكل طبقة. وهذا يعني أن النموذج يحدد `has_relative_bias` لكل طبقة.

يختلف نص البرنامج النصي للتحويل أيضًا لأن النموذج تم حفظه بتنسيق أحدث لنقاط تفتيش t5x.

# مثال الاستخدام

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")

>>> inputs = tokenizer(
...     "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>.",
...     return_tensors="pt",
... )
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs))
['<pad><extra_id_0>nyone who<extra_id_1> drink<extra_id_2> a<extra_id_3> alcohol<extra_id_4> A<extra_id_5> A. This<extra_id_6> I<extra_id_7><extra_id_52><extra_id_53></s>']
```

<Tip>

راجع [صفحة وثائق T5](t5) للحصول على مزيد من النصائح وأمثلة التعليمات البرمجية ومفكرات Jupyter.

</Tip>

## UMT5Config

[[autodoc]] UMT5Config

## UMT5Model

[[autodoc]] UMT5Model

- forward

## UMT5ForConditionalGeneration

[[autodoc]] UMT5ForConditionalGeneration

- forward

## UMT5EncoderModel

[[autodoc]] UMT5EncoderModel

- forward

## UMT5ForSequenceClassification

[[autodoc]] UMT5ForSequenceClassification

- forward

## UMT5ForTokenClassification

[[autodoc]] UMT5ForTokenClassification

- forward

## UMT5ForQuestionAnswering

[[autodoc]] UMT5ForQuestionAnswering

- forward