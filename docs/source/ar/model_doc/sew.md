# SEW

## نظرة عامة

اقترح SEW (Squeezed and Efficient Wav2Vec) في ورقة "Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition" بواسطة Felix Wu وKwangyoun Kim وJing Pan وKyu Han وKilian Q. Weinberger وYoav Artzi.

ملخص الورقة هو كما يلي:

*هذه الورقة هي دراسة لمقايضات الأداء والكفاءة في النماذج مسبقة التدريب للتعرف التلقائي على الكلام (ASR). نركز على wav2vec 2.0، ونقوم بتنظيم عدة تصاميم معمارية تؤثر على أداء النموذج وكفاءته. من خلال جمع جميع ملاحظاتنا، نقدم SEW (Squeezed and Efficient Wav2vec)، وهو تصميم معماري لنماذج مسبقة التدريب مع تحسينات كبيرة على أبعاد الأداء والكفاءة عبر مجموعة متنوعة من إعدادات التدريب. على سبيل المثال، في إعداد التدريب شبه الخاضع للإشراف 100h-960h على LibriSpeech، يحقق SEW تسريعًا في الاستدلال يبلغ 1.9x مقارنة بـ wav2vec 2.0، مع انخفاض نسبته 13.5% في معدل خطأ الكلمة. وبزمن استدلال مماثل، يقلل SEW من معدل خطأ الكلمة بنسبة 25-50% عبر أحجام نماذج مختلفة.*

تمت المساهمة بهذا النموذج من قبل [anton-l](https://huggingface.co/anton-l).

## نصائح الاستخدام

- SEW هو نموذج كلام يقبل مصفوفة عائمة تتوافق مع الشكل الموجي الخام لإشارة الكلام.

- يتم ضبط دقة SEWForCTC باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].

## الموارد

- [دليل مهام التصنيف الصوتي](../tasks/audio_classification)

- [دليل مهام التعرف التلقائي على الكلام](../tasks/asr)

## SEWConfig

[[autodoc]] SEWConfig

## SEWModel

[[autodoc]] SEWModel

- forward

## SEWForCTC

[[autodoc]] SEWForCTC

- forward

## SEWForSequenceClassification

[[autodoc]] SEWForSequenceClassification

- forward