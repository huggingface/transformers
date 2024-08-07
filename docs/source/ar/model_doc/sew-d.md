# SEW-D

## نظرة عامة

اقترح SEW-D (Squeezed and Efficient Wav2Vec with Disentangled attention) في ورقة "Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition" من قبل فيليكس وو، وكوانجيون كيم، وجينغ بان، وكيو هان، وكيليان كيو. واينبرجر، ويواف أرتزي.

ملخص الورقة هو ما يلي:

*هذه الورقة هي دراسة لمفاضلات الأداء والكفاءة في النماذج المُدربة مسبقًا للتعرف التلقائي على الكلام (ASR). نركز على Wav2vec 2.0، ونقوم بتنظيم عدة تصاميم معمارية تؤثر على أداء النموذج وكفاءته. من خلال جمع جميع ملاحظاتنا، نقدم SEW (Squeezed and Efficient Wav2vec)، وهو تصميم معماري للنموذج المُدرب مسبقًا مع تحسينات كبيرة على أبعاد الأداء والكفاءة عبر مجموعة متنوعة من إعدادات التدريب. على سبيل المثال، في إعداد التدريب شبه المُشرف على LibriSpeech باستخدام 100 ساعة و960 ساعة، يحقق SEW تسريعًا في الاستدلال يبلغ 1.9 مرة مقارنة بـ Wav2vec 2.0، مع انخفاض نسبته 13.5% في معدل خطأ الكلمة. وبزمن استدلال مماثل، يقلل SEW من معدل خطأ الكلمة بنسبة 25-50% عبر أحجام نماذج مختلفة.*

تمت المساهمة بهذا النموذج من قبل [anton-l](https://huggingface.co/anton-l).

## نصائح الاستخدام

- SEW-D هو نموذج كلام يقبل مصفوفة أرقام عشرية مطابقة للموجة الصوتية الخام لإشارة الكلام.

- يتم ضبط دقة SEWDForCTC باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].

## الموارد

- [دليل مهام التصنيف الصوتي](../tasks/audio_classification)

- [دليل مهام التعرف التلقائي على الكلام (ASR)](../tasks/asr)

## SEWDConfig

[[autodoc]] SEWDConfig

## SEWDModel

[[autodoc]] SEWDModel

- forward

## SEWDForCTC

[[autodoc]] SEWDForCTC

- forward

## SEWDForSequenceClassification

[[autodoc]] SEWDForSequenceClassification

- forward