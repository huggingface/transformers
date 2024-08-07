# UniSpeech-SAT

## نظرة عامة

تم اقتراح نموذج UniSpeech-SAT في ورقة بحثية بعنوان "UniSpeech-SAT: Universal Speech Representation Learning with Speaker Aware Pre-Training" بواسطة Sanyuan Chen وآخرون.

ملخص الورقة البحثية هو كما يلي:

*"يعد التعلم الذاتي الخاضع للإشراف (SSL) هدفًا طويل الأمد لمعالجة الكلام، حيث أنه يستخدم بيانات غير معلمة على نطاق واسع ويتجنب التصنيف البشري المكثف. شهدت السنوات الأخيرة نجاحات كبيرة في تطبيق التعلم الذاتي الخاضع للإشراف في التعرف على الكلام، في حين لم يتم سوى القليل من الاستكشاف في تطبيق SSL لنمذجة الخصائص الصوتية للمتكلم. في هذه الورقة، نهدف إلى تحسين إطار عمل SSL الحالي لتعلم تمثيل المتحدث. تم تقديم طريقتين لتعزيز استخراج معلومات المتحدث غير الخاضعة للإشراف. أولاً، نطبق التعلم متعدد المهام على إطار عمل SSL الحالي، حيث نقوم بدمج الخسارة التمييزية على مستوى العبارة مع دالة الهدف SSL. ثانيًا، من أجل التمييز بشكل أفضل بين المتحدثين، نقترح استراتيجية خلط العبارات لزيادة البيانات، حيث يتم إنشاء عبارات متداخلة إضافية دون إشراف ويتم دمجها أثناء التدريب. قمنا بدمج الأساليب المقترحة في إطار عمل HuBERT. أظهرت نتائج التجارب على معيار SUPERB أن النظام المقترح يحقق أداءً متميزًا في تعلم التمثيل العام، خاصة في المهام الموجهة نحو تحديد هوية المتحدث. تم إجراء دراسة لتحليل الأجزاء المكونة للتحقق من فعالية كل طريقة مقترحة. وأخيرًا، قمنا بزيادة حجم مجموعة البيانات التدريبية إلى 94 ألف ساعة من البيانات الصوتية العامة وحققنا تحسنًا إضافيًا في الأداء في جميع مهام SUPERB."*

تمت المساهمة بهذا النموذج من قبل [patrickvonplaten]. يمكن العثور على الكود الخاص بالمؤلفين [هنا](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT).

## نصائح الاستخدام

- UniSpeechSat هو نموذج كلام يقبل مصفوفة أرقام عائمة مطابقة للموجة الصوتية الخام للإشارة الصوتية. يرجى استخدام [`Wav2Vec2Processor`] لاستخراج الميزات.

- يمكن ضبط نموذج UniSpeechSat باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].

- يعمل نموذج UniSpeechSat بشكل جيد بشكل خاص في مهام التحقق من المتحدث وتحديد هوية المتحدث وفصل المتحدث.

## الموارد

- [دليل مهام التصنيف الصوتي](../tasks/audio_classification)

- [دليل مهام التعرف التلقائي على الكلام](../tasks/asr)

## UniSpeechSatConfig

[[autodoc]] UniSpeechSatConfig

## المخرجات الخاصة بنموذج UniSpeechSat

[[autodoc]] models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput

## نموذج UniSpeechSat

[[autodoc]] UniSpeechSatModel

- forward

## UniSpeechSatForCTC

[[autodoc]] UniSpeechSatForCTC

- forward

## UniSpeechSatForSequenceClassification

[[autodoc]] UniSpeechSatForSequenceClassification

- forward

## UniSpeechSatForAudioFrameClassification

[[autodoc]] UniSpeechSatForAudioFrameClassification

- forward

## UniSpeechSatForXVector

[[autodoc]] UniSpeechSatForXVector

- forward

## UniSpeechSatForPreTraining

[[autodoc]] UniSpeechSatForPreTraining

- forward