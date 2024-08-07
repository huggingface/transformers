# UniSpeech

## نظرة عامة

تم اقتراح نموذج UniSpeech في [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) بواسطة Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.

الملخص من الورقة هو ما يلي:

*في هذه الورقة، نقترح نهجًا موحدًا للتعلم المسبق يسمى UniSpeech لتعلم تمثيلات الكلام باستخدام كل من البيانات الموسومة وغير الموسومة، حيث يتم إجراء التعلم الصوتي CTC الإشرافي والتعلم الذاتي الخاضع للرقابة على أساس المقارنة بطريقة تعلم المهام المتعددة. يمكن للتمثيلات الناتجة التقاط معلومات أكثر ارتباطًا بالهياكل الصوتية وتحسين التعميم عبر اللغات والمجالات. نقيم فعالية UniSpeech لتعلم التمثيل متعدد اللغات على مجموعة CommonVoice العامة. وتُظهر النتائج أن UniSpeech يتفوق على التعلم المسبق للإشراف والتعلم التحويلي المُشرف للتعرف على الكلام بحد أقصى 13.4٪ و 17.8٪ من التخفيضات النسبية لمعدل خطأ الهاتف، على التوالي (المتوسط ​​عبر جميع لغات الاختبار). كما تم توضيح قابلية نقل UniSpeech في مهمة التعرف على الكلام ذات التحول النطاقي، أي انخفاض نسبي قدره 6٪ في معدل خطأ الكلمة مقارنة بالنهج السابق.*

تمت المساهمة بهذا النموذج من قبل [باتريكفونبلاتين] (https://huggingface.co/patrickvonplaten). يمكن العثور على كود المؤلفين [هنا] (https://github.com/microsoft/UniSpeech/tree/main/UniSpeech).

## نصائح الاستخدام

- UniSpeech هو نموذج كلام يقبل مصفوفة عائمة تتوافق مع الشكل الموجي الخام لإشارة الكلام. يرجى استخدام [`Wav2Vec2Processor`] لاستخراج الميزات.

- يمكن ضبط نموذج UniSpeech باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].

## الموارد

- [دليل مهام التصنيف الصوتي](../tasks/audio_classification)

- [دليل مهام التعرف التلقائي على الكلام (ASR)](../tasks/asr)

## UniSpeechConfig

[[autodoc]] UniSpeechConfig

## مخرجات UniSpeech المحددة

[[autodoc]] models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput

## UniSpeechModel

[[autodoc]] UniSpeechModel

- forward

## UniSpeechForCTC

[[autodoc]] UniSpeechForCTC

- forward

## UniSpeechForSequenceClassification

[[autodoc]] UniSpeechForSequenceClassification

- forward

## UniSpeechForPreTraining

[[autodoc]] UniSpeechForPreTraining

- forward