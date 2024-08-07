# WavLM

## نظرة عامة

تم اقتراح نموذج WavLM في [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) بواسطة Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei.

الملخص من الورقة هو ما يلي:

*يحقق التعلم الذاتي الإشراف (SSL) نجاحًا كبيرًا في التعرف على الكلام، في حين لم يتم سوى القليل من الاستكشاف لمهام معالجة الكلام الأخرى. نظرًا لأن إشارة الكلام تحتوي على معلومات متعددة الجوانب بما في ذلك هوية المتحدث، والسمات اللغوية، والمحتوى المنطوق، وما إلى ذلك، فإن تعلم التمثيلات الشاملة لجميع مهام الكلام أمر صعب. في هذه الورقة، نقترح نموذجًا جديدًا مسبق التدريب، يسمى WavLM، لحل مهام الكلام النهائية الكاملة. تم بناء WavLM بناءً على إطار HuBERT، مع التركيز على كل من نمذجة المحتوى المنطوق والحفاظ على هوية المتحدث. أولاً، نقوم بتزويد بنية Transformer بانحياز الموضع النسبي المبوب لتحسين قدرته على مهام التعرف. من أجل التمييز بشكل أفضل بين المتحدثين، نقترح استراتيجية تدريب خلط العبارات، حيث يتم إنشاء عبارات متداخلة إضافية دون إشراف وتضمينها أثناء تدريب النموذج. أخيرًا، نقوم بزيادة حجم مجموعة البيانات التدريبية من 60 ألف ساعة إلى 94 ألف ساعة. يحقق WavLM Large أداءً متميزًا في معيار SUPERB، ويحقق تحسينات كبيرة لمهام معالجة الكلام المختلفة في معاييرها التمثيلية.*

يمكن العثور على نقاط التفتيش ذات الصلة في https://huggingface.co/models?other=wavlm.

تمت المساهمة بهذا النموذج من قبل [patrickvonplaten](https://huggingface.co/patrickvonplaten). يمكن العثور على كود المؤلفين [هنا](https://github.com/microsoft/unilm/tree/master/wavlm).

## نصائح الاستخدام

- WavLM هو نموذج كلام يقبل مصفوفة عائمة تتوافق مع الشكل الموجي الخام لإشارة الكلام. يرجى استخدام [`Wav2Vec2Processor`] لاستخراج الميزات.
- يمكن ضبط نموذج WavLM باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].
- يعمل WavLM بشكل جيد بشكل خاص في مهام التحقق من المتحدث، وتحديد هوية المتحدث، وفصل المتحدث.

## الموارد

- [دليل مهام تصنيف الصوت](../tasks/audio_classification)
- [دليل مهام التعرف على الكلام التلقائي](../tasks/asr)

## WavLMConfig

[[autodoc]] WavLMConfig

## WavLMModel

[[autodoc]] WavLMModel

- forward

## WavLMForCTC

[[autodoc]] WavLMForCTC

- forward

## WavLMForSequenceClassification

[[autodoc]] WavLMForSequenceClassification

- forward

## WavLMForAudioFrameClassification

[[autodoc]] WavLMForAudioFrameClassification

- forward

## WavLMForXVector

[[autodoc]] WavLMForXVector

- forward