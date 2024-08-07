# Data2Vec

## نظرة عامة

اقترح نموذج Data2Vec في ورقة بحثية بعنوان "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language" بواسطة Alexei Baevski و Wei-Ning Hsu و Qiantong Xu و Arun Babu و Jiatao Gu و Michael Auli.

يقترح Data2Vec إطار عمل موحد للتعلم الذاتي الإشراف عبر طرائق بيانات مختلفة - النص والصوت والصور. والأهم من ذلك، أن الأهداف المتوقعة للتعليم المسبق هي تمثيلات كامنة متخصصة في السياق للإدخالات، بدلاً من أهداف مستقلة عن السياق خاصة بالطرائق.

الملخص من الورقة هو ما يلي:

* "في حين أن الفكرة العامة للتعلم الذاتي الإشراف متطابقة عبر الطرائق، إلا أن الخوارزميات والأهداف الفعلية تختلف اختلافًا كبيرًا لأنها طُورت مع وضع طريقة واحدة في الاعتبار. وللاقتراب أكثر من التعلم الذاتي الإشراف العام، نقدم data2vec، وهو إطار عمل يستخدم نفس طريقة التعلم إما للكلام أو معالجة اللغات الطبيعية أو رؤية الكمبيوتر. وتتمثل الفكرة الأساسية في التنبؤ بالتمثيلات الكامنة للبيانات الكاملة للإدخال بناءً على عرض مقنع للإدخال في إعداد التقطير الذاتي باستخدام بنية Transformer القياسية.

بدلاً من التنبؤ بالأهداف الخاصة بالطرائق مثل الكلمات أو الرموز المرئية أو وحدات الكلام البشري التي تكون محلية في طبيعتها، يتنبأ نموذج data2vec بالتمثيلات الكامنة المتخصصة في السياق والتي تحتوي على معلومات من الإدخال بالكامل. وتظهر التجارب على المعايير المرجعية الرئيسية للتعرف على الكلام وتصنيف الصور وفهم اللغة الطبيعية حالة جديدة من الفن أو الأداء التنافسي للأساليب السائدة. النماذج والتعليمات البرمجية متاحة على www.github.com/pytorch/fairseq/tree/master/examples/data2vec.

هذه النماذج ساهم بها [edugp](https://huggingface.co/edugp) و [patrickvonplaten](https://huggingface.co/patrickvonplaten). وساهم [sayakpaul](https://github.com/sayakpaul) و [Rocketknight1](https://github.com/Rocketknight1) في Data2Vec للرؤية في TensorFlow.

يمكن العثور على التعليمات البرمجية الأصلية (للنص والكلام) هنا: [https://github.com/pytorch/fairseq/tree/main/examples/data2vec](https://github.com/pytorch/fairseq/tree/main/examples/data2vec).

يمكن العثور على التعليمات البرمجية الأصلية للرؤية هنا: [https://github.com/facebookresearch/data2vec_vision/tree/main/beit](https://github.com/facebookresearch/data2vec_vision/tree/main/beit).

## نصائح الاستخدام

- تم تدريب كل من Data2VecAudio و Data2VecText و Data2VecVision باستخدام نفس طريقة التعلم الذاتي.

- بالنسبة لـ Data2VecAudio، تكون المعالجة المسبقة مماثلة لـ [`Wav2Vec2Model`]، بما في ذلك استخراج الميزات.

- بالنسبة لـ Data2VecText، تكون المعالجة المسبقة مماثلة لـ [`RobertaModel`]، بما في ذلك التمييز.

- بالنسبة لـ Data2VecVision، تكون المعالجة المسبقة مماثلة لـ [`BeitModel`]، بما في ذلك استخراج الميزات.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام Data2Vec.

<PipelineTag pipeline="image-classification"/>

- [`Data2VecVisionForImageClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

- للضبط الدقيق لـ [`TFData2VecVisionForImageClassification`] على مجموعة بيانات مخصصة، راجع [دفتر الملاحظات](https://colab.research.google.com/github/sayakpaul/TF-2.0-Hacks/blob/master/data2vec_vision_image_classification.ipynb) هذا.

**موارد وثائق Data2VecText**

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)

- [دليل مهمة تصنيف الرموز](../tasks/token_classification)

- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)

- [دليل مهمة النمذجة اللغوية السببية](../tasks/language_modeling)

- [دليل مهمة النمذجة اللغوية المقنعة](../tasks/masked_language_modeling)

- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

**موارد وثائق Data2VecAudio**

- [دليل مهمة تصنيف الصوت](../tasks/audio_classification)

- [دليل مهمة التعرف التلقائي على الكلام](../tasks/asr)

**موارد وثائق Data2VecVision**

- [تصنيف الصور](../tasks/image_classification)

- [القطاعية الدلالية](../tasks/semantic_segmentation)

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب Pull Request وسنراجعه! ويفضل أن يوضح المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

## Data2VecTextConfig

[[autodoc]] Data2VecTextConfig

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig

## Data2VecVisionConfig

[[autodoc]] Data2VecVisionConfig

<frameworkcontent>
<pt>

## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel

- forward

## Data2VecAudioForAudioFrameClassification

[[autodoc]] Data2VecAudioForAudioFrameClassification

- forward

## Data2VecAudioForCTC

[[autodoc]] Data2VecAudioForCTC

- forward

## Data2VecAudioForSequenceClassification

[[autodoc]] Data2VecAudioForSequenceClassification

- forward

## Data2VecAudioForXVector


[[autodoc]] Data2VecAudioForXVector

- forward

## Data2VecTextModel

[[autodoc]] Data2VecTextModel

- forward

## Data2VecTextForCausalLM

[[autodoc]] Data2VecTextForCausalLM

- forward

## Data2VecTextForMaskedLM

[[autodoc]] Data2VecTextForMaskedLM

- forward

## Data2VecTextForSequenceClassification


[[autodoc]] Data2VecTextForSequenceClassification

- forward

## Data2VecTextForMultipleChoice

[[autodoc]] Data2VecTextForMultipleChoice

- forward

## Data2VecTextForTokenClassification

[[autodoc]] Data2VecTextForTokenClassification

- forward

## Data2VecTextForQuestionAnswering

[[autodoc]] Data2VecTextForQuestionAnswering

- forward

## Data2VecVisionModel

[[autodoc]] Data2VecVisionModel

- forward

## Data2VecVisionForImageClassification

[[autodoc]] Data2VecVisionForImageClassification

- forward

## Data2VecVisionForSemanticSegmentation

[[autodoc]] Data2VecVisionForSemanticSegmentation

- forward

</pt>
<tf>

## TFData2VecVisionModel

[[autodoc]] TFData2VecVisionModel

- call

## TFData2VecVisionForImageClassification

[[autodoc]] TFData2VecVisionForImageClassification

- call

## TFData2VecVisionForSemanticSegmentation

[[autodoc]] TFData2VecVisionForSemanticSegmentation

- call

</tf>
</frameworkcontent>