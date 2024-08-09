# MobileBERT

## نظرة عامة

اقترح نموذج MobileBERT في ورقة بحثية بعنوان "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices" بواسطة Zhiqing Sun وآخرون. إنه محول ثنائي الاتجاه يعتمد على نموذج BERT، والذي تم ضغطه وتسريعه باستخدام عدة طرق.

الملخص من الورقة البحثية هو كما يلي:

*حققت معالجة اللغة الطبيعية (NLP) مؤخرًا نجاحًا كبيرًا من خلال استخدام نماذج ضخمة مسبقة التدريب تحتوي على مئات الملايين من المعلمات. ومع ذلك، تعاني هذه النماذج من أحجام ضخمة ووقت استجابة مرتفع، مما يجعل من الصعب نشرها على الأجهزة المحمولة المحدودة الموارد. في هذه الورقة، نقترح MobileBERT لضغط وتسريع نموذج BERT الشهير. مثل BERT الأصلي، فإن MobileBERT لا يعتمد على المهام، أي أنه يمكن تطبيقه بشكل عام على مختلف مهام NLP اللاحقة من خلال الضبط الدقيق البسيط. بشكل أساسي، MobileBERT هو إصدار مضغوط من BERT_LARGE، ولكنه مزود بهياكل عنق الزجاجة وتوازن مدروس بعناية بين الاهتمامات الذاتية والشبكات الأمامية الخلفية. لتدريب MobileBERT، نقوم أولاً بتدريب نموذج المعلم المصمم خصيصًا، وهو نموذج BERT_LARGE المدمج بهيكل عنق الزجاجة المعكوس. بعد ذلك، نقوم بنقل المعرفة من هذا المعلم إلى MobileBERT. تظهر الدراسات التجريبية أن MobileBERT أصغر 4.3 مرة وأسرع 5.5 مرة من BERT_BASE مع تحقيق نتائج تنافسية على معايير معروفة جيدًا. في مهام الاستدلال اللغوي الطبيعي لـ GLUE، يحقق MobileBERT نتيجة GLUEscore تبلغ 77.7 (أقل بـ 0.6 من BERT_BASE)، و62 مللي ثانية من وقت الاستجابة على هاتف Pixel 4. في مهمة SQuAD v1.1/v2.0 للإجابة على الأسئلة، يحقق MobileBERT نتيجة 90.0/79.2 على مجموعة البيانات الاختبارية (أعلى بـ 1.5/2.1 من BERT_BASE).*

تمت المساهمة بهذا النموذج بواسطة [vshampor](https://huggingface.co/vshampor). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/google-research/tree/master/mobilebert).

## نصائح الاستخدام

- MobileBERT هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بإضافة حشو إلى المدخلات من اليمين بدلاً من اليسار.

- يشبه MobileBERT نموذج BERT، وبالتالي يعتمد على هدف نمذجة اللغة المقنعة (MLM). لذلك، فهو فعال في التنبؤ بالرموز المقنعة وفي فهم اللغة الطبيعية بشكل عام، ولكنه ليس الأمثل لتوليد النصوص. النماذج التي تم تدريبها بهدف نمذجة اللغة السببية (CLM) أفضل في هذا الصدد.

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار من متعدد](../tasks/multiple_choice)

## MobileBertConfig

[[autodoc]] MobileBertConfig

## MobileBertTokenizer

[[autodoc]] MobileBertTokenizer

## MobileBertTokenizerFast

[[autodoc]] MobileBertTokenizerFast

## مخرجات MobileBert المحددة

[[autodoc]] models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput

[[autodoc]] models.mobilebert.modeling_tf_mobilebert.TFMobileBertForPreTrainingOutput

<frameworkcontent>
<pt>

## MobileBertModel

[[autodoc]] MobileBertModel

- forward

## MobileBertForPreTraining

[[autodoc]] MobileBertForPreTraining

- forward

## MobileBertForMaskedLM

[[autodoc]] MobileBertForMaskedLM

- forward

## MobileBertForNextSentencePrediction

[[autodoc]] MobileBertForNextSentencePrediction

- forward

## MobileBertForSequenceClassification

[[autodoc]] MobileBertForSequenceClassification

- forward

## MobileBertForMultipleChoice

[[autodoc]] MobileBertForMultipleChoice

- forward

## MobileBertForTokenClassification

[[autodoc]] MobileBertForTokenClassification

- forward

## MobileBertForQuestionAnswering

[[autodoc]] MobileBertForQuestionAnswering

- forward

</pt>
<tf>

## TFMobileBertModel

[[autodoc]] TFMobileBertModel

- call

## TFMobileBertForPreTraining

[[autodoc]] TFMobileBertForPreTraining

- call

## TFMobileBertForMaskedLM

[[autodoc]] TFMobileBertForMaskedLM

- call

## TFMobileBertForNextSentencePrediction

[[autodoc]] TFMobileBertForNextSentencePrediction

- call

## TFMobileBertForSequenceClassification


[[autodoc]] TFMobileBertForSequenceClassification

- call

## TFMobileBertForMultipleChoice


[[autodoc]] TFMobileBertForMultipleChoice

- call

## TFMobileBertForTokenClassification

[[autodoc]] TFMobileBertForTokenClassification

- call

## TFMobileBertForQuestionAnswering

[[autodoc]] TFMobileBertForQuestionAnswering

- call

</tf>

</frameworkcontent>