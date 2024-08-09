# XLM

## نظرة عامة

اقترح نموذج XLM في [التدريب المسبق للنموذج اللغوي متعدد اللغات](https://arxiv.org/abs/1901.07291) بواسطة غيوم لامبل وأليكسيس كونو. إنه محول تم تدريبه مسبقًا باستخدام أحد الأهداف التالية:

- هدف نمذجة اللغة السببية (CLM) (تنبؤ الرمز التالي)
- هدف نمذجة اللغة المقنعة (MLM) (على غرار BERT)
- هدف نمذجة اللغة الترجمية (TLM) (امتداد لنمذجة اللغة المقنعة BERT لمدخلات اللغات المتعددة)

الملخص من الورقة هو ما يلي:

*أظهرت الدراسات الحديثة كفاءة التدريب المسبق التوليدي لفهم اللغة الإنجليزية الطبيعية. في هذا العمل، نقوم بتوسيع هذا النهج ليشمل لغات متعددة ونظهر فعالية التدريب المسبق متعدد اللغات. نقترح طريقتين لتعلم نماذج اللغة متعددة اللغات (XLMs): واحدة غير خاضعة للإشراف تعتمد فقط على البيانات أحادية اللغة، وأخرى خاضعة للإشراف تستفيد من البيانات المتوازية مع هدف نموذج اللغة متعدد اللغات الجديد. نحصل على نتائج متقدمة في التصنيف متعدد اللغات، والترجمة الآلية غير الخاضعة للإشراف والخاضعة للإشراف. في XNLI، يدفع نهجنا حالة الفن بمكسب مطلق قدره 4.9% من الدقة. في الترجمة الآلية غير الخاضعة للإشراف، نحصل على 34.3 BLEU على WMT'16 German-English، مما يحسن حالة الفن السابقة بأكثر من 9 BLEU. في الترجمة الآلية الخاضعة للإشراف، نحصل على حالة فنية جديدة تبلغ 38.5 BLEU على WMT'16 Romanian-English، متجاوزة بذلك أفضل نهج سابق بأكثر من 4 BLEU. سيتم إتاحة التعليمات البرمجية الخاصة بنا والنماذج التي تم تدريبها مسبقًا للجمهور.*

تمت المساهمة بهذا النموذج من قبل [thomwolf](https://huggingface.co/thomwolf). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/XLM/).

## نصائح الاستخدام

- يحتوي XLM على العديد من نقاط التفتيش، التي تم تدريبها باستخدام أهداف مختلفة: CLM أو MLM أو TLM. تأكد من تحديد الهدف الصحيح لمهمتك (على سبيل المثال، نقاط تفتيش MLM غير مناسبة للجيل).

- يحتوي XLM على نقاط تفتيش متعددة اللغات والتي تستفيد من معلمة `lang` محددة. تحقق من الصفحة [متعددة اللغات](../multilingual) لمزيد من المعلومات.

- نموذج محول تم تدريبه على عدة لغات. هناك ثلاثة أنواع مختلفة من التدريب لهذا النموذج ويوفر المكتبة نقاط تفتيش لكل منها:

1. نمذجة اللغة السببية (CLM) والتي تعد تدريبًا تلقائيًا تقليديًا (لذلك يمكن أن يكون هذا النموذج في القسم السابق أيضًا). يتم اختيار إحدى اللغات لكل عينة تدريب، ومدخل النموذج هو عبارة عن 256 رمزًا، والتي قد تمتد عبر عدة وثائق بإحدى هذه اللغات.

2. نمذجة اللغة المقنعة (MLM) والتي تشبه RoBERTa. يتم اختيار إحدى اللغات لكل عينة تدريب، ومدخل النموذج هو عبارة عن 256 رمزًا، والتي قد تمتد عبر عدة وثائق بإحدى هذه اللغات، مع التعتيم الديناميكي للرموز.

3. مزيج من MLM و TLM. يتكون ذلك من ضم جملة في لغتين مختلفتين، مع التعتيم العشوائي. للتنبؤ بأحد الرموز المقنعة، يمكن للنموذج استخدام كل من السياق المحيط باللغة 1 والسياق الذي توفره اللغة 2.

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## تكوين XLM

[[autodoc]] XLMConfig

## محلل ترميز XLM

[[autodoc]] XLMTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## مخرجات XLM المحددة

[[autodoc]] models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput

<frameworkcontent>
<pt>

## نموذج XLM

[[autodoc]] XLMModel

- forward

## XLMWithLMHeadModel

[[autodoc]] XLMWithLMHeadModel

- forward

## XLMForSequenceClassification

[[autodoc]] XLMForSequenceClassification

- forward

## XLMForMultipleChoice

[[autodoc]] XLMForMultipleChoice

- forward

## XLMForTokenClassification

[[autodoc]] XLMForTokenClassification

- forward

## XLMForQuestionAnsweringSimple

[[autodoc]] XLMForQuestionAnsweringSimple

- forward

## XLMForQuestionAnswering

[[autodoc]] XLMForQuestionAnswering

- forward

</pt>
<tf>

## TFXLMModel

[[autodoc]] TFXLMModel

- call

## TFXLMWithLMHeadModel

[[autodoc]] TFXLMWithLMHeadModel

- call

## TFXLMForSequenceClassification

[[autodoc]] TFXLMForSequenceClassification

- call

## TFXLMForMultipleChoice

[[autodoc]] TFXLMForMultipleChoice

- call

## TFXLMForTokenClassification

[[autodoc]] TFXLMForTokenClassification

- call

## TFXLMForQuestionAnsweringSimple

[[autodoc]] TFXLMForQuestionAnsweringSimple

- call

</tf>
</frameworkcontent>