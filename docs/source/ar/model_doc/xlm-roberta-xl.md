# XLM-RoBERTa-XL

## نظرة عامة
اقترح نمان جويال، وجينغفي دو، وميل أوت، وجيري أنانثارامان، وأليكسيس كونو نموذج XLM-RoBERTa-XL في ورقتهم البحثية بعنوان "محولات ذات نطاق أكبر للنمذجة اللغوية المقنعة متعددة اللغات".

فيما يلي الملخص المستخرج من الورقة البحثية:

*أظهرت الدراسات الحديثة فعالية النمذجة اللغوية المقنعة عبر اللغات لفهم اللغات عبر اللغات. في هذه الدراسة، نقدم نتائج نموذجين مقنعين للغة متعددة اللغات أكبر، يحتويان على 3.5 مليار و 10.7 مليار معامل. يتفوق نموذجانا الجديدان، المسميان XLM-R XL و XLM-R XXL، على XLM-R بنسبة 1.8% و 2.4% على التوالي في متوسط الدقة على مجموعة بيانات XNLI. يتفوق نموذجنا أيضًا على نموذج RoBERTa-Large في عدة مهام باللغة الإنجليزية من معيار GLUE بنسبة 0.3% في المتوسط، في حين أنه يدعم 99 لغة إضافية. يشير هذا إلى أن النماذج الأولية ذات السعة الأكبر قد تحقق أداءً قويًا في اللغات ذات الموارد العالية مع تحسين اللغات منخفضة الموارد بشكل كبير. نجعل أكوادنا ونماذجنا متاحة للعموم*.

تمت المساهمة بهذا النموذج من قبل Soonhwan-Kwon و stefan-it. يمكن العثور على الكود الأصلي هنا.

## نصائح الاستخدام
XLM-RoBERTa-XL هو نموذج متعدد اللغات تم تدريبه على 100 لغة مختلفة. على عكس بعض النماذج متعددة اللغات XLM، فإنه لا يتطلب وسائد اللغة لتحديد اللغة المستخدمة، ويجب أن يكون قادرًا على تحديد اللغة الصحيحة من معرفات الإدخال.

## الموارد
- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## XLMRobertaXLConfig

[[autodoc]] XLMRobertaXLConfig

## XLMRobertaXLModel

[[autodoc]] XLMRobertaXLModel

- forward

## XLMRobertaXLForCausalLM

[[autodoc]] XLMRobertaXLForCausalLM

- forward

## XLMRobertaXLForMaskedLM

[[autodoc]] XLMRobertaXLForMaskedLM

- forward

## XLMRobertaXLForSequenceClassification

[[autodoc]] XLMRobertaXLForSequenceClassification

- forward

## XLMRobertaXLForMultipleChoice

[[autodoc]] XLMRobertaXLForMultipleChoice

- forward

## XLMRobertaXLForTokenClassification

[[autodoc]] XLMRobertaXLForTokenClassification

- forward

## XLMRobertaXLForQuestionAnswering

[[autodoc]] XLMRobertaXLForQuestionAnswering

- forward