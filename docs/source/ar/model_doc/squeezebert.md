# SqueezeBERT

## نظرة عامة

اقترح نموذج SqueezeBERT في [SqueezeBERT: ماذا يمكن لمعالجة اللغات الطبيعية أن تتعلم من رؤية الكمبيوتر حول كفاءة الشبكات العصبية؟](https://arxiv.org/abs/2006.11316) بواسطة Forrest N. Iandola و Albert E. Shaw و Ravi Krishna و Kurt W. Keutzer. إنه محول ثنائي الاتجاه مشابه لنموذج BERT. الفرق الرئيسي بين بنية BERT وبنية SqueezeBERT هو أن SqueezeBERT يستخدم [التقسيمات المجمعة](https://blog.yani.io/filter-group-tutorial) بدلاً من الطبقات المتصلة بالكامل لطبقات Q و K و V و FFN.

الملخص من الورقة هو ما يلي:

*يقرأ البشر ويكتبون مئات المليارات من الرسائل كل يوم. علاوة على ذلك، وبفضل توفر مجموعات البيانات الكبيرة وأنظمة الحوسبة والنماذج الأفضل للشبكات العصبية، حققت تقنية معالجة اللغات الطبيعية (NLP) تقدمًا كبيرًا في فهم هذه الرسائل وتدقيقها وإدارتها. وبالتالي، هناك فرصة كبيرة لنشر NLP في العديد من التطبيقات لمساعدة مستخدمي الويب والشبكات الاجتماعية والشركات. ونحن نعتبر الهواتف الذكية والأجهزة المحمولة الأخرى منصات أساسية لنشر نماذج NLP على نطاق واسع. ومع ذلك، فإن نماذج الشبكات العصبية NLP عالية الدقة اليوم مثل BERT و RoBERTa مكلفة للغاية من الناحية الحسابية، حيث تستغرق BERT-base 1.7 ثانية لتصنيف مقتطف نصي على هاتف Pixel 3 الذكي. في هذا العمل، نلاحظ أن الأساليب مثل التقسيمات المجمعة حققت تسريعًا كبيرًا لشبكات رؤية الكمبيوتر، ولكن العديد من هذه التقنيات لم يعتمد من قبل مصممي الشبكات العصبية NLP. نحن نوضح كيفية استبدال العديد من العمليات في طبقات الاهتمام الذاتي بالتقسيمات المجمعة، ونستخدم هذه التقنية في بنية شبكة جديدة تسمى SqueezeBERT، والتي تعمل بشكل أسرع 4.3x من BERT-base على Pixel 3 مع تحقيق دقة تنافسية على مجموعة اختبار GLUE. سيتم إصدار كود SqueezeBERT.*

تمت المساهمة بهذا النموذج من قبل [forresti](https://huggingface.co/forresti).

## نصائح الاستخدام

- يستخدم SqueezeBERT تضمين الموضع المطلق، لذلك يُنصح عادةً بإضافة حشو إلى الإدخالات من اليمين بدلاً من اليسار.

- يشبه SqueezeBERT نموذج BERT، وبالتالي يعتمد على هدف نمذجة اللغة المقنعة (MLM). لذلك، فهو فعال في التنبؤ بالرموز المميزة المقنعة وفي فهم اللغة الطبيعية بشكل عام، ولكنه ليس الأمثل لتوليد النصوص. النماذج التي تم تدريبها بهدف نمذجة اللغة السببية (CLM) أفضل في هذا الصدد.

- للحصول على أفضل النتائج عند الضبط الدقيق لمهام تصنيف التسلسل، يوصى بالبدء من نقطة التحقق *squeezebert/squeezebert-mnli-headless*.

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)

- [دليل مهام تصنيف الرموز](../tasks/token_classification)

- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)

- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)

- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## SqueezeBertConfig

[[autodoc]] SqueezeBertConfig

## SqueezeBertTokenizer

[[autodoc]] SqueezeBertTokenizer

- build_inputs_with_special_tokens

- get_special_tokens_mask

- create_token_type_ids_from_sequences

- save_vocabulary

## SqueezeBertTokenizerFast

[[autodoc]] SqueezeBertTokenizerFast

## SqueezeBertModel

[[autodoc]] SqueezeBertModel

## SqueezeBertForMaskedLM

[[autodoc]] SqueezeBertForMaskedLM

## SqueezeBertForSequenceClassification

[[autodoc]] SqueezeBertForSequenceClassification

## SqueezeBertForMultipleChoice

[[autodoc]] SqueezeBertForMultipleChoice

## SqueezeBertForTokenClassification

[[autodoc]] SqueezeBertForTokenClassification

## SqueezeBertForQuestionAnswering

[[autodoc]] SqueezeBertForQuestionAnswering