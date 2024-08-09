# Nyströmformer

## نظرة عامة

تم اقتراح نموذج Nyströmformer في [*Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*](https://arxiv.org/abs/2102.03902) بواسطة Yunyang Xiong وآخرون.

مقدمة الورقة البحثية هي كما يلي:

* برزت محولات الطاقة كأداة قوية لمجموعة واسعة من مهام معالجة اللغة الطبيعية. تعد آلية الاهتمام الذاتي أحد المكونات الرئيسية التي تقود الأداء المتميز لمحولات الطاقة، حيث تقوم بتشفير تأثير اعتماد الرموز الأخرى على كل رمز محدد. على الرغم من الفوائد، إلا أن التعقيد التربيعي للاهتمام الذاتي على طول تسلسل الإدخال حد من تطبيقه على التسلسلات الأطول - وهو موضوع تتم دراسته بنشاط في المجتمع. لمعالجة هذا القيد، نقترح Nyströmformer - وهو نموذج يظهر قابلية توسع مواتية كدالة لطول التسلسل. تقوم فكرتنا على تكييف طريقة Nyström لتقريب الاهتمام الذاتي القياسي بتعقيد O(n). تمكّن قابلية توسع Nyströmformer من تطبيقها على تسلسلات أطول تحتوي على آلاف الرموز. نقوم بتقييمات لمهام متعددة لأسفل على معيار GLUE ومراجعات IMDB بطول تسلسل قياسي، ونجد أن أداء Nyströmformer لدينا مماثل، أو في بعض الحالات، أفضل قليلاً من الاهتمام الذاتي القياسي. في مهام تسلسل طويل في معيار Long Range Arena (LRA)، يعمل Nyströmformer بشكل موات نسبيًا مقارنة بطرق الاهتمام الذاتي الأخرى. الكود الخاص بنا متاح على هذا الرابط https.*

تمت المساهمة بهذا النموذج من قبل [novice03](https://huggingface.co/novice03). يمكن العثور على الكود الأصلي [هنا](https://github.com/mlpen/Nystromformer).

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## NystromformerConfig

[[autodoc]] NystromformerConfig

## NystromformerModel

[[autodoc]] NystromformerModel

- forward

## NystromformerForMaskedLM

[[autodoc]] NystromformerForMaskedLM

- forward

## NystromformerForSequenceClassification

[[autodoc]] NystromformerForSequenceClassification

- forward

## NystromformerForMultipleChoice

[[autodoc]] NystromformerForMultipleChoice

- forward

## NystromformerForTokenClassification

[[autodoc]] NystromformerForTokenClassification

- forward

## NystromformerForQuestionAnswering

[[autodoc]] NystromformerForQuestionAnswering

- forward