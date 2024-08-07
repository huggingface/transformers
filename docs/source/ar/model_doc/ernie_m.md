# ErnieM

<Tip warning={true}>
يتم الاحتفاظ بهذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب (Pull Requests) جديدة لتغيير شفرته.
إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2.
يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.
</Tip>

## نظرة عامة
تم اقتراح نموذج ErnieM في الورقة البحثية [ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora](https://arxiv.org/abs/2012.15674) بواسطة Xuan Ouyang و Shuohuan Wang و Chao Pang و Yu Sun و Hao Tian و Hua Wu و Haifeng Wang.

ملخص الورقة البحثية هو كما يلي:

أظهرت الدراسات الحديثة أن النماذج متعددة اللغات المُدربة مسبقًا تحقق أداءً مثيرًا للإعجاب في المهام متعددة اللغات التنازلية. وتستفيد هذه التحسينات من تعلم كمية كبيرة من النصوص أحادية اللغة والنصوص المتوازية. وعلى الرغم من الاعتراف العام بأن النصوص المتوازية مهمة لتحسين أداء النموذج، إلا أن الطرق الحالية مقيدة غالبًا بحجم النصوص المتوازية، خاصة بالنسبة للغات منخفضة الموارد. في هذه الورقة، نقترح ERNIE-M، وهي طريقة تدريب جديدة تشجع النموذج على محاذاة تمثيل عدة لغات مع النصوص أحادية اللغة، للتغلب على القيود التي يفرضها حجم النصوص المتوازية على أداء النموذج. وتتمثل رؤيتنا الأساسية في دمج الترجمة من اللغة الأخرى إلى اللغة الأصلية في عملية التدريب المسبق. نقوم بتوليد أزواج من الجمل المتوازية الزائفة على نص أحادي اللغة لتمكين تعلم المحاذاة الدلالية بين اللغات المختلفة، وبالتالي تعزيز النمذجة الدلالية للنماذج متعددة اللغات. وتظهر النتائج التجريبية أن ERNIE-M يتفوق على النماذج متعددة اللغات الحالية ويحقق نتائج جديدة رائدة في مختلف المهام التنازلية متعددة اللغات.

تمت المساهمة بهذا النموذج من قبل [Susnato Dhar](https://huggingface.co/susnato). يمكن العثور على الشفرة الأصلية [هنا](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_m).

## نصائح الاستخدام

- Ernie-M هو نموذج مشابه لـ BERT، وبالتالي فهو عبارة عن مكدس لترميز المحول Transformer Encoder.
- بدلاً من استخدام MaskedLM للمرحلة السابقة للتدريب (كما هو الحال في BERT)، استخدم المؤلفون تقنيتين جديدتين: `Cross-attention Masked Language Modeling` و `Back-translation Masked Language Modeling`. حاليًا، لم يتم تنفيذ هذين الهدفين من أهداف LMHead هنا.
- إنه نموذج لغوي متعدد اللغات.
- لم يتم استخدام التنبؤ بالجملة التالية في عملية التدريب المسبق.

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف العلامات](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## ErnieMConfig

[[autodoc]] ErnieMConfig

## ErnieMTokenizer

[[autodoc]] ErnieMTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## ErnieMModel

[[autodoc]] ErnieMModel

- forward

## ErnieMForSequenceClassification

[[autodoc]] ErnieMForSequenceClassification

- forward

## ErnieMForMultipleChoice

[[autodoc]] ErnieMForMultipleChoice

- forward

## ErnieMForTokenClassification

[[autodoc]] ErnieMForTokenClassification

- forward

## ErnieMForQuestionAnswering

[[autodoc]] ErnieMForQuestionAnswering

- forward

## ErnieMForInformationExtraction

[[autodoc]] ErnieMForInformationExtraction

- forward