# CamemBERT

## نظرة عامة
تم اقتراح نموذج CamemBERT في ورقة بحثية بعنوان [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) بواسطة Louis Martin وآخرون. وهو مبني على نموذج RoBERTa الذي أصدرته شركة فيسبوك في عام 2019. وقد تم تدريب النموذج على 138 جيجابايت من النصوص باللغة الفرنسية.

ملخص الورقة البحثية هو كما يلي:

*تعتبر النماذج اللغوية المعالجة مسبقًا منتشرة الآن على نطاق واسع في معالجة اللغات الطبيعية. وعلى الرغم من نجاحها، إلا أن معظم النماذج المتاحة تم تدريبها إما على بيانات باللغة الإنجليزية أو على ربط بيانات من عدة لغات. وهذا يجعل الاستخدام العملي لمثل هذه النماذج -في جميع اللغات باستثناء الإنجليزية- محدودًا جدًا. بهدف معالجة هذه المشكلة بالنسبة للغة الفرنسية، نقدم CamemBERT، وهو إصدار باللغة الفرنسية من النماذج ثنائية الاتجاه للمتحولين (BERT). نقيس أداء CamemBERT مقارنة بالنماذج متعددة اللغات في مهام تدفق متعددة، وهي تحديد أجزاء الكلام، وتحليل الإعراب، وتعريف الكيانات المسماة، والاستدلال اللغوي الطبيعي. يحسن CamemBERT حالة الفن في معظم المهام التي تم أخذها في الاعتبار. نقوم بإطلاق النموذج المعالج مسبقًا لـ CamemBERT على أمل تعزيز البحث والتطبيقات اللاحقة لمعالجة اللغة الطبيعية باللغة الفرنسية.*

تمت المساهمة بهذا النموذج من قبل فريق ALMAnaCH (Inria). يمكن العثور على الكود الأصلي [هنا](https://camembert-model.fr/).

<Tip>

هذا التنفيذ هو نفسه في RoBERTa. راجع [توثيق RoBERTa](roberta) للحصول على أمثلة الاستخدام بالإضافة إلى المعلومات المتعلقة بالمدخلات والمخرجات.

</Tip>

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## CamembertConfig

[[autodoc]] CamembertConfig

## CamembertTokenizer

[[autodoc]] CamembertTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## CamembertTokenizerFast

[[autodoc]] CamembertTokenizerFast

<frameworkcontent>

<pt>

## CamembertModel

[[autodoc]] CamembertModel

## CamembertForCausalLM

[[autodoc]] CamembertForCausalLM

## CamembertForMaskedLM

[[autodoc]] CamembertForMaskedLM

## CamembertForSequenceClassification

[[autodoc]] CamembertForSequenceClassification

## CamembertForMultipleChoice

[[autodoc]] CamembertForMultipleChoice

## CamembertForTokenClassification

[[autodoc]] CamembertForTokenClassification

## CamembertForQuestionAnswering

[[autodoc]] CamembertForQuestionAnswering

</pt>

<tf>

## TFCamembertModel

[[autodoc]] TFCamembertModel

## TFCamembertForCausalLM

[[autodoc]] TFCamembertForCausalLM

## TFCamembertForMaskedLM

[[autodoc]] TFCamembertForMaskedLM

## TFCamembertForSequenceClassification

[[autodoc]] TFCamembertForSequenceClassification

## TFCamembertForMultipleChoice

[[autodoc]] TFCamembertForMultipleChoice

## TFCamembertForTokenClassification

[[autodoc]] TFCamembertForTokenClassification

## TFCamembertForQuestionAnswering

[[autodoc]] TFCamembertForQuestionAnswering

</tf>

</frameworkcontent>