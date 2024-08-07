# RoCBert

## نظرة عامة

اقترح نموذج RoCBert في "RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining" بواسطة HuiSu و WeiweiShi و XiaoyuShen و XiaoZhou و TuoJi و JiaruiFang و JieZhou. وهو نموذج لغة صيني مسبق التدريب يتميز بالمتانة في مواجهة مختلف أشكال الهجمات الضارة.

فيما يلي الملخص المستخرج من الورقة البحثية:

*حققت نماذج اللغة كبيرة الحجم والمدربة مسبقًا نتائج متميزة في مهام معالجة اللغة الطبيعية. ومع ذلك، فقد ثبت أنها عرضة للهجمات الضارة، خاصة بالنسبة للغات المقطعية مثل الصينية. في هذا العمل، نقترح ROCBERT: نموذج Bert صيني مسبق التدريب يتميز بالمتانة في مواجهة مختلف أشكال الهجمات الضارة مثل اضطراب الكلمات، والمترادفات، والأخطاء الإملائية، وما إلى ذلك. تم تدريب النموذج مسبقًا باستخدام هدف التعلم التمييزي الذي يزيد من اتساق التصنيف تحت أمثلة ضارة مختلفة تم توليفها. يأخذ النموذج كمدخلات معلومات متعددة الوسائط تشمل السمات الدلالية والمقاطع البصرية. ونحن نثبت أهمية جميع هذه الميزات لمتانة النموذج، حيث يمكن تنفيذ الهجوم بجميع الأشكال الثلاثة. يفوق أداء ROCBERT الخطوط الأساسية القوية في 5 مهام فهم اللغة الصينية في ظل ثلاثة خوارزميات ضارة دون التضحية بالأداء على مجموعة الاختبار النظيفة. كما أنه يحقق أفضل أداء في مهمة اكتشاف المحتوى الضار في ظل الهجمات التي يصنعها الإنسان.*

تمت المساهمة بهذا النموذج من قبل [weiweishi](https://huggingface.co/weiweishi).

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام النمذجة اللغوية السببية](../tasks/language_modeling)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## RoCBertConfig

[[autodoc]] RoCBertConfig

- all

## RoCBertTokenizer

[[autodoc]] RoCBertTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## RoCBertModel

[[autodoc]] RoCBertModel

- forward

## RoCBertForPreTraining

[[autodoc]] RoCBertForPreTraining

- forward

## RoCBertForCausalLM

[[autodoc]] RoCBertForCausalLM

- forward

## RoCBertForMaskedLM

[[autodoc]] RoCBertForMaskedLM

- forward

## RoCBertForSequenceClassification

[[autodoc]] transformers.RoCBertForSequenceClassification

- forward

## RoCBertForMultipleChoice

[[autodoc]] transformers.RoCBertForMultipleChoice

- forward

## RoCBertForTokenClassification

[[autodoc]] transformers.RoCBertForTokenClassification

- forward

## RoCBertForQuestionAnswering

[[autodoc]] RoCBertForQuestionAnswering

- forward