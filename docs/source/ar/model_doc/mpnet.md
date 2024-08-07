# MPNet

## نظرة عامة
اقترح نموذج MPNet في ورقة "MPNet: Masked and Permuted Pre-training for Language Understanding" بواسطة كايتاو سونغ، و شو تان، و تاو تشين، و جيانفينغ لو، و تاي-يان ليو.

يعتمد MPNet طريقةً جديدةً للتعلم المسبق، تسمى نمذجة اللغة المقنعة والمرتبة، للاستفادة من مزايا نمذجة اللغة المقنعة والمرتبة لفهم اللغة الطبيعية.

ملخص الورقة هو كما يلي:

> "يعتمد BERT نمذجة اللغة المقنعة (MLM) للتعلم المسبق وهو أحد أكثر نماذج التعلم المسبق نجاحًا. نظرًا لأن BERT يتجاهل الاعتماد المتبادل بين الرموز المتوقعة، فقد قدم XLNet نمذجة اللغة المرتبة (PLM) للتعلم المسبق لمعالجة هذه المشكلة. ومع ذلك، لا يستفيد XLNet من المعلومات الكاملة لموضع الجملة، وبالتالي يعاني من عدم الاتساق في الموضع بين التعلم المسبق والضبط الدقيق. في هذه الورقة، نقترح MPNet، وهي طريقة تعلم مسبق جديدة تستفيد من مزايا BERT و XLNet وتتجنب قيودها. يستفيد MPNet من الاعتماد المتبادل بين الرموز المتوقعة من خلال نمذجة اللغة المرتبة (مقابل MLM في BERT)، ويأخذ معلومات الموضع المساعدة كإدخال لجعل النموذج يرى جملة كاملة، وبالتالي تقليل عدم الاتساق في الموضع (مقابل PLM في XLNet). نقوم بتدريب MPNet مسبقًا على مجموعة بيانات واسعة النطاق (أكثر من 160 جيجابايت من نصوص الفيلق) ونضبطها بشكل دقيق على مجموعة متنوعة من المهام السفلية (مثل GLUE و SQuAD، إلخ). تُظهر النتائج التجريبية أن MPNet يتفوق على MLM و PLM بهامش كبير، ويحقق نتائج أفضل في هذه المهام مقارنة بطرق التعلم المسبق السابقة (مثل BERT و XLNet و RoBERTa) في ظل نفس إعداد النموذج."

يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/MPNet).

## نصائح الاستخدام
لا يحتوي MPNet على `token_type_ids`، لذا لا تحتاج إلى الإشارة إلى الرمز الذي ينتمي إلى الجزء. ما عليك سوى فصل أجزاءك باستخدام رمز الفصل `tokenizer.sep_token` (أو `[sep]`).

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## MPNetConfig

[[autodoc]] MPNetConfig

## MPNetTokenizer

[[autodoc]] MPNetTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## MPNetTokenizerFast

[[autodoc]] MPNetTokenizerFast

<frameworkcontent>
<pt>

## MPNetModel

[[autodoc]] MPNetModel

- forward

## MPNetForMaskedLM

[[autodoc]] MPNetForMaskedLM

- forward

## MPNetForSequenceClassification

[[autodoc]] MPNetForSequenceClassification

- forward

## MPNetForMultipleChoice

[[autodoc]] MPNetForMultipleChoice

- forward

## MPNetForTokenClassification

[[autodoc]] MPNetForTokenClassification

- forward

## MPNetForQuestionAnswering

[[autodoc]] MPNetForQuestionAnswering

- forward

</pt>
<tf>

## TFMPNetModel

[[autodoc]] TFMPNetModel

- call

## TFMPNetForMaskedLM

[[autodoc]] TFMPNetForMaskedLM

- call

## TFMPNetForSequenceClassification

[[autodoc]] TFMPNetForSequenceClassification

- call

## TFMPNetForMultipleChoice

[[autodoc]] TFMPNetForMultipleChoice

- call

## TFMPNetForTokenClassification

[[autodoc]] TFMPNetForTokenClassification

- call

## TFMPNetForQuestionAnswering

[[autodoc]] TFMPNetForQuestionAnswering

- call

</tf>

</frameworkcontent>